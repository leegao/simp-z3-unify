from .types import BOOL, INT, STR
from .ast_nodes import (
    Assign, BinOp, Const, FieldGet, FieldSet, FuncCall, FuncDecl, IfStmt,
    IteExpr, LetDecl, Program, ReturnStmt, StrConcat, StrLen, StrSubstr,
    StructDecl, StructInit, UnOp, Var, WhileStmt, typecheck_program,
)
from .passes import buggy_opt_pass, const_fold_pass, cse_pass, dce_pass
from .symexec import SymbolicExecutor
from .unifier import verify_equiv_bu_wlau


def build_self_hosted_simp_parser() -> Program:
    """
    A self-hosted s-imp lexer & expression/statement parser written IN s-imp,
    using native `str` (Z3 StringSort) and `struct` (Z3 Datatype):
      - struct Token { kind: int, lexeme: str, next_pos: int }
      - struct ParseNode { op_kind: int, lhs_tok: str, rhs_tok: str, next_pos: int, ok: bool }
      - fn next_token(src: str, pos: int) -> Token
      - fn parse_assign_expr(src: str) -> ParseNode
    Token kinds: 0=EOF, 1=LET, 2=EQ, 3=PLUS, 4=STAR, 5=IDENT/ATOM
    """
    tok_decl = StructDecl("Token", (("kind", INT), ("lexeme", STR), ("next_pos", INT)))
    node_decl = StructDecl(
        "ParseNode",
        (("op_kind", INT), ("lhs_tok", STR), ("rhs_tok", STR), ("next_pos", INT), ("ok", BOOL)),
    )
    tok_ty = tok_decl.as_type()
    node_ty = node_decl.as_type()

    # fn next_token(src: str, pos: int) -> Token
    next_token = FuncDecl(
        name="next_token",
        params=[("src", STR), ("pos", INT)],
        ret_ty=tok_ty,
        body=[
            LetDecl("p", INT, BinOp("+", Var("pos"), Const(0, INT))),
            LetDecl("n", INT, StrLen(Var("src"))),
            # Skip up to 2 leading spaces
            WhileStmt(
                cond=BinOp(
                    "&&",
                    BinOp("<", Var("p"), Var("n")),
                    BinOp("==", StrSubstr(Var("src"), Var("p"), Const(1, INT)), Const(" ", STR)),
                ),
                body=[Assign("p", BinOp("+", Var("p"), Const(1, INT)))],
                unroll_bound=2,
            ),
            # Default EOF token
            LetDecl(
                "res",
                tok_ty,
                StructInit("Token", (("kind", Const(0, INT)), ("lexeme", Const("", STR)), ("next_pos", Var("p")))),
            ),
            IfStmt(
                cond=BinOp("<", Var("p"), Var("n")),
                then_body=[
                    LetDecl("ch", STR, StrSubstr(Var("src"), Var("p"), Const(1, INT))),
                    # Redundant concat "" + ch to exercise native string constant folding
                    LetDecl("ch_norm", STR, StrConcat(Const("", STR), Var("ch"))),
                    LetDecl("dead_len", INT, StrLen(Const("let", STR))),  # dead code for DCE + StrLen fold
                    IfStmt(
                        cond=BinOp("==", StrSubstr(Var("src"), Var("p"), Const(3, INT)), Const("let", STR)),
                        then_body=[
                            Assign(
                                "res",
                                StructInit(
                                    "Token",
                                    (
                                        ("kind", Const(1, INT)),
                                        ("lexeme", StrConcat(Const("l", STR), Const("et", STR))),
                                        ("next_pos", BinOp("+", Var("p"), Const(3, INT))),
                                    ),
                                ),
                            )
                        ],
                        else_body=[
                            LetDecl(
                                "k",
                                INT,
                                IteExpr(
                                    BinOp("==", Var("ch_norm"), Const("=", STR)),
                                    Const(2, INT),
                                    IteExpr(
                                        BinOp("==", Var("ch_norm"), Const("+", STR)),
                                        Const(3, INT),
                                        IteExpr(
                                            BinOp("==", Var("ch_norm"), Const("*", STR)),
                                            Const(4, INT),
                                            Const(5, INT),
                                        ),
                                    ),
                                ),
                            ),
                            Assign(
                                "res",
                                StructInit(
                                    "Token",
                                    (
                                        ("kind", Var("k")),
                                        ("lexeme", Var("ch_norm")),
                                        ("next_pos", BinOp("+", Var("p"), Const(1, INT))),
                                    ),
                                ),
                            ),
                        ],
                    ),
                ],
                else_body=[],
            ),
            ReturnStmt(Var("res")),
        ],
    )

    # fn parse_assign_expr(src: str) -> ParseNode
    # Parses simple `a + b` or `a * b` expression into a ParseNode struct!
    parse_expr = FuncDecl(
        name="parse_assign_expr",
        params=[("src", STR)],
        ret_ty=node_ty,
        body=[
            LetDecl("t1", tok_ty, FuncCall("next_token", (Var("src"), Const(0, INT)))),
            LetDecl("t2", tok_ty, FuncCall("next_token", (Var("src"), FieldGet(Var("t1"), "next_pos")))),
            LetDecl("t3", tok_ty, FuncCall("next_token", (Var("src"), FieldGet(Var("t2"), "next_pos")))),
            # Build ParseNode and exercise FieldGet/FieldSet + struct peepholes
            LetDecl(
                "is_binop",
                BOOL,
                BinOp(
                    "||",
                    BinOp("==", FieldGet(Var("t2"), "kind"), Const(3, INT)),
                    BinOp("==", FieldGet(Var("t2"), "kind"), Const(4, INT)),
                ),
            ),
            LetDecl(
                "valid",
                BOOL,
                BinOp(
                    "&&",
                    BinOp("&&", BinOp("==", FieldGet(Var("t1"), "kind"), Const(5, INT)), Var("is_binop")),
                    BinOp("==", FieldGet(Var("t3"), "kind"), Const(5, INT)),
                ),
            ),
            LetDecl(
                "node",
                node_ty,
                StructInit(
                    "ParseNode",
                    (
                        ("op_kind", FieldGet(Var("t2"), "kind")),
                        ("lhs_tok", StrConcat(Const("", STR), FieldGet(Var("t1"), "lexeme"))),
                        ("rhs_tok", StrConcat(FieldGet(Var("t3"), "lexeme"), Const("", STR))),
                        ("next_pos", FieldGet(Var("t3"), "next_pos")),
                        ("ok", Const(False, BOOL)),
                    ),
                ),
            ),
            # Functional struct update: node.ok = valid
            LetDecl("node2", node_ty, FieldSet(Var("node"), "ok", Var("valid"))),
            # Duplicate pure projection for CSE
            LetDecl("ok_check", BOOL, FieldGet(Var("node2"), "ok")),
            LetDecl("ok_dup", BOOL, FieldGet(Var("node2"), "ok")),
            LetDecl("final_ok", BOOL, BinOp("&&", Var("ok_check"), Var("ok_dup"))),
            ReturnStmt(FieldSet(Var("node2"), "ok", Var("final_ok"))),
        ],
    )

    return Program([next_token, parse_expr], structs=[tok_decl, node_decl])


def run_self_parser_verification():
    base = build_self_hosted_simp_parser()
    typecheck_program(base)

    # Verify individual passes + full pipeline on the self-hosted s-imp parser
    passes = [
        ("1. ConstFold (String+Struct Peepholes)", const_fold_pass),
        ("2. CSE (Struct Field Projection Reuse)", cse_pass),
        ("3. DCE (Dead StrLen Elimination)", dce_pass),
    ]
    cur = base
    for name, pfn in passes:
        nxt = pfn(cur)
        typecheck_program(nxt)
        se_l = SymbolicExecutor(cur, rec_depth=2)
        se_r = SymbolicExecutor(nxt, rec_depth=2)
        init_env, preconds = se_l.make_initial_inputs("parse_assign_expr")
        se_r.share_uf_tables_from(se_l)
        lhs, _ = se_l.lower_function("parse_assign_expr", init_env)
        rhs, _ = se_r.lower_function("parse_assign_expr", init_env)
        res = verify_equiv_bu_wlau(lhs, rhs, preconds=preconds, local_timeout_ms=100, final_timeout_ms=2000)
        print(
            f"=== Self-Hosted Parser | {name} ===\n"
            f"  Status: {'EQUIVALENT (UNSAT)' if res.equivalent else res.z3_final_status.upper()} | "
            f"DAG nodes: {res.nodes_before} -> {res.nodes_after} | "
            f"Discharged: {len(res.discharged)} | Time: {res.elapsed_ms:.2f} ms"
        )
        for i, dp in enumerate(res.discharged[:4]):
            print(f"    [{i+1}] ({dp.method}) {dp.hole_name}: {dp.lhs_str} ~ {dp.rhs_str}")
        assert res.equivalent, f"Self-hosted parser failed pass {name}"
        cur = nxt

    # Full pipeline check: Original ~ ConstFold+CSE+DCE
    se_l = SymbolicExecutor(base, rec_depth=2)
    se_r = SymbolicExecutor(cur, rec_depth=2)
    init_env, preconds = se_l.make_initial_inputs("parse_assign_expr")
    se_r.share_uf_tables_from(se_l)
    lhs, _ = se_l.lower_function("parse_assign_expr", init_env)
    rhs, _ = se_r.lower_function("parse_assign_expr", init_env)
    res_all = verify_equiv_bu_wlau(lhs, rhs, preconds=preconds, local_timeout_ms=100, final_timeout_ms=2000)
    print(
        f"=== Self-Hosted Parser | 4. Full Pipeline (Orig ~ ConstFold+CSE+DCE) ===\n"
        f"  Status: {'EQUIVALENT (UNSAT)' if res_all.equivalent else res_all.z3_final_status.upper()} | "
        f"DAG nodes: {res_all.nodes_before} -> {res_all.nodes_after} | "
        f"Discharged: {len(res_all.discharged)} | Time: {res_all.elapsed_ms:.2f} ms"
    )
    assert res_all.equivalent
    print("SELF-HOSTED s-imp PARSER VERIFICATION PASSED!")


if __name__ == "__main__":
    run_self_parser_verification()
