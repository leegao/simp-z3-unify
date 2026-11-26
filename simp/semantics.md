# Operational Semantics of *simp*

This document defines the **big-step (natural) operational semantics** of *simp*
and its **syntax-directed type system**. Both presentations follow the abstract
syntax in [`simp_ast.py`](./simp_ast.py) and the concrete surface syntax in
[`simp.lark`](./simp.lark); the type rules mirror `typecheck_program` exactly.

*s-imp* is a small imperative language with integers, floats, booleans, native
strings, arrays, dictionaries, pointers, structs, and a rigid host-Python
extension (imports, attribute access, calls, methods, and `box`/`unbox`
bridges). All host-Python interaction is modeled as **uninterpreted** operations.

---

## 1. Abstract Syntax

### 1.1 Types

$$
\begin{aligned}
\tau ::= & \mathsf{bool} \mid \mathsf{int} \mid \mathsf{float} \mid \mathsf{str}
        \mid \mathsf{void} \\
      \mid& \mathsf{ptr}\langle\tau\rangle
        \mid \mathsf{array}\langle\tau\rangle
        \mid \mathsf{dict}\langle\tau,\tau\rangle
        \mid \mathsf{pyobj}\langle t\rangle
        \mid \mathsf{struct}\,S
\end{aligned}
$$

where $t$ is a *tag* (a host-Python module/class name, default $\mathsf{object}$)
and $S$ ranges over struct names declared by `struct` declarations. A struct
declaration has the form

$$
\mathsf{struct}\\,S\\,\\{\\, f_1:\tau_1,\ \dots,\ f_n:\tau_n \\,\\}.
$$

### 1.2 Expressions

$$
\begin{aligned}
e ::= & c \qquad\qquad\qquad\\,\\, \text{(constant: } \mathsf{bool},\mathsf{int},\mathsf{float},\mathsf{str}\text{)}\\
      \mid& x \qquad\qquad\qquad\\,\\, \text{(variable)}\\
      \mid& \ominus\\, e \mid\\, !\\,e \qquad\\, \text{(unary } \ominus\in\\{-,!\\}\text{)}\\
      \mid& e_1 \oplus e_2 \qquad\quad \text{(binary } \oplus\in\\{+,-,*,/,==,\neq,<,\le,>, \ge,\land,\lor\\}\text{)}\\
      \mid& e_1 \\,?\\, e_2 : e_3 \qquad\\, \text{(ite)}\\
      \mid& e_1[e_2] \qquad\qquad\\, \text{(array load)}\\
      \mid& \mathsf{store}(e_1,e_2,e_3) \quad \text{(functional array store)}\\
      \mid& e_1 \mathbin{@} e_2 \qquad\qquad \text{(pointer offset)}\\
      \mid& *\\,e \qquad\qquad\qquad\\, \text{(pointer dereference)}\\
      \mid& \mathsf{builtin}(\bar e) \qquad\quad \text{(}\exp,\log,\sin,\cos,\sqrt{}\text{)}\\
      \mid& f(\bar e) \qquad\qquad\qquad \text{(function call)}\\
      \mid& \mathsf{pyobj}\langle t\rangle \qquad\qquad \text{(module reference)}\\
      \mid& e.\mathsf{attr} \qquad\qquad\\, \text{(attribute / field projection)}\\
      \mid& \mathsf{pybox}(e) \mid \mathsf{pyunbox}\langle\tau\rangle(e) \quad \text{(boxing bridges)}\\
      \mid& e_1 \mathbin{+\!\!+} e_2 \qquad\quad \text{(string concatenation)}\\
      \mid& \mathsf{len}(e) \mid \mathsf{substr}(e_1,e_2,e_3) \quad \text{(string builtins)}\\
      \mid& S\{\,\overline{f_i:e_i}\,\} \qquad\quad \text{(struct construction)}\\
      \mid& \mathsf{dict}\langle\tau_K,\tau_V\rangle\{\,\overline{k_i:v_i}\,\} \quad \text{(dict construction)}\\
      \mid& \mathsf{setfield}(e_1,f,e_2) \quad \text{(functional field update)}
\end{aligned}
$$

The surface forms `e.f`, `f(args)`, `m.f(args)`, `box(e)`, `unbox<T>(e)` are
resolved to the appropriate node during parsing in a type-directed way; the
abstract syntax above is the resolved form.

### 1.3 Statements

$$
\begin{aligned}
s ::= & \mathsf{let}\\,x:\tau = e; \qquad\quad \text{(declaration)}\\
      \mid& x = e; \qquad\qquad\qquad\\, \text{(assignment)}\\
      \mid& x[e_1] = e_2; \qquad\qquad \text{(array store)}\\
      \mid& *\\,e_1 = e_2; \qquad\qquad\\, \text{(pointer store)}\\
      \mid& \mathsf{if}\\,(e)\\,\\{s\\}\\,\mathsf{else}\\,\\{s\\} \quad \text{(conditional)}\\
      \mid& \mathsf{while}\\,(e)\\,\\{s\\} \qquad\quad \text{(loop)}\\
      \mid& \mathsf{return}\\,e; \qquad\qquad\\, \text{(return)}
\end{aligned}
$$

A **block** is a sequence $s_1;\\,s_2;\dots;s_n$ (possibly empty).

### 1.4 Programs

$$
\begin{aligned}
P ::= \ \bigl(&\ \overline{\mathsf{import}}\ m_i\ \mathsf{as}\ a_i,\ \
      &\ \overline{\mathsf{struct}\ S_j},\ \
      &\ \overline{\mathsf{fn}}\ f_k\bigl(\vec{x}:\vec{\tau}\bigr)\to\tau_k\ \\{s\\}\ \bigr)
\end{aligned}
$$

A function's return type defaults to $\mathsf{void}$ when the `-> τ` clause is
omitted.

---

## 2. Semantic Domains

$$
\begin{aligned}
v \in \mathit{Val} ::= & n \in \mathbb{Z} \mid r \in \mathbb{R} \mid b \in \\{\mathsf{true},\mathsf{false}\\} \mid s \in \Sigma^{*}\\
      \mid& \ell \in \mathit{Loc} \qquad\text{(pointer)}\\
      \mid& A \in \mathbb{Z} \rightharpoonup \mathit{Val} \qquad\text{(array, McCarthy-style)}\\
      \mid& D \in \mathit{Key} \rightharpoonup \mathit{Val} \qquad\text{(dict, McCarthy-style)}\\
      \mid& o \in \mathit{PyObj} \qquad\text{(opaque host object)}\\
      \mid& \\{\\,f_1\mapsto v_1,\dots,f_n\mapsto v_n\\,\\} \qquad\text{(struct)}
\end{aligned}
$$

$$
\sigma \in \mathit{Store} = \mathit{Var} \rightharpoonup \mathit{Val},
\qquad
H \in \mathit{Heap} = \mathit{Loc} \rightharpoonup \mathit{Val}.
$$

We write $\sigma[x\mapsto v]$ and $H[\ell\mapsto v]$ for the usual updates, and
use the McCarthy array operations

$$
\mathsf{select}(A,i) = A(i),
\qquad
\mathsf{store}(A,i,v) = A\bigl[i\mapsto v\bigr].
$$

A **result** of statement execution is $r \in \\{\bot,\ \mathsf{ret}(v)\\}$, where
$\bot$ denotes *fall-through* (normal completion) and $\mathsf{ret}(v)$ denotes
an early return carrying value $v$.

Two judgments are used:

$$
\boxed{\\,\langle e,\sigma,H\rangle \Downarrow v\\,}
\qquad\text{(expression $e$ evaluates to $v$)}
$$

$$
\boxed{\\,\langle s,\sigma,H\rangle \Downarrow \langle\sigma',H',r\rangle\\,}
\qquad\text{(statement $s$ transforms the state)}
$$

Expressions are **pure**: they read $\sigma$ and $H$ but never mutate them
(array/field "stores" are functional and yield new values).

---

## 3. Big-Step Operational Semantics

### 3.1 Expression Evaluation

**Constants and variables.**

$$
\frac{\strut c\ \text{is a literal of type}\ \tau}
     {\langle c,\sigma,H\rangle \Downarrow c}
\qquad
\frac{\strut \sigma(x)=v}
     {\langle x,\sigma,H\rangle \Downarrow v}
$$

**Unary operators.**

$$
\frac{\langle e,\sigma,H\rangle \Downarrow b}
     {\langle !\\,e,\sigma,H\rangle \Downarrow \neg b}
\qquad
\frac{\langle e,\sigma,H\rangle \Downarrow n}
     {\langle -\\,e,\sigma,H\rangle \Downarrow -n}
\qquad
\frac{\langle e,\sigma,H\rangle \Downarrow r}
     {\langle -\\,e,\sigma,H\rangle \Downarrow -r}
$$

**Binary operators.** For arithmetic and comparison we write $\oplus$ for the
corresponding mathematical operation on $\mathbb{Z}$ or $\mathbb{R}$:

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow v_1 \qquad
      \langle e_2,\sigma,H\rangle \Downarrow v_2}
     {\langle e_1 \oplus e_2,\sigma,H\rangle \Downarrow v_1 \oplus v_2}
\quad
\oplus \in \\{+,-,*,/\\}
$$

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow v_1 \qquad
      \langle e_2,\sigma,H\rangle \Downarrow v_2}
     {\langle e_1 \oplus e_2,\sigma,H\rangle \Downarrow (v_1 \oplus v_2)}
\quad
\oplus \in \\{<,\le,>,\ge,==,\neq\\}
$$

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow b_1 \qquad
      \langle e_2,\sigma,H\rangle \Downarrow b_2}
     {\langle e_1 \wedge e_2,\sigma,H\rangle \Downarrow b_1 \wedge b_2}
\qquad
\frac{\langle e_1,\sigma,H\rangle \Downarrow b_1 \qquad
      \langle e_2,\sigma,H\rangle \Downarrow b_2}
     {\langle e_1 \vee e_2,\sigma,H\rangle \Downarrow b_1 \vee b_2}
$$

**Conditional expression.**

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow \mathsf{true} \qquad
      \langle e_2,\sigma,H\rangle \Downarrow v}
     {\langle e_1\\,?\\,e_2:e_3,\sigma,H\rangle \Downarrow v}
\qquad
\frac{\langle e_1,\sigma,H\rangle \Downarrow \mathsf{false} \qquad
      \langle e_3,\sigma,H\rangle \Downarrow v}
     {\langle e_1\\,?\\,e_2:e_3,\sigma,H\rangle \Downarrow v}
$$

**Arrays.**

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow A \qquad
      \langle e_2,\sigma,H\rangle \Downarrow n}
     {\langle e_1[e_2],\sigma,H\rangle \Downarrow \mathsf{select}(A,n)}
$$

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow A \quad
      \langle e_2,\sigma,H\rangle \Downarrow n \quad
      \langle e_3,\sigma,H\rangle \Downarrow v}
     {\langle \mathsf{store}(e_1,e_2,e_3),\sigma,H\rangle \Downarrow \mathsf{store}(A,n,v)}
$$

**Dicts.** A dictionary is a finite map $D$ keyed by $\mathit{Key} = \Sigma^{*}\cup\mathbb{Z}$.
The intro form builds $D$ by successive functional stores over an empty map;
subscript load/store are the same $\mathsf{select}/\mathsf{store}$ operations as
arrays (with a general key domain). Reads of absent keys yield the map's default
value (runtime-checked).

$$
\frac{\langle \bar k,\sigma,H\rangle \Downarrow \bar v_k \quad
      \langle \bar e,\sigma,H\rangle \Downarrow \bar v}
     {\langle \mathsf{dict}\langle\tau_K,\tau_V\rangle\{\overline{k_i:e_i}\},\sigma,H\rangle \Downarrow
      \mathsf{store}(\dots\mathsf{store}(D_\varnothing,k_1,v_1)\dots,k_n,v_n)}
$$

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow D \qquad
      \langle e_2,\sigma,H\rangle \Downarrow k}
     {\langle e_1[e_2],\sigma,H\rangle \Downarrow \mathsf{select}(D,k)}
\qquad
\frac{\langle e_1,\sigma,H\rangle \Downarrow D \quad
      \langle e_2,\sigma,H\rangle \Downarrow k \quad
      \langle e_3,\sigma,H\rangle \Downarrow v}
     {\langle \mathsf{store}(e_1,e_2,e_3),\sigma,H\rangle \Downarrow \mathsf{store}(D,k,v)}
$$

**Pointers.** A pointer value is a location; offset arithmetic is on locations.

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow \ell \qquad
      \langle e_2,\sigma,H\rangle \Downarrow n}
     {\langle e_1 \mathbin{@} e_2,\sigma,H\rangle \Downarrow \ell + n}
\qquad
\frac{\langle e,\sigma,H\rangle \Downarrow \ell \qquad H(\ell)=v}
     {\langle *\\,e,\sigma,H\rangle \Downarrow v}
$$

**Builtins and function calls.** Builtins ($\exp,\log,\sin,\cos,\sqrt{}$) are
modeled as (mathematical or uninterpreted) unary functions $B$:

$$
\frac{\langle e,\sigma,H\rangle \Downarrow r}
     {\langle \mathsf{builtin}(e),\sigma,H\rangle \Downarrow B(r)}
$$

Function calls evaluate arguments left-to-right, bind them to the parameters,
and execute the body (see §3.3):

$$
\frac{\langle \bar e,\sigma,H\rangle \Downarrow \bar v \qquad
      \langle f(\overline{x:\tau})\to\tau'\\{s\\},\ \sigma[x_i\mapsto v_i],\ H\rangle
      \Downarrow \langle\sigma',H',\mathsf{ret}(v)\rangle}
     {\langle f(\bar e),\sigma,H\rangle \Downarrow v}
$$

**Host-Python extension.** Module references, attributes, calls, methods, and the
boxing bridges are **rigid uninterpreted operations**. We write
$\mathcal{U}^{\mathsf{attr}}$, $\mathcal{U}^{\mathsf{call}}$,
$\mathcal{U}^{\mathsf{method}}$, $\mathcal{U}^{\mathsf{box}}$,
$\mathcal{U}^{\mathsf{unbox}}$ for these uninterpreted functions:

$$
\frac{\strut a\ \text{is an import alias}}
     {\langle \mathsf{pyobj}\langle t\rangle,\sigma,H\rangle \Downarrow o_a}
\qquad
\frac{\langle e,\sigma,H\rangle \Downarrow o}
     {\langle e.\mathsf{attr},\sigma,H\rangle \Downarrow \mathcal{U}^{\mathsf{attr}}(o)}
$$

$$
\frac{\langle e,\sigma,H\rangle \Downarrow o}
     {\langle \mathsf{pybox}(e),\sigma,H\rangle \Downarrow \mathcal{U}^{\mathsf{box}}(o)}
\qquad
\frac{\langle e,\sigma,H\rangle \Downarrow o}
     {\langle \mathsf{pyunbox}\langle\tau\rangle(e),\sigma,H\rangle \Downarrow \mathcal{U}^{\mathsf{unbox}}(o)}
$$

$$
\frac{\langle e_0,\sigma,H\rangle \Downarrow o_0 \quad
      \langle \bar e,\sigma,H\rangle \Downarrow \bar v}
     {\langle e_0(\bar e),\sigma,H\rangle \Downarrow \mathcal{U}^{\mathsf{call}}(o_0,\bar v)}
\qquad
\frac{\langle e_0,\sigma,H\rangle \Downarrow o_0 \quad
      \langle \bar e,\sigma,H\rangle \Downarrow \bar v}
     {\langle e_0.m(\bar e),\sigma,H\rangle \Downarrow \mathcal{U}^{\mathsf{method}}(o_0,\bar v)}
$$

**Strings.**

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow s_1 \quad
      \langle e_2,\sigma,H\rangle \Downarrow s_2}
     {\langle e_1 \mathbin{+\!\!+} e_2,\sigma,H\rangle \Downarrow s_1 \cdot s_2}
\qquad
\frac{\langle e,\sigma,H\rangle \Downarrow s}
     {\langle \mathsf{len}(e),\sigma,H\rangle \Downarrow |s|}
$$

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow s \quad
      \langle e_2,\sigma,H\rangle \Downarrow i \quad
      \langle e_3,\sigma,H\rangle \Downarrow j}
     {\langle \mathsf{substr}(e_1,e_2,e_3),\sigma,H\rangle \Downarrow s[i\\,..\\,i+j]}
$$

**Structs.**

$$
\frac{\langle \bar e,\sigma,H\rangle \Downarrow \bar v}
     {\langle S\\{\overline{f_i:e_i}\\},\sigma,H\rangle \Downarrow
      \\{\\,f_1\mapsto v_1,\dots,f_n\mapsto v_n\\,\\}}
$$

$$
\frac{\langle e,\sigma,H\rangle \Downarrow \\{f_1\mapsto v_1,\dots,f_n\mapsto v_n\\}}
     {\langle e.f_i,\sigma,H\rangle \Downarrow v_i}
$$

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow \\{f_1\mapsto v_1,\dots,f_n\mapsto v_n\\} \quad
      \langle e_2,\sigma,H\rangle \Downarrow v}
     {\langle \mathsf{setfield}(e_1,f_i,e_2),\sigma,H\rangle \Downarrow
      \\{f_1\mapsto v_1,\dots,f_i\mapsto v,\dots,f_n\mapsto v_n\\}}
$$

### 3.2 Statement Execution

**Declaration and assignment.**

$$
\frac{\langle e,\sigma,H\rangle \Downarrow v}
     {\langle \mathsf{let}\\,x:\tau=e;,\sigma,H\rangle
      \Downarrow \langle\sigma[x\mapsto v],H,\bot\rangle}
\qquad
\frac{\langle e,\sigma,H\rangle \Downarrow v}
     {\langle x=e;,\sigma,H\rangle
      \Downarrow \langle\sigma[x\mapsto v],H,\bot\rangle}
$$

**Array and pointer stores.**

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow n \quad
      \langle e_2,\sigma,H\rangle \Downarrow v \quad
      \sigma(x)=A}
     {\langle x[e_1]=e_2;,\sigma,H\rangle
      \Downarrow \langle\sigma[x\mapsto \mathsf{store}(A,n,v)],H,\bot\rangle}
$$

$$
\frac{\langle e_1,\sigma,H\rangle \Downarrow \ell \quad
      \langle e_2,\sigma,H\rangle \Downarrow v}
     {\langle *\\,e_1=e_2;,\sigma,H\rangle
      \Downarrow \langle\sigma,H[\ell\mapsto v],\bot\rangle}
$$

**Sequencing.** A block $s_1;\dots;s_n$ is executed left-to-right; a return
short-circuits the remainder.

$$
\frac{\langle s_1,\sigma,H\rangle \Downarrow \langle\sigma_1,H_1,\mathsf{ret}(v)\rangle}
     {\langle s_1;\bar s,\sigma,H\rangle \Downarrow \langle\sigma_1,H_1,\mathsf{ret}(v)\rangle}
$$

$$
\frac{\langle s_1,\sigma,H\rangle \Downarrow \langle\sigma_1,H_1,\bot\rangle \qquad
      \langle \bar s,\sigma_1,H_1\rangle \Downarrow \langle\sigma',H',r\rangle}
     {\langle s_1;\bar s,\sigma,H\rangle \Downarrow \langle\sigma',H',r\rangle}
\qquad
\frac{\strut \text{(empty block)}}
     {\langle \varepsilon,\sigma,H\rangle \Downarrow \langle\sigma,H,\bot\rangle}
$$

**Conditional.**

$$
\frac{\langle e,\sigma,H\rangle \Downarrow \mathsf{true} \qquad
      \langle s_1,\sigma,H\rangle \Downarrow \langle\sigma',H',r\rangle}
     {\langle \mathsf{if}\\,(e)\\,\\{s_1\\}\\,\mathsf{else}\\,\\{s_2\\},\sigma,H\rangle
      \Downarrow \langle\sigma',H',r\rangle}
$$

$$
\frac{\langle e,\sigma,H\rangle \Downarrow \mathsf{false} \qquad
      \langle s_2,\sigma,H\rangle \Downarrow \langle\sigma',H',r\rangle}
     {\langle \mathsf{if}\\,(e)\\,\\{s_1\\}\\,\mathsf{else}\\,\\{s_2\\},\sigma,H\rangle
      \Downarrow \langle\sigma',H',r\rangle}
$$

**While.** The standard big-step loop rule (the `unroll` bound is a
bounded-model-checking artifact and does not appear in the pure semantics):

$$
\frac{\langle e,\sigma,H\rangle \Downarrow \mathsf{false}}
     {\langle \mathsf{while}\\,(e)\\,\\{s\\},\sigma,H\rangle
      \Downarrow \langle\sigma,H,\bot\rangle}
$$

$$
\frac{\langle e,\sigma,H\rangle \Downarrow \mathsf{true} \quad
      \langle s,\sigma,H\rangle \Downarrow \langle\sigma_1,H_1,\bot\rangle \quad
      \langle \mathsf{while}\\,(e)\\,\\{s\\},\sigma_1,H_1\rangle
      \Downarrow \langle\sigma',H',r\rangle}
     {\langle \mathsf{while}\\,(e)\\,\\{s\\},\sigma,H\rangle
      \Downarrow \langle\sigma',H',r\rangle}
$$

**Return.**

$$
\frac{\langle e,\sigma,H\rangle \Downarrow v}
     {\langle \mathsf{return}\\,e;,\sigma,H\rangle
      \Downarrow \langle\sigma,H,\mathsf{ret}(v)\rangle}
$$

### 3.3 Program Execution

A program is executed by evaluating a designated entry function $f$ on argument
values $\bar v$. The initial store binds parameters to arguments (and, for
pointer parameters, supplies a fresh heap); the body is executed and must
terminate with $\mathsf{ret}(v)$:

$$
\frac{\langle \mathsf{fn}\\,f(\overline{x:\tau})\to\tau'\\{s\\},\ \sigma_0[x_i\mapsto v_i],\ H_0\rangle
      \Downarrow \langle\sigma',H',\mathsf{ret}(v)\rangle}
     {\langle P,\ f,\ \bar v\rangle \Downarrow v}
$$

---

## 4. Type System

The type system is **syntax-directed** and mirrors `typecheck_program`. We
judge expressions and statements under a *type environment* $\Gamma$ (mapping
variables to types), together with fixed global tables:

- $\Sigma$ — function signatures $f : (\tau_1,\dots,\tau_n)\to\tau$,
- $\Delta$ — import aliases $a \mapsto t$ (module tag),
- $\mathcal{S}$ — struct definitions $S \mapsto \\{f_i:\tau_i\\}$.

### 4.1 Type Compatibility

Because all host-Python objects share one opaque sort, we use a **compatibility**
relation $\sim$ rather than strict equality:

$$
\frac{\strut \tau_1 = \tau_2}{\tau_1 \sim \tau_2}
\qquad
\frac{\strut}{\mathsf{pyobj}\langle t_1\rangle \sim \mathsf{pyobj}\langle t_2\rangle}
$$

Thus two `pyobj` types are always compatible regardless of tag; all other types
must be *equal*. This relation is used for `==`/`!=`, `let`, assignment,
`ite`, array/field stores, call arguments, and returns. Conditions (`if`,
`while`, `!`) require the type to be **exactly** $\mathsf{bool}$.

### 4.2 Expression Typing

$$
\boxed{\\,\Gamma;\Sigma;\Delta;\mathcal{S} \vdash e : \tau\\,}
$$

**Constants / variables.**

$$
\frac{\strut c\ \text{has type}\ \tau}{\Gamma;\Sigma;\Delta;\mathcal{S}\vdash c:\tau}
\qquad
\frac{\strut \Gamma(x)=\tau}{\Gamma;\Sigma;\Delta;\mathcal{S}\vdash x:\tau}
$$

**Unary.**

$$
\frac{\Gamma;\Sigma;\Delta;\mathcal{S}\vdash e:\mathsf{bool}}
     {\Gamma;\Sigma;\Delta;\mathcal{S}\vdash !\\,e:\mathsf{bool}}
\qquad
\frac{\Gamma;\Sigma;\Delta;\mathcal{S}\vdash e:\tau \quad \tau\in\\{\mathsf{int},\mathsf{float}\\}}
     {\Gamma;\Sigma;\Delta;\mathcal{S}\vdash -\\,e:\tau}
$$

**Binary.**

$$
\frac{\Gamma\vdash e_1:\tau \quad \Gamma\vdash e_2:\tau \quad
      \tau\in\\{\mathsf{int},\mathsf{float}\\}}
     {\Gamma\vdash e_1\oplus e_2:\tau}
\quad \oplus\in\\{+,-,*,/\\}
$$

$$
\frac{\Gamma\vdash e_1:\tau \quad \Gamma\vdash e_2:\tau \quad
      \tau\in\\{\mathsf{int},\mathsf{float}\\}}
     {\Gamma\vdash e_1\oplus e_2:\mathsf{bool}}
\quad \oplus\in\\{<,\le,>,\ge\\}
$$

$$
\frac{\Gamma\vdash e_1:\tau_1 \quad \Gamma\vdash e_2:\tau_2 \quad \tau_1\sim\tau_2}
     {\Gamma\vdash e_1\oplus e_2:\mathsf{bool}}
\quad \oplus\in\\{==,\neq\\}
$$

$$
\frac{\Gamma\vdash e_1:\mathsf{bool} \quad \Gamma\vdash e_2:\mathsf{bool}}
     {\Gamma\vdash e_1\oplus e_2:\mathsf{bool}}
\quad \oplus\in\\{\land,\lor\\}
$$

(Here $\Gamma\vdash e:\tau$ abbreviates
$\Gamma;\Sigma;\Delta;\mathcal{S}\vdash e:\tau$.)

**Conditional expression.**

$$
\frac{\Gamma\vdash e_1:\mathsf{bool} \quad
      \Gamma\vdash e_2:\tau_1 \quad
      \Gamma\vdash e_3:\tau_2 \quad
      \tau_1\sim\tau_2}
     {\Gamma\vdash e_1\\,?\\,e_2:e_3:\tau_1}
$$

**Arrays.**

$$
\frac{\Gamma\vdash e_1:\mathsf{array}\langle\tau\rangle \quad
      \Gamma\vdash e_2:\mathsf{int}}
     {\Gamma\vdash e_1[e_2]:\tau}
\qquad
\frac{\Gamma\vdash e_1:\mathsf{array}\langle\tau\rangle \quad
      \Gamma\vdash e_2:\mathsf{int} \quad
      \Gamma\vdash e_3:\tau' \quad \tau'\sim\tau}
     {\Gamma\vdash \mathsf{store}(e_1,e_2,e_3):\mathsf{array}\langle\tau\rangle}
$$

**Dicts.** Keys must be $\mathsf{str}$ or $\mathsf{int}$; subscript load/store
reuse the array rules with the key type in place of $\mathsf{int}$.

$$
\frac{\Gamma\vdash e_1:\mathsf{dict}\langle\tau_K,\tau_V\rangle \quad
      \Gamma\vdash e_2:\tau_K}
     {\Gamma\vdash e_1[e_2]:\tau_V}
\qquad
\frac{\Gamma\vdash e_1:\mathsf{dict}\langle\tau_K,\tau_V\rangle \quad
      \Gamma\vdash e_2:\tau_K \quad
      \Gamma\vdash e_3:\tau' \quad \tau'\sim\tau_V}
     {\Gamma\vdash \mathsf{store}(e_1,e_2,e_3):\mathsf{dict}\langle\tau_K,\tau_V\rangle}
$$

$$
\frac{\tau_K\in\{\mathsf{str},\mathsf{int}\} \quad
      \Gamma\vdash \bar k:\tau_K \quad
      \Gamma\vdash \bar e:\bar\tau \quad
      \forall i.\ \tau_i\sim\tau_V}
     {\Gamma\vdash \mathsf{dict}\langle\tau_K,\tau_V\rangle\{\overline{k_i:e_i}\}:\mathsf{dict}\langle\tau_K,\tau_V\rangle}
$$

**Pointers.**

$$
\frac{\Gamma\vdash e_1:\mathsf{ptr}\langle\tau\rangle \quad
      \Gamma\vdash e_2:\mathsf{int}}
     {\Gamma\vdash e_1\mathbin{@}e_2:\mathsf{ptr}\langle\tau\rangle}
\qquad
\frac{\Gamma\vdash e:\mathsf{ptr}\langle\tau\rangle}
     {\Gamma\vdash *\\,e:\tau}
$$

**Builtins and function calls.**

$$
\frac{\Gamma\vdash \bar e:\bar\tau \quad
      (\tau_1,\dots,\tau_n)\to\tau \in \mathit{BUILTIN\_SIGS} \quad
      \bar\tau = \bar\tau_{\text{sig}}}
     {\Gamma\vdash \mathsf{builtin}(\bar e):\tau}
$$

$$
\frac{\Gamma\vdash \bar e:\bar\tau \quad
      f:(\tau_1,\dots,\tau_n)\to\tau \in \Sigma \quad
      |\bar e|=n \quad \forall i.\ \tau_i\sim\tau_{\text{sig},i}}
     {\Gamma\vdash f(\bar e):\tau}
$$

**Host-Python extension.**

$$
\frac{a\mapsto t \in \Delta}
     {\Gamma\vdash \mathsf{pyobj}\langle t\rangle:\mathsf{pyobj}\langle t\rangle}
\qquad
\frac{\Gamma\vdash e:\mathsf{pyobj}\langle t\rangle}
     {\Gamma\vdash e.\mathsf{attr}:\tau_{\mathsf{ret}}}
$$

$$
\frac{\Gamma\vdash e_0:\mathsf{pyobj}\langle t\rangle \quad
      \Gamma\vdash \bar e:\bar\tau}
     {\Gamma\vdash e_0(\bar e):\tau_{\mathsf{ret}}}
\qquad
\frac{\Gamma\vdash e_0:\mathsf{pyobj}\langle t\rangle \quad
      \Gamma\vdash \bar e:\bar\tau}
     {\Gamma\vdash e_0.m(\bar e):\tau_{\mathsf{ret}}}
$$

$$
$$\frac{\Gamma\vdash e:\tau \quad \tau\in\{\mathsf{bool},\mathsf{int},\mathsf{float}\}}
     {\Gamma\vdash \mathsf{pybox}(e):\mathsf{pyobj}\langle\mathsf{object}\rangle}
\qquad
\frac{\Gamma\vdash e:\mathsf{pyobj}\langle t\rangle \quad
      \tau\in\{\mathsf{bool},\mathsf{int},\mathsf{float}\}\cup\{\mathsf{struct}\,S,\mathsf{dict}\langle\tau_K,\tau_V\rangle\}}
     {\Gamma\vdash \mathsf{pyunbox}\langle\tau\rangle(e):\tau}
$$

Unboxing is **runtime-checked**: the target may be a primitive, a struct, or a
dict; the conversion itself is a rigid uninterpreted operation
$\mathcal{U}^{\mathsf{unbox}}$ (no algebraic laws are assumed).

**Strings.**

$$
\frac{\Gamma\vdash e_1:\mathsf{str} \quad \Gamma\vdash e_2:\mathsf{str}}
     {\Gamma\vdash e_1\mathbin{+\!\!+}e_2:\mathsf{str}}
\qquad
\frac{\Gamma\vdash e:\mathsf{str}}
     {\Gamma\vdash \mathsf{len}(e):\mathsf{int}}
$$

$$
\frac{\Gamma\vdash e_1:\mathsf{str} \quad
      \Gamma\vdash e_2:\mathsf{int} \quad
      \Gamma\vdash e_3:\mathsf{int}}
     {\Gamma\vdash \mathsf{substr}(e_1,e_2,e_3):\mathsf{str}}
$$

**Structs.**

$$
\frac{S\mapsto\\{f_1:\tau_1,\dots,f_n:\tau_n\\}\in\mathcal{S} \quad
      \Gamma\vdash \bar e:\bar\tau' \quad
      |\bar e|=n \quad \forall i.\ \tau'_i\sim\tau_i}
     {\Gamma\vdash S\\{\overline{f_i:e_i}\\}:S}
$$

$$
\frac{\Gamma\vdash e:S \quad S\mapsto\\{f_i:\tau_i\\}\in\mathcal{S}}
     {\Gamma\vdash e.f_i:\tau_i}
\qquad
\frac{\Gamma\vdash e_1:S \quad S\mapsto\\{f_i:\tau_i\\}\in\mathcal{S} \quad
      \Gamma\vdash e_2:\tau' \quad \tau'\sim\tau_i}
     {\Gamma\vdash \mathsf{setfield}(e_1,f_i,e_2):S}
$$

### 4.3 Statement Typing

$$
\boxed{\\,\Gamma;\Sigma;\Delta;\mathcal{S}\vdash s \dashv \Gamma'\\,}
$$

where $\Gamma'$ is the environment *after* the statement (only `let` extends
$\Gamma$; other statements leave it unchanged). The return type $\tau_{\mathsf{ret}}$
of the enclosing function is an additional parameter, written as a subscript.

**Declaration.**

$$
\frac{\Gamma\vdash e:\tau_0 \quad \tau_0\sim\tau}
     {\Gamma;\dots\vdash \mathsf{let}\\,x:\tau=e;\ \dashv\ \Gamma[x\mapsto\tau]}
$$

**Assignment.**

$$
\frac{\Gamma(x)=\tau \quad \Gamma\vdash e:\tau' \quad \tau'\sim\tau}
     {\Gamma;\dots\vdash x=e;\ \dashv\ \Gamma}
$$

**Array / pointer store.**

$$
\frac{\Gamma(x)=\mathsf{array}\langle\tau\rangle \quad
      \Gamma\vdash e_1:\mathsf{int} \quad
      \Gamma\vdash e_2:\tau' \quad \tau'\sim\tau}
     {\Gamma;\dots\vdash x[e_1]=e_2;\ \dashv\ \Gamma}
$$

$$
\frac{\Gamma\vdash e_1:\mathsf{ptr}\langle\tau\rangle \quad
      \Gamma\vdash e_2:\tau' \quad \tau'\sim\tau}
     {\Gamma;\dots\vdash *\\,e_1=e_2;\ \dashv\ \Gamma}
$$

**Conditional.**

$$
\frac{\Gamma\vdash e:\mathsf{bool} \quad
      \Gamma;\dots\vdash s_1\dashv\Gamma_1 \quad
      \Gamma;\dots\vdash s_2\dashv\Gamma_2}
     {\Gamma;\dots\vdash \mathsf{if}\\,(e)\\,\\{s_1\\}\\,\mathsf{else}\\,\\{s_2\\}\ \dashv\ \Gamma}
$$

**While.**

$$
\frac{\Gamma\vdash e:\mathsf{bool} \quad
      \Gamma;\dots\vdash s\dashv\Gamma'}
     {\Gamma;\dots\vdash \mathsf{while}\\,(e)\\,\\{s\\}\ \dashv\ \Gamma}
$$

**Return.**

$$
\frac{\Gamma\vdash e:\tau \quad \tau\sim\tau_{\mathsf{ret}}}
     {\Gamma;\dots\vdash_{\tau_{\mathsf{ret}}} \mathsf{return}\\,e;\ \dashv\ \Gamma}
$$

### 4.4 Program Typing

A program is well-typed when:

1. **No duplicate names.** All function names are distinct (and distinct from
   struct names); all struct field names within a struct are distinct.
2. **Signatures are consistent.** $\Sigma$ collects each
   $f:(\tau_1,\dots,\tau_n)\to\tau$ from the declarations.
3. **Bodies are well-typed.** For each function
   $\mathsf{fn}\\,f(\overline{x:\tau})\to\tau_{\mathsf{ret}}\\{s\\}$, with
   $\Gamma_0 = \\{x_i\mapsto\tau_i\\}$:

$$
\frac{\Gamma_0;\Sigma;\Delta;\mathcal{S}\vdash s\dashv\Gamma' \quad
      \text{(every path returns a value compatible with }\tau_{\mathsf{ret}}\text{)}}
     {\Sigma;\Delta;\mathcal{S}\vdash \mathsf{fn}\\,f(\overline{x:\tau})\to\tau_{\mathsf{ret}}\\{s\\}}
$$

The full program judgment is

$$
\boxed{\\,\vdash P\\,}
\qquad\text{iff all imports, structs, and functions satisfy the above.}
$$

---

## 5. Remarks

- **Purity of expressions.** Expressions never mutate $\sigma$ or $H$; the
  "store" and "setfield" constructs are *functional* updates that produce new
  array/struct values. Only statements mutate the store and heap.
- **Dictionaries as arrays.** A `dict<K,V>` is encoded in Z3 as
  `Array(K, V)`; the intro form `dict<K,V>{k:v,...}` is a chain of functional
  stores over a fixed empty base array, and `d[k]` / `d[k]=v` are the same
  `Select`/`Store` operations used for arrays. Keys are restricted to `str` or
  `int`.
- **Pointers as locations.** There is no allocation construct; pointer values
  originate from parameters (a fresh heap is supplied at call time) and are
  manipulated only by `@` (offset) and `*` (deref / store).
- **Host-Python is rigid.** Every host interaction is an uninterpreted function
  $\mathcal{U}^{(\cdot)}$. This is what makes the Z3 encoding sound: the
  semantics commits to *no* algebraic laws for host objects beyond congruence.
- **Bounded loops and recursion.** The operational semantics above is the
  *unbounded* standard semantics. The verifier
  ([`symexec.py`](../verify/symexec.py)) instead uses **bounded unrolling**
  (`unroll_bound`, default $4$) plus a modular uninterpreted-function summary at
  the depth cap — a sound-by-construction approximation used for
  bounded equivalence checking, not a redefinition of the language.
- **Type vs. operational `bool`.** Type rules for conditions demand *exactly*
  $\mathsf{bool}$, whereas value-level compatibility $\sim$ (used for data
  flow) additionally equates any two `pyobj` types.
