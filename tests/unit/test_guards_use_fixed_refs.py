"""T-0094 criterion 10 and T-0094c: no scope or contract guard in ``tests/`` decides pass or fail against a moving
git ref, and none compares a finished task's base with the working tree.

A guard that diffs against ``main``, ``origin/<x>``, ``main:<path>``, a range ending in ``main``/``HEAD`` or
``merge-base HEAD <branch>`` is vacuous on main (``merge-base(HEAD, main)`` is ``HEAD``) or goes red when a later task
legitimately edits a guarded file. Scope guards compare a closed range ``BASE..TIP`` of two literal commits
(``tests/_git_range.py``); invariants meant to hold forever pin content (a sha256 literal).

The detector walks the AST of every ``tests/**/*.py`` and looks only at strings that are arguments of a process call
(``subprocess.run``/``check_output``/``check_call``/``call``/``Popen``/``getoutput``/``getstatusoutput``,
``os.system``/``os.popen``) or of a *git helper*: a function or method whose body runs an argv starting with ``"git"``
(a display, ``["git"] + x``, ``[*GIT, ...]``, a name bound to such a list, or a ``"git ..."`` shell string), or calls
another helper (``self.``/``cls.`` included); resolved within the module and through ``tests`` imports. An argument
contributes its literals: constants; f-strings rendered whole (a field is its bound literal, else an opaque
placeholder); names bound to literals (module, function, ``for`` targets, ``pytest.mark.parametrize`` values,
``from tests... import NAME``); ``alias.NAME`` of a ``tests`` module; and, one level, the ``return`` literals of a
module-level function called in the argument. Shell strings starting ``git `` are ``shlex``-split. Strings come from
call arguments only: docstrings, comments, ``hasattr(m, "main")`` and the like are never arguments of such a call.

Exceptions (T-0094 criterion 1): ``rev-parse HEAD`` provenance, unless the name it is assigned to reaches another git
call in the same function (T-0094c hole 8); calls into a scratch repo, recognised by position only (a tmp-derived name
after ``-C``, in ``cwd=``/``repo=``, after ``git init``, or as the first argument of a helper that passes it there;
T-0094c hole 7); and the two monotone ``HEAD``-as-descendant controls in ``test_cd_tap_composition.py``, allowlisted by
exact location (module, function, constant).

The moving-tip check (T-0094c) flags real-repo git calls comparing a literal commit that is an ancestor of
``CLOSED_AS_OF`` with the working tree: ``diff`` with exactly one revision before ``--`` (helpers inlined, their
parameters substituted); ``show <rev>:<path>`` in a function that also reads the same path with
``read_text()``/``read_bytes()``; ``status --porcelain --``/``ls-files --others --`` beside such a diff. Frozen-copy
and oracle ``show`` rows (a fixed blob against a fixed copy) are not flagged; unresolvable revisions are not flagged.

Not worth closing (the closed-range helper plus review cover them): split literals (``"ma" + "in"``,
``"".join(...)``); refs read at run time (environment variables, config files, ``git config``, command output other
than ``rev-parse HEAD``); ``getattr(subprocess, "run")``, ``importlib`` and other dynamic dispatch; multi-level data
flow (a return value of a return value, a helper's helper's parameter beyond three levels).
"""

from __future__ import annotations

import ast
import functools
import itertools
import re
import shlex
import subprocess
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests import _git_range as gr

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests"
SUBPROCESS_FUNCS = frozenset(
    {"run", "check_output", "check_call", "call", "Popen", "getoutput", "getstatusoutput"}
)
OS_FUNCS = frozenset({"system", "popen"})
#: (module path, enclosing function, constant): the two monotone controls of T-0069 ruling F1 (``HEAD`` only as a
#: descendant of a literal merge commit; T-0094 criterion 1).
ALLOWED = frozenset(
    {
        (
            "tests/cd/integration/test_cd_tap_composition.py",
            "test_f1_t0069_range_is_real_history_and_the_range_check_has_power",
            "HEAD",
        ),
    }
)
#: The seven moving-base guard sites at 6a8600b (T-0094 inventory G1-G7): module and enclosing function.
G_SITES_AT_6A8600B = {
    "G1": (
        "tests/unit/test_toolcall_imports.py",
        "test_conductor_llm_contract_and_plugin_unchanged_from_the_integration_fork_point",
    ),
    "G2": (
        "tests/unit/test_toolcall_native_parse.py",
        "test_router_score_tools_and_the_tool_file_are_byte_identical_to_the_fork_point",
    ),
    "G3": (
        "tests/unit/test_service_docs_are_true.py",
        "test_the_m1_checklist_is_byte_identical_to_mains",
    ),
    "G4": (
        "tests/contract/test_vllm_tap_fields_gpu.py",
        "test_fenced_paths_are_untouched_relative_to_main",
    ),
    "G5": (
        "tests/cd/unit/test_t0087_probe_offline.py",
        "test_t0087_changes_nothing_under_local_astra_plugin_services_or_stored_logs",
    ),
    "G6": (
        "tests/cd/unit/test_toolcall_native_grammar.py",
        "test_t0084a_changes_nothing_outside_decoding_toolcall",
    ),
    "G7": (
        "tests/cd/integration/test_twopass_native_driver.py",
        "test_t0084c_changes_nothing_outside_twopass",
    ),
}

#: The T-0094b merge: every task up to it is done, so no real-repo git call may compare a literal commit that is its
#: ancestor with the working tree. The orchestrator bumps it (here and in ``.claude/rules/tests.md``) in the merge
#: commit of any task that converts or closes scope guards (T-0094c ruling Q-c1); a later latent guard is caught only
#: then.
#: Bumped 2026-10-22 to the T-0131 merge, whose scope rows were closed to fixed commits before it merged.
CLOSED_AS_OF = "39c7cfb"

#: What an f-string field renders to when its expression does not resolve to literals (T-0094c hole 1).
_PH = "\x1a"
#: Revision suffixes: ``~n``/``^n``, peeling ``^{commit}``/``^{tree}`` and reflog/upstream ``@{...}`` (hole 1).
_SUFFIX = r"(?:[~^]\d*|\^\{\w*\}|@\{[^}]*\})*"
#: One revision-like range end: not starting with ``.``, ``/`` or ``-`` (so ``../x`` is a path, not a range).
_REVTOK = rf"[\w^~{{}}@{_PH}-][\w./^~{{}}@{_PH}-]*"
_MOVING_END = (
    rf"(?:main|HEAD|@|origin/[\w./-]*|refs/(?:heads|remotes)/[\w./-]+|(?:FETCH|ORIG|MERGE)_HEAD"
    rf"|T-\d{{4}}[a-z]?-[\w-]+){_SUFFIX}"
)
_TAIL = r"(?::.*)?$"
#: One regex per moving form, keyed so a reviewer can drop each in turn (T-0094c "Tests to write"). A match on the
#: whole (stripped) string flags it.
_RULES: dict[str, re.Pattern[str]] = {
    # T-0094
    "main": re.compile(rf"^main{_SUFFIX}{_TAIL}"),
    "origin": re.compile(rf"^origin/[\w./-]+{_SUFFIX}{_TAIL}"),
    "range": re.compile(
        rf"^(?:{_MOVING_END}\.\.\.?(?:{_REVTOK})?|(?:{_REVTOK})?\.\.\.?{_MOVING_END})$"
    ),
    # T-0094c hole 1: an empty range end is HEAD; reflog/upstream forms; FETCH/ORIG/MERGE_HEAD; ref namespaces
    "empty-end": re.compile(rf"^{_REVTOK}(?<!/)\.\.\.?$"),
    "empty-start": re.compile(rf"^\.\.\.?{_REVTOK}$"),
    "head-reflog": re.compile(rf"^HEAD(?:[~^]\d*|\^\{{\w*\}})*@\{{[^}}]*\}}{_SUFFIX}{_TAIL}"),
    "upstream": re.compile(rf"^@\{{[^}}]*\}}{_SUFFIX}{_TAIL}"),
    "bare-at": re.compile(rf"^@(?:[~^]\d*)*{_TAIL}"),
    "special-heads": re.compile(rf"^(?:FETCH|ORIG|MERGE)_HEAD{_SUFFIX}{_TAIL}"),
    "refs": re.compile(rf"^refs/(?:heads|remotes)/[\w./-]+{_SUFFIX}{_TAIL}"),
    "origin-empty": re.compile(rf"^origin/{_TAIL}"),
    # T-0094c hole 2: a task branch (how G1 hid)
    "task-branch": re.compile(rf"^T-\d{{4}}[a-z]?-[\w.-]+?{_SUFFIX}{_TAIL}"),
}
#: ``HEAD`` (with ``~n``/``^n``/``^{...}``): moving unless the same call is ``rev-parse`` (provenance).
_HEAD = re.compile(rf"^HEAD(?:[~^]\d*|\^\{{\w*\}})*{_TAIL}")
_HEX = re.compile(r"[0-9a-f]{7,40}")
#: Options of ``git`` itself that take a value (skipped when finding the subcommand).
_GLOBAL_OPTS = frozenset({"-C", "-c", "--git-dir", "--work-tree", "--namespace"})


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    function: str
    constant: str


# --------------------------------------------------------------------------- the detector


def _callee(call: ast.Call) -> tuple[str | None, str]:
    """``(alias, name)`` for ``alias.name(...)``, ``(None, name)`` for ``name(...)``."""
    f = call.func
    if isinstance(f, ast.Name):
        return None, f.id
    if isinstance(f, ast.Attribute):
        return (f.value.id if isinstance(f.value, ast.Name) else "?"), f.attr
    return "?", ""


_FN = (ast.FunctionDef, ast.AsyncFunctionDef)
_SELF = (None, "self", "cls")


@dataclass(eq=False)
class _Fn:
    """What the detector needs of one function (or of the module body, ``<module>``), collected in one pass."""

    name: str
    outer: str  # the top-level function it sits in: the scope of the scratch-repo taint
    node: ast.AST
    parent: _Fn | None = None
    calls: list[ast.Call] = field(default_factory=list)
    #: (targets, names in the value, the value)
    assigns: list[tuple[set[str], set[str], ast.expr]] = field(default_factory=list)


@dataclass(eq=False)
class _Module:
    path: str
    tree: ast.Module
    fns: list[_Fn]
    proc: dict[str, set[str]]  # "sp_mods", "sp_funcs", "os_mods", "os_funcs"
    helpers: set[str] = field(default_factory=set)
    by_name: dict[str, list[_Fn]] = field(default_factory=dict)


@dataclass(frozen=True)
class _Tok:
    """One argv element: its possible literal values (``None``: unknown), its source node, and whether it is an
    unexpanded ``*x`` (zero or more unknown elements)."""

    vals: tuple[str, ...] | None
    node: ast.AST | None
    star: bool = False

    @property
    def lit(self) -> str | None:
        return self.vals[0] if self.vals is not None and len(self.vals) == 1 else None


def _index(tree: ast.Module) -> tuple[list[_Fn], dict[str, set[str]]]:
    """Each call and assignment, attributed to its innermost enclosing function, and the names ``subprocess``/``os``
    are reachable by (one walk of the tree)."""
    fns = [_Fn("<module>", "<module>", tree)]
    proc: dict[str, set[str]] = {k: set() for k in ("sp_mods", "sp_funcs", "os_mods", "os_funcs")}

    def visit(node: ast.AST, cur: _Fn, outer: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FN):
                top = child.name if outer == "<module>" else outer
                fn = _Fn(child.name, top, child, cur)
                fns.append(fn)
                visit(child, fn, top)
                continue
            if isinstance(child, ast.Call):
                cur.calls.append(child)
            elif isinstance(child, (ast.Assign, ast.AnnAssign)) and child.value is not None:
                targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                cur.assigns.append(
                    (
                        {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)},
                        {n.id for n in ast.walk(child.value) if isinstance(n, ast.Name)},
                        child.value,
                    )
                )
            elif isinstance(child, ast.Import):
                for a in child.names:
                    if a.name in ("subprocess", "os"):
                        proc["sp_mods" if a.name == "subprocess" else "os_mods"].add(
                            a.asname or a.name
                        )
            elif isinstance(child, ast.ImportFrom) and child.module in ("subprocess", "os"):
                known, key = (
                    (SUBPROCESS_FUNCS, "sp_funcs")
                    if child.module == "subprocess"
                    else (OS_FUNCS, "os_funcs")
                )
                proc[key] |= {a.asname or a.name for a in child.names if a.name in known}
            visit(child, cur, outer)

    visit(tree, fns[0], "<module>")
    return fns, proc


def _str_constants(node: ast.AST) -> Iterator[str]:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub.value


def _literal_bindings(
    body: Iterable[ast.stmt], seed: dict[str, list[str]] | None = None
) -> dict[str, list[str]]:
    """Names bound (in this scope, not nested functions) to literal strings or literal collections of strings,
    including the targets of ``for`` loops over literal tuples/lists; ``seed`` is the enclosing scope's bindings."""
    out: dict[str, list[str]] = dict(seed or {})

    def literal(v: ast.AST) -> list[str] | None:
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return [v.value]
        if isinstance(v, (ast.Tuple, ast.List, ast.Set)):
            strs = list(_str_constants(v))
            return strs if strs else None
        if isinstance(v, ast.Name) and v.id in out:
            return out[v.id]
        return None

    def visit(stmts: Iterable[ast.stmt]) -> None:
        for st in stmts:
            if isinstance(st, (*_FN, ast.ClassDef)):
                continue
            if isinstance(st, ast.Assign):
                lit = literal(st.value)
                if lit is not None:
                    for t in st.targets:
                        if isinstance(t, ast.Name):
                            out[t.id] = lit
            elif isinstance(st, ast.AnnAssign) and st.value is not None:
                lit = literal(st.value)
                if lit is not None and isinstance(st.target, ast.Name):
                    out[st.target.id] = lit
            elif isinstance(st, ast.For):
                lit = literal(st.iter)
                if lit is not None:
                    for t in ast.walk(st.target):
                        if isinstance(t, ast.Name):
                            out[t.id] = lit
            for fld in ("body", "orelse", "finalbody", "handlers"):
                inner = getattr(st, fld, None)
                if isinstance(inner, list):
                    visit(
                        [s for s in inner if isinstance(s, ast.stmt)]
                        + [b for h in inner if isinstance(h, ast.ExceptHandler) for b in h.body]
                    )

    visit(body)
    return out


def _parametrized(fnode: ast.AST, lits: dict[str, list[str]]) -> dict[str, list[str]]:
    """T-0094c hole 3: ``@pytest.mark.parametrize("a,b", [(x, y), ...])`` binds ``a`` to the literal strings in the
    first position of each row, ``b`` to the second (one name: every string of the values)."""
    out: dict[str, list[str]] = {}
    for d in getattr(fnode, "decorator_list", []):
        if not (
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and d.func.attr == "parametrize"
            and len(d.args) >= 2
        ):
            continue
        a0, values = d.args[0], d.args[1]
        if isinstance(a0, ast.Constant) and isinstance(a0.value, str):
            names = [n.strip() for n in a0.value.split(",") if n.strip()]
        else:
            names = list(_str_constants(a0))
        if len(names) > 1 and isinstance(values, (ast.List, ast.Tuple)):
            for row in values.elts:
                items = (
                    row.elts
                    if isinstance(row, (ast.Tuple, ast.List))
                    else row.args
                    if isinstance(row, ast.Call)
                    else []
                )
                for n, item in zip(names, items, strict=False):
                    out.setdefault(n, []).extend(_str_constants(item))
        else:
            flat = (
                lits.get(values.id, [])
                if isinstance(values, ast.Name)
                else list(_str_constants(values))
            )
            for n in names:
                out.setdefault(n, []).extend(flat)
    return {k: v for k, v in out.items() if v}


def _params(f: _Fn) -> list[str]:
    if not isinstance(f.node, _FN):
        return []
    a = f.node.args
    return [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)] + [
        p.arg for p in (a.vararg, a.kwarg) if p is not None
    ]


def _scratch_names(fns: list[_Fn], seed: set[str] | None = None) -> set[str]:
    """``tmp_path``-like names (or ``seed``) and every name assigned from an expression that mentions one
    (iterated), over the given functions (a top-level function and the functions nested in it)."""
    if seed is None:
        tainted = {"tmp_path", "tmp_path_factory", "tmpdir"}
        for f in fns:
            if isinstance(f.node, _FN):
                tainted |= {a.arg for a in f.node.args.args if a.arg.startswith("tmp")}
    else:
        tainted = set(seed)
    assigns = [(t, n) for f in fns for t, n, _ in f.assigns]
    grew = True
    while grew:
        new = {t for targets, names in assigns if names & tainted for t in targets} - tainted
        grew = bool(new)
        tainted |= new
    return tainted


def _names(node: ast.AST | None) -> set[str]:
    return set() if node is None else {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _shell(s: str) -> list[str]:
    """T-0094c hole 5: the tokens of a shell string ``"git ..."``."""
    try:
        return shlex.split(s)
    except ValueError:
        return s.split()


def _is_split(e: ast.AST | None) -> bool:
    """``<string>.split()`` with no arguments."""
    return (
        isinstance(e, ast.Call)
        and isinstance(e.func, ast.Attribute)
        and e.func.attr == "split"
        and isinstance(e.func.value, (ast.Constant, ast.JoinedStr))
        and not e.args
    )


def _moving(constant: str, call_strings: list[str]) -> bool:
    s = constant.strip()
    if any(r.match(s) for r in _RULES.values()):
        return True
    if _HEAD.match(s):
        return "rev-parse" not in call_strings  # provenance: ``rev-parse HEAD``
    return False


def _dotted(rel: str) -> str:
    return ".".join(Path(rel).with_suffix("").parts)


def _norm(path: str) -> str:
    return "/".join(p for p in path.split("/") if p not in ("", "."))


_ANCESTOR: dict[tuple[str, str], bool] = {}


def _is_closed(rev: str, closed_as_of: str) -> bool:
    """A literal commit that is an ancestor of ``closed_as_of`` (one ``merge-base --is-ancestor`` per literal)."""
    if not _HEX.fullmatch(rev):
        return False
    key = (rev, closed_as_of)
    if key not in _ANCESTOR:
        proc = subprocess.run(
            ["git", "-C", str(REPO), "merge-base", "--is-ancestor", rev, closed_as_of],
            capture_output=True,
        )
        _ANCESTOR[key] = proc.returncode == 0
    return _ANCESTOR[key]


class _Scan:
    """The parsed ``sources`` (``{repo-relative path: source}``) and the resolvers both checks share."""

    def __init__(self, sources: dict[str, str]) -> None:
        self.sources = sources
        self.rel_of = {_dotted(rel): rel for rel in sources}
        self.mods: dict[str, _Module] = {}
        self._mlits: dict[str, dict[str, list[str]]] = {}
        self._flits: dict[int, dict[str, list[str]]] = {}
        self._imports: dict[str, tuple[dict[str, str], dict[str, tuple[str, str]]]] = {}
        self._tmp: dict[tuple[str, str], set[str]] = {}
        self._repo_first: dict[int, bool] = {}
        self._parse_all()

    # ------------------------------------------------------------------ parsing and helpers

    def _parse(self, rel: str) -> _Module:
        tree = ast.parse(self.sources[rel])
        fns, proc = _index(tree)
        m = _Module(rel, tree, fns, proc)
        for f in fns[1:]:
            m.by_name.setdefault(f.name, []).append(f)
        return m

    def _parse_all(self) -> None:
        """Parse the sources that can run git and resolve git helpers within and across ``tests`` modules (a
        fixpoint). A module without ``subprocess``/``os`` process calls in its text reaches git only through a helper
        imported from a module that defines one, so it is parsed only if its text names such a module."""
        procs = ("subprocess", "os.system", "os.popen", "from os import")
        self.mods = {
            rel: self._parse(rel)
            for rel, src in self.sources.items()
            if any(p in src for p in procs)
        }
        while True:
            grew = True
            while grew:
                grew = False
                for m in self.mods.values():
                    imported = {
                        name
                        for name, (mod, orig) in self.imports(m)[1].items()
                        if (src_m := self.mods.get(self.rel_of[mod])) and orig in src_m.helpers
                    }
                    helpers = self._local_helpers(m, imported) | imported
                    if helpers != m.helpers:
                        m.helpers = helpers
                        grew = True
            stems = sorted({Path(rel).stem for rel, m in self.mods.items() if m.helpers})
            word = (
                re.compile(r"\b(?:" + "|".join(map(re.escape, stems)) + r")\b") if stems else None
            )
            more = [
                rel
                for rel, src in self.sources.items()
                if rel not in self.mods and word is not None and word.search(src)
            ]
            if not more:
                return
            self.mods |= {rel: self._parse(rel) for rel in more}

    def _local_helpers(self, m: _Module, imported: set[str]) -> set[str]:
        """Functions (and methods) of this module that run git: a process call whose argv starts with ``"git"``, or
        a call to a helper (``self.``/``cls.`` included, T-0094c hole 6)."""
        direct = {
            f.name
            for f in m.fns[1:]
            if any(self._is_proc(c, m) and self._runs_git(c, m, f) for c in f.calls)
        }
        bare = {f.name: {n for a, n in map(_callee, f.calls) if a in _SELF} for f in m.fns[1:]}
        helpers = set(direct)
        grew = True
        while grew:
            known = helpers | imported
            new = {name for name, called in bare.items() if name not in helpers and called & known}
            grew = bool(new)
            helpers |= new
        return helpers

    def _tree(self, rel: str) -> ast.Module:
        m = self.mods.get(rel)
        return m.tree if m is not None else ast.parse(self.sources[rel])

    def imports(self, m: _Module) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
        """``(alias -> tests module, name -> (tests module, name))`` from the module's top-level imports."""
        if m.path not in self._imports:
            aliases: dict[str, str] = {}
            names: dict[str, tuple[str, str]] = {}
            for node in m.tree.body:
                if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    for a in node.names:
                        full = f"{node.module}.{a.name}"
                        if full in self.rel_of:
                            aliases[a.asname or a.name] = full
                        elif node.module in self.rel_of:
                            names[a.asname or a.name] = (node.module, a.name)
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name in self.rel_of and a.asname:
                            aliases[a.asname] = a.name
            self._imports[m.path] = (aliases, names)
        return self._imports[m.path]

    def mlits(self, rel: str) -> dict[str, list[str]]:
        if rel not in self._mlits:
            self._mlits[rel] = _literal_bindings(self._tree(rel).body)
        return self._mlits[rel]

    def flits(self, m: _Module, f: _Fn) -> dict[str, list[str]]:
        """Literal bindings visible in ``f``: the enclosing scope's (minus ``f``'s parameters), ``parametrize``
        values, then ``f``'s own assignments and loops."""
        if f.parent is None:
            return self.mlits(m.path)
        if id(f) not in self._flits:
            seed = dict(self.flits(m, f.parent))
            for p in _params(f):
                seed.pop(p, None)
            seed |= _parametrized(f.node, seed)
            self._flits[id(f)] = _literal_bindings(f.node.body, seed)  # type: ignore[attr-defined]
        return self._flits[id(f)]

    def _fn_named(self, m: _Module, f: _Fn | None, name: str) -> _Fn | None:
        cands = m.by_name.get(name, [])
        scope = f
        while scope is not None:
            for c in cands:
                if c.parent is scope:
                    return c
            scope = scope.parent
        return cands[0] if cands else None

    def _top_fn(self, m: _Module, name: str) -> _Fn | None:
        return next((c for c in m.by_name.get(name, []) if c.parent is m.fns[0]), None)

    def helper(self, call: ast.Call, m: _Module, f: _Fn) -> tuple[_Module, _Fn] | None:
        """The git helper ``call`` runs (local, method, imported by name or through a module alias), if any."""
        alias, name = _callee(call)
        if alias in _SELF:
            if name not in m.helpers:
                return None
            local = self._fn_named(m, f, name)
            if local is not None:
                return m, local
            imp = self.imports(m)[1].get(name)
            target = self.mods.get(self.rel_of[imp[0]]) if imp else None
            fn = self._fn_named(target, None, imp[1]) if target and imp else None
            return (target, fn) if target and fn else None
        dotted = self.imports(m)[0].get(alias or "")
        target = self.mods.get(self.rel_of[dotted]) if dotted else None
        if target is None or name not in target.helpers:
            return None
        fn = self._fn_named(target, None, name)
        return (target, fn) if fn else None

    def _is_proc(self, call: ast.Call, m: _Module) -> bool:
        """``subprocess.run``/``check_output``/.../``getoutput``/``getstatusoutput`` and ``os.system``/``os.popen``
        (T-0094c hole 5), through a module alias or a ``from`` import."""
        f, p = call.func, m.proc
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            return (f.value.id in p["sp_mods"] and f.attr in SUBPROCESS_FUNCS) or (
                f.value.id in p["os_mods"] and f.attr in OS_FUNCS
            )
        return isinstance(f, ast.Name) and (f.id in p["sp_funcs"] or f.id in p["os_funcs"])

    def is_git_call(self, call: ast.Call, m: _Module, f: _Fn) -> bool:
        return self._is_proc(call, m) or self.helper(call, m, f) is not None

    def _runs_git(self, call: ast.Call, m: _Module, f: _Fn) -> bool:
        toks = self.argv(call, m, f, {})
        return bool(toks) and toks[0].lit == "git"  # type: ignore[index]

    # ------------------------------------------------------------------ values and argv

    def values(
        self, e: ast.AST, m: _Module, f: _Fn, env: dict[str, object], calls: bool = True
    ) -> tuple[str, ...] | None:
        """The literal strings ``e`` may evaluate to, or ``None``: constants, f-strings (rendered whole), names
        bound to literals (module, function, loop, ``parametrize``, ``from tests... import``), ``alias.NAME`` of a
        ``tests`` module and, one level, ``f()`` of a module-level function (its ``return`` literals)."""
        if isinstance(e, ast.Constant):
            return (e.value,) if isinstance(e.value, str) else None
        if isinstance(e, ast.JoinedStr):
            return self._render(e, m, f, env)
        if isinstance(e, ast.Name):
            if e.id in env:
                t = env[e.id]
                return t.vals if isinstance(t, _Tok) else None
            lits = self.flits(m, f)
            if e.id in lits:
                return tuple(lits[e.id])
            imp = self.imports(m)[1].get(e.id)
            got = self.mlits(self.rel_of[imp[0]]).get(imp[1]) if imp else None
            return tuple(got) if got else None
        if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name):
            dotted = self.imports(m)[0].get(e.value.id)
            got = self.mlits(self.rel_of[dotted]).get(e.attr) if dotted else None
            return tuple(got) if got else None
        if isinstance(e, ast.IfExp):
            a, b = self.values(e.body, m, f, env, calls), self.values(e.orelse, m, f, env, calls)
            return a + b if a is not None and b is not None else None
        if calls and isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
            fn = self._top_fn(m, e.func.id)
            if fn is None:
                return None
            out: list[str] = []
            for node in self._returns(fn.node):
                got = self.values(node, m, fn, {}, calls=False)
                out += got or ()
            return tuple(out) if out else None
        return None

    @staticmethod
    def _returns(fnode: ast.AST) -> Iterator[ast.expr]:
        stack = list(ast.iter_child_nodes(fnode))
        while stack:
            node = stack.pop()
            if isinstance(node, (*_FN, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(node, ast.Return) and node.value is not None:
                yield node.value
            stack.extend(ast.iter_child_nodes(node))

    def _render(
        self, e: ast.JoinedStr, m: _Module, f: _Fn, env: dict[str, object]
    ) -> tuple[str, ...]:
        """An f-string as whole strings: each ``{...}`` is its bound literal(s), else the placeholder."""
        parts: list[tuple[str, ...]] = []
        for v in e.values:
            if isinstance(v, ast.Constant):
                parts.append((str(v.value),))
            elif isinstance(v, ast.FormattedValue):
                parts.append(self.values(v.value, m, f, env) or (_PH,))
        return tuple("".join(c) for c in itertools.islice(itertools.product(*parts), 64))

    def strings(self, e: ast.AST, m: _Module, f: _Fn) -> Iterator[str]:
        """Every literal string an argument expression carries (the moving-ref check); shell strings starting
        ``git `` also yield their tokens."""
        got = self.values(e, m, f, {})
        if got is not None:
            for s in got:
                yield s
                if s.startswith("git "):
                    yield from _shell(s)
            if not isinstance(e, ast.Call):
                return
        if isinstance(e, ast.Name):
            # T-0094c hole 5: ``cmd = "git diff main".split()`` then ``subprocess.run(cmd)``
            d = self._display(e.id, m, f)
            if _is_split(d):
                yield from self.strings(d, m, f)  # type: ignore[arg-type]
            return
        for child in ast.iter_child_nodes(e):
            yield from self.strings(child, m, f)

    def _display(self, name: str, m: _Module, f: _Fn) -> ast.expr | None:
        """The list/tuple display (or ``[...] + x``, or ``"git ...".split()``) ``name`` was last assigned in ``f`` or
        an enclosing scope."""
        scope: _Fn | None = f
        while scope is not None:
            for targets, _, value in reversed(scope.assigns):
                if targets == {name} and (
                    isinstance(value, (ast.List, ast.Tuple, ast.BinOp)) or _is_split(value)
                ):
                    return value
            scope = scope.parent
        return None

    def _star(
        self, v: ast.expr, m: _Module, f: _Fn, env: dict[str, object], depth: int
    ) -> list[_Tok]:
        if isinstance(v, ast.Name):
            if v.id in env:
                t = env[v.id]
                return list(t) if isinstance(t, list) else [_Tok(None, v, True)]
            d = self._display(v.id, m, f)
            if d is not None and depth < 4:
                got = self.expand(d, m, f, env, depth + 1)
                if got is not None:
                    return got
            vals = self.values(v, m, f, env)
            if vals is not None:
                return [_Tok((s,), v) for s in vals]
        if (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Name)
            and v.func.id in ("list", "tuple", "sorted")
            and len(v.args) == 1
        ):
            return self._star(v.args[0], m, f, env, depth)
        return [_Tok(None, v, True)]

    def expand(
        self, e: ast.expr, m: _Module, f: _Fn, env: dict[str, object], depth: int = 0
    ) -> list[_Tok] | None:
        """An argv expression as tokens: list/tuple displays, ``*x`` of a bound display or literal collection,
        ``[...] + x``, a name bound to a display (T-0094c hole 6), and shell strings (hole 5)."""
        if isinstance(e, (ast.List, ast.Tuple)):
            out: list[_Tok] = []
            for el in e.elts:
                if isinstance(el, ast.Starred):
                    out += self._star(el.value, m, f, env, depth)
                else:
                    out.append(_Tok(self.values(el, m, f, env), el))
            return out
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
            left = self.expand(e.left, m, f, env, depth)
            if left is None:
                return None
            return left + (
                self.expand(e.right, m, f, env, depth) or self._star(e.right, m, f, env, depth)
            )
        if isinstance(e, ast.Name):
            if isinstance(env.get(e.id), list):
                return list(env[e.id])  # type: ignore[call-overload]
            d = self._display(e.id, m, f)
            return self.expand(d, m, f, env, depth + 1) if d is not None and depth < 4 else None
        if (
            isinstance(e, ast.Call)
            and isinstance(e.func, ast.Attribute)
            and e.func.attr == "split"
            and not e.args
        ):
            e = e.func.value
        if isinstance(e, (ast.Constant, ast.JoinedStr)):
            got = self.values(e, m, f, env)
            if got is not None and len(got) == 1 and got[0].startswith("git "):
                return [_Tok((t,), e) for t in _shell(got[0])]
        return None

    def argv(self, call: ast.Call, m: _Module, f: _Fn, env: dict[str, object]) -> list[_Tok] | None:
        return self.expand(call.args[0], m, f, env) if call.args else None

    # ------------------------------------------------------------------ scratch repos and provenance

    def tmp_names(self, m: _Module, f: _Fn) -> set[str]:
        key = (m.path, f.outer)
        if key not in self._tmp:
            self._tmp[key] = _scratch_names([g for g in m.fns if g.outer == f.outer])
        return self._tmp[key]

    def is_scratch(self, call: ast.Call, m: _Module, f: _Fn, tainted: set[str]) -> bool:
        """T-0094c hole 7: a scratch-repo call only when a tmp-derived name selects the repo: the value after
        ``-C``, ``cwd=``/``repo=``, ``git init <x>``, or a helper's first positional parameter that the helper passes
        to one of those. A tmp name anywhere else (``stdout=``) does not exempt the call."""
        if any(k.arg in ("cwd", "repo") and _names(k.value) & tainted for k in call.keywords):
            return True
        if self._is_proc(call, m):
            toks = self.argv(call, m, f, {}) or []
            for i, t in enumerate(toks):
                if t.lit == "-C" and i + 1 < len(toks) and _names(toks[i + 1].node) & tainted:
                    return True
                if t.lit == "init" and any(_names(u.node) & tainted for u in toks[i + 1 :]):
                    return True
            return False
        target = self.helper(call, m, f)
        return (
            target is not None
            and self._first_selects_repo(*target)
            and bool(call.args)
            and not isinstance(call.args[0], ast.Starred)
            and bool(_names(call.args[0]) & tainted)
        )

    def _first_selects_repo(self, m: _Module, h: _Fn) -> bool:
        if id(h) not in self._repo_first:
            self._repo_first[id(h)] = False  # recursion guard
            params = [p for p in _params(h) if p not in ("self", "cls")]
            if params and isinstance(h.node, _FN) and h.node.args.args:
                taint = _scratch_names([h], {params[0]})
                self._repo_first[id(h)] = any(
                    self.is_git_call(c, m, h) and self.is_scratch(c, m, h, taint) for c in h.calls
                )
        return self._repo_first[id(h)]

    # ------------------------------------------------------------------ the moving-ref check

    def moving_refs(self) -> list[Hit]:
        hits: list[Hit] = []
        for rel, m in self.mods.items():
            for f in m.fns:
                calls = [c for c in f.calls if self.is_git_call(c, m, f)]
                if not calls:
                    continue
                tainted = self.tmp_names(m, f)
                real = [c for c in calls if not self.is_scratch(c, m, f, tainted)]
                strs = {
                    id(c): [
                        s
                        for a in (*c.args, *(k.value for k in c.keywords))
                        for s in self.strings(a, m, f)
                    ]
                    for c in real
                }
                for c in real:
                    hits += [
                        Hit(rel, c.lineno, f.name, s.replace(_PH, "{}"))
                        for s in strs[id(c)]
                        if _moving(s, strs[id(c)])
                    ]
                hits += self._provenance_hits(rel, f, real, strs)
        return sorted(set(hits), key=lambda h: (h.path, h.line, h.constant))

    @staticmethod
    def _provenance_hits(
        rel: str, f: _Fn, real: list[ast.Call], strs: dict[int, list[str]]
    ) -> list[Hit]:
        """T-0094c hole 8: a name assigned from ``rev-parse HEAD`` (in this function) passed to another git call."""
        rp = {
            id(c)
            for c in real
            if "rev-parse" in strs[id(c)] and any(_HEAD.match(s) for s in strs[id(c)])
        }
        if not rp:
            return []
        prov = {
            t
            for targets, _, value in f.assigns
            if any(id(n) in rp for n in ast.walk(value))
            for t in targets
        }
        grew = bool(prov)
        while grew:
            new = {t for targets, names, _ in f.assigns if names & prov for t in targets} - prov
            grew = bool(new)
            prov |= new
        return [
            Hit(rel, c.lineno, f.name, f"rev-parse HEAD -> {n}")
            for c in real
            if id(c) not in rp
            for n in sorted(_names(c) & prov)
        ]

    # ------------------------------------------------------------------ the moving-tip check

    def _bind(
        self, call: ast.Call, m: _Module, f: _Fn, env: dict[str, object], h: _Fn
    ) -> dict[str, object]:
        """The helper's parameters bound to the call's arguments (``*args`` to a token list)."""
        assert isinstance(h.node, _FN)
        a = h.node.args
        pos = [p.arg for p in (*a.posonlyargs, *a.args)]
        if pos and pos[0] in ("self", "cls") and _callee(call)[0] in ("self", "cls"):
            pos = pos[1:]
        toks: list[_Tok] = []
        for arg in call.args:
            if isinstance(arg, ast.Starred):
                toks += self._star(arg.value, m, f, env, 0)
            else:
                toks.append(_Tok(self.values(arg, m, f, env), arg))
        out: dict[str, object] = {}
        i = 0
        for p in pos:
            if i >= len(toks) or toks[i].star:
                break
            out[p] = toks[i]
            i += 1
        if a.vararg is not None:
            out[a.vararg.arg] = toks[i:]
        for k in call.keywords:
            if k.arg:
                out[k.arg] = _Tok(self.values(k.value, m, f, env), k.value)
        return out

    def argvs(
        self, call: ast.Call, m: _Module, f: _Fn, env: dict[str, object], depth: int = 0
    ) -> list[list[_Tok]]:
        """Every git argv ``call`` runs, through helpers (parameters substituted; three levels)."""
        if self._is_proc(call, m):
            toks = self.argv(call, m, f, env)
            return [toks] if toks and toks[0].lit == "git" else []
        target = self.helper(call, m, f)
        if target is None or depth >= 3 or not (call.args or call.keywords):
            # a helper called with no arguments has nothing to substitute: its own body is scanned in its own right
            return []
        m2, h = target
        env2 = self._bind(call, m, f, env, h)
        return [
            toks
            for c in h.calls
            if self.is_git_call(c, m2, h)
            for toks in self.argvs(c, m2, h, env2, depth + 1)
        ]

    @staticmethod
    def _classify(toks: list[_Tok], closed_as_of: str) -> Iterator[tuple[str, str, str]]:
        """``(form, constant, path)``: (a) ``diff`` with exactly one revision (before ``--``) that is a closed
        literal commit; (b) ``show <closed>:<path>``; (c) ``status --porcelain --`` / ``ls-files --others --``."""
        i = 1
        while i < len(toks):
            v = toks[i].lit
            if v in _GLOBAL_OPTS:
                i += 2
            elif toks[i].star or (v is not None and v.startswith("-")):
                i += 1
            else:
                break
        if i >= len(toks):
            return
        sub, rest = toks[i].lit, toks[i + 1 :]
        lits = [t.lit for t in rest]
        before = rest[: lits.index("--")] if "--" in lits else rest
        if sub == "diff":
            revs = [t for t in before if not t.star and not (t.lit or "").startswith("-")]
            if len(revs) == 1:
                for v in revs[0].vals or ():
                    if _is_closed(v, closed_as_of):
                        yield "a", v, ""
        elif sub == "show":
            for t in before:
                for v in t.vals or ():
                    rev, _, path = v.partition(":")
                    if path and _is_closed(rev, closed_as_of):
                        yield "b", v, path
        elif "--" in lits and (
            (sub == "status" and any((x or "").startswith("--porcelain") for x in lits))
            or (sub == "ls-files" and ("--others" in lits or "-o" in lits))
        ):
            yield "c", "--porcelain" if sub == "status" else "--others", ""

    def _path_strs(self, e: ast.AST, m: _Module, f: _Fn, depth: int = 0) -> list[str]:
        """The repo-relative paths a ``Path`` expression may denote (``REPO / "a" / "b.py"`` is ``a/b.py``)."""
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Div):
            left, right = (
                self._path_strs(e.left, m, f, depth),
                self._path_strs(e.right, m, f, depth),
            )
            return [f"{a}/{b}" for a in left for b in right][:16]
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "Path":
            parts = [self._path_strs(a, m, f, depth) for a in e.args] or [[""]]
            return ["/".join(c) for c in itertools.islice(itertools.product(*parts), 16)]
        got = self.values(e, m, f, {}, calls=False)
        if got is not None:
            return list(got)
        if isinstance(e, ast.Name) and depth < 3:
            scope: _Fn | None = f
            while scope is not None:
                for targets, _, value in reversed(scope.assigns):
                    if targets == {e.id}:
                        return self._path_strs(value, m, f, depth + 1)
                scope = scope.parent
        return [""]

    def moving_tips(self, closed_as_of: str) -> list[Hit]:
        hits: list[Hit] = []
        for rel, m in self.mods.items():
            for f in m.fns:
                calls = [c for c in f.calls if self.is_git_call(c, m, f)]
                if not calls:
                    continue
                tainted = self.tmp_names(m, f)
                found: dict[str, list[tuple[Hit, str]]] = {"a": [], "b": [], "c": []}
                for c in calls:
                    if self.is_scratch(c, m, f, tainted):
                        continue
                    for toks in self.argvs(c, m, f, {}):
                        for form, const, path in self._classify(toks, closed_as_of):
                            found[form].append((Hit(rel, c.lineno, f.name, const), path))
                if not found["a"] and not found["b"]:
                    continue
                hits += [h for h, _ in found["a"]]
                if found["a"]:
                    hits += [h for h, _ in found["c"]]
                if found["b"]:
                    reads = {
                        _norm(p)
                        for c in f.calls
                        if isinstance(c.func, ast.Attribute)
                        and c.func.attr in ("read_text", "read_bytes")
                        for p in self._path_strs(c.func.value, m, f)
                    }
                    hits += [
                        h
                        for h, path in found["b"]
                        if any(r == _norm(path) or r.endswith("/" + _norm(path)) for r in reads)
                    ]
        return sorted(set(hits), key=lambda h: (h.path, h.line, h.constant))


def scan(sources: dict[str, str]) -> list[Hit]:
    """Every moving-ref constant passed to git in ``sources`` (``{repo-relative path: source}``), before the
    allowlist."""
    return _Scan(sources).moving_refs()


def tip_scan(sources: dict[str, str], closed_as_of: str | None = None) -> list[Hit]:
    """T-0094c moving-tip half: real-repo git calls comparing a literal commit that is an ancestor of
    ``closed_as_of`` (default: the module's ``CLOSED_AS_OF``, read at call time) with the working tree."""
    return _Scan(sources).moving_tips(CLOSED_AS_OF if closed_as_of is None else closed_as_of)


def _tree_sources() -> dict[str, str]:
    return {
        str(p.relative_to(REPO)): p.read_text(encoding="utf-8")
        for p in sorted(TESTS.rglob("*.py"))
        if "__pycache__" not in p.parts
    }


@functools.cache
def _tree_scan() -> _Scan:
    """One parse of this tree, shared by the two tree rows."""
    return _Scan(_tree_sources())


def _unallowed(hits: list[Hit]) -> list[Hit]:
    return [h for h in hits if (h.path, h.function, h.constant) not in ALLOWED]


# --------------------------------------------------------------------------- the rows


def test_no_guard_compares_against_a_moving_ref() -> None:
    """Criterion 10 (and 1): on this tree no git call in ``tests/`` passes a moving ref, apart from the allowlist; and
    every allowlist entry is still needed (no stale exemption)."""
    hits = _tree_scan().moving_refs()
    bad = _unallowed(hits)
    assert bad == [], "moving refs passed to git:\n" + "\n".join(
        f"{h.path}:{h.line} ({h.function}): {h.constant!r}" for h in bad
    )
    used = {(h.path, h.function, h.constant) for h in hits}
    assert used >= ALLOWED, f"stale allowlist entries: {sorted(ALLOWED - used)}"


def test_the_detector_flags_every_known_moving_ref_site_at_6a8600b() -> None:
    """Criterion 10, too wide: over the 6a8600b blobs of the seven G modules the detector flags each G site, in its
    own function; skips if 6a8600b is absent."""
    gr.require_commits(gr.LATER)
    sources = {}
    for path, _fn in G_SITES_AT_6A8600B.values():
        blob = subprocess.run(
            ["git", "-C", str(REPO), "show", f"6a8600b:{path}"],
            capture_output=True,
            text=True,
            check=True,
        )
        sources[path] = blob.stdout
    hits = _unallowed(scan(sources))
    flagged = {(h.path, h.function) for h in hits}
    missed = {g: site for g, site in G_SITES_AT_6A8600B.items() if site not in flagged}
    assert missed == {}, f"not flagged at 6a8600b: {missed}"
    assert {
        h.constant for h in hits if (h.path, h.function) in set(G_SITES_AT_6A8600B.values())
    } >= {
        "main",
        "HEAD",
        "main:hud/README.md",
    }


_IGNORED = '''
"""A docstring quoting ``git show main:hud/README.md`` and ``merge-base HEAD main``."""
import subprocess
from pathlib import Path

REPO = Path(".")


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)


def test_attribute_names(m):
    # a comment: git diff main..HEAD
    assert hasattr(m, "main")
    assert getattr(m, "main", None) is not None
    print("main", "HEAD", "origin/main")


def test_provenance():
    head = _git("rev-parse", "HEAD").stdout
    short = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True).stdout
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True).stdout
    return head, short, dirty


def test_scratch_repo(tmp_path):
    git = ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    subprocess.run([*git, "checkout", "-q", "main"], check=True)
    top = tmp_path / "repo"
    subprocess.run(["git", "-C", str(top), "diff", "main..HEAD"], check=True)
    subprocess.run([*git, "merge-base", "HEAD", "main"], check=True)


def test_fixed_range():
    return _git("diff", "--name-only", "1e06a24", "f0a0561", "--", "local_astra")
'''

_FLAGGED = """
import subprocess
from subprocess import check_output as co

REF = "main"
BASES = ("T-0062-perstep-cd", "origin/T-0062-perstep-cd", "main")


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def _diff(a, b):
    return _git("diff", "--name-only", a, b)


def test_a():
    _git("merge-base", "HEAD", "x")


def test_b():
    subprocess.run(["git", "show", "main:hud/README.md"], check=True)


def test_c():
    _diff("abc1234", "HEAD")


def test_d():
    for ref in BASES:
        _git("rev-parse", "--verify", ref)


def test_e():
    co(["git", "diff", "abc1234..main"])


def test_f():
    _git("diff", "--name-only", REF)


def test_g():
    _git("log", "origin/main")
"""


def test_the_detector_ignores_prose_attribute_names_provenance_and_scratch_repos() -> None:
    """Criterion 10, too narrow: ``hasattr(m, "main")``, ``getattr(m, "main", None)``, a docstring or comment quoting
    ``git show main:``, ``rev-parse HEAD`` provenance, ``status --porcelain``, scratch-repo calls and a closed literal
    range are not flagged. Adversarial companion: each moving form in ``_FLAGGED`` is, once per function."""
    assert scan({"tests/x/test_ignored.py": _IGNORED}) == []
    hits = scan({"tests/x/test_flagged.py": _FLAGGED})
    by_fn = {h.function for h in hits}
    assert by_fn == {"test_a", "test_b", "test_c", "test_d", "test_e", "test_f", "test_g"}, hits


def test_the_detector_resolves_git_helpers_imported_from_another_tests_module() -> None:
    """A guard that calls a git helper defined in another ``tests`` module (as ``tests/_git_range.py``) is scanned
    too: ``gr.changed("abc1234", "main", ...)`` and ``changed(..., "HEAD", ...)`` are flagged."""
    helper = (
        "import subprocess\n"
        "def _git(*a):\n    return subprocess.run(['git', *a])\n"
        "def changed(base, tip, *paths):\n    return _git('diff', '--name-only', base, tip, '--', *paths)\n"
    )
    user = (
        "from tests import _helper as gr\nfrom tests._helper import changed\n"
        "def test_x():\n    gr.changed('abc1234', 'main', 'local_astra')\n"
        "def test_y():\n    changed('abc1234', 'HEAD', 'plugin')\n"
        "def test_z():\n    gr.changed('abc1234', 'def5678', 'plugin')\n"
    )
    hits = scan({"tests/_helper.py": helper, "tests/x/test_user.py": user})
    assert {(h.function, h.constant) for h in hits} == {("test_x", "main"), ("test_y", "HEAD")}, (
        hits
    )


# --------------------------------------------------------------------------- T-0094c: the review's holes

#: A constants-only ``tests`` module (no ``subprocess``): the target of ``R.BRANCH`` and ``from ... import BRANCH``.
_REFS_B = """
BRANCH = "main"
FIXED = "abc1234"
"""

_FLAGGED_B = """
import os
import subprocess

import pytest

from tests.x import _refs as R
from tests.x._refs import BRANCH

REPO = "/repo"
GIT = ["git", "-C", REPO]
PEELED = "main"


def _git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def _git_concat(*args):
    return subprocess.check_output(["git"] + list(args))


def _git_star(*args):
    return subprocess.run([*GIT, *args], capture_output=True)


def _git_named(*args):
    cmd = ["git", *args]
    return subprocess.run(cmd, capture_output=True)


def _ref():
    return "main"


def test_h1_empty_end_range():
    _git("diff", "--name-only", "abc1234..", "--", "local_astra")


def test_h1_empty_end_symmetric_range():
    _git("log", "abc1234...")


def test_h1_empty_start_range():
    _git("log", "..abc1234")


def test_h1_peeled_suffix():
    _git("cat-file", "-e", "main^{commit}")


def test_h1_upstream():
    _git("diff", "@{u}")


def test_h1_reflog():
    _git("diff", "HEAD@{1}")


def test_h1_previous_branch():
    _git("diff", "@{-1}")


def test_h1_bare_at():
    _git("diff", "@")


def test_h1_special_head():
    _git("diff", "FETCH_HEAD")
    _git("diff", "ORIG_HEAD")
    _git("diff", "MERGE_HEAD")


def test_h1_refs_namespace():
    _git("diff", "refs/heads/feature")


def test_h1_bare_origin_prefix():
    _git("diff", "origin/")


def test_h1_fstring_bound_literal():
    _git("cat-file", "-e", f"{PEELED}^{{commit}}")


def test_h1_fstring_empty_end(base):
    _git("diff", f"{base}..")


def test_h2_task_branch():
    _git("diff", "--name-only", "T-0062-perstep-cd", "--", "local_astra")


def test_h2_task_branch_with_suffixes():
    _git("diff", "T-0062-perstep-cd~1")
    _git("diff", "T-0062-perstep-cd@{u}")
    _git("cat-file", "-e", "T-0062-perstep-cd^{commit}")


@pytest.mark.parametrize("ref", ["abc1234", "main"])
def test_h3_parametrize(ref):
    _git("diff", "--name-only", ref, "--", "local_astra")


@pytest.mark.parametrize("path,ref", [("local_astra", "origin/main")])
def test_h3_parametrize_tuple(path, ref):
    _git("diff", ref, "--", path)


def test_h4_module_attribute():
    _git("diff", R.BRANCH)


def test_h4_imported_name():
    _git("diff", BRANCH)


def test_h4_return_value():
    _git("diff", _ref())


def test_h5_shell_true():
    subprocess.run("git diff --name-only main -- local_astra", shell=True)


def test_h5_getoutput():
    subprocess.getoutput("git log -1 HEAD~1")


def test_h5_getstatusoutput():
    subprocess.getstatusoutput("git diff origin/main")


def test_h5_os_system():
    os.system("git diff main")


def test_h5_os_popen():
    os.popen("git show main:hud/README.md")


def test_h5_split():
    subprocess.run("git diff main".split(), check=True)


def test_h5_named_split():
    cmd = "git diff main".split()
    subprocess.run(cmd, check=True)


def test_h6_concat_argv():
    _git_concat("diff", "main")


def test_h6_starred_git_list():
    _git_star("diff", "main")


def test_h6_named_argv():
    _git_named("diff", "main")


class TestMethods:
    def run_git(self, *args):
        return subprocess.run(["git", *args])

    def test_h6_method(self):
        self.run_git("diff", "main")


def _git_logged(out, *args):
    return subprocess.run(["git", "-C", REPO, *args], stdout=out)


def test_h7_helper_first_parameter_is_not_the_repo(tmp_path):
    _git_logged((tmp_path / "log").open("w"), "diff", "main")


def test_h7_tmp_elsewhere(tmp_path):
    subprocess.run(["git", "-C", str(REPO), "diff", "main"], stdout=(tmp_path / "x").open("w"))


def test_h8_provenance_taint():
    head = _git("rev-parse", "HEAD").stdout.strip()
    _git("diff", "--name-only", head, "--", "local_astra")
"""

#: Hole number (T-0094c "Holes to close") -> the ``_FLAGGED_B`` functions that only that hole's rule flags.
_HOLES = {
    "1-ranges-and-suffixes": {
        "test_h1_empty_end_range",
        "test_h1_empty_end_symmetric_range",
        "test_h1_empty_start_range",
        "test_h1_peeled_suffix",
        "test_h1_upstream",
        "test_h1_reflog",
        "test_h1_previous_branch",
        "test_h1_bare_at",
        "test_h1_special_head",
        "test_h1_refs_namespace",
        "test_h1_bare_origin_prefix",
        "test_h1_fstring_empty_end",
        "test_h1_fstring_bound_literal",
    },
    "2-task-branch": {"test_h2_task_branch", "test_h2_task_branch_with_suffixes"},
    "3-parametrize": {"test_h3_parametrize", "test_h3_parametrize_tuple"},
    "4-attribute-and-return": {
        "test_h4_module_attribute",
        "test_h4_imported_name",
        "test_h4_return_value",
    },
    "5-shell-strings": {
        "test_h5_shell_true",
        "test_h5_getoutput",
        "test_h5_getstatusoutput",
        "test_h5_os_system",
        "test_h5_os_popen",
        "test_h5_split",
        "test_h5_named_split",
    },
    "6-helper-shapes": {
        "test_h6_concat_argv",
        "test_h6_starred_git_list",
        "test_h6_named_argv",
        "test_h6_method",
    },
    "7-scratch-by-position": {
        "test_h7_tmp_elsewhere",
        "test_h7_helper_first_parameter_is_not_the_repo",
    },
    "8-provenance-taint": {"test_h8_provenance_taint"},
}

_IGNORED_B = """
import subprocess

import pytest

BASE = "e760cb7"
TIP = "154614d"
REPO = "/repo"


def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)


def _git_in(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def test_closed_fstring_range():
    _git("diff", "--name-only", f"{BASE}..{TIP}", "--", "local_astra")


def test_fixed_start_and_a_placeholder_end(ev):
    _git("log", "--format=%H", f"806da12..{ev.sha}")


def test_a_task_id_that_is_not_a_ref():
    _git("log", "--grep", "T-0092b", "abc1234..def5678")
    _git("log", "--grep=T-0092b-perstep", "abc1234..def5678")
    _git("diff", "--name-only", "abc1234", "def5678", "--", "docs/tasks/T-0094c-detector.md")


def test_scratch_calls_by_position(tmp_path):
    repo = tmp_path / "r"
    git = ["git", "-C", str(repo)]
    subprocess.run(["git", "-C", str(repo), "diff", "main"], check=True)
    subprocess.run(["git", "diff", "main"], cwd=repo, check=True)
    subprocess.run([*git, "merge-base", "HEAD", "main"], check=True)
    _git_in(repo, "diff", "HEAD~1..")
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path / "s")], check=True)


def test_rev_parse_head_for_a_record_only(tmp_path):
    head = _git("rev-parse", "HEAD").stdout.strip()
    record = {"head": head}
    (tmp_path / "r.json").write_text(str(record))


@pytest.mark.parametrize("path", ["local_astra/conductor", "plugin", "docs/tasks/T-0094c.md"])
def test_parametrize_over_paths(path):
    _git("diff", "--name-only", BASE, TIP, "--", path)


def test_peeled_literal_commits(sha):
    _git("cat-file", "-e", f"{sha}^{{commit}}")
    _git("cat-file", "-e", f"{BASE}^{{commit}}")
"""


def _scan_b() -> list[Hit]:
    return scan({"tests/x/test_flagged_b.py": _FLAGGED_B, "tests/x/_refs.py": _REFS_B})


@pytest.mark.parametrize("hole", sorted(_HOLES))
def test_the_detector_flags_each_review_hole(hole: str) -> None:
    """T-0094c criterion 1: every fixture function of this hole is flagged, each asserted by name. Each function
    holds a form the T-0094 detector missed (see the review note in ``docs/tasks/T-0094.md``)."""
    flagged = {h.function for h in _scan_b()}
    missed = sorted(_HOLES[hole] - flagged)
    assert missed == [], f"hole {hole}: not flagged: {missed}"


def test_the_review_hole_fixtures_flag_exactly_their_functions() -> None:
    """T-0094c criterion 1, exactness: no helper of ``_FLAGGED_B`` (``_git_concat``, ``run_git``, ``_ref`` ...) is
    flagged, so a hit means the guard function, and the T-0094 ``_FLAGGED`` set stays exact (its own row)."""
    flagged = {h.function for h in _scan_b()}
    assert flagged == set().union(*_HOLES.values()), sorted(flagged ^ set().union(*_HOLES.values()))


def test_the_detector_ignores_closed_fstring_ranges_and_positional_scratch_calls() -> None:
    """T-0094c criterion 2, too narrow: ``f"{BASE}..{TIP}"`` with literals, ``f"806da12..{ev.sha}"``, ``T-0092b``
    as a ``--grep`` value or in a path, scratch calls with tmp in the repo-selecting position, ``rev-parse HEAD``
    kept for a record, a parametrize over paths and peeled literal commits are not flagged."""
    hits = scan({"tests/x/test_ignored_b.py": _IGNORED_B})
    assert hits == [], hits


def test_a_real_repo_call_mentioning_tmp_elsewhere_is_scanned() -> None:
    """T-0094c criterion 3 (hole 7, adversarial): tmp in ``stdout=`` does not make a real-repo call a scratch call."""
    src = (
        "import subprocess\nfrom pathlib import Path\nREPO = Path('.')\n"
        "def test_x(tmp_path):\n"
        '    subprocess.run(["git", "-C", str(REPO), "diff", "main"], stdout=(tmp_path / "x").open("w"))\n'
    )
    hits = scan({"tests/x/test_tmp_elsewhere.py": src})
    assert [(h.function, h.line, h.constant) for h in hits] == [("test_x", 5, "main")], hits


# --------------------------------------------------------------------------- T-0094c: the moving-tip check

#: (module path, enclosing function, revision) exempt from the moving-tip check, by exact location. The one entry is
#: not a guard: a one-shot capture script (never imported by a test) whose precondition is "run on a checkout of
#: 9669258", so comparing that base with the working tree is its purpose (T-0088 criterion 2's golden).
TIP_ALLOWED = frozenset(
    {("tests/fixtures/llm/capture_request_bodies_9669258.py", "main", "9669258")}
)
#: The T-0094b inventory sites at 2c584d9 (``docs/tasks/T-0094b.md``) the moving-tip check must flag.
L_SITES_AT_2C584D9 = {
    "L1": (
        "tests/cd/unit/test_toolcall_rescore_exact.py",
        "test_t0086_changes_nothing_it_does_not_own",
    ),
    "L6": (
        "tests/unit/test_toolcall_live.py",
        "test_c8_the_unchanged_files_are_unchanged_since_the_base",
    ),
    "L6-protocols": (
        "tests/unit/test_toolcall_live.py",
        "test_c8_protocols_changes_only_by_the_tool_executor_protocol",
    ),
    "L7": (
        "tests/unit/test_session_tools_off_golden.py",
        "test_the_files_criterion_7_names_are_unchanged_from_the_base",
    ),
    "L8": (
        "tests/unit/test_session_tool_turn.py",
        "test_c8_the_owned_scope_is_unchanged_from_the_base",
    ),
    "L11": (
        "tests/cd/integration/test_cd_tap_composition.py",
        "test_t0011_criteria_1_2_15_modules_are_unedited",
    ),
    "L12": ("tests/contract/test_vllm_cd_wiring_gpu.py", "test_the_fence_modules_are_unedited"),
    "L16": (
        "tests/cd/unit/test_plugin_cd_imports.py",
        "test_llm_contract_is_unchanged_from_the_merge_base",
    ),
}
#: Modules the L sites import their base literals from (``S.BASE_SHA``, ``from ... import BASE_SHA``).
_L_SUPPORT = ("tests/unit/_toolcall_live_support.py", "tests/unit/_tool_turn_support.py")

_TIP_FIXTURE = """
import subprocess
from pathlib import Path

REPO = Path("/repo")
OLD = "2c584d9"
LATER = "9dc9c94"
FABRICATED = "0123456789abcdef0123456789abcdef01234567"


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout


def _changed(base, *paths):
    return _git("diff", "--name-only", base, "--", *paths).split()


def test_single_revision_diff():
    assert _git("diff", "--name-only", OLD, "--", "local_astra") == ""
    assert _git("status", "--porcelain", "--", "local_astra") == ""
    assert _git("ls-files", "--others", "--exclude-standard", "--", "local_astra") == ""


def test_single_revision_through_a_helper():
    assert _changed(OLD, "plugin") == []


def test_blob_against_the_working_file():
    old = _git("show", f"{OLD}:local_astra/contracts/llm.py")
    assert old == (REPO / "local_astra" / "contracts" / "llm.py").read_text()


def test_closed_range():
    assert _git("diff", "--name-only", OLD, "6a8600b", "--", "local_astra") == ""


def test_status_without_a_single_revision_diff():
    assert _git("status", "--porcelain", "--", "local_astra") == ""


def test_frozen_copy():
    old = _git("show", f"{OLD}:local_astra/toolcall/score.py")
    assert old == (REPO / "tests" / "unit" / "_frozen_score.py").read_text()


def test_bases_after_closed_as_of():
    assert _git("diff", "--name-only", LATER, "--", "local_astra") == ""
    assert _changed(FABRICATED, "plugin") == []


def test_unresolvable_base(base):
    assert _git("diff", base, "--", "local_astra") == ""


def test_scratch_repo(tmp_path):
    subprocess.run(["git", "-C", str(tmp_path), "diff", OLD, "--", "x"], check=True)
"""


#: The mechanism rows pass this literal, never the bumpable ``CLOSED_AS_OF``: a Q-c1 bump past 9dc9c94 would make the
#: fixture's "later" base an ancestor and turn the too-narrow control red for a reason that is not a regression.
_FIXTURE_CLOSED_AS_OF = "ce34516"


def _tip_b() -> list[Hit]:
    gr.require_commits(_FIXTURE_CLOSED_AS_OF, gr.LATER_B, "9dc9c94")
    return tip_scan({"tests/x/test_tip.py": _TIP_FIXTURE}, closed_as_of=_FIXTURE_CLOSED_AS_OF)


def test_no_finished_task_base_is_compared_with_the_working_tree() -> None:
    """T-0094c criterion 4: on this tree no real-repo git call compares a literal commit that is an ancestor of
    ``CLOSED_AS_OF`` with the working tree (a one-revision ``diff``; a ``show <rev>:<path>`` beside a read of the
    same live file; ``status``/``ls-files --others`` beside such a diff)."""
    gr.require_commits(CLOSED_AS_OF)
    hits = _tree_scan().moving_tips(CLOSED_AS_OF)
    bad = [h for h in hits if (h.path, h.function, h.constant) not in TIP_ALLOWED]
    assert bad == [], "finished-task bases compared with the working tree:\n" + "\n".join(
        f"{h.path}:{h.line} ({h.function}): {h.constant!r}" for h in bad
    )
    used = {(h.path, h.function, h.constant) for h in hits}
    assert used >= TIP_ALLOWED, f"stale moving-tip allowlist entries: {sorted(TIP_ALLOWED - used)}"


def test_the_moving_tip_check_flags_the_t0094b_sites_at_2c584d9() -> None:
    """T-0094c criterion 4, too wide: over the 2c584d9 blobs of the T-0094b inventory modules (and the support
    modules their base literals live in) the check flags L1, L6 (scope and protocols), L7, L8, L11, L12 and L16,
    each in its own function, and the ``status``/``ls-files --others`` halves of L1, L7, L11 and L12."""
    gr.require_commits(gr.LATER_B, _FIXTURE_CLOSED_AS_OF)
    paths = sorted({p for p, _ in L_SITES_AT_2C584D9.values()} | set(_L_SUPPORT))
    sources = {p: _blob(gr.LATER_B, p) for p in paths}
    hits = tip_scan(sources, closed_as_of=_FIXTURE_CLOSED_AS_OF)
    flagged = {(h.path, h.function) for h in hits}
    missed = {k: v for k, v in L_SITES_AT_2C584D9.items() if v not in flagged}
    assert missed == {}, f"not flagged at 2c584d9: {missed}"
    halves = {(h.path, h.function) for h in hits if not re.fullmatch(r"[0-9a-f]{7,40}", h.constant)}
    assert {L_SITES_AT_2C584D9[k] for k in ("L1", "L7", "L11", "L12")} <= halves, hits


def test_the_moving_tip_check_ignores_bases_after_closed_as_of() -> None:
    """T-0094c criterion 4, too narrow: a base after ``CLOSED_AS_OF`` (9dc9c94, the T-0094c task-file commit) or a
    fabricated 40-hex SHA, an unresolvable base, a closed range, ``status`` without a one-revision diff, a frozen
    copy and a scratch repo are not flagged; the positive controls in the same fixture are (exactly these lines)."""
    hits = _tip_b()
    got = {(h.function, h.line, h.constant) for h in hits}
    assert got == {
        ("test_single_revision_diff", 20, "2c584d9"),
        ("test_single_revision_diff", 21, "--porcelain"),
        ("test_single_revision_diff", 22, "--others"),
        ("test_single_revision_through_a_helper", 26, "2c584d9"),
        ("test_blob_against_the_working_file", 30, "2c584d9:local_astra/contracts/llm.py"),
    }, sorted(got)


def test_a_bump_of_closed_as_of_leaves_the_fixture_rows_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review fix 1 (ruling Q-c1): with ``CLOSED_AS_OF`` bumped to a later commit (9dc9c94, after which the fixture's
    "later" base is an ancestor) the fixture rows give the same hits, because they pass their own literal; and the
    bump is not vacuous: the default-argument scan then flags the 9dc9c94 line."""
    before = _tip_b()
    monkeypatch.setattr(sys.modules[__name__], "CLOSED_AS_OF", "9dc9c94")
    assert _tip_b() == before
    bumped = {(h.function, h.constant) for h in tip_scan({"tests/x/test_tip.py": _TIP_FIXTURE})}
    assert ("test_bases_after_closed_as_of", "9dc9c94") in bumped, sorted(bumped)


def _blob(rev: str, path: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:{path}"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        pytest.skip(f"{rev}:{path} is absent from this clone: {proc.stderr.strip()}")
    return proc.stdout


# --------------------------------------------------------------------------- the shared helper


def _scratch_repo(tmp_path: Path) -> tuple[str, str]:
    git = ["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for rel in ("pkg/a.py", "pkg/b.py", "pkg/native.py", "other/c.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("base\n")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "base"], check=True)
    rev = [*git, "rev-parse", "HEAD"]
    base = subprocess.run(rev, capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "pkg/a.py").write_text("edited\n")
    (tmp_path / "pkg/b.py").unlink()
    (tmp_path / "pkg/native_extra.py").write_text("new\n")
    (tmp_path / "pkg/native").mkdir()
    (tmp_path / "pkg/native/x.py").write_text("new\n")
    (tmp_path / "other/c.py").write_text("edited\n")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "tip"], check=True)
    tip = subprocess.run(rev, capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "pkg/native.py").write_text("uncommitted\n")  # the working tree is not the range
    return base, tip


def test_changed_reports_every_modified_deleted_and_added_file_under_its_paths_only(
    tmp_path: Path,
) -> None:
    """The closed-range predicate replacing the retired exemption rows' "still caught" halves: an edit, a deletion,
    a prefix sibling and a file in a subdirectory of an owned name are all reported, a path outside the set is not,
    and an uncommitted working-tree edit is not part of the range."""
    base, tip = _scratch_repo(tmp_path)
    assert gr.changed(base, tip, "pkg", repo=tmp_path) == [
        "pkg/a.py",
        "pkg/b.py",
        "pkg/native/x.py",
        "pkg/native_extra.py",
    ]
    assert gr.changed(base, tip, "pkg", "other", repo=tmp_path)[-1] == "pkg/native_extra.py"
    assert "other/c.py" in gr.changed(base, tip, "pkg", "other", repo=tmp_path)
    assert gr.changed(base, tip, "pkg", diff_filter="MDR", repo=tmp_path) == [
        "pkg/a.py",
        "pkg/b.py",
    ]
    assert gr.changed(base, base, "pkg", repo=tmp_path) == []
    with pytest.raises(AssertionError):
        gr.changed(base, tip, repo=tmp_path)


def test_require_commits_skips_naming_an_absent_sha() -> None:
    """Criterion 3: an absent commit skips with the SHA in the reason; it never passes silently.

    Precondition (T-0157 criterion 11): ``LATER`` itself must be present, else the skip below would name it and
    not the absent SHA; in a clone without history this row skips naming ``LATER``."""
    gr.require_commits(gr.LATER)
    with pytest.raises(pytest.skip.Exception, match="0000000deadbeef"):
        gr.require_commits(gr.LATER, "0000000deadbeef")


def test_prove_range_rejects_a_wrong_base_or_tip() -> None:
    """Criterion 3: the range proof checks BASE = merge-base(M^1, M^2) and TIP = M^2, so the declared fork point
    1e06a24 is refused as T-0084b's base, and a TIP from another merge is refused."""
    gr.require_commits("9737368", "4a4513a", "a1ef4dd", "1e06a24", "f0a0561")
    gr.prove_range("4a4513a", "a1ef4dd", "9737368")
    with pytest.raises(AssertionError, match="merge-base"):
        gr.prove_range("1e06a24", "a1ef4dd", "9737368")
    with pytest.raises(AssertionError):
        gr.prove_range("1e06a24", "f0a0561", "9737368")
