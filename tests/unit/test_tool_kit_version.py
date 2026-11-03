"""T-0092c, carried from the T-0092b review: ``ToolKit.tools_version``, the tools sha on the kit.

docs/tasks/T-0092c.md ("Carried from the T-0092b review"): ``ToolKit`` gains ``tools_version``, set in
``cli.load_tool_kit`` from the same ``ToolSet`` that built ``kit.tools``; ``_tools_regime`` and ``cli.tools_line`` read
it, and the session stops re-reading ``tools_v1.json`` (T-0092b criterion 2: "reads none of these files"). This
amends ruling H-A1, whose field pin is here.

The oracle for "read from the kit" is a kit whose ``tools_version`` is *not* the file's sha (``FAKE``): a reader that
goes back to the file prints ``bbfc53ccdd80`` instead. Regime: CPU, the T-0088a subset detokeniser, no event loop for
the CLI rows.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

from local_astra.config import Profile
from tests.unit import _tools_switch_support as T

#: H-A1's six fields (T-0092b), plus the one T-0092c adds.
H_A1 = {"mode", "tools", "executor", "payloads", "calibration", "repair_off"}
FAKE = "feedfacecafe" + "0" * 52


def need_field() -> None:
    cls = T.kit_class()
    names = {f.name for f in dataclasses.fields(cls)}
    assert T.KIT_VERSION_FIELD in names, (
        f"T-0092c (carried): ToolKit gains {T.KIT_VERSION_FIELD!r}; its fields are {sorted(names)}"
    )


# T-0125 pin move (Pins, amendment 2026-10-27): + web_search
def test_the_kit_fields_are_h_a1_plus_tools_version():
    """The H-A1 field pin, widened by exactly ``tools_version`` (named as a moving pin in the report) and, by the
    T-0114 re-scope (docs/tasks/T-0114.md, "Re-scope after implementation"), by ``search``, and, by T-0111g R8
    (docs/tasks/T-0111g.md, Rulings, "on the group B implementation"), by ``live_browser``."""
    fields = dataclasses.fields(T.kit_class())
    # T-0140 re-pin (A6, docs/tasks/T-0140.md): exactly one optional field typed ``ToolSet | None`` with default
    # ``None`` (v1); its name is the implementer's, so it is found by its annotation and subtracted
    tool_set = [f.name for f in fields if "ToolSet" in str(f.type) and f.default is None]
    assert len(tool_set) == 1, (
        f"T-0140 A6: ToolKit gains one `<name>: ToolSet | None = None`; got {tool_set}"
    )
    names = {f.name for f in fields} - set(tool_set)
    widened = {
        "tools_version",
        "search",
        "live_browser",
        "web_search",
    }  # web_search: T-0125 criterion 3
    web = [f for f in fields if f.name == "web_search"]
    assert web and web[0].default is None, (
        "T-0125 criterion 3: `web_search: Literal['ddgs'] | None = None`"
    )
    assert names == H_A1 | widened, (
        f"T-0092c/T-0114/T-0111g R8/T-0125: ToolKit is H-A1's fields plus tools_version, search, live_browser and "
        f"web_search; extra "
        f"{sorted(names - H_A1 - widened)}, missing {sorted((H_A1 | widened) - names)}"
    )


def test_load_tool_kit_takes_the_version_from_the_toolset_that_built_the_tools(
    tmp_path, monkeypatch
):
    """``cli.load_tool_kit`` sets ``tools_version`` from the ``ToolSet`` it built ``kit.tools`` from: with
    ``tools.load`` answering a set whose version is ``FAKE``, the kit says ``FAKE`` (and repair turns off, since the
    payload manifest no longer matches -- that is T-0092b's staleness rule, not under test here)."""
    from local_astra import cli
    from local_astra.toolcall import native, tools

    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    real = tools.load()
    monkeypatch.setattr(tools, "load", lambda *a, **k: dataclasses.replace(real, version=FAKE))
    kit, _det = cli.load_tool_kit(Profile(name="lean"))
    assert getattr(kit, T.KIT_VERSION_FIELD, None) == FAKE, (
        f"T-0092c: kit.tools_version comes from the ToolSet; got {getattr(kit, T.KIT_VERSION_FIELD, '<absent>')!r}"
    )
    assert kit.tools == tuple(native.tools_field(real))


def test_load_tool_kit_records_the_shipped_tools_sha(tmp_path, monkeypatch):
    from local_astra import cli

    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    kit, _det = cli.load_tool_kit(Profile(name="lean"))
    assert getattr(kit, T.KIT_VERSION_FIELD, None) == T.tools_sha256(), getattr(
        kit, T.KIT_VERSION_FIELD, None
    )


def test_tools_line_reads_the_kit(tmp_path):
    """``cli.tools_line`` prints the kit's sha, not the file's."""
    from local_astra import cli

    need_field()
    line = cli.tools_line(T.make_kit(tools_version=FAKE))
    assert f"native:{FAKE[:12]}" in line and T.TOOLS_SHA12 not in line, line


async def test_the_regime_reads_the_kit(tmp_path):
    """``Regime.tools`` is ``native:<kit.tools_version[:12]>:dry_run``."""
    need_field()
    value = await T.regime_tools_of(T.make_kit(tools_version=FAKE), tmp_path)
    assert value == f"native:{FAKE[:12]}:dry_run", (
        f"T-0092c: _tools_regime reads kit.tools_version; got {value!r}"
    )


async def test_the_session_never_reads_the_tools_file(tmp_path, monkeypatch):
    """T-0092b criterion 2 as amended: a dry-run session (construction and a toolless turn) never calls
    ``tools.load`` itself. Calls from other modules (the T-0091 judge) are not this row's business; the caller of
    each call is read off the stack, so the row does not dictate how the session imports anything."""
    from local_astra.toolcall import tools

    kit = T.make_kit()
    real = tools.load
    callers: list[str] = []

    def spy(*a: Any, **k: Any) -> Any:
        callers.append(str(sys._getframe(1).f_globals.get("__name__")))
        return real(*a, **k)

    monkeypatch.setattr(tools, "load", spy)
    import local_astra.conductor.session as session_mod

    if getattr(session_mod, "toolset", None) is tools:
        pass  # the module attribute is patched above; a direct binding of ``load`` would be caught below
    for name, value in list(vars(session_mod).items()):
        if value is real:
            monkeypatch.setattr(session_mod, name, spy)
    await T.regime_tools_of(kit, tmp_path)
    own = [c for c in callers if c.startswith("local_astra.conductor")]
    assert own == [], (
        f"T-0092c: the session reads tools_v1.json itself ({len(own)} calls from {sorted(set(own))})"
    )


def test_load_tool_kit_reads_the_tool_set_once(tmp_path, monkeypatch):
    """I13 (reviewer 2): ``cli.load_tool_kit`` reads the tool set once, so ``kit.tools`` and ``kit.tools_version`` can
    never come from two different reads. The spy answers the real set on the first call and, on any later call, a
    set with another version *and* one declaration fewer -- so a second read shows up in the call count, and also in
    the kit (whichever field took the second read disagrees with the first set). Only reads made by ``local_astra.cli``
    itself count (the caller is read off the stack): other modules' reads are not this row's business."""
    from local_astra import cli
    from local_astra.toolcall import native, tools

    calls = T.Calls()
    T.patch_detok_load(monkeypatch, calls, tmp_path)
    real = tools.load()
    assert len(real.declarations) > 1, "premise: the shipped set has more than one tool"
    second = dataclasses.replace(real, version=FAKE, declarations=real.declarations[:-1])
    callers: list[str] = []

    def spy(*a: Any, **k: Any) -> Any:
        # only the CLI's own reads are counted: the payload staleness check reads the set from another module
        caller = str(sys._getframe(1).f_globals.get("__name__"))
        callers.append(caller)
        return second if caller == "local_astra.cli" and callers.count(caller) > 1 else real

    real_load = tools.load
    monkeypatch.setattr(tools, "load", spy)
    for name, value in list(vars(cli).items()):
        if value is real_load:  # a direct ``from ... import load`` binding is spied on too
            monkeypatch.setattr(cli, name, spy)
    kit, _det = cli.load_tool_kit(Profile(name="lean"))
    own = [c for c in callers if c == "local_astra.cli"]
    assert len(own) == 1, (
        f"I13: cli.load_tool_kit reads the tool set once; it read it {len(own)} times"
    )
    assert (kit.tools, getattr(kit, T.KIT_VERSION_FIELD, None)) == (
        tuple(native.tools_field(real)),
        real.version,
    ), "I13: kit.tools and kit.tools_version come from the same (first) read"
