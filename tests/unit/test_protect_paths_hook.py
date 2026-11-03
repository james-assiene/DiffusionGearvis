"""Acceptance tests for the Edit/Write PreToolUse guard, ``.claude/hooks/protect_paths.sh``.

The hook's contract (``docs/09-agent-workflow.md`` §3, CLAUDE.md "Two agents never edit the same
file concurrently"):

* stdin is the hook JSON ``{"cwd", "agent_type", "tool_input": {"file_path"}}`` (or
  ``notebook_path`` for ``NotebookEdit``);
* **exit 0 = allowed**, **non-zero = denied**. That status is the verdict. Wording was once
  asserted only where a criterion is *about* the wording (criterion 6's fail-closed message and
  criterion 7's "no task branch" message, each docstring says why -- T-0026's group 10 convention
  in ``test_guard_bash_hook.py``).
* Since T-0145, every deny row of group 5 also asserts **why** it was denied: a short, stable
  substring of the rule's own message (one constant per rule, below), and for each implementer
  row the absence of the three ownership messages. The exit status alone could not tell "denied by
  the rule this row is about" from "denied by some other rule": T-0143's reviewer removed the
  implementer's ``tests/*`` deny and no row went red, because the role-rule sandbox's owned
  section (``- **/*``) parsed as empty and the ownership check denied every row for its own
  reason. A deny for the wrong reason looks exactly like a pass.

T-0029 is filed against one defect with two halves, both measured on 2026-09-14 on ``main``
@ 7989c81 in a throwaway repo, with the hook invoked as ``bash protect_paths.sh`` and the real
hook JSON on stdin. For the task file::

    ## Owned files
    - `local_astra/x.py` *(new — the module)*

    **Files this task needs and does not own:**
    - local_astra/other.py

the live hook answers::

    BLOCKED (implementer): local_astra/x.py is not in the owned files of task T-0099.   <- owned
    ALLOWED                local_astra/other.py                                         <- not owned

The verdict is exactly inverted, because the section is read line by line: a leading ``-`` and the
*outer* backticks are stripped and whatever is left becomes a bash ``case`` glob, so an annotated
bullet becomes a glob that matches nothing, while the ``\\n##`` section terminator does not stop at
the "does not own" list, whose plain bullets become grants.

The two halves are what the tests are organised around:

* over-*denial* (criteria 2, 8): every bullet form a planner writes must yield its path;
* over-*grant* (criteria 3, 4, 5, 6): nothing else ever may -- not a not-owned bullet, not prose,
  not a lone ``*``, and not an empty parse.

Criterion 7's rows are regression pins: they pass on ``main`` today and carry no ``pending``
marker. Everything else in this module is red until T-0029 lands.

Every case builds its own throwaway git repo and its own ``docs/tasks/`` tree, so nothing here
depends on this repo's branch, task file or dirty state.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests._home_jail import GIT_SELECTORS

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / ".claude" / "hooks" / "protect_paths.sh"
BASH_HOOK = REPO / ".claude" / "hooks" / "guard_bash.sh"

# Git in a hermetic state: no user config, no global hooks, deterministic identity, and (T-0186 criterion 3, belt
# and braces beside the conftest strip) none of git's repo selectors, so `git init <tmp>` never re-initialises a repo
# an exported GIT_DIR names.
GIT_ENV = {
    **{k: v for k, v in os.environ.items() if k not in GIT_SELECTORS},
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_AUTHOR_NAME": "protect-test",
    "GIT_AUTHOR_EMAIL": "protect-test@example.invalid",
    "GIT_COMMITTER_NAME": "protect-test",
    "GIT_COMMITTER_EMAIL": "protect-test@example.invalid",
}


def _git(cwd: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
        env=GIT_ENV,
    )
    return out.stdout.strip()


@dataclass
class Sandbox:
    """A throwaway git repo laid out like this one, with a ``docs/tasks/`` tree."""

    top: Path

    def on_task(self, task_id: str, body: str) -> None:
        """Write ``docs/tasks/<task_id>.md`` and switch to that task's branch.

        The branch name is where ``_lib.sh`` reads ``TASK`` from, so the two must be set
        together or the hook is answering about a different task than the test wrote.
        """
        (self.top / "docs" / "tasks" / f"{task_id}.md").write_text(body)
        _git(self.top, "switch", "-q", "-c", f"{task_id}-probe")

    def on_branch(self, name: str) -> None:
        _git(self.top, "switch", "-q", "-c", name)


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    top = tmp_path / "repo"
    for rel in (
        "local_astra/conductor",
        "local_astra/contracts",
        "tests/unit",
        "docs/tasks",
        "scripts",
        ".claude/hooks",
        ".claude/agent-memory/implementer",
    ):
        (top / rel).mkdir(parents=True, exist_ok=True)
    (top / "README.md").write_text("clean\n")
    (top / "CLAUDE.md").write_text("rules\n")
    (top / "uv.lock").write_text("lock\n")
    (top / "services").mkdir(parents=True, exist_ok=True)
    (top / "services" / "versions.lock").write_text("pins\n")
    (top / "docs" / "03-test-plan.md").write_text("plan\n")
    _git(top.parent, "init", "-q", str(top))
    _git(top, "add", "-A")
    _git(top, "commit", "-qm", "init")
    return Sandbox(top=top)


def run_hook(
    path: str, sandbox: Sandbox, role: str = "implementer", *, notebook: bool = False
) -> subprocess.CompletedProcess[str]:
    key = "notebook_path" if notebook else "file_path"
    payload = json.dumps({"cwd": str(sandbox.top), "agent_type": role, "tool_input": {key: path}})
    return subprocess.run(
        ["bash", str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(sandbox.top),
        env=GIT_ENV,
    )


def run_bash_hook(
    command: str, sandbox: Sandbox, role: str = "implementer"
) -> subprocess.CompletedProcess[str]:
    payload = json.dumps(
        {"cwd": str(sandbox.top), "agent_type": role, "tool_input": {"command": command}}
    )
    return subprocess.run(
        ["bash", str(BASH_HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(sandbox.top),
        env=GIT_ENV,
    )


def assert_allowed(res: subprocess.CompletedProcess[str], path: str, why: str) -> None:
    assert res.returncode == 0, (
        f"hook DENIED (exit {res.returncode}) an edit it must allow ({why}):\n"
        f"{path}\nstderr: {res.stderr.strip()}"
    )


def assert_denied(res: subprocess.CompletedProcess[str], path: str, why: str) -> None:
    assert res.returncode != 0, f"hook ALLOWED (exit 0) an edit it must deny ({why}):\n{path}"


# --------------------------------------------------------------------------------------------
# Group 1 -- criterion 2: every bullet form a planner writes yields its path.
#
# Each row is (owned-files section body, path that must be allowed). Every row also asserts that
# an unrelated in-repo path stays denied, so a parser that answers "allow" to everything cannot
# satisfy this group (landmine 3).
# --------------------------------------------------------------------------------------------

OWNED = "local_astra/conductor/audio_capture.py"
NOT_OWNED = "local_astra/conductor/session.py"

# Each row is (section body, path that must be allowed, path that must stay denied). The third
# element defaults to ``NOT_OWNED``; the ``double-star-glob`` row overrides it because
# ``local_astra/conductor/session.py`` is *inside* ``local_astra/conductor/**`` -- asserting both
# there would be a contradiction, not a pin. Its denied path is another package's file, which the
# glob genuinely does not cover.
OUTSIDE_THE_CONDUCTOR = "local_astra/contracts/protocols.py"

BULLET_FORMS = [
    pytest.param(f"- {OWNED}", OWNED, NOT_OWNED, id="bare"),
    pytest.param(f"- `{OWNED}`", OWNED, NOT_OWNED, id="backticked"),
    pytest.param(f"- `{OWNED}` *(new — the module)*", OWNED, NOT_OWNED, id="backticked-annotated"),
    pytest.param(
        f"- `{OWNED}` *(new — capture stream, clock, xrun/backpressure policy,\n"
        "  the gate, `CaptureBlock`/`CaptureStats`)*",
        OWNED,
        NOT_OWNED,
        id="annotation-wrapped",
    ),
    pytest.param(f"* `{OWNED}` *(from T-0006)*", OWNED, NOT_OWNED, id="star-bullet"),
    pytest.param(f"+ `{OWNED}`", OWNED, NOT_OWNED, id="plus-bullet"),
    pytest.param(f"- the capture side:\n  - `{OWNED}` *(new)*", OWNED, NOT_OWNED, id="indented"),
    pytest.param(
        "- `local_astra/conductor/**` *(the whole package)*",
        OWNED,
        OUTSIDE_THE_CONDUCTOR,
        id="double-star-glob",
    ),
    pytest.param(
        "- `scripts/audio_rig.sh` *(from T-0006: expose the monitor)*",
        "scripts/audio_rig.sh",
        NOT_OWNED,
        id="hyphen-in-name",
    ),
    # Ruling A9 (confirmation review): T-0008 owns a `- Makefile` bullet. A path-shaped test
    # that keys on a dot or a slash denies its implementer a file the task file plainly lists,
    # which is the over-denial half of the original defect coming back through the fix. A bullet
    # of exactly one token is a path whatever it looks like; prose bullets are multi-token.
    pytest.param("- Makefile", "Makefile", NOT_OWNED, id="extensionless-single-token"),
    pytest.param("- ./Makefile", "Makefile", NOT_OWNED, id="dot-slash-prefix"),
]


def _task_file(section: str) -> str:
    """A minimal but realistic task file whose ``## Owned files`` section is ``section``."""
    return (
        "# T-0099 — the probe task\n\n"
        "**Status:** planned\n\n"
        "## Acceptance criteria\n1. Something.\n\n"
        f"## Owned files\n{section}\n\n"
        "## Out of scope\n- Everything else.\n"
    )


@pytest.mark.parametrize(("section", "path", "denied"), BULLET_FORMS)
def test_every_bullet_form_yields_its_path(
    section: str, path: str, denied: str, sandbox: Sandbox
) -> None:
    """Criterion 2: ownership does not depend on how a planner formatted a bullet.

    Today the section is read whole-line, so ``- `p` *(annotation)*`` becomes the glob
    ``p` *(annotation)*`` -- which matches no path, and therefore denies **every** edit to a file
    the task legitimately owns. The ``annotation-wrapped`` row is the exact shape T-0021 was
    written in and the defect this task exists for (ruling A15).

    The second assertion is the guard against landmine 3 ("return allow-everything"): a parser
    that grants the path by granting everything fails here. The denied path is per row, because
    a row whose bullet is a ``**`` glob must be probed with a path *outside* that glob -- probing
    it with one inside would assert the row's own bullet does not work.
    """
    sandbox.on_task("T-0099", _task_file(section))
    assert_allowed(run_hook(path, sandbox), path, "the bullet lists this path")
    assert_denied(run_hook(denied, sandbox), denied, "no bullet in the section covers this path")


# The pre-HEAD ``## Owned files`` block of docs/tasks/T-0021.md, verbatim
# (``git show 7989c81~1:docs/tasks/T-0021.md``), down to and including the next heading. It is the
# shape that forced ruling A15's workaround: annotated bullets, wrapped annotations, prose above
# the list, a "does not own" list below it, and a ``###`` heading after that.
T0021_OWNED_BLOCK = """## Owned files

T-0006 owns four of these today. **This task may not start until T-0006 is merged**; ownership then
transfers (sequential, never concurrent — CLAUDE.md). If T-0006 is still open when an implementer
reaches for these, stop and say so.

- `local_astra/conductor/audio_capture.py` *(new — capture stream, clock, xrun/backpressure policy,
  the gate, `CaptureBlock`/`CaptureStats`)*
- `local_astra/conductor/audio_io.py` *(from T-0006: `capture()` delegates here; playback untouched)*
- `local_astra/conductor/audio_device.py` *(from T-0006: the input side of criterion 1)*
- `local_astra/conductor/audio_resample.py` *(from T-0006: add the 48 → 24 kHz decimation stage
  beside the existing 24 → 48 kHz one; do not alter the upsampler's behaviour — criterion 12a covers
  it)*
- `scripts/audio_rig.sh` *(from T-0006: expose/verify `astra-test-sink.monitor` as a capture source;
  `pactl` is still absent — `pw-cli`/`pw-loopback`/`pw-dump` only)*

**Files this task needs and does not own:**
- `local_astra/contracts/protocols.py` — **T-0017** (open). `AudioIO` is not widened: `capture()`
  keeps its `AsyncIterator[bytes]` signature and `capture_frames()`, `stt_blocks()`,
  `capture_stats()` live on the concrete class only, exactly as T-0006 did for `played_ms()`.
  Widening is a follow-on for T-0017's owner or T-0007.
- `local_astra/contracts/speech.py` — same owner. `STT_BLOCK_SAMPLES = 1920` is defined in
  `audio_capture.py`; promoting it to `contracts/speech.py` beside `CAPTURE_BLOCK_MS` is a spec
  delta for the orchestrator, not an edit here.
- `local_astra/conductor/session.py` — **T-0007/T-0011/T-0020**. Not touched. The seam is below.
- `local_astra/conductor/stt_client.py`, `local_astra/reflex/**` — **T-0007**.
- `pyproject.toml` — T-0006 claims it narrowly. Expected diff here: **none** (the backend and its
  mypy override were settled by T-0006). If capture genuinely needs a change, it is one line and the
  orchestrator serialises it against whatever else is in flight; do not take the file speculatively.
- `tests/**`, `tests/fakes/**`, `tests/fixtures/**` — the test-author's (docs/tasks/README.md).

### Seam to `session.py` (T-0007), stated so T-0007 can be planned against it
"""

T0021_OWNED = [
    "local_astra/conductor/audio_capture.py",
    "local_astra/conductor/audio_io.py",
    "local_astra/conductor/audio_device.py",
    "local_astra/conductor/audio_resample.py",
    "scripts/audio_rig.sh",
]
# Listed under "Files this task needs and does not own" in the same block.
T0021_NOT_OWNED = [
    "local_astra/contracts/protocols.py",
    "local_astra/contracts/speech.py",
    "local_astra/conductor/session.py",
    "local_astra/conductor/stt_client.py",
    "pyproject.toml",
]


@pytest.mark.parametrize("path", T0021_OWNED)
def test_the_t0021_owned_block_is_parsed(path: str, sandbox: Sandbox) -> None:
    """The regression pin for ruling A15: the real block, verbatim, as it was written.

    All five paths T-0021 owns are allowed. On ``main`` the embedded python turns each of these
    bullets into a glob carrying its annotation, so the implementer of T-0021 could not edit a
    single file the task gave it -- which is why A15 forced the "exact bare paths" workaround into
    the task file. Criterion 8 pins the workaround still working; this pins that it is no longer
    needed.
    """
    sandbox.on_task("T-0021", T0021_OWNED_BLOCK)
    assert_allowed(run_hook(path, sandbox), path, "an annotated bullet lists this path")


@pytest.mark.parametrize("path", T0021_NOT_OWNED)
def test_the_t0021_not_owned_paths_stay_denied(path: str, sandbox: Sandbox) -> None:
    """The other half of A15: the same block's "does not own" list is not a grant.

    ``local_astra/conductor/session.py`` is the named one in the task; the rest of that list is
    parametrised with it because a section terminator that stops at the wrong place leaks all of
    them at once.
    """
    sandbox.on_task("T-0021", T0021_OWNED_BLOCK)
    assert_denied(run_hook(path, sandbox), path, "the block lists this under 'does not own'")


def test_the_bare_path_workaround_keeps_working(sandbox: Sandbox) -> None:
    """Criterion 8: nobody has to re-edit a task file for this fix.

    The *current* (already rewritten) T-0021 block is read from this repo and must yield the same
    five paths as its annotated predecessor above. Read live rather than copied so that the day
    someone drops the workaround, this test follows the file instead of pinning a stale copy.
    """
    live = (REPO / "docs" / "tasks" / "T-0021.md").read_text()
    if "audio_capture.py" not in live:
        pytest.skip("docs/tasks/T-0021.md no longer lists the audio capture modules")
    sandbox.on_task("T-0021", live)
    for path in T0021_OWNED:
        assert_allowed(run_hook(path, sandbox), path, "a bare bullet lists this path")
    assert_denied(run_hook(NOT_OWNED, sandbox), NOT_OWNED, "listed under 'does not own'")


# --------------------------------------------------------------------------------------------
# Group 2 -- criterion 3: where the section ends. The adversarial group.
# --------------------------------------------------------------------------------------------

NOT_OWN_MARKERS = [
    pytest.param("**Files this task needs and does not own:**", id="marker-does-not-own"),
    pytest.param(
        "**Files this task deliberately does not touch — say so and stop rather than editing:**",
        id="marker-deliberately-does-not-touch",
    ),
    pytest.param(
        "**Files this task needs and does not own — say so and stop rather than editing:**",
        id="marker-does-not-own-with-trailer",
    ),
    # Ruling A4: the marker rule is case-insensitive, so a planner who typed the phrase in a
    # different case still gets a section end rather than a grant.
    pytest.param(
        "**files this task NEEDS and does not OWN:**", id="marker-does-not-own-mixed-case"
    ),
    # Ruling A7: T-0011's live spelling. It is not "does not own" or "deliberately does not
    # touch" at all -- it is an *ownership conflict* list that disclaims files ("T-0011 claims
    # no conductor file"), and with only the two prescribed spellings the parser walked straight
    # into it. The rule that decides is CLAUDE.md's "two agents never edit the same file", not
    # the letter of criterion 3: a list whose purpose is to say "not mine" can never be a grant.
    pytest.param(
        "**Ownership conflicts, resolved explicitly (T-0011 claims no conductor file):**",
        id="marker-ownership-conflicts",
    ),
]

AFTER_THE_MARKER = [
    pytest.param(f"- {NOT_OWNED}", id="bare-bullet"),
    pytest.param(f"- `{NOT_OWNED}`", id="backticked-bullet"),
    pytest.param(f"- `{NOT_OWNED}` — **T-0007/T-0011**. Not touched.", id="annotated-bullet"),
]


@pytest.mark.parametrize("marker", NOT_OWN_MARKERS)
@pytest.mark.parametrize("after", AFTER_THE_MARKER)
def test_a_not_owned_bullet_is_never_granted(marker: str, after: str, sandbox: Sandbox) -> None:
    """Criterion 3, and the most likely way this task is got wrong.

    The cheap fix for the over-denial half is to widen the bullet parser and forget where the
    section ends -- which turns every "Files this task needs and does not own" list into a grant,
    i.e. it converts a hook that denied too much into one that grants exactly the files another
    task owns. That is strictly worse than the defect, because the over-denial is loud (the
    implementer is stopped) and the over-grant is silent (two agents edit one file).

    Measured on ``main`` @ 7989c81 with the probe file in the module docstring::

        BLOCKED (implementer): local_astra/x.py is not in the owned files of task T-0099.   <- owned
        ALLOWED                local_astra/other.py                                         <- not owned

    Both halves are asserted here: the owned path stays allowed **and** the path below the marker
    stays denied, so the row cannot be satisfied by swinging back to deny-everything.
    """
    section = f"- `{OWNED}` *(new — the module)*\n\n{marker}\n{after}\n"
    sandbox.on_task("T-0099", _task_file(section))
    assert_allowed(run_hook(OWNED, sandbox), OWNED, "listed above the marker")
    assert_denied(run_hook(NOT_OWNED, sandbox), NOT_OWNED, "listed below the 'does not own' marker")


def test_the_section_ends_at_the_next_heading_of_any_level(sandbox: Sandbox) -> None:
    """Criterion 3, first half: ``^#{1,6}\\s`` ends the section, not only ``\\n##``.

    On ``main`` the terminator is the literal ``\\n##``, which happens to match ``\\n###`` too --
    the only reason T-0021's not-own list stopped where it did. A task whose section is followed
    by an ``#### `` heading, or by no heading at all, leaks everything after it.
    """
    section = f"- `{OWNED}`\n\n#### A note on scope\n- {NOT_OWNED}\n"
    sandbox.on_task("T-0099", _task_file(section))
    assert_allowed(run_hook(OWNED, sandbox), OWNED, "listed in the section")
    assert_denied(run_hook(NOT_OWNED, sandbox), NOT_OWNED, "listed after a level-4 heading")


# --------------------------------------------------------------------------------------------
# Group 3 -- criteria 4 and 5: prose never becomes a pattern, and no parse over-grants.
# --------------------------------------------------------------------------------------------

HOSTILE_SECTION = """\
This task owns the conductor's capture side. **Do not** touch the session.

- **everything under `local_astra/**`**
- *
- | file | owner |
- | ---- | ----- |
- see https://example.invalid/docs/tasks/T-0099 for the rationale
- — the annotation without a path
- **T-0017** (open) — the protocol file stays where it is
- `local_astra/conductor/audio_capture.py` *(new — the module)*

```bash
- /etc/passwd
cat > /etc/shadow
```

- ../outside_the_repo.py
- /absolute/path.py
"""

HOSTILE_DENIED = [
    pytest.param("/etc/passwd", id="etc-passwd"),
    pytest.param("/etc/shadow", id="etc-shadow-from-a-fenced-block"),
    pytest.param(NOT_OWNED, id="an-unrelated-in-repo-path"),
    pytest.param("local_astra/contracts/protocols.py", id="another-tasks-contract"),
    # Was ``CLAUDE.md``, which the implementer's ``.claude/*|CLAUDE.md`` role rule denies before
    # the owned list is read, so the row could never see a parser over-grant (T-0145 review: a
    # parser accepting a lone ``*`` reddened its six siblings and left it green). A top-level
    # CLAUDE file no role rule names is decided by the ownership check alone. Id kept (T-0145 9b).
    pytest.param("CLAUDE.local.md", id="claude-md"),
    pytest.param("../outside_the_repo.py", id="a-dot-dot-bullet"),
    pytest.param("/absolute/path.py", id="an-absolute-bullet"),
]


@pytest.mark.parametrize("path", HOSTILE_DENIED)
def test_prose_in_the_section_grants_nothing(path: str, sandbox: Sandbox) -> None:
    """Criteria 4 and 5: a non-bullet line, and a bullet whose first token is not path-shaped,
    yield nothing -- no pattern that matches a path the file does not list.

    The hostile section carries every shape that could become a wildcard by accident: bold markup
    around a glob (``- **everything under `local_astra/**`**``), a lone ``*`` (which as a bash
    ``case`` pattern matches every path there is), two table rows, a URL, an em-dash annotation
    with no path at all, a fenced code block containing ``- /etc/passwd``, and bullets that are
    absolute or contain ``..``.

    T-0145: each row also asserts the ownership reason. The question is whether the *parser*
    granted the path, so a deny by any other rule says nothing about it.
    """
    sandbox.on_task("T-0099", _task_file(HOSTILE_SECTION))
    res = run_hook(path, sandbox)
    assert_denied(res, path, "no bullet in the section legitimately lists it")
    assert NOT_OWNED_RULE in res.stderr, (
        f"denied, but not by the ownership check, so this row cannot see a parser over-grant.\n"
        f"stderr: {res.stderr.strip()}"
    )


def test_the_real_bullet_in_a_hostile_section_is_still_owned(sandbox: Sandbox) -> None:
    """Criterion 4's other side: hostile neighbours do not cost the one real bullet its grant.

    Without this, "deny everything whenever anything looks odd" would pass the whole group -- and
    the over-denial half of the defect would survive the fix.
    """
    sandbox.on_task("T-0099", _task_file(HOSTILE_SECTION))
    assert_allowed(run_hook(OWNED, sandbox), OWNED, "the one path-shaped bullet in the section")


def _live_task_files() -> list[pytest.param]:
    files = sorted((REPO / "docs" / "tasks").glob("T-*.md"))
    return [pytest.param(f, id=f.stem) for f in files]


LIVE_TASK_FILES = _live_task_files()
# A path no task file lists, and that no legitimate glob in any of them can match: it is not under
# local_astra/, tests/, docs/, scripts/ or .claude/. If a parse grants it, the parse produced a
# wildcard out of prose.
SENTINEL = "zz_probe/zz_owned_by_nobody.py"


@pytest.mark.skipif(not LIVE_TASK_FILES, reason="no docs/tasks/T-*.md in the repo")
@pytest.mark.parametrize("task_file", LIVE_TASK_FILES)
def test_no_task_file_in_the_repo_over_grants(task_file: Path, sandbox: Sandbox) -> None:
    """Criterion 4's corpus half, and criterion 5: the whole live corpus, one file per row.

    This is the test that would have caught the defect on the day T-0021 was written: it drives
    the parser over every task file anyone has actually authored, rather than over shapes a test
    author thought of. Two probes per file:

    * ``SENTINEL`` -- a repo-relative path under a directory no task file mentions. Any parse that
      yields a bare ``*``, an empty pattern, or a pattern built from a prose line will match it.
    * ``local_astra/conductor/turn_machine.py`` -- the file CLAUDE.md singles out as pure and
      invariant-bearing. Checked only for task files whose text does not mention it at all and
      carries no ``conductor`` glob, so the oracle for "genuinely one of the bullets" is the raw
      text of the file rather than a second copy of the parser.
    """
    sandbox.on_task("T-0099", task_file.read_text())
    res = run_hook(SENTINEL, sandbox)
    assert_denied(res, SENTINEL, f"{task_file.name} lists nothing under zz_probe/")
    text = task_file.read_text()
    if "turn_machine" not in text and "conductor/**" not in text and "conductor/*" not in text:
        turn_machine = "local_astra/conductor/turn_machine.py"
        assert_denied(
            run_hook(turn_machine, sandbox),
            turn_machine,
            f"{task_file.name} never names the turn machine",
        )


# --------------------------------------------------------------------------------------------
# Group 4 -- criterion 6: fail closed, with a name.
# --------------------------------------------------------------------------------------------

BROKEN_SECTIONS = [
    pytest.param(None, id="task-file-missing"),
    pytest.param(
        "# T-0099 — the probe task\n\n## Acceptance criteria\n1. Something.\n",
        id="owned-heading-missing",
    ),
    pytest.param(
        "# T-0099 — the probe task\n\n## Owned files\n\n(to be decided with the orchestrator)\n\n"
        "## Out of scope\n- Everything.\n",
        id="section-parses-to-zero-globs",
    ),
]


@pytest.mark.parametrize("body", BROKEN_SECTIONS)
def test_a_broken_or_missing_owned_section_denies(body: str | None, sandbox: Sandbox) -> None:
    """Criterion 6: an unreadable ownership list denies, and the refusal says where to look.

    Today an empty parse silently allows **everything** -- the open hole this criterion closes:
    ``owned`` comes back empty and the ``[ -n "$owned" ]`` guard skips the check entirely, so a
    task whose section is a placeholder grants the implementer the whole repo.

    Wording is asserted here because the criterion is about the wording: "denied with a message
    naming the task file and the section" is what makes the refusal actionable instead of a wall
    (T-0026 group 10's convention). Both the file name and the section name are required, in
    whatever sentence the implementer writes.
    """
    if body is None:
        sandbox.on_branch("T-0099-probe")
    else:
        sandbox.on_task("T-0099", body)
    res = run_hook(OWNED, sandbox)
    assert_denied(res, OWNED, "the owned-files list cannot be read")
    err = res.stderr.lower()
    assert "t-0099.md" in err, (
        f"the refusal must name the task file it failed to read.\nstderr: {res.stderr.strip()}"
    )
    assert "owned files" in err, (
        f"the refusal must name the section it expected.\nstderr: {res.stderr.strip()}"
    )


def test_a_parser_failure_denies_without_a_traceback(sandbox: Sandbox) -> None:
    """Criterion 6, second sentence: if the parser itself errors, the hook denies and names
    ``_lib.sh``; no bare traceback reaches the model.

    The failure is induced the only way a test can induce it from outside: a task file that is not
    text the parser can decode (a NUL-containing byte string). Whatever the parser does with it,
    the two contract points hold -- exit non-zero, and a sentence rather than a stack trace.
    """
    (sandbox.top / "docs" / "tasks").mkdir(parents=True, exist_ok=True)
    (sandbox.top / "docs" / "tasks" / "T-0099.md").write_bytes(
        b"## Owned files\n- local_astra/\x00\xff\xfe.py\n\x00\x00\x00"
    )
    sandbox.on_branch("T-0099-probe")
    res = run_hook(OWNED, sandbox)
    assert_denied(res, OWNED, "the parser could not read the task file")
    assert "Traceback (most recent call last)" not in res.stderr, (
        f"a python traceback must never reach the model.\nstderr: {res.stderr.strip()}"
    )
    # Ruling A3: the refusal names where the parser lives, because a failure in the shared
    # parser denies every implementer's edit in every worktree at once and the agent that hits
    # it must be sent to the right file rather than to its own task.
    assert "_lib.sh" in res.stderr, (
        f"the refusal must name the file whose parser failed.\nstderr: {res.stderr.strip()}"
    )


# --------------------------------------------------------------------------------------------
# Group 5 -- criterion 7: every other rule in the hook is unchanged. Regression pins, all green
# on ``main`` today, so **no pending marker**: if one of these goes red, T-0029 changed who may
# write what, which is explicitly out of scope.
# --------------------------------------------------------------------------------------------


# The role-rule sandbox's ``## Owned files`` section (T-0145). It owns, one bare backticked path
# per bullet, every path a role rule below protects, plus two owned paths no role rule protects
# (the positive control). So for the implementer the ownership check would *allow* each protected
# path, and only the role rule can deny it. It used to be ``- **/*``, which the parser rejects
# (a leading ``*`` is not path-shaped, ruling A9): the list parsed empty and the ownership check
# denied every implementer row for its own reason ("lists no path"), whatever the role rules said.
# No annotations, so the premise does not depend on the annotation parser groups 1 and 7 pin.
ROLE_RULE_OWNED = "\n".join(
    f"- `{p}`"
    for p in (
        "tests/unit/test_x.py",
        ".claude/hooks/_lib.sh",
        ".claude/agent-memory/reviewer/notes.md",
        "CLAUDE.md",
        "docs/03-test-plan.md",
        "services/versions.lock",
        "uv.lock",
        "local_astra/conductor/session.py",
        "docs/02-architecture.md",
    )
)

# One short, stable substring per deny rule of ``protect_paths.sh`` (T-0145): pin the reason, not
# the sentence, so a rewording in the hook is a one-line edit here.
TESTS_RULE = "may not edit tests/"
CONFIG_RULE = "agent configuration"
TEST_PLAN_RULE = "test plan"
LOCK_RULE = "version locks"
TEST_AUTHOR_RULE = "test-author may edit tests/ only"
PLANNER_RULE = "planner may write task files only"
READ_ONLY_RULE = "is read-only"
NOT_OWNED_RULE = "not in the owned files"
# Every message the ownership half of the implementer branch can produce: "cannot read the
# '## Owned files' section", "... lists no path", "<path> is not in the owned files of task ...",
# and "no task branch". An implementer row denied by a role rule must show none of them.
OWNERSHIP_MESSAGES = ("Owned files", NOT_OWNED_RULE, "no task branch")


@dataclass(frozen=True)
class RoleRule:
    role: str
    path: str
    allowed: bool
    # The task owns every protected path (``ROLE_RULE_OWNED``), so only the role rule can deny.
    owned_section: str | None = ROLE_RULE_OWNED
    notebook: bool = False
    # The substring of the deny message of the rule that must decide this row (None when allowed).
    expect_reason: str | None = None


ROLE_RULES = [
    # Criterion 3 (T-0145): the always-protected lock rule, for every role. For the planner and
    # the test-author their own role rule would also deny, and for ``none`` nothing else would,
    # so the reason assertion is what makes these rows about the lock rule.
    pytest.param(
        RoleRule("implementer", "services/versions.lock", False, expect_reason=LOCK_RULE),
        id="versions-lock",
    ),
    pytest.param(RoleRule("implementer", "uv.lock", False, expect_reason=LOCK_RULE), id="uv-lock"),
    pytest.param(
        RoleRule("planner", "uv.lock", False, expect_reason=LOCK_RULE), id="uv-lock-planner"
    ),
    pytest.param(
        RoleRule("planner", "services/versions.lock", False, expect_reason=LOCK_RULE),
        id="versions-lock-planner",
    ),
    pytest.param(
        RoleRule("test-author", "uv.lock", False, expect_reason=LOCK_RULE),
        id="uv-lock-test-author",
    ),
    pytest.param(
        RoleRule("test-author", "services/versions.lock", False, expect_reason=LOCK_RULE),
        id="versions-lock-test-author",
    ),
    pytest.param(
        RoleRule("none", "uv.lock", False, expect_reason=LOCK_RULE), id="uv-lock-main-session"
    ),
    pytest.param(
        RoleRule("none", "services/versions.lock", False, expect_reason=LOCK_RULE),
        id="versions-lock-main-session",
    ),
    pytest.param(
        RoleRule("implementer", ".claude/agent-memory/implementer/notes.md", True),
        id="own-memory-implementer",
    ),
    pytest.param(
        RoleRule("reviewer", ".claude/agent-memory/reviewer/notes.md", True),
        id="own-memory-reviewer",
    ),
    pytest.param(
        RoleRule(
            "implementer",
            ".claude/agent-memory/reviewer/notes.md",
            False,
            expect_reason=CONFIG_RULE,
        ),
        id="another-roles-memory",
    ),
    pytest.param(
        RoleRule("implementer", "tests/unit/test_x.py", False, expect_reason=TESTS_RULE),
        id="implementer-tests",
    ),
    # The adversarial row (T-0145): a task that owns exactly the test file and nothing else, so
    # the ownership check has the plainest possible reason to allow it. Only the tests rule may
    # deny it. This is the row the escaped mutant M1 (the ``tests/*`` deny removed) must turn red.
    pytest.param(
        RoleRule(
            "implementer",
            "tests/unit/test_x.py",
            False,
            owned_section="- `tests/unit/test_x.py`",
            expect_reason=TESTS_RULE,
        ),
        id="implementer-tests-owned",
    ),
    pytest.param(
        RoleRule("implementer", ".claude/hooks/_lib.sh", False, expect_reason=CONFIG_RULE),
        id="implementer-claude",
    ),
    pytest.param(
        RoleRule("implementer", "CLAUDE.md", False, expect_reason=CONFIG_RULE),
        id="implementer-claude-md",
    ),
    pytest.param(
        RoleRule("implementer", "docs/03-test-plan.md", False, expect_reason=TEST_PLAN_RULE),
        id="implementer-test-plan",
    ),
    pytest.param(RoleRule("test-author", "tests/unit/test_x.py", True), id="test-author-tests"),
    pytest.param(
        RoleRule(
            "test-author",
            "local_astra/conductor/session.py",
            False,
            expect_reason=TEST_AUTHOR_RULE,
        ),
        id="test-author-outside-tests",
    ),
    pytest.param(RoleRule("planner", "docs/tasks/T-0100.md", True), id="planner-task-file"),
    pytest.param(
        RoleRule("planner", "local_astra/conductor/session.py", False, expect_reason=PLANNER_RULE),
        id="planner-outside-tasks",
    ),
    pytest.param(
        RoleRule(
            "reviewer", "local_astra/conductor/session.py", False, expect_reason=READ_ONLY_RULE
        ),
        id="reviewer-ro",
    ),
    pytest.param(
        RoleRule(
            "verifier", "local_astra/conductor/session.py", False, expect_reason=READ_ONLY_RULE
        ),
        id="verifier-ro",
    ),
    pytest.param(
        RoleRule(
            "launch-reviewer",
            "local_astra/conductor/session.py",
            False,
            expect_reason=READ_ONLY_RULE,
        ),
        id="launch-reviewer-ro",
    ),
    pytest.param(
        RoleRule(
            "explorer", "local_astra/conductor/session.py", False, expect_reason=READ_ONLY_RULE
        ),
        id="explorer-ro",
    ),
    pytest.param(
        RoleRule(
            "implementer", "tests/unit/test_x.py", False, notebook=True, expect_reason=TESTS_RULE
        ),
        id="notebook-path-honoured",
    ),
    pytest.param(RoleRule("none", "local_astra/conductor/session.py", True), id="main-session"),
]


@pytest.mark.parametrize("rule", ROLE_RULES)
def test_role_rules_are_unchanged(rule: RoleRule, sandbox: Sandbox) -> None:
    """Criterion 7: T-0029 changes *how the owned list is read*, never who may write what.

    Every row passes on ``main`` and carries no ``pending`` marker. The task branch owns every
    path the role rules protect (``ROLE_RULE_OWNED``; until T-0145 it was ``- **/*``, which parsed
    as an empty list), so the ownership check would allow each of them, and the only thing that
    can deny is the role rule. In particular the implementer refusals (``tests/``, ``.claude/``,
    ``CLAUDE.md``, the test plan) must still fire *before* the owned-files check, even though this
    task file owns those paths.

    T-0145: a deny row asserts the substring of its own rule's message, and asserts that no
    ownership message appears. Without that, a row denied by the ownership check (or by another
    role's rule) reads as a pass, which is how the M1 mutant escaped.
    """
    body = _task_file(rule.owned_section) if rule.owned_section else _task_file(f"- {OWNED}")
    sandbox.on_task("T-0099", body)
    res = run_hook(rule.path, sandbox, role=rule.role, notebook=rule.notebook)
    if rule.allowed:
        assert_allowed(res, rule.path, f"{rule.role} may write this")
    else:
        assert_denied(res, rule.path, f"{rule.role} may not write this")
        assert rule.expect_reason is not None, "every deny row names the rule that decides it"
        assert rule.expect_reason in res.stderr, (
            f"denied, but not by the rule this row is about (expected {rule.expect_reason!r}).\n"
            f"stderr: {res.stderr.strip()}"
        )
        stray = [m for m in OWNERSHIP_MESSAGES if m in res.stderr]
        assert stray == [], (
            f"the ownership check, not the role rule, produced this deny ({stray!r}).\n"
            f"stderr: {res.stderr.strip()}"
        )


ROLE_RULE_OWNED_EXPECTED = {
    "tests/unit/test_x.py",
    ".claude/hooks/_lib.sh",
    ".claude/agent-memory/reviewer/notes.md",
    "CLAUDE.md",
    "docs/03-test-plan.md",
    "services/versions.lock",
    "uv.lock",
    "local_astra/conductor/session.py",
    "docs/02-architecture.md",
}


def test_the_role_rule_sandbox_owns_every_protected_path(tmp_path: Path) -> None:
    """T-0145 criterion 1: the premise of group 5, pinned on its own.

    If a fixture edit makes ``ROLE_RULE_OWNED`` parse to something else -- empty, as ``- **/*``
    did -- this row fails with that diagnosis instead of every reason row going red at once. The
    expected set is written by hand from criterion 1, not derived from the constant.
    """
    task_file = tmp_path / "T-0099.md"
    task_file.write_text(_task_file(ROLE_RULE_OWNED))
    res = owned_globs(task_file)
    assert res.returncode == 0, (
        f"the parser failed on the role-rule sandbox's task file.\n"
        f"exit {res.returncode}\nstderr: {res.stderr.strip()}"
    )
    tokens = [line for line in res.stdout.splitlines() if line != ""]
    assert tokens, (
        "the role-rule sandbox's owned section parses EMPTY, so every implementer row of "
        "test_role_rules_are_unchanged is denied by the ownership check, not by its role rule"
    )
    assert len(tokens) == len(set(tokens)) and set(tokens) == ROLE_RULE_OWNED_EXPECTED, (
        f"the role-rule sandbox owns {sorted(tokens)!r}; "
        f"criterion 1 requires {sorted(ROLE_RULE_OWNED_EXPECTED)!r}"
    )


@pytest.mark.parametrize(
    ("path", "allowed"),
    [
        pytest.param("local_astra/conductor/session.py", True, id="owned-module"),
        pytest.param("docs/02-architecture.md", True, id="owned-doc-beside-the-test-plan"),
        pytest.param("local_astra/contracts/protocols.py", False, id="unlisted-module"),
    ],
)
def test_the_role_rule_sandbox_grants_what_it_lists(
    path: str, allowed: bool, sandbox: Sandbox
) -> None:
    """T-0145 criterion 4: the positive control, in the too-wide direction.

    In the same sandbox the implementer may edit two owned paths no role rule protects, and is
    refused an unlisted path *by the ownership check*. Together they prove the owned list parses,
    grants exactly what it lists, and that the ``docs/03-test-plan.md`` refusal above is a rule
    about that file, not a ``docs/*`` rule (``docs/02-architecture.md`` must stay editable).
    """
    sandbox.on_task("T-0099", _task_file(ROLE_RULE_OWNED))
    res = run_hook(path, sandbox, role="implementer")
    if allowed:
        assert_allowed(res, path, "the role-rule sandbox owns it and no role rule protects it")
    else:
        assert_denied(res, path, "the role-rule sandbox does not list it")
        assert NOT_OWNED_RULE in res.stderr, (
            f"an unlisted path must be refused by the ownership check.\n"
            f"stderr: {res.stderr.strip()}"
        )


def test_an_implementer_with_no_task_branch_is_told_to_make_one(sandbox: Sandbox) -> None:
    """Criterion 7: the "no task branch" refusal, message included.

    Wording is asserted here for the same reason as in criterion 6: the message *is* the
    mechanism -- it is the one place the agent is told the exact command that unblocks it, and
    docs/09 names that command. Passes today; a regression pin.
    """
    (sandbox.top / "docs" / "tasks" / "T-0099.md").write_text(_task_file(f"- {OWNED}"))
    sandbox.on_branch("scratch-no-task-id")
    res = run_hook(OWNED, sandbox, role="implementer")
    assert_denied(res, OWNED, "an implementer off a task branch has no ownership at all")
    assert "git switch -c" in res.stderr, (
        f"the refusal must name the command that unblocks it.\nstderr: {res.stderr.strip()}"
    )


def test_a_path_outside_the_worktree_is_not_matched_as_a_relative_path(sandbox: Sandbox) -> None:
    """Criterion 7, last clause (docs/09 §hooks): ``rel`` is only relative when the path really is
    under ``$TOP``.

    An edit to ``/somewhere/else/local_astra/conductor/audio_capture.py`` must not be granted by a
    task that owns ``local_astra/conductor/audio_capture.py`` *inside this worktree*. Passes today
    (``${path#"$TOP"/}`` leaves a foreign path absolute, and no glob is anchored to ``/``); the pin
    exists because a parser rewrite is exactly when someone normalises paths and loses it.
    """
    sandbox.on_task("T-0099", _task_file(f"- {OWNED}"))
    outside = str(Path(sandbox.top.parent) / "elsewhere" / OWNED)
    assert_denied(run_hook(outside, sandbox), outside, "the path is not in this worktree")


# --------------------------------------------------------------------------------------------
# Group 6 -- criterion 1: both hooks read ownership through the same parser.
# Stage 2 of the task; red until the Bash guard grows its ownership rule.
# --------------------------------------------------------------------------------------------

PATHOLOGICAL = _task_file(
    f"- `{OWNED}` *(new — capture stream, clock,\n"
    "  the gate)*\n"
    "- prose that is not a bullet at all\n"
    "\n"
    "**Files this task needs and does not own:**\n"
    f"- `{NOT_OWNED}` — **T-0007**. Not touched.\n"
)


@pytest.mark.parametrize(
    ("path", "expect_allowed"),
    [pytest.param(OWNED, True, id="owned"), pytest.param(NOT_OWNED, False, id="not-owned")],
)
def test_both_hooks_agree_on_the_same_task_file(
    path: str, expect_allowed: bool, sandbox: Sandbox
) -> None:
    """Criterion 1: one parser, in ``_lib.sh``, used by both hooks (docs/09 §"All hooks source
    ``_lib.sh``").

    The same pathological task file -- annotated bullet, wrapped annotation, a prose line, and a
    "does not own" list -- is put to ``protect_paths.sh`` as an Edit and to ``guard_bash.sh`` as a
    ``cat > path`` heredoc. Two parsers would drift the moment one of them is fixed; this row is
    what makes "one parser" observable from outside, since a test cannot see the function.
    """
    sandbox.on_task("T-0099", PATHOLOGICAL)
    edit = run_hook(path, sandbox, role="implementer")
    command = f"cat > {path} <<'PY'\nx = 1\nPY"
    bash = run_bash_hook(command, sandbox, role="implementer")
    if expect_allowed:
        assert_allowed(edit, path, "the annotated bullet lists this path")
        assert_allowed(bash, command, "the annotated bullet lists this path")
    else:
        assert_denied(edit, path, "listed under 'does not own'")
        assert_denied(bash, command, "listed under 'does not own'")
    assert (edit.returncode == 0) == (bash.returncode == 0), (
        "the two hooks disagree about the same task file and the same path -- there are two "
        f"owned-files parsers.\nEdit: exit {edit.returncode} {edit.stderr.strip()}\n"
        f"Bash: exit {bash.returncode} {bash.stderr.strip()}"
    )


# --------------------------------------------------------------------------------------------
# Group 7 -- ruling A2: the parser's own output, pinned token by token.
#
# Criterion 1 names a callable entry point, so the owned list is observable directly instead of
# only through a verdict. Every expectation below is written by hand from the bullet, never
# computed from the section: a test that derived the expected token by stripping backticks would
# be the parser a second time.
# --------------------------------------------------------------------------------------------

LIB = REPO / ".claude" / "hooks" / "_lib.sh"


def owned_globs(task_file: Path) -> subprocess.CompletedProcess[str]:
    """Run the parser entry point on ``task_file`` (ruling A2).

    ``bash _lib.sh owned-globs <task-file>`` prints one glob per line to stdout. stdin is closed
    rather than left open: ``_lib.sh`` reads the hook JSON from stdin when it is *sourced*, and a
    subcommand invocation that blocked on a terminal would hang every caller.
    """
    return subprocess.run(
        ["bash", str(LIB), "owned-globs", str(task_file)],
        input="",
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env=GIT_ENV,
        timeout=30,
    )


def parsed_tokens(section: str, tmp_path: Path, name: str = "T-0099.md") -> list[str]:
    task_file = tmp_path / name
    task_file.write_text(_task_file(section))
    res = owned_globs(task_file)
    assert res.returncode == 0, (
        f"the parser entry point failed on a well-formed task file.\n"
        f"exit {res.returncode}\nstderr: {res.stderr.strip()}"
    )
    return [line for line in res.stdout.splitlines() if line != ""]


CAPTURE = "local_astra/conductor/audio_capture.py"

BULLET_FORM_TOKENS = [
    pytest.param(f"- {OWNED}", [CAPTURE], id="bare"),
    pytest.param(f"- `{OWNED}`", [CAPTURE], id="backticked"),
    pytest.param(f"- `{OWNED}` *(new — the module)*", [CAPTURE], id="backticked-annotated"),
    pytest.param(
        f"- `{OWNED}` *(new — capture stream, clock, xrun/backpressure policy,\n"
        "  the gate, `CaptureBlock`/`CaptureStats`)*",
        [CAPTURE],
        id="annotation-wrapped",
    ),
    pytest.param(f"* `{OWNED}` *(from T-0006)*", [CAPTURE], id="star-bullet"),
    pytest.param(f"+ `{OWNED}`", [CAPTURE], id="plus-bullet"),
    pytest.param(f"- the capture side:\n  - `{OWNED}` *(new)*", [CAPTURE], id="indented"),
    pytest.param(
        "- `local_astra/conductor/**` *(the whole package)*",
        ["local_astra/conductor/**"],
        id="double-star-glob",
    ),
    pytest.param(
        "- `scripts/audio_rig.sh` *(from T-0006: expose the monitor)*",
        ["scripts/audio_rig.sh"],
        id="hyphen-in-name",
    ),
    pytest.param(
        "- `docs/03-test-plan.md` *(the thresholds table only)*",
        ["docs/03-test-plan.md"],
        id="hyphen-and-dot-in-name",
    ),
    pytest.param(
        f"- `{OWNED}`\n\n**Files this task needs and does not own:**\n- `{NOT_OWNED}` — T-0007.",
        [CAPTURE],
        id="marker-ends-the-list",
    ),
    pytest.param(HOSTILE_SECTION, [CAPTURE], id="hostile-section-yields-one-token"),
    pytest.param(
        "- `local_astra/conductor/**`\n- `scripts/gpu/**`\n- `tests/unit/test_x.py`",
        ["local_astra/conductor/**", "scripts/gpu/**", "tests/unit/test_x.py"],
        id="three-bullets-in-order",
    ),
    # Ruling A9, at the token level: the bullet yields `Makefile`, and the `./` spelling yields
    # the same token -- not `./Makefile`, which would match nothing once the hook compares it
    # against a repo-relative path.
    pytest.param("- Makefile", ["Makefile"], id="extensionless-single-token"),
    pytest.param("- ./Makefile", ["Makefile"], id="dot-slash-prefix"),
    # The other side of the same rule: a multi-token prose bullet is still prose.
    pytest.param(
        "- Makefile\n- the build is not owned by this task", ["Makefile"], id="prose-is-multi-token"
    ),
]


@pytest.mark.parametrize(("section", "expected"), BULLET_FORM_TOKENS)
def test_the_parser_entry_point_prints_exactly_the_expected_globs(
    section: str, expected: list[str], tmp_path: Path
) -> None:
    """Ruling A2 / criterion 1: the token list itself, not just the verdict it produces.

    A verdict test cannot tell "the path is owned" from "everything is owned": a parser that
    emitted ``local_astra/conductor/audio_capture.py`` *and* a stray ``*`` would satisfy every
    allow/deny row above and grant the whole repo. Equality against a hand-written list is what
    makes landmines 2 (take the whole remainder of the bullet) and 5 (leak prose) visible as a
    diff rather than as a silent extra grant.

    ``hostile-section-yields-one-token`` is the sharp end: nine lines of bold markup, a lone
    ``*``, table rows, a URL, a fenced block and two out-of-tree bullets must contribute nothing,
    and the single path-shaped bullet must contribute exactly itself.
    """
    tokens = parsed_tokens(section, tmp_path)
    assert tokens == expected, (
        f"the parser printed {tokens!r} for this section; the bullets name {expected!r}"
    )


def test_the_parser_entry_point_prints_the_t0021_block_as_five_paths(tmp_path: Path) -> None:
    """Ruling A2 on the real defect: the verbatim pre-HEAD T-0021 block, token for token.

    The list is the one the task file's *Owned files* section names -- five paths, in the order
    they are written, with their annotations, wrapped continuation lines and the "does not own"
    list below them contributing nothing. This is the whole of ruling A15 in one assertion.
    """
    task_file = tmp_path / "T-0021.md"
    task_file.write_text(T0021_OWNED_BLOCK)
    res = owned_globs(task_file)
    assert res.returncode == 0, (
        f"the parser entry point failed on T-0021's own task file.\n"
        f"exit {res.returncode}\nstderr: {res.stderr.strip()}"
    )
    assert [line for line in res.stdout.splitlines() if line != ""] == T0021_OWNED


def test_the_parser_entry_point_reports_an_unreadable_file_without_a_traceback(
    tmp_path: Path,
) -> None:
    """Ruling A2 + criterion 6 at the entry point: a total function or a named failure.

    Neither a missing file nor an undecodable one may produce a traceback or a stray glob on
    stdout. Exit status is left to the implementer (the caller's contract is criterion 6's
    *deny*, pinned through the hook above); what is pinned here is that stdout never carries a
    pattern in that case -- an empty parse that printed something would be an over-grant.
    """
    missing = owned_globs(tmp_path / "T-0404.md")
    assert missing.stdout.strip() == "", (
        f"a missing task file must yield no globs.\nstdout: {missing.stdout!r}"
    )
    assert "Traceback (most recent call last)" not in missing.stderr, (
        f"a python traceback must never reach the model.\nstderr: {missing.stderr.strip()}"
    )
    undecodable = tmp_path / "T-0405.md"
    undecodable.write_bytes(b"## Owned files\n- local_astra/\x00\xff\xfe.py\n")
    res = owned_globs(undecodable)
    assert res.stdout.strip() == "", (
        f"an undecodable task file must yield no globs.\nstdout: {res.stdout!r}"
    )
    assert "Traceback (most recent call last)" not in res.stderr, (
        f"a python traceback must never reach the model.\nstderr: {res.stderr.strip()}"
    )


# --------------------------------------------------------------------------------------------
# Group 8 -- ruling A7: the live task file the prescribed marker spellings missed.
# --------------------------------------------------------------------------------------------

# Paths T-0011's "Ownership conflicts, resolved explicitly" list disclaims, with the task that
# does own them. They are read from the prose of that list, not from a parser.
T0011_DISCLAIMED = [
    pytest.param("local_astra/conductor/session.py", id="session-py-belongs-to-T-0007-T-0020"),
    pytest.param("hud/server.py", id="hud-belongs-to-T-0008"),
    pytest.param("services/vllm.sh", id="vllm-sh-belongs-to-T-0015"),
    pytest.param("local_astra/contracts/codec.py", id="codec-py-is-explicitly-not-touched"),
]

# Two paths the same section genuinely lists as owned, so the row below cannot be satisfied by
# denying everything in that file.
T0011_OWNED = [
    "plugin/astra_vllm_plugin/astra_vllm_plugin/sampler.py",
    "local_astra/bench/b16_tap.py",
]


@pytest.mark.parametrize("path", T0011_DISCLAIMED)
def test_the_t0011_ownership_conflict_list_is_not_a_grant(path: str, sandbox: Sandbox) -> None:
    """Ruling A7, on the live ``docs/tasks/T-0011.md`` rather than on a synthetic marker.

    T-0011's conflict list names ``local_astra/conductor/session.py`` (T-0007/T-0020),
    ``hud/**`` (T-0008), ``services/vllm.sh`` (T-0015) and ``contracts/codec.py`` precisely to
    disclaim them -- "T-0011 claims **no** conductor file". A section end that only knows the two
    prescribed spellings hands T-0011's implementer all four, which is the concurrency invariant
    broken by the very mechanism meant to enforce it. Driven from the real file so that the day
    someone rewords that heading, this test follows the file.
    """
    live = REPO / "docs" / "tasks" / "T-0011.md"
    if "Ownership conflicts" not in live.read_text():
        pytest.skip("docs/tasks/T-0011.md no longer carries an ownership-conflict list")
    sandbox.on_task("T-0011", live.read_text())
    assert_denied(run_hook(path, sandbox), path, "T-0011's conflict list disclaims this path")


def test_t0011_still_owns_what_it_claims(sandbox: Sandbox) -> None:
    """The other half of A7: ending the section at the conflict list must not cost T-0011 the
    files above it. Without this row, "deny everything in T-0011.md" would pass the group above.
    """
    live = REPO / "docs" / "tasks" / "T-0011.md"
    if "b16_tap.py" not in live.read_text():
        pytest.skip("docs/tasks/T-0011.md no longer lists the b16 tap")
    sandbox.on_task("T-0011", live.read_text())
    for path in T0011_OWNED:
        assert_allowed(run_hook(path, sandbox), path, "T-0011's owned list names this path")


def test_the_parser_yields_no_conductor_or_hud_path_for_t0011() -> None:
    """Ruling A7 at the token level (A2's entry point): the corpus assertion.

    The verdict rows above probe four paths; this one pins that *nothing* T-0011 parses to can
    reach the conductor or the HUD -- including a glob nobody thought to probe. The expectation
    is written from the task file's own sentence ("T-0011 claims **no** conductor file" / "claims
    no HUD file"), not from the parser's output.
    """
    live = REPO / "docs" / "tasks" / "T-0011.md"
    if "Ownership conflicts" not in live.read_text():
        pytest.skip("docs/tasks/T-0011.md no longer carries an ownership-conflict list")
    res = owned_globs(live)
    assert res.returncode == 0, (
        f"the parser entry point failed on a live task file.\n"
        f"exit {res.returncode}\nstderr: {res.stderr.strip()}"
    )
    tokens = [line for line in res.stdout.splitlines() if line != ""]
    assert tokens, "T-0011 lists owned files; an empty parse would deny its implementer everything"
    offenders = [t for t in tokens if t.startswith(("local_astra/conductor/", "hud/", "services/"))]
    assert offenders == [], (
        "T-0011 claims no conductor, HUD or services file; the parser handed it "
        f"{offenders!r}.\nfull token list: {tokens!r}"
    )


# --------------------------------------------------------------------------------------------
# Group 9 -- human ruling 2026-10-22: the test-author and the reviewer may write in a session
# scratchpad (``/tmp/claude-<uid>/<project>/<session>/scratchpad/``), and nowhere else outside
# their rules. Blocking it made them improvise heredocs, inline stdin scripts and temporary
# harnesses under ``tests/``. Both directions are pinned: the grant, and every way it must not
# widen (other roles, other /tmp paths, ``..``, and a scratchpad symlinked into the repo).
# --------------------------------------------------------------------------------------------


@pytest.fixture
def scratchpad() -> Iterator[Path]:
    """A real ``/tmp/claude-*/<project>/<session>/scratchpad`` directory, removed afterwards."""
    root = Path(tempfile.mkdtemp(prefix="claude-protect-test-", dir="/tmp"))
    pad = root / "proj" / "sess" / "scratchpad"
    pad.mkdir(parents=True)
    yield pad
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize("role", ["test-author", "reviewer", "implementer"])
def test_scratchpad_writes_are_allowed_for_test_author_reviewer_and_implementer(
    role: str, scratchpad: Path, sandbox: Sandbox
) -> None:
    target = str(scratchpad / "mutants" / "M1" / "probe.py")
    assert_allowed(run_hook(target, sandbox, role=role), target, f"{role} may write its scratchpad")


@pytest.mark.parametrize("role", ["planner", "verifier", "launch-reviewer", "explorer"])
def test_scratchpad_grant_does_not_extend_to_other_roles(
    role: str, scratchpad: Path, sandbox: Sandbox
) -> None:
    sandbox.on_task("T-0099", _task_file(f"- {OWNED}"))
    target = str(scratchpad / "probe.py")
    assert_denied(run_hook(target, sandbox, role=role), target, f"{role} has no scratchpad grant")


@pytest.mark.parametrize("role", ["test-author", "reviewer", "implementer"])
def test_other_tmp_paths_stay_denied(role: str, scratchpad: Path, sandbox: Sandbox) -> None:
    for target in (
        str(scratchpad.parent / "notes.md"),  # the session directory, not its scratchpad
        "/tmp/elsewhere/scratchpad/probe.py",  # not a claude scratchpad
        str(scratchpad.parent.parent.parent / "scratchpad" / "probe.py"),  # too shallow
    ):
        assert_denied(run_hook(target, sandbox, role=role), target, "not a session scratchpad")


@pytest.mark.parametrize("role", ["test-author", "reviewer", "implementer"])
def test_dotdot_out_of_the_scratchpad_is_denied(
    role: str, scratchpad: Path, sandbox: Sandbox
) -> None:
    target = f"{scratchpad}/../../../../..{sandbox.top}/local_astra/conductor/session.py"
    assert_denied(run_hook(target, sandbox, role=role), target, "`..` leaves the scratchpad")
    # Out of the scratchpad but not into the repo: only the containment check can refuse this one.
    home = f"{scratchpad}/../../../../../home/someone/.bashrc"
    assert_denied(run_hook(home, sandbox, role=role), home, "`..` leaves the scratchpad")


@pytest.mark.parametrize("role", ["test-author", "reviewer", "implementer"])
def test_a_scratchpad_symlinked_into_the_repo_is_denied(
    role: str, scratchpad: Path, sandbox: Sandbox
) -> None:
    link_pad = scratchpad.parent / "linked" / "scratchpad"
    link_pad.parent.mkdir(parents=True)
    link_pad.symlink_to(sandbox.top / "local_astra")
    target = str(link_pad / "conductor" / "session.py")
    assert_denied(
        run_hook(target, sandbox, role=role), target, "the scratchpad resolves into the repo"
    )
