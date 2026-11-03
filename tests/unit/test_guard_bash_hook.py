"""Acceptance tests for the Bash PreToolUse guard, ``.claude/hooks/guard_bash.sh``.

The hook's contract (``docs/09-agent-workflow.md``, CLAUDE.md "Any command that spends GPU time
or writes to ``artifacts/`` is reviewed (``/launch``) **before** it runs, and never runs against a
dirty tree"):

* stdin is the hook JSON ``{"cwd", "agent_type", "tool_input": {"command"}}``;
* **exit 0 = allowed**, **non-zero = blocked**. That status is the whole contract -- nothing here
  asserts on message wording.

Two defects are pinned below, both observed live on 2026-09-13:

**Defect A (false negative, the serious one).** The GPU trigger keys on command *spelling*
(``astra bench``, ``vllm serve``, ``nohup ``, a redirect into ``artifacts/`` ...), not on effect. A
command that drives a running engine under a different spelling spends real GPU time while
consuming no launch token *and* skipping the dirty-tree refusal entirely.

**Defect B (false positive).** The patterns match anywhere in the command text, including inside
quoted arguments and heredoc bodies, so writing *about* a GPU command -- a commit message, a
``grep`` pattern -- is blocked.

Groups 1 and 2 pin the two defects, group 3 the behaviour the fix must not break. Groups 4 and 5
came out of the review of the first implementation (T-0016 rulings T1-T3): group 4 is the D1
distinction -- a heredoc body is inert text only when nothing executes it -- and group 5 the
false positives found after the original tests were written. Group 6 came out of the same review
(rulings I1, I2, I9, D5): the ``cd`` bypass, ``scripts/locked/``, read-only work on a harness,
and the ungated demo directory. It was the spec for the implementer's second pass and carried
``pending(task="T-0016")`` until that pass landed (ruling R1). Group 7 and the quoted-path rows
of ``GPU_SPENDING`` are the third review's findings (I10-I12): a quoted gated path, a wrapped or
sourced harness, and the two service-start spellings that need a clean tree rather than a token.
Groups 8-10 are T-0026: a wrapper or interpreter flag whose argument is a separate token
(``nice -n 10``, ``timeout -s KILL 600``, ``bash -euo pipefail``) currently ends the peel, so the
harness behind it runs unreviewed; the over-block twin of I10 (a harness path named in prose) gets
its regression pin; and the two refusal messages are pinned to the obligations the rules actually
impose. Those three groups (and group 12 below) are the only places in this file that assert on
wording, and the tests that do say why in their docstrings.

Group 12 is T-0145: three deny rules that no row decided alone -- the tests-overwrite rule, the
implementer's push/PR rule, and four of the six destructive-git forms. Each deny row there asserts
its rule's message, because for an implementer the tests-overwrite rule overlaps the ownership
rule: a row asserting only the exit status stays green when the tests rule is removed, since the
ownership rule then denies for its own reason. A deny for the wrong reason looks like a pass.

Group 11 is T-0029: the Bash guard gains an ownership rule for ``ROLE=implementer``, because
until it had one an implementer that reached for ``cat > x.py <<'PY'``, ``tee``, ``sed -i`` or
``cp`` wrote any file in the repo with no ownership check at all (measured 2026-09-14 on ``main``
@ 7989c81: all four ALLOWED), while ``protect_paths.sh`` refused the same write as an Edit.

**Deliberately unasserted in the ownership rule (T-0029 ruling A9, residuals in the hook
header).** ``>|`` (clobber-override) and ``>&word`` (the word form of a descriptor duplication,
which in bash can name a *file* rather than a descriptor), and a write target behind any variable
other than ``$TOP``/``$PWD``. The variable case is recorded by one row in the other direction --
``variable-target-outside-top`` pins that such a target is skipped rather than guessed at -- but
nothing here asserts what happens when a variable does expand into the repo, because the hook
cannot know that without running the shell. ``truncate``, ``install`` and ``dd of=`` are the same
family and belong to D-a below.

**Deliberately unasserted, in either direction (T-0029 deferred item D-a).** A heredoc fed to an
interpreter that then writes a file -- ``python3 - <<PY ... open("local_astra/x.py","w") ... PY``,
``perl -e``, ``node -e``, ``install``, ``dd of=``, ``patch -p1 <``, ``git apply``,
``git restore <path>``, ``truncate`` -- and the whole indirection family of T-0016 I4-I7. The
ownership rule does not see them, for the same reason the GPU trigger accepts ``cat x | bash``:
the guard is a speed bump, not a sandbox. Asserting either verdict here would pin a decision
nobody has taken; it is recorded rather than tested, the way group 7 treats indirection.

Every case builds its own throwaway git repo and its own launch-token state, so nothing here
depends on whether this repo happens to be dirty or to have a pending ``/launch`` approval.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / ".claude" / "hooks" / "guard_bash.sh"

# Git in a hermetic state: no user config, no global hooks, deterministic identity.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_AUTHOR_NAME": "guard-test",
    "GIT_AUTHOR_EMAIL": "guard-test@example.invalid",
    "GIT_COMMITTER_NAME": "guard-test",
    "GIT_COMMITTER_EMAIL": "guard-test@example.invalid",
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
    """A throwaway git repo laid out like this one, plus its launch-token slot."""

    top: Path
    token: Path

    def approve(self, command: str) -> None:
        """Write the single-use launch token the way ``/launch`` does."""
        self.token.parent.mkdir(parents=True, exist_ok=True)
        self.token.write_text(command)

    def on_task(self, task_id: str, owned: list[str]) -> None:
        """Write ``docs/tasks/<task_id>.md`` with an ``## Owned files`` section and switch to the
        task's branch (T-0029).

        The branch name is where ``_lib.sh`` reads ``TASK`` from and the task file is where
        ownership is read from, so the two are set together or the hook answers about a task the
        test never wrote. The bullets are annotated on purpose: an annotated bullet is the shape
        T-0029's parser exists for, and a rule that only works for bare paths would pass a
        bare-path fixture while denying every real task's owned files.
        """
        tasks = self.top / "docs" / "tasks"
        tasks.mkdir(parents=True, exist_ok=True)
        bullets = "\n".join(f"- `{o}` *(new — the module)*" for o in owned)
        tasks.joinpath(f"{task_id}.md").write_text(
            f"# {task_id} — the probe task\n\n## Owned files\n{bullets}\n\n"
            "**Files this task needs and does not own:**\n- local_astra/x.py\n\n"
            "## Out of scope\n- Everything else.\n"
        )
        _git(self.top, "add", "-A")
        _git(self.top, "commit", "-qm", f"task {task_id}")
        _git(self.top, "switch", "-q", "-c", f"{task_id}-probe")

    def on_branch(self, name: str) -> None:
        _git(self.top, "switch", "-q", "-c", name)

    def make_dirty(self, rel: str = "README.md") -> str:
        """Edit a tracked file and return its repo-relative path.

        The path is returned so a test can look for it in a refusal message: ruling I12 requires
        a dirty-tree refusal to name what is dirty, and a message that only says "tree is dirty"
        is the one that gets an agent to re-run blind.
        """
        (self.top / rel).write_text("edited by an agent while the bench was running\n")
        return rel


def _x3_relpath() -> str:
    """Where the x3 prefix-cache harness lives, as a repo-relative posix path.

    Looked up rather than hard-coded: T-0016's drafted fix moves GPU-spending harnesses under
    ``scripts/gpu/``, and these tests must follow the harness rather than pin the path it had on
    the day the defect was observed. The fallback is the path the task mandates.
    """
    found = sorted(REPO.glob("scripts/**/x3_prefix_cache.sh"))
    return found[0].relative_to(REPO).as_posix() if found else "scripts/gpu/x3_prefix_cache.sh"


X3 = _x3_relpath()
# A harness that does not exist yet: the rule must be structural (a directory), not a list of
# filenames someone remembered to add (T-0016 acceptance criterion 2).
NEW_HARNESS = "scripts/gpu/x9_render_sweep.sh"
# A python harness under the same convention, used by the I9 read-only cases below. It is a
# harness rather than ``scripts/gpu/ask.py`` on purpose: ruling D5 moves the interactive demo
# scripts out of ``scripts/gpu/``, so a test that linted ``scripts/gpu/ask.py`` would pin a path
# the same task deletes.
GPU_PY_HARNESS = "scripts/gpu/x9_render_sweep.py"
# ``/launch`` step 2 copies a long script here and runs it from here (ruling I2).
LOCKED_COPY = "scripts/locked/x3_prefix_cache.sh"
# Ruling D5: interactive demo scripts are ungated and live here.
DEMO_SCRIPT = "scripts/demo/ask.py"

HARNESS_BODY = (
    "#!/usr/bin/env bash\n"
    "API=http://127.0.0.1:8001\n"
    'curl -s "$API/v1/chat/completions" -d @/tmp/req.json\n'
)
PY_BODY = (
    "#!/usr/bin/env python3\n"
    "import urllib.request\n"
    'urllib.request.urlopen("http://127.0.0.1:8001/v1/chat/completions", b"{}")\n'
)


@pytest.fixture
def sandbox(tmp_path: Path) -> Sandbox:
    top = tmp_path / "repo"
    (top / "logs").mkdir(parents=True)
    (top / "artifacts").mkdir()
    (top / "README.md").write_text("clean\n")
    # Realistic GPU-spending repo scripts: they send inference requests to a running engine.
    # ``LOCKED_COPY`` is the same harness where ``/launch`` step 2 puts it; ``DEMO_SCRIPT`` is the
    # ungated interactive demo (ruling D5). The hook only reads command text, but the sandbox
    # mirrors the layout the rulings describe so the paths under test are not fictional.
    for rel in (X3, NEW_HARNESS, GPU_PY_HARNESS, LOCKED_COPY, DEMO_SCRIPT):
        script = top / rel
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(PY_BODY if rel.endswith(".py") else HARNESS_BODY)
        script.chmod(0o755)
    _git(top.parent, "init", "-q", str(top))
    _git(top, "add", "-A")
    _git(top, "commit", "-qm", "init")
    common = _git(top, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Sandbox(top=top, token=Path(common) / "local_astra" / "launch_approved")


def run_hook(
    command: str, sandbox: Sandbox, role: str = "none"
) -> subprocess.CompletedProcess[str]:
    payload = json.dumps(
        {
            "cwd": str(sandbox.top),
            "agent_type": role,
            "tool_input": {"command": command},
        }
    )
    return subprocess.run(
        ["bash", str(HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(sandbox.top),
        env=GIT_ENV,
    )


def assert_blocked(res: subprocess.CompletedProcess[str], command: str, why: str) -> None:
    assert res.returncode != 0, f"hook ALLOWED (exit 0) a command it must block ({why}):\n{command}"


def assert_allowed(res: subprocess.CompletedProcess[str], command: str, why: str) -> None:
    assert res.returncode == 0, (
        f"hook BLOCKED (exit {res.returncode}) a command it must allow ({why}):\n"
        f"{command}\nstderr: {res.stderr.strip()}"
    )


# --------------------------------------------------------------------------------------------
# Group 1 -- Defect A: commands that spend GPU time but are currently allowed.
# Each must be blocked when no launch token exists. The sandbox has no token, so "blocked" here
# means exactly "the hook recognised this as GPU-spending and demanded /launch review".
# --------------------------------------------------------------------------------------------

# The shape of the command run live on 2026-09-13: absolute paths, a pipe into tee, and a log
# under logs/ rather than artifacts/ -- so not one pattern fires. ``{top}`` is the sandbox repo
# root and ``{x3}`` the harness's current home (see _x3_relpath).
X3_AS_RUN_LIVE = "bash {top}/{x3} 2>&1 | tee {top}/logs/x3-prefix-cache.log"

GPU_SPENDING = [
    pytest.param(X3_AS_RUN_LIVE, id="x3-harness-as-run-live"),
    pytest.param("bash {x3} > logs/x3-prefix-cache.log 2>&1", id="x3-harness-relative-path"),
    pytest.param("./{x3}", id="x3-harness-executed-directly"),
    pytest.param("bash " + NEW_HARNESS + " --n 5", id="unknown-harness-under-scripts-gpu"),
    pytest.param(
        "docker run --rm --gpus all -p 127.0.0.1:8001:8000 "
        "-v /models:/models vllm-astra:pinned "
        "python3 -m vllm.entrypoints.openai.api_server --model /models/diffusion-gemma",
        id="docker-run-gpus",
    ),
    # Ruling I10: quoting the path does not make it a different path. All five spellings below
    # were ALLOWED by the first implementation, because quoted spans are blanked before the
    # path trigger looks -- the defect-B cure eating the defect-A fix for a second time (D1 was
    # the first). An agent quotes a path by reflex whenever it might contain a space.
    pytest.param('bash "{x3}"', id="quoted-path-double"),
    pytest.param("bash '{x3}'", id="quoted-path-single"),
    pytest.param('bash "$TOP/{x3}"', id="quoted-path-shell-variable"),
    pytest.param('bash "{top}/{x3}"', id="quoted-path-absolute"),
    pytest.param('cd scripts/gpu && bash "x3_prefix_cache.sh"', id="quoted-path-after-cd"),
    pytest.param('bash "' + LOCKED_COPY + '"', id="quoted-path-under-scripts-locked"),
]


@pytest.mark.parametrize("template", GPU_SPENDING)
def test_gpu_spending_command_needs_a_launch_token(template: str, sandbox: Sandbox) -> None:
    """Defect A: effect, not spelling, decides whether /launch review is required.

    None of these match the current regex, so today they run unreviewed against a possibly dirty
    tree -- the exact hole that let the x3 prefix-cache run happen this session. The sandbox holds
    no approval, so "blocked" here means precisely "recognised as GPU-spending".

    ``unknown-harness-under-scripts-gpu`` is the adversarial one: a harness that does not exist in
    this repo and whose name nobody has added to any list. If it is allowed while the x3 cases are
    blocked, the fix is another spelling table and the defect will come back with the next script.
    """
    command = template.format(top=sandbox.top, x3=X3)
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "spends GPU time with no launch token")


def test_gpu_spending_command_is_refused_on_a_dirty_tree(sandbox: Sandbox) -> None:
    """Defect A, second half: the live run also skipped the dirty-tree check.

    Even *with* a matching approval, a GPU-spending command must be refused while the tree is
    dirty ("never measure a tree an agent is editing"). Today the x3 invocation never reaches this
    check at all, so the number it printed was measured against a tree two agents were editing.
    """
    command = X3_AS_RUN_LIVE.format(top=sandbox.top, x3=X3)
    sandbox.approve(command)
    sandbox.make_dirty()
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "approved but the tree is dirty")


def test_a_post_to_the_running_engine_needs_a_launch_token(sandbox: Sandbox) -> None:
    """Defect A, the bare shape: an inference POST typed straight into the shell.

    This is what the x3 harness *does*; running it without the script wrapper spends the same GPU
    time. T-0016's drafted fix lists the engine-port trigger as optional ("weigh this ... or leave
    it out and rely on the path convention"), so this case is the one deliberate judgement call in
    the file: if the implementer decides against it, that decision belongs in the task file rather
    than in a quietly deleted test. The read-only counterpart (GET /v1/models) must stay allowed --
    see test_polling_the_engine_for_status_is_allowed.
    """
    command = (
        "curl -sX POST http://127.0.0.1:8001/v1/chat/completions "
        "-H 'Content-Type: application/json' -d @/tmp/req.json"
    )
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "an inference request is GPU time")


# --------------------------------------------------------------------------------------------
# Group 2 -- Defect B: commands that must be allowed but are blocked because a trigger word
# appears inside a quoted argument or a heredoc body. All three are real, from this session.
# --------------------------------------------------------------------------------------------

COMMIT_MENTIONING_NOHUP = """git commit -q -F - <<'MSG'
Document the /launch skill

The skill reviews the command before it runs, including the nohup  form used to
detach a long bench, and records the approval as a single-use token.
MSG"""

COMMIT_QUOTING_A_VLLM_ERROR = """git commit -q -F - <<'MSG'
Guard the diffusion path

The engine died on the image path; the log line was
  vllm serve: EngineDeadError (AttributeError: no attribute '_enable_mm_lora')
which kills EngineCore rather than the single request.
MSG"""

GREP_FOR_A_GPU_PHRASE = "grep -rn 'astra bench' docs/ .claude/"

COMMIT_MENTIONING_ARTIFACTS = (
    'git commit -q -m "X3: tee artifacts/x3.json was the wrong regime for B11"'
)

TEXT_ONLY_MENTIONS = [
    pytest.param(COMMIT_MENTIONING_NOHUP, id="commit-message-contains-nohup"),
    pytest.param(COMMIT_QUOTING_A_VLLM_ERROR, id="commit-message-quotes-vllm-serve"),
    pytest.param(GREP_FOR_A_GPU_PHRASE, id="grep-pattern-contains-astra-bench"),
    pytest.param(COMMIT_MENTIONING_ARTIFACTS, id="commit-message-mentions-tee-artifacts"),
]


@pytest.mark.parametrize("command", TEXT_ONLY_MENTIONS)
def test_trigger_words_inside_quotes_and_heredocs_are_not_gpu_commands(
    command: str, sandbox: Sandbox
) -> None:
    """Defect B: writing *about* a GPU command spends no GPU time.

    A commit whose message mentions ``nohup `` or quotes a ``vllm serve:`` error, and a read-only
    ``grep`` whose search pattern is ``astra bench``, must all run without a launch token.
    """
    assert not sandbox.token.exists(), "these commands must not need an approval at all"
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "the trigger word is inert text, not an executed program")


# --------------------------------------------------------------------------------------------
# Group 3 -- regression guards. These pass today; a fix must keep them passing.
# --------------------------------------------------------------------------------------------

VLLM_SERVE = "vllm serve /models/diffusion-gemma --port 8001 --attention-backend FLASHINFER"


def test_vllm_serve_without_a_token_is_blocked(sandbox: Sandbox) -> None:
    res = run_hook(VLLM_SERVE, sandbox)
    assert_blocked(res, VLLM_SERVE, "GPU command, no launch token")


def test_the_bootstrap_self_check_payload_is_still_blocked(sandbox: Sandbox) -> None:
    """``scripts/bootstrap.sh:42`` feeds this exact command and reports ok only when it is blocked.

    Same command, same ``agent_type``. If this stops being blocked, bootstrap starts reporting
    ``guard_bash.sh let an unapproved GPU command through`` on every fresh checkout.
    """
    command = "uv run astra bench B3"
    res = run_hook(command, sandbox, role="implementer")
    assert_blocked(res, command, "bootstrap's hook self-check")


def test_pytest_dash_m_gpu_without_a_token_is_blocked(sandbox: Sandbox) -> None:
    command = "uv run pytest -q -m gpu"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "selects the gpu marker, no launch token")


C_PAYLOAD_LAUNCHES = [
    # The whole command is `sh -c '<engine start>'`: nothing outside the quotes is a trigger, so
    # this case and only this case pins the `-c` exception. Deleting the exception (guard_bash.sh
    # line 26/30, so that quoted spans are always blanked) leaves the hook looking at `sh -c` and
    # it must go red here. Every other launch shape below carries a second trigger.
    pytest.param(
        "sh -c 'exec vllm serve /models/diffusion-gemma --host 127.0.0.1 --port 8001'",
        id="bare-sh-c-payload-only-trigger",
    ),
    # How we actually start the engine on the 5090 box: the docker group wrapper, everything in
    # one double-quoted payload (`services/vllm.sh`).
    pytest.param(
        'sg docker -c "docker run -d --gpus all -p 127.0.0.1:8001:8000 '
        'vllm/vllm-openai:v0.29.0 vllm serve /models/dg"',
        id="sg-docker-c-payload",
    ),
    # ``services/vllm.sh:81``: the engine launch lives inside the container entrypoint's -c
    # argument. Kept as a regression shape; it also trips `docker run --gpus`, so it does not by
    # itself pin the `-c` exception.
    pytest.param(
        "docker run -d --gpus all --entrypoint bash vllm/vllm-openai:v0.29.0 "
        '-c "pip install -e /plugin && exec vllm serve /models/dg --host 127.0.0.1 --port 8001"',
        id="docker-entrypoint-c-payload",
    ),
]


@pytest.mark.parametrize("command", C_PAYLOAD_LAUNCHES)
def test_a_gpu_command_inside_a_quoted_c_payload_is_still_blocked(
    command: str, sandbox: Sandbox
) -> None:
    """The adversarial case against the defect-B fix.

    Stripping quoted spans to cure defect B (trigger words inside a commit message) must not
    blind the hook to how we actually launch: every real engine start in this repo puts the
    command inside the quoted argument of a ``-c``. The first case is deliberately minimal --
    remove the payload and the command is inert -- because the earlier version of this test
    carried ``docker run --gpus all`` outside the quotes, so it stayed green even with the ``-c``
    exception deleted (T-0016 ruling T2).
    """
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "an engine launch, merely written inside quotes")


def test_polling_the_engine_for_status_is_allowed(sandbox: Sandbox) -> None:
    """Reading the engine's status spends no GPU time; we do it constantly.

    Guards the over-blocking direction of the defect-A fix: a trigger on "any request to an engine
    port" would put a launch review in front of ``/v1/models`` and ``/metrics``.
    """
    for command in (
        "curl -s http://127.0.0.1:8001/v1/models",
        "curl -s http://127.0.0.1:8001/metrics | grep prefix_cache",
    ):
        res = run_hook(command, sandbox)
        assert_allowed(res, command, "read-only status poll")


def test_pytest_dash_m_not_gpu_is_allowed(sandbox: Sandbox) -> None:
    command = 'uv run pytest tests/unit tests/contract -q -m "not gpu"'
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "the fast suite explicitly excludes gpu")


def test_matching_token_on_a_clean_tree_allows_the_run(sandbox: Sandbox) -> None:
    sandbox.approve(VLLM_SERVE)
    res = run_hook(VLLM_SERVE, sandbox)
    assert_allowed(res, VLLM_SERVE, "approved by /launch, tree clean")


def test_the_launch_token_is_single_use(sandbox: Sandbox) -> None:
    """An approval buys one run: the second identical invocation is blocked again."""
    sandbox.approve(VLLM_SERVE)
    first = run_hook(VLLM_SERVE, sandbox)
    assert_allowed(first, VLLM_SERVE, "approved by /launch, tree clean")
    assert not sandbox.token.exists(), "the token must be consumed by the run it approved"
    second = run_hook(VLLM_SERVE, sandbox)
    assert_blocked(second, VLLM_SERVE, "the approval was already spent")


def test_a_token_for_a_different_command_does_not_transfer(sandbox: Sandbox) -> None:
    sandbox.approve("uv run astra bench B11")
    res = run_hook(VLLM_SERVE, sandbox)
    assert_blocked(res, VLLM_SERVE, "approval was for a different command")


def test_dirty_tree_is_refused_even_with_a_matching_token(sandbox: Sandbox) -> None:
    sandbox.approve(VLLM_SERVE)
    sandbox.make_dirty()
    res = run_hook(VLLM_SERVE, sandbox)
    assert_blocked(res, VLLM_SERVE, "never measure a tree an agent is editing")


def test_staged_changes_also_count_as_dirty(sandbox: Sandbox) -> None:
    sandbox.approve(VLLM_SERVE)
    sandbox.make_dirty()
    _git(sandbox.top, "add", "-A")
    res = run_hook(VLLM_SERVE, sandbox)
    assert_blocked(res, VLLM_SERVE, "staged edits are edits")


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("git status --short", id="git-status"),
        pytest.param("cat artifacts/turns.jsonl | head -3", id="read-of-artifacts"),
        pytest.param("uv run ruff check .", id="lint"),
        pytest.param("rg -n 'moshi' docs/02-architecture.md", id="ripgrep-in-docs"),
    ],
)
def test_ordinary_commands_are_allowed(command: str, sandbox: Sandbox) -> None:
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "no GPU time, no artifacts/ write")


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("git reset --hard HEAD~1", id="reset-hard"),
        pytest.param("git push --force origin main", id="push-force"),
        pytest.param("uv run pytest -q --deselect tests/unit/test_speech_gate.py", id="deselect"),
    ],
)
def test_the_other_guards_still_fire(command: str, sandbox: Sandbox) -> None:
    """Not about GPU, but they share the hook: a fix must not disarm them."""
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "destructive git / test silencing")


# --------------------------------------------------------------------------------------------
# Group 4 -- D1: a heredoc body is inert text *only* when nothing executes it.
#
# Curing defect B means the hook stops reading heredoc bodies. Ruling D1 says that must not
# re-open defect A: `bash <<'SH' ... SH` runs its body, so the body is the command. The pair of
# tests below is what is pinned -- the *distinction*, not "heredocs are blocked" (which would ban
# commit messages again) and not "heredocs are ignored" (which hides an engine launch in the one
# construct the guard was taught to skip). Mutating the hook to blank heredoc bodies
# unconditionally must turn the first test red and leave the second green.
# --------------------------------------------------------------------------------------------

ENGINE_START_IN_A_BASH_HEREDOC = """bash <<'SH'
set -euo pipefail
exec vllm serve /models/diffusion-gemma --port 8001 --attention-backend FLASHINFER
SH"""

HARNESS_IN_A_BASH_HEREDOC = """bash <<'SH'
bash {x3}
SH"""

# The pipeline form: the interpreter is named after the heredoc's own line, not before it.
HARNESS_IN_A_PIPED_HEREDOC = """cat <<'SH' | bash
bash {x3}
SH"""

# Unquoted delimiter (the body is expanded by the shell) -- the same execution, different syntax.
ENGINE_START_IN_AN_SH_HEREDOC = """sh <<EOF
exec vllm serve /models/dg --port 8001
EOF"""

# The tab-stripping form `<<-`; the body is indented with a real tab.
HARNESS_IN_A_TAB_HEREDOC = "bash <<-SH\n\tbash {x3} --n 3\nSH"

HEREDOC_FED_TO_AN_INTERPRETER = [
    pytest.param(ENGINE_START_IN_A_BASH_HEREDOC, id="bash-heredoc-starts-the-engine"),
    pytest.param(HARNESS_IN_A_BASH_HEREDOC, id="bash-heredoc-runs-the-x3-harness"),
    pytest.param(HARNESS_IN_A_PIPED_HEREDOC, id="heredoc-piped-into-bash"),
    pytest.param(ENGINE_START_IN_AN_SH_HEREDOC, id="sh-heredoc-unquoted-delimiter"),
    pytest.param(HARNESS_IN_A_TAB_HEREDOC, id="bash-heredoc-tab-indented-form"),
]


@pytest.mark.parametrize("template", HEREDOC_FED_TO_AN_INTERPRETER)
def test_a_heredoc_fed_to_an_interpreter_is_still_a_gpu_command(
    template: str, sandbox: Sandbox
) -> None:
    """D1: the body of a heredoc that an interpreter executes is the command it runs.

    Adversarial against the defect-B cure: "strip heredoc bodies always" would make every one of
    these invisible, which is a launch that spends VRAM with no token and no dirty-tree check --
    defect A again, wearing the costume the guard was told to ignore.
    """
    command = template.format(x3=X3)
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "an interpreter executes this heredoc body")


COMMIT_HEREDOC_DESCRIBING_THE_HARNESS = """git commit -q -F - <<'MSG'
T-0016: record how the x3 prefix-cache number was produced

Ran `bash {x3}` against the engine started by
`vllm serve /models/dg`; the log went to logs/ rather than artifacts/, which is
why the guard never fired.
MSG"""

FILE_WRITE_HEREDOC_DESCRIBING_THE_HARNESS = """cat >> notes.md <<'EOF'
To reproduce: start the engine, then run bash {x3} --n 3 and read the p50 line.
EOF"""

HEREDOC_FED_TO_A_NON_INTERPRETER = [
    pytest.param(COMMIT_HEREDOC_DESCRIBING_THE_HARNESS, id="commit-message-heredoc"),
    pytest.param(FILE_WRITE_HEREDOC_DESCRIBING_THE_HARNESS, id="file-write-heredoc"),
]


@pytest.mark.parametrize("template", HEREDOC_FED_TO_A_NON_INTERPRETER)
def test_a_heredoc_fed_to_a_non_interpreter_is_inert_text(template: str, sandbox: Sandbox) -> None:
    """The other half of D1: ``git commit -F -`` and ``cat >> file`` record text, they run none.

    Written as the pair of the test above so that what is pinned is the distinction between
    "something executes this body" and "something stores it" -- not "heredocs are blocked", which
    is defect B, and which is how this file's own task description kept getting refused.
    """
    command = template.format(x3=X3)
    assert not sandbox.token.exists(), "these commands must not need an approval at all"
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "nothing executes a commit message or a file being appended to")


# --------------------------------------------------------------------------------------------
# Group 5 -- false positives found after the original tests were written (T-0016 ruling T3).
# All four must be allowed with no token. Each names the clause it pins, because the pytest
# ``-m gpu`` trigger has two independent anchors and a single command only exercises one.
# --------------------------------------------------------------------------------------------


def test_a_curl_with_max_time_next_to_an_nvidia_smi_query_is_allowed(sandbox: Sandbox) -> None:
    """The launch-reviewer's find, 2026-09-13: ``-m`` inside ``--max-time`` plus ``gpu`` inside
    ``--query-gpu`` used to satisfy the pytest marker clause between them.

    Pins the *token* anchor: ``-m`` must be a word of its own, not any ``-m``-ish substring.
    """
    command = (
        "curl -s --max-time 5 http://127.0.0.1:8001/metrics | tail -2; "
        "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv"
    )
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "a status poll and a VRAM read; no marker selection anywhere")


def test_a_pytest_run_whose_log_name_mentions_gpu_is_allowed(sandbox: Sandbox) -> None:
    """The fast suite, teed to a log named after the box it ran on.

    ``pytest`` really is invoked here, so the pytest half of the clause is satisfied and only the
    ``-m`` token anchor stands between this command and a launch review: without it the ``-m`` of
    ``--maxfail`` reaches forward to the ``gpu`` in the log filename.
    """
    command = (
        "uv run pytest tests/unit tests/contract -q --maxfail=1 2>&1 "
        "| tee logs/fast-suite-on-the-gpu-box.log"
    )
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "the fast suite selects no marker and writes under logs/")


def test_a_commit_message_mentioning_a_marker_run_is_allowed(sandbox: Sandbox) -> None:
    """Defect B, the form that blocked the writing of this task file.

    The message contains a literal ``pytest -m gpu``. Pins the other anchor: the marker clause
    must require ``pytest`` to be *invoked* (matched against the normalised command), not merely
    written -- a commit is not a test run.
    """
    command = 'git commit -q -m "T-0016: pytest -m gpu now needs a launch token"'
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "the marker selection is quoted text in a commit message")


def test_a_grep_for_gpu_with_a_match_limit_is_allowed(sandbox: Sandbox) -> None:
    """``grep -m 1 -n gpu ...``: a genuine ``-m`` token whose argument is followed by ``gpu``.

    The complement of the test above: here ``-m`` *is* a word of its own, so only the "pytest is
    actually invoked" half keeps this read-only search out of the launch queue.
    """
    command = "grep -m 1 -n gpu logs/x3-prefix-cache.log"
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "grep -m is a match limit, not a pytest marker selector")


# --------------------------------------------------------------------------------------------
# Group 6 -- rulings I1, I2, I9 and D5, written as the spec for the implementer's second pass
# and activated when it landed (ruling R1). Where a ruling has an over-block direction (I9, D5)
# a test beside it pins that direction, so the blocking half cannot be satisfied by erasing the
# distinction it rests on: `bash -n` on a harness must stay free while `bash` on it does not,
# and `scripts/demo/` must stay free while `scripts/gpu/` does not.
# --------------------------------------------------------------------------------------------

READ_ONLY_WORK_UNDER_SCRIPTS_GPU = [
    pytest.param("uv run ruff check " + GPU_PY_HARNESS, id="ruff-check"),
    pytest.param("bash -n {x3}", id="bash-syntax-check"),
    # The clustered form: `-en` must still hit the shell `-n` early return, before any rule that
    # reads a cluster's last letter to decide whether the next token is an argument (T-0026).
    pytest.param("bash -en {x3}", id="bash-syntax-check-clustered"),
    pytest.param("python3 -m py_compile " + GPU_PY_HARNESS, id="py-compile"),
]


@pytest.mark.parametrize("template", READ_ONLY_WORK_UNDER_SCRIPTS_GPU)
def test_read_only_work_under_scripts_gpu_is_allowed(template: str, sandbox: Sandbox) -> None:
    """Ruling I9: the directory convention must gate *running* a harness, not touching it.

    Linting, syntax-checking and byte-compiling spend no GPU time, and an agent editing a harness
    does all three -- but each of these command words (``uv``, ``bash``, ``python3``) is on the
    interpreter list, so today the path alone condemns them. If they need a launch token, the
    person fixing the harness is the person who learns to route around the hook, which is how the
    original defects were found. ``.claude/rules/gpu-services.md`` already promises this; the hook
    must keep the promise. The distinguishing rule the ruling names: the path must be the
    interpreter's first non-flag argument, or the command word an execution rather than a lint.
    """
    command = template.format(x3=X3)
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "reading or checking a harness is not running it")


NON_INTERPRETER_READS_UNDER_SCRIPTS_GPU = [
    pytest.param("cat {x3}", id="cat"),
    pytest.param("sed -n '1,20p' {x3}", id="sed-print"),
    pytest.param("wc -l {x3} " + GPU_PY_HARNESS, id="wc"),
    pytest.param("git add {x3} && git commit -q -m 'move the harness'", id="git-add-and-commit"),
]


@pytest.mark.parametrize("template", NON_INTERPRETER_READS_UNDER_SCRIPTS_GPU)
def test_reading_a_harness_with_a_non_interpreter_stays_allowed(
    template: str, sandbox: Sandbox
) -> None:
    """The half of ruling I9 the hook already satisfies -- pinned so a fix for the other half
    (allowing ``bash -n``) is not implemented by weakening the path trigger to a word list.

    ``cat``/``sed``/``wc``/``git`` are not interpreters, so the path convention already lets them
    through; that must survive whatever makes ``bash -n`` pass.
    """
    command = template.format(x3=X3)
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "the harness is data here, not a program")


CD_INTO_SCRIPTS_GPU = [
    pytest.param("cd scripts/gpu && ./x3_prefix_cache.sh", id="cd-then-direct-execution"),
    pytest.param("cd scripts/gpu && bash x3_prefix_cache.sh", id="cd-then-bash"),
    pytest.param(
        "bash -c 'cd scripts/gpu && ./x9_render_sweep.sh --n 5'", id="cd-inside-a-c-payload"
    ),
]


@pytest.mark.parametrize("command", CD_INTO_SCRIPTS_GPU)
def test_cd_into_scripts_gpu_does_not_escape_the_convention(command: str, sandbox: Sandbox) -> None:
    """Ruling I1: a path convention that a ``cd`` defeats is not a convention.

    Agents in this repo write ``cd X && ...`` on nearly every call, so this is the likeliest
    accidental bypass, not an exotic one.
    """
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "the harness still runs; only the working directory moved")


def test_a_rerun_from_scripts_locked_needs_its_own_token(sandbox: Sandbox) -> None:
    """Ruling I2: ``/launch`` step 2 copies long scripts to ``scripts/locked/`` and runs them there.

    The copy spends exactly the GPU time the original does, and the token is single-use, so a
    second run out of ``scripts/locked/`` must be refused rather than invisible.
    """
    command = "bash " + LOCKED_COPY + " 2>&1 | tee logs/x.log"
    assert not sandbox.token.exists(), "the first run's token is spent by the time this happens"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "scripts/locked/ is the same harness under another path")


DEMO_INVOCATIONS = [
    pytest.param("uv run python " + DEMO_SCRIPT + ' "hello"', id="demo-one-question"),
    pytest.param(
        "uv run --with mss python " + DEMO_SCRIPT + " --screen --listen --speak",
        id="demo-screen-listen-speak",
    ),
]


@pytest.mark.parametrize("command", DEMO_INVOCATIONS)
def test_the_interactive_demo_scripts_are_ungated(command: str, sandbox: Sandbox) -> None:
    """Ruling D5, hook half: a command under ``scripts/demo/`` needs no launch token.

    It sends one request to an engine that is already running and produces no number anyone will
    quote; one ``/launch`` review per spoken question is the friction that gets a hook switched
    off. This is the over-block guard for the D5 move: the natural way to implement "but the demo
    still needs the engine" is to widen the trigger or add a name-based exemption, and both would
    fail here. The other half of D5 -- the files actually living under ``scripts/demo/``, since
    the ruling allows no name-based exemption -- is the test below.
    """
    assert not sandbox.token.exists(), "a demo question must not need an approval at all"
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "the demo directory is outside the measurement convention")


def test_the_demo_scripts_have_moved_out_of_the_gpu_convention() -> None:
    """Ruling D5, repository half: ``ask.py`` and ``_listen.py`` live under ``scripts/demo/``.

    The ruling is "no name-based exemption anywhere: the directory is the rule", so the only way
    the demo can be ungated *and* the convention stay honest is for the files to move. This is
    the one case in the file that reads the real repository rather than a sandbox, because the
    claim is about this repository's layout: ``scripts/gpu/`` means "spends GPU time to produce a
    measurement", and a demo script sitting there makes the directory mean two things.

    Ruling T-0007 A7: this is a subset check, not a closed list -- ``scripts/demo/`` is allowed to
    grow (T-0007 adds ``scripts/demo/session_rung.py``); what D5 fixes is where these two live.
    """
    demo = REPO / "scripts" / "demo"
    moved = sorted(p.name for p in demo.glob("*.py")) if demo.is_dir() else []
    assert {"_listen.py", "ask.py"} <= set(moved), (
        f"ruling D5: the interactive demo scripts belong in scripts/demo/, found {moved}"
    )
    strays = sorted(p.name for p in (REPO / "scripts" / "gpu").glob("*.py"))
    assert "ask.py" not in strays and "_listen.py" not in strays, (
        f"scripts/gpu/ is for measurement harnesses; the demo scripts are still there: {strays}"
    )


# --------------------------------------------------------------------------------------------
# Group 7 -- rulings I11 and I12 from the second review.
# --------------------------------------------------------------------------------------------

WRAPPED_HARNESS_RUNS = [
    # `timeout N cmd` is how a long harness is bounded; the wrapper takes one token of its own.
    pytest.param("timeout 600 bash {x3}", id="timeout-then-bash"),
    pytest.param("timeout 600 {x3}", id="timeout-then-the-script"),
    # `source`/`.` run the file in the *current* shell: no child process, same GPU time.
    pytest.param("source {x3}", id="source-the-harness"),
    pytest.param(". {x3}", id="dot-the-harness"),
]


@pytest.mark.parametrize("template", WRAPPED_HARNESS_RUNS)
def test_a_wrapped_or_sourced_harness_still_needs_a_launch_token(
    template: str, sandbox: Sandbox
) -> None:
    """Ruling I11: ``timeout`` joins the wrapper list and ``source``/``.`` count as running it.

    Both are ordinary spellings rather than exotic evasions -- ``timeout`` is what bounds a
    harness that might hang, and ``source`` is what someone writes when they want the script's
    variables afterwards. The rest of the indirection family (``cat | bash``, ``bash < file``,
    ``xargs``, ``find -exec``, ``make``, ``eval "$(cat ...)"``, ``docker exec``, a symlink) is a
    documented residual, not a promise: the guard is a speed bump, not a sandbox, so those are
    deliberately *not* asserted here in either direction.
    """
    command = template.format(top=sandbox.top, x3=X3)
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "the harness runs; the wrapper only changes who runs it")


SERVICE_STARTS = [
    pytest.param("uv run astra up --profile lean", id="astra-up"),
    pytest.param("systemctl --user start astra-vllm", id="systemctl-start-astra"),
]


@pytest.mark.parametrize("command", SERVICE_STARTS)
def test_starting_a_service_on_a_clean_tree_needs_no_token(command: str, sandbox: Sandbox) -> None:
    """Ruling I12 (= D2 as the human decided it): a service start needs a clean tree, not a token.

    The hazard is not that an engine was started, it is measuring later against a tree that was
    dirty when it started. A token cannot be had without a ``/launch`` reviewer pass, so gating
    ``astra up`` on one puts a review in front of every service start -- which is the friction
    D2 was decided to remove, and CLAUDE.md lists ``astra up`` among the routine commands.
    """
    assert not sandbox.token.exists(), "a service start must not need an approval at all"
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "clean tree, service start: the cheap invariant is satisfied")


@pytest.mark.parametrize("command", SERVICE_STARTS)
def test_starting_a_service_on_a_dirty_tree_is_refused_and_names_what_is_dirty(
    command: str, sandbox: Sandbox
) -> None:
    """The other direction of I12, and the only place in this file that reads the message.

    Wording is normally none of this file's business (exit status is the contract), but here the
    refusal *is* the mechanism: "tree is dirty" alone leaves the agent guessing, and the four
    times this session someone routed around the gate, they did it after a refusal that did not
    say what to fix. The dirty file is a harness rather than README.md so that the assertion
    cannot be satisfied by a boilerplate sentence that happens to mention a common filename.
    """
    dirty = sandbox.make_dirty(NEW_HARNESS)
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "an engine started against a tree an agent is editing")
    assert dirty in res.stderr, (
        f"the refusal must name what is dirty ({dirty}); the agent has to know what to commit.\n"
        f"stderr: {res.stderr.strip()}"
    )


# --------------------------------------------------------------------------------------------
# Group 8 -- T-0026 item 1: a flag whose argument is a separate token must not end the peel.
#
# `runs_gated` finds the program by consuming tokens that start with `-`, taking two tokens only
# for the flags it knows swallow their argument. Every other flag consumes one token, so the
# flag's *argument* becomes the candidate program: after `nice -n` the peel inspects `10`, after
# `bash -euo` it inspects `pipefail`. Neither is a gated path, so the verdict is "not gated" and
# the harness runs with no token and no dirty-tree check -- T-0016 defect A, reopened by the
# ordinary spellings a careful person uses to bound a long run (`nice`, `timeout`,
# `set -euo pipefail`), not by an evasion.
#
# The ten rows that were ALLOWED on main @ aad95b0 carried `pending(task="T-0026")` until the
# implementer's pass began (T-0016 ruling R1); the rest are the mutation guards that keep the fix
# from being "one global list of value-taking flags", which would make `bash -u <harness>`
# (nounset, no argument) invisible while curing `nice -n 10`.
# --------------------------------------------------------------------------------------------

WRAPPED_HARNESS_WITH_A_VALUED_FLAG = [
    # Wrappers (criterion 1). `nice -n 10` is how a long harness is deprioritised, `timeout -s
    # KILL 600` how it is bounded, `stdbuf -o 0` how its output is made to stream.
    pytest.param("nice -n 10 bash {x3}", id="nice-n-then-bash"),
    pytest.param("nice -n 10 {x3}", id="nice-n-then-the-script"),
    pytest.param("sudo -u user bash {x3}", id="sudo-u-user"),
    pytest.param("timeout -s KILL 600 bash {x3}", id="timeout-s-kill-then-duration"),
    pytest.param("env -u FOO bash {x3}", id="env-u-var"),
    pytest.param("ionice -c 2 -n 7 bash {x3}", id="ionice-c-class-n-level"),
    pytest.param("stdbuf -o 0 bash {x3}", id="stdbuf-o-0"),
    # Interpreters (criterion 2). The clustered form is the common one and the one the T-0016
    # follow-up line misses: `-euo` is not `-o`, so adding `-o` to a flag list by exact
    # membership leaves this row allowed.
    pytest.param("bash -euo pipefail {x3}", id="bash-euo-pipefail"),
    pytest.param("bash -o pipefail {x3}", id="bash-o-pipefail"),
    pytest.param("bash -o pipefail -e {x3}", id="bash-o-pipefail-then-e"),
]


@pytest.mark.parametrize("template", WRAPPED_HARNESS_WITH_A_VALUED_FLAG)
def test_a_flag_taking_a_separate_argument_does_not_end_the_peel(
    template: str, sandbox: Sandbox
) -> None:
    """T-0026 criteria 1-2 (T-0016 defect A; rules file "Where GPU-spending commands live").

    Every command here runs the x3 harness. Measured on main @ aad95b0, every one of them is
    ALLOWED: the peel loses the program at the flag's argument and the ``scripts/gpu/`` trigger
    never fires, so the harness spends GPU time with no ``/launch`` token *and* no dirty-tree
    check -- the same hole T-0016 closed for the unwrapped spelling. These are not evasions:
    ``nice``/``timeout``/``sudo -u``/``set -euo pipefail`` are how a person bounds a long run,
    which is why they are in scope where ``cat <harness> | bash`` is a documented residual.
    """
    command = template.format(top=sandbox.top, x3=X3)
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "the harness runs; only the flag moved the program along")


@pytest.mark.parametrize(
    "template", [pytest.param("timeout -k 30 600 bash {x3}", id="timeout-k-30-600")]
)
def test_timeout_k_then_a_duration_stays_blocked(template: str, sandbox: Sandbox) -> None:
    """Blocked today, and blocked for a reason that the fix could remove by accident.

    ``timeout -k 30 600 bash <harness>`` is caught on main only because ``-k`` swallows nothing:
    the peel then meets ``30`` and ``600``, and ``timeout``'s duration loop eats both, landing on
    ``bash``. The moment ``-k`` is taught to take its argument, that accident disappears and the
    row has to be carried by the new rule instead. Kept as a row of its own so the change of
    mechanism is visible: it is the one spelling in group 8 that is green before the fix.
    """
    command = template.format(top=sandbox.top, x3=X3)
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "the harness runs under a kill-after timeout")


FLAGS_THAT_TAKE_NO_ARGUMENT = [
    pytest.param("bash -u {x3}", id="bash-u-nounset"),
    pytest.param("bash -s {x3}", id="bash-s-read-from-stdin"),
    pytest.param("sudo -n bash {x3}", id="sudo-n-non-interactive"),
    # The same mistake from the other end: these three are ALLOWED on main @ 308e51d because the
    # shell path consults the *shared* ``VALUED`` list, where ``-p`` is uv's ``--python``
    # shorthand and ``-C`` is a chdir flag. To a shell they are ``privileged`` and ``noclobber``
    # and take no argument, so the list makes the harness the flag's argument (reviewer A).
    pytest.param("bash -p {x3}", id="bash-p-privileged"),
    pytest.param("bash -C {x3}", id="bash-C-noclobber"),
    pytest.param("sh -p {x3}", id="sh-p-privileged"),
    # A4: the bare form, with no interpreter between the flag and the harness. The `sudo -n bash
    # <h>` row above cannot discriminate -- if `-n` wrongly swallowed `bash`, the harness would
    # become the program and the path trigger would fire anyway -- so only this spelling goes red
    # when `sudo`'s table gains `-n`.
    pytest.param("sudo -n {x3}", id="sudo-n-bare-no-interpreter"),
    # A1: `-W` is a *python* flag with a separate argument (`-W ignore`); to perl it enables
    # warnings and to ruby it sets a warning level in the same token, and neither takes the next
    # token. A python-shaped table applied to every non-shell interpreter swallows the harness.
    pytest.param("perl -W {x3}", id="perl-W-warnings"),
    pytest.param("ruby -W {x3}", id="ruby-W-warning-level"),
    pytest.param("node -W {x3}", id="node-W-not-a-node-flag"),
]


@pytest.mark.parametrize("template", FLAGS_THAT_TAKE_NO_ARGUMENT)
def test_a_flag_that_takes_no_argument_does_not_swallow_the_program(
    template: str, sandbox: Sandbox
) -> None:
    """Adversarial against the cheapest fix for criteria 1-2 (T-0026 criterion 3).

    All three are BLOCKED on main and must stay blocked. The cheapest way to cure ``nice -n 10``
    and ``bash -euo pipefail`` is to drop the letters ``-n -u -s -k -o`` into the single
    ``VALUED`` list -- and ``VALUED`` is read by *both* the wrapper peel and the interpreter flag
    loop, where the same letter means something else:

    * ``bash -u`` is ``nounset`` and takes no argument, so the next token is the script;
    * ``bash -s`` means "read the program from stdin" and takes no argument either;
    * ``sudo -n`` is non-interactive, unlike ``nice -n 10`` where ``-n`` is a niceness level;
    * ``bash -p`` is ``privileged`` and ``bash -C`` is ``noclobber``, neither of which takes an
      argument -- but the *shared* ``VALUED`` list holds ``-p`` (uv's ``--python``) and ``-C``
      (a chdir flag), so a shell that consults it swallows the harness. Measured ALLOWED on main
      @ 308e51d: the same defect as ``-u``/``-s``, reached by keeping a list rather than adding
      one, and found by review rather than by this test -- which is why the three rows are here;
    * ``sudo -n <harness>`` with no interpreter in between (ruling A4): the ``sudo -n bash ...``
      row above stays green even when ``sudo``'s table wrongly gains ``-n``, because the harness
      then becomes the program and the path trigger fires on it anyway. Only the bare form can
      tell the two apart;
    * ``perl -W``, ``ruby -W`` and ``node -W`` (ruling A1): ``-W`` takes a separate argument for
      *python* (``-W ignore``) and not for perl (enable warnings), ruby (a level in the same
      token) or node (which has no ``-W`` at all, so the token after it is still the script), so
      a python-shaped table applied to "every interpreter that is not a shell" eats the harness.

    Under one global list each of these swallows the harness, the path trigger never sees it, and
    the fix has opened a second hole while closing the first. The question the implementer must
    answer is "which flags take a separate argument **for this program**", and the shell's answer
    is not the shared list's. Do not delete a row here to make a list pass; that is the failure
    mode they exist to catch.
    """
    command = template.format(top=sandbox.top, x3=X3)
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "this flag takes no argument, so the harness is still the program")


# Flag forms beyond the ten rows T-0026 prescribed, all found unpinned by reviewer A. Each is
# ALLOWED on main @ 308e51d and BLOCKED by the branch, so each is carried by a specific piece of
# the fix rather than by the general shape -- which is exactly why they need rows of their own.
FLAG_FORMS_BEYOND_THE_PRESCRIBED_ROWS = [
    # A long/short flag of a *non*-shell interpreter: `-W ignore` is a warning filter and the
    # program is the token after its argument. The per-program split has to cover python too,
    # not only the shells and the wrappers.
    pytest.param("python3 -W ignore " + GPU_PY_HARNESS, id="python3-W-ignore"),
    # `+o` is how a shell option is turned *off*; the flag loop must treat `+` as a flag start at
    # all, which the `startswith("-")` test on main does not.
    pytest.param("bash +o nounset {x3}", id="bash-plus-o-nounset"),
    # The long form of a shell flag with a separate argument: a cluster rule keyed on the last
    # letter of a short group says nothing about `--rcfile`.
    pytest.param("bash --rcfile x {x3}", id="bash-rcfile-then-the-harness"),
    # The optional item in T-0026's implementation notes, which the implementer took: a cluster
    # ending in `c` re-targets to the payload, so `bash -ec '<harness run>'` is the harness run.
    # The task said "if you take it, ask the test-author for one" -- this is that row.
    pytest.param("bash -ec 'bash {x3}'", id="bash-ec-cluster-retarget"),
]


@pytest.mark.parametrize("template", FLAG_FORMS_BEYOND_THE_PRESCRIBED_ROWS)
def test_a_flag_form_beyond_the_prescribed_rows_still_does_not_end_the_peel(
    template: str, sandbox: Sandbox
) -> None:
    """Criteria 1-2 again, for the spellings the task's row list did not enumerate.

    The ten prescribed rows are examples of a rule, not the rule itself; a fix that satisfies
    exactly those ten and nothing else is a spelling table again -- the thing T-0016 removed. Each
    row here runs the harness through a form the prescribed ten do not reach: a non-shell
    interpreter's valued flag, the ``+`` spelling of a shell option, a long shell flag, and a
    cluster ending in ``c`` that re-targets to its payload. All four were ALLOWED on main and are
    the reviewer's additions; pinned so a later simplification of the flag tables has to notice
    them.
    """
    command = template.format(top=sandbox.top, x3=X3)
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "the harness runs; the flag only moved the program along")


UNKNOWN_WRAPPER_FLAGS = [
    # `nice -5` is the numeric shorthand for the adjustment, not a flag with an argument.
    pytest.param("nice -5 {x3}", id="nice-numeric-shorthand"),
    pytest.param("nice --verbose {x3}", id="nice-long-verbose"),
    pytest.param("env --ignore-environment {x3}", id="env-ignore-environment"),
    pytest.param("timeout --foreground {x3}", id="timeout-foreground"),
]


@pytest.mark.parametrize("template", UNKNOWN_WRAPPER_FLAGS)
def test_an_unknown_wrapper_flag_consumes_one_token(template: str, sandbox: Sandbox) -> None:
    """Ruling A5: the unknown-flag default is "consumes one token", and it fails closed.

    A per-program flag table is an enumeration, so it will always be incomplete; what makes the
    incompleteness safe is the default. A flag the table does not know must leave the program
    where it is, so an unlisted *no-argument* flag keeps the harness visible -- these four rows.
    The opposite default ("consume two tokens unless the flag is known to take none") is the
    tempting simplification, and it makes every one of these ALLOWED: the harness becomes the
    flag's argument. They are green on main and on the branch; what they pin is the direction of
    the default, which no other row states.

    The cost of this default is recorded rather than hidden: an *unknown* flag that really does
    take a separate argument (``nice -q 10 <harness>``, ``sudo -T 600 <harness>``) still ends the
    peel and is ALLOWED. That fails open, so it is a limit of the table, not of the default --
    extend the table when such a spelling is met.
    """
    command = template.format(top=sandbox.top, x3=X3)
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "the flag takes no argument, so the harness is still the program")


A_WRAPPER_THAT_BUILDS_ITS_COMMAND_LINE = [
    # `-I {}` names the replacement string; the token after it is the replacement marker, not
    # the program. Without `xargs` in the wrapper table the peel stops at `{}` and the harness
    # is invisible.
    pytest.param("xargs -I {{}} bash {x3}", id="xargs-I-replace-string"),
    # `-n 1` is a batch size, the same shape as `nice -n 10`: the peel stops at `1`.
    pytest.param("xargs -n 1 bash {x3}", id="xargs-n-batch-size"),
]


@pytest.mark.parametrize("template", A_WRAPPER_THAT_BUILDS_ITS_COMMAND_LINE)
def test_a_command_building_wrapper_does_not_hide_the_harness_behind_its_flags(
    template: str, sandbox: Sandbox
) -> None:
    """Ruling A6: the ``xargs`` spellings whose flags take a separate argument are BLOCKED.

    ``xargs`` is named in T-0026's *Out of scope* list as a documented indirection residual, and
    that stays true of the forms where the program never appears on the command line at all
    (``... | xargs bash`` reading the harness path from stdin): the guard is a speed bump, not a
    sandbox. These two rows are the other half -- the harness *is* written on the line, in a
    program position, and only ``-I {}``/``-n 1`` stand between the peel and it. That is the
    same defect as ``nice -n 10 bash <harness>``, so leaving it open would mean the fix cured one
    spelling of "a flag ate the program" and not its neighbour; ruling A6 therefore tightens the
    residual this far and no further, and the tightening is noted for the rules file when T-0014
    and T-0015 release it.

    Both rows are carried specifically by ``xargs``'s entry in the wrapper flag table: delete it
    and the peel stops at ``{}`` and at ``1`` respectively, and both go ALLOWED.
    """
    command = template.format(top=sandbox.top, x3=X3)
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "the harness is on the line; only the flag moved the program on")


def test_parse_only_wins_over_the_cluster_retarget(sandbox: Sandbox) -> None:
    """Ruling A2, the precedence the ``-ec`` row above could otherwise erase.

    ``bash -nc '<payload>'`` parses the payload and runs nothing -- ``-n`` is parse-only and it
    applies to the whole invocation, including a ``-c`` string. The re-target added for
    ``bash -ec '<harness run>'`` reads a cluster's last letter, and ``-nc`` ends in ``c`` too, so
    the two rules meet here and the order decides: the ``-n`` early return must run first, exactly
    as ruling I9 and the hook's own header require. Get the order wrong and syntax-checking a
    snippet that mentions a harness demands a launch token, which is the over-block that teaches
    people to route around the guard.
    """
    command = f"bash -nc 'bash {X3}'"
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "-n parses the payload; nothing runs")


READ_ONLY_CHECKS_UNDER_A_WRAPPER = [
    pytest.param("nice -n 10 bash -n {x3}", id="nice-then-bash-syntax-check"),
    pytest.param("nice -n 10 python3 -m py_compile " + GPU_PY_HARNESS, id="nice-then-py-compile"),
]


@pytest.mark.parametrize("template", READ_ONLY_CHECKS_UNDER_A_WRAPPER)
def test_a_read_only_check_under_a_wrapper_is_still_free(template: str, sandbox: Sandbox) -> None:
    """T-0026 criterion 4: ruling I9 survives the wider peel.

    These are allowed today for the wrong reason -- the peel stops at ``10`` and never reaches
    the harness at all. After the fix the peel does reach it, and what must keep them free is the
    shell ``-n`` early return and the ``-m`` re-target, which run *before* any new flag handling.
    Moving the ``-n`` return after a cluster rule turns ``bash -n``/``bash -en`` into a token
    demand and teaches the person editing a harness to route around the hook (T-0016 I9, R4).
    """
    command = template.format(top=sandbox.top, x3=X3)
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "a syntax check under a wrapper reads the harness, it runs none")


def test_a_wrapper_does_not_gate_the_demo_directory(sandbox: Sandbox) -> None:
    """T-0026 criterion 5: the fix widens *where the program is found*, not *what is gated*.

    ``scripts/demo/`` is ungated by ruling D5, and finding the program through ``nice -n 10``
    must not change that verdict -- the peel's job is to identify the program, the directory
    convention alone decides whether that program needs a token.
    """
    command = "nice -n 10 uv run python " + DEMO_SCRIPT + " 'hi'"
    assert not sandbox.token.exists(), "a demo question must not need an approval at all"
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "the demo directory is outside the measurement convention")


# --------------------------------------------------------------------------------------------
# Group 9 -- T-0026 item 2: the over-block twin of ruling I10.
# --------------------------------------------------------------------------------------------

HARNESS_PATH_IN_PROSE = [
    pytest.param(
        'git commit -q -m "T-0026: bash {x3} ran with no token; that is defect A"',
        id="commit-message-names-the-harness",
    ),
    pytest.param("git log --oneline --grep 'bash {x3}'", id="git-log-grep-for-the-harness"),
    pytest.param("rg -n '{x3}' docs/", id="ripgrep-the-harness-path-in-docs"),
    pytest.param('echo "see {x3} for the regime" >> notes.md', id="echo-the-path-into-a-note"),
]


@pytest.mark.parametrize("template", HARNESS_PATH_IN_PROSE)
def test_naming_a_harness_path_in_prose_is_not_running_it(template: str, sandbox: Sandbox) -> None:
    """T-0026 criterion 6: the over-block twin of I10, unpinned when T-0016 merged.

    I10 says a *quoted gated path in a program position* is still gated, so ``bash "{x3}"`` keeps
    its quotes for the path verdict. The natural over-shoot of that rule is to treat any quoted
    mention of the path as a program, which would re-block commit messages, ``git log --grep``
    and ``rg`` -- defect B, and the habit CLAUDE.md's "Observed mistakes" warns about, because an
    agent that has to reword a commit message to appease the guard learns to route around it.
    These pass on main; the criterion is that they still pass after the peel is widened, since
    widening *where the program is found* is exactly the change that could start reading the
    third word of a commit message as a program. The commit-message *heredoc* form is already
    pinned by ``test_a_heredoc_fed_to_a_non_interpreter_is_inert_text``.

    Deliberately not asserted in either direction: a ``cd ... && ...`` written inside a quoted
    message, which ``resolve_cd`` can read as a real ``cd`` (measured BLOCKED on main). It fails
    closed and is a documented residual, treated the way group 7 treats the indirection family.
    """
    command = template.format(top=sandbox.top, x3=X3)
    assert not sandbox.token.exists(), "writing about a harness must not need an approval"
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "the path is prose here; nothing executes a commit message")


# --------------------------------------------------------------------------------------------
# Group 10 -- T-0026 item 3: the two refusals, and what each is allowed to demand.
#
# This file otherwise asserts exit status only. These two tests read stderr for the same reason
# group 7's dirty-tree test does: for the *Engine starts* and *Routine service starts* classes
# the message is the mechanism -- it is what tells the agent which gate it hit and what to do
# next, and the two classes differ only in the message. A refusal that invents an obligation the
# rules do not impose is how a gate starts to look arbitrary; a refusal that collapses the two
# classes into one sentence erases the D2/I12 split the human ruled on.
# --------------------------------------------------------------------------------------------


def test_the_engine_start_refusal_names_only_the_obligations_the_rules_impose(
    sandbox: Sandbox,
) -> None:
    """T-0026 criterion 7. ``.claude/rules/gpu-services.md`` attributes a ``PROGRESS.md`` *Runs*
    entry to the *Measurement* class; *Engine starts* need "the same token and clean tree".

    The message on main demands all three, so an agent starting the engine is told to write a
    *Runs* entry that nothing in the rules asks of it. The ruling is that the **message** changes,
    not the spec: the rules file is owned by two open tasks (T-0014, T-0015), and the refusal must
    not be the place a new obligation is invented.
    """
    command = VLLM_SERVE
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "an engine start takes VRAM and needs a token")
    assert "/launch" in res.stderr, (
        f"the refusal must name the review that unblocks it.\nstderr: {res.stderr.strip()}"
    )
    assert "clean tree" in res.stderr.lower(), (
        f"the refusal must name the clean-tree invariant.\nstderr: {res.stderr.strip()}"
    )
    assert "runs entry" not in res.stderr.lower(), (
        "the rules attribute the PROGRESS.md Runs entry to the Measurement class, not to engine "
        f"starts; the refusal must not invent it.\nstderr: {res.stderr.strip()}"
    )


@pytest.mark.parametrize("command", SERVICE_STARTS)
def test_a_routine_start_refusal_does_not_ask_for_a_launch_review(
    command: str, sandbox: Sandbox
) -> None:
    """T-0026 criterion 8, and the guard on criterion 7: the two refusals stay distinguishable.

    Ruling I12 (the human's D2): a routine service start needs *only* a clean tree -- no token, no
    reviewer pass, no *Runs* entry. The message is the mechanism here too: the only thing that
    writes a token is ``/launch``, so a refusal that mentions ``/launch`` or a token sends the
    agent to a review it does not need, which is the friction D2 was decided to remove. Paired
    with the engine-start test above so that what is pinned is the *distinction*: criterion 7 must
    not be satisfied by collapsing both classes into one sentence.
    """
    dirty = sandbox.make_dirty(NEW_HARNESS)
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "a service started against a tree an agent is editing")
    assert dirty in res.stderr, (
        f"the refusal must name what is dirty ({dirty}).\nstderr: {res.stderr.strip()}"
    )
    assert "/launch" not in res.stderr, (
        "a routine start needs no reviewer pass (I12); naming /launch sends the agent to a review "
        f"that will not happen.\nstderr: {res.stderr.strip()}"
    )
    assert "token" not in res.stderr.lower(), (
        f"a routine start needs no token (I12).\nstderr: {res.stderr.strip()}"
    )


# --------------------------------------------------------------------------------------------
# Group 11 -- T-0029: ownership is enforced on the common Bash write shapes too.
#
# ``protect_paths.sh`` refuses an implementer's Edit to a file its task does not own; until
# T-0029 the same implementer could write that file through the shell with no check whatsoever.
# The rule fires for ``ROLE=implementer`` only, and it reads the *same* owned-files parser the
# Edit hook does (criterion 1, pinned from the other side in
# ``tests/unit/test_protect_paths_hook.py::test_both_hooks_agree_on_the_same_task_file``).
# --------------------------------------------------------------------------------------------

OWNED_PATH = "local_astra/only_this.py"
UNOWNED_PATH = "local_astra/x.py"

OWNERSHIP_WRITE_SHAPES = [
    pytest.param("cat > {p} <<'PY'\nx = 1\nPY", id="heredoc-redirect"),
    pytest.param("cat >> {p} <<'PY'\nx = 2\nPY", id="append-redirect"),
    pytest.param("printf 'x = 1\\n' | tee {p}", id="tee"),
    pytest.param("printf 'x = 1\\n' | tee -a {p}", id="tee-append"),
    pytest.param("sed -i s/a/b/ {p}", id="sed-i"),
    pytest.param("cp /tmp/a {p}", id="cp"),
    pytest.param("mv /tmp/a {p}", id="mv"),
    pytest.param("rm {p}", id="rm"),
    pytest.param(
        "cd local_astra && cat > x.py <<'PY'\nx = 1\nPY",
        id="redirect-after-cd",
    ),
    pytest.param("cat > \"{p}\" <<'PY'\nx = 1\nPY", id="quoted-target-double"),
    pytest.param("cat > '{p}' <<'PY'\nx = 1\nPY", id="quoted-target-single"),
    pytest.param("cat > \"$TOP/{p}\" <<'PY'\nx = 1\nPY", id="quoted-target-variable"),
    # The blocked twin of ``cp-into-an-owned-directory`` (ruling A8): a trailing-slash
    # destination resolves to one target per source, so copying into a directory the task does
    # not own is still a write the task does not own. Without this row, the A8 fix could be
    # "stop naming directory destinations at all", which would silence the whole shape.
    pytest.param("cp /tmp/a /tmp/b local_astra/conductor/", id="cp-into-an-unowned-directory"),
]


@pytest.mark.parametrize("template", OWNERSHIP_WRITE_SHAPES)
def test_writing_a_file_the_task_does_not_own_is_blocked(template: str, sandbox: Sandbox) -> None:
    """T-0029 criterion 9: the shell is not a way around file ownership.

    The task on this branch owns ``local_astra/only_this.py`` and explicitly lists
    ``local_astra/x.py`` under "does not own", so every shape here writes a file another task
    owns -- which is the concurrency invariant CLAUDE.md states as "two agents never edit the
    same file concurrently", and which ``protect_paths.sh`` already enforces for Edit/Write.
    Measured on ``main`` @ 7989c81: the heredoc, ``sed -i``, ``tee`` and ``cp`` rows were all
    ALLOWED; only ``tests/*.py`` writes were shape-blocked, by a rule that is about tests rather
    than about ownership.

    ``redirect-after-cd`` is T-0016 I1 applied to writes (the ``cd`` is resolved before the
    target is read) and the three ``quoted-target-*`` rows are I10: a quoted path is still a
    path, so the ownership scan must read the quote-preserving view and strip quotes per token.
    An implementation that scans the blanked ``scan`` view passes the unquoted rows and silently
    loses all three quoted ones -- the failure mode that has now bitten this hook twice.
    """
    command = template.format(p=UNOWNED_PATH)
    sandbox.on_task("T-0099", [OWNED_PATH])
    res = run_hook(command, sandbox, role="implementer")
    assert_blocked(res, command, "the task does not own the write target")


def test_the_ownership_refusal_names_the_path_and_the_task(sandbox: Sandbox) -> None:
    """T-0029 criterion 9, last sentence: "the refusal names the path and the task, like
    ``protect_paths.sh``'s".

    One of the three places this file asserts on wording (the others are group 10), for the same
    reason: an agent that is told only "blocked" cannot tell an ownership refusal from a GPU one,
    and the next thing it does is re-run the command a different way. The Edit hook's message
    names both; the Bash one must too, or the two gates teach different lessons for the same
    mistake.
    """
    command = f"cat > {UNOWNED_PATH} <<'PY'\nx = 1\nPY"
    sandbox.on_task("T-0099", [OWNED_PATH])
    res = run_hook(command, sandbox, role="implementer")
    assert_blocked(res, command, "the task does not own the write target")
    assert UNOWNED_PATH in res.stderr, (
        f"the refusal must name the path it refused.\nstderr: {res.stderr.strip()}"
    )
    assert "T-0099" in res.stderr, (
        f"the refusal must name the task whose owned list was consulted.\n"
        f"stderr: {res.stderr.strip()}"
    )


COMMIT_MESSAGE_MENTIONING_A_WRITE = (
    "git commit -q -m 'T-0029: wrote > local_astra/x.py by hand; the guard did not notice'"
)

COMMIT_HEREDOC_MENTIONING_A_WRITE = """git commit -q -F - <<'MSG'
Guard writes by ownership

The measured hole was that `cat > local_astra/x.py <<PY ... PY`, `sed -i s/a/b/
local_astra/x.py` and `cp /tmp/a local_astra/x.py` were all allowed for an implementer
whose task owned neither path.
MSG"""

# Rows are (command template, role, the task's owned list). The owned list is per row because the
# directory rows need a glob owner; everything else owns the single file ``OWNED_PATH``.
OWNS_THE_FILE = [OWNED_PATH]
OWNS_THE_CONDUCTOR = ["local_astra/conductor/**"]

OWNERSHIP_ALLOWED = [
    pytest.param(
        "cat > {owned} <<'PY'\nx = 1\nPY",
        "implementer",
        OWNS_THE_FILE,
        id="owned-target-heredoc",
    ),
    pytest.param(
        "printf 'x = 1\\n' | tee {owned}",
        "implementer",
        OWNS_THE_FILE,
        id="owned-target-tee",
    ),
    pytest.param(
        "cat > /tmp/scratch_notes.py <<'PY'\nx = 1\nPY",
        "implementer",
        OWNS_THE_FILE,
        id="outside-the-repo",
    ),
    pytest.param(
        "cat > {p} <<'PY'\nx = 1\nPY", "none", OWNS_THE_FILE, id="role-none-writes-anything"
    ),
    pytest.param("cat {p}", "implementer", OWNS_THE_FILE, id="read-only-cat"),
    pytest.param(
        "grep -n '>' {p}", "implementer", OWNS_THE_FILE, id="grep-for-a-redirect-character"
    ),
    pytest.param(
        COMMIT_MESSAGE_MENTIONING_A_WRITE,
        "implementer",
        OWNS_THE_FILE,
        id="commit-message-mentioning-a-redirect",
    ),
    pytest.param(
        COMMIT_HEREDOC_MENTIONING_A_WRITE,
        "implementer",
        OWNS_THE_FILE,
        id="heredoc-body-mentioning-a-redirect",
    ),
    pytest.param("mkdir -p local_astra/perception", "implementer", OWNS_THE_FILE, id="mkdir"),
    # Ruling A8 (review REVISE): three over-blocks the first implementation had, plus the
    # redirection an agent types a hundred times a day. Each is a *file descriptor* or a flag
    # argument that a naive "the token after the operator is a file" scan reads as a write target
    # the task does not own -- and each therefore refused a command that writes nothing it does
    # not own. They are the nuisance half of criterion 10: an ownership gate that blocks
    # ``rm -f x 2>/dev/null`` is one an agent learns to route around.
    pytest.param(
        "rm -f {owned} 2>/dev/null", "implementer", OWNS_THE_FILE, id="fd-redirect-after-rm"
    ),
    pytest.param("sed -i -e s/a/b/ {owned}", "implementer", OWNS_THE_FILE, id="sed-i-with-e"),
    pytest.param(
        "cp /tmp/a /tmp/b local_astra/conductor/",
        "implementer",
        OWNS_THE_CONDUCTOR,
        id="cp-into-an-owned-directory",
    ),
    pytest.param(
        "uv run pytest tests -q > /dev/null 2>&1",
        "implementer",
        OWNS_THE_FILE,
        id="devnull-redirect",
    ),
    # Ruling A9: a target still carrying a variable other than $TOP/$PWD is unknowable at hook
    # time -- the hook does not run the shell, so `$OUT` could be anywhere. It is skipped, which
    # is fail-open for ownership exactly as the rule's header declares (fail-closed is for GPU).
    # The alternative, guessing that it resolves under the repo, denied a write to /tmp.
    pytest.param(
        "OUT=/tmp/z; echo hi > $OUT/f.txt",
        "implementer",
        OWNS_THE_FILE,
        id="variable-target-outside-top",
    ),
]


@pytest.mark.parametrize(("template", "role", "owned"), OWNERSHIP_ALLOWED)
def test_ownership_does_not_block_the_ordinary_cases(
    template: str, role: str, owned: list[str], sandbox: Sandbox
) -> None:
    """T-0029 criterion 10: the ownership rule must not become a nuisance.

    **Adversarial rows**: ``commit-message-mentioning-a-redirect`` and
    ``heredoc-body-mentioning-a-redirect``. The cheapest possible implementation greps the raw
    command text for ``>`` and blocks half of git -- defect B all over again, and the habit
    CLAUDE.md's "Observed mistakes" warns about, because an agent that has to reword a commit
    message to appease the guard learns to route around the guard. The same distinction groups 4
    and 9 already pin for the GPU trigger: a heredoc body fed to a non-interpreter is inert text,
    and quoted prose is prose.

    ``role-none-writes-anything`` matters for a second reason: this task's own work is done from
    the main session, where ``ROLE`` is ``none``. A rule that fired there would deny the
    implementer of T-0029 the ability to edit the hook it is fixing.
    """
    command = template.format(p=UNOWNED_PATH, owned=OWNED_PATH)
    sandbox.on_task("T-0099", owned)
    res = run_hook(command, sandbox, role=role)
    assert_allowed(res, command, "ownership has nothing to say about this command")


def test_an_implementer_off_a_task_branch_cannot_write_through_bash(sandbox: Sandbox) -> None:
    """T-0029 criterion 10's last sentence: consistency with ``protect_paths.sh``.

    With no ``T-NNNN`` branch there is no owned list to consult, so an implementer has no
    ownership at all; the Edit hook already says so and names the command that fixes it
    (``git switch -c T-NNNN-<slug>``). The Bash guard must give the same message rather than a
    second, differently-worded refusal for the same situation -- and must not fall through to
    "allow", which is how the hole under repair was opened in the first place.
    """
    command = f"cat > {UNOWNED_PATH} <<'PY'\nx = 1\nPY"
    sandbox.on_branch("scratch-no-task-id")
    res = run_hook(command, sandbox, role="implementer")
    assert_blocked(res, command, "an implementer off a task branch owns nothing")
    assert "git switch -c" in res.stderr, (
        "the refusal must name the same command protect_paths.sh names.\n"
        f"stderr: {res.stderr.strip()}"
    )


# --------------------------------------------------------------------------------------------
# Group 12 -- T-0145: three deny rules, each pinned by rows only that rule can decide.
#
# Every deny row asserts the substring of its rule's message (one constant per rule, so a
# rewording in the hook is a one-line edit here), and every allow row is the too-wide twin of a
# deny row: a rule widened past its ruling turns the twin red.
# --------------------------------------------------------------------------------------------

TESTS_RULE_REASON = "do not delete, move or overwrite tests"
PUSH_RULE_REASON = "does not push"
DESTRUCTIVE_GIT_REASON = "destructive or history-rewriting git"
# The ownership rule's refusal; it must never be what denies an implementer a tests/ write.
OWNERSHIP_REASON = "not in the owned files"

TEST_FILE = "tests/unit/test_x.py"

TEST_OVERWRITE_SHAPES = [
    pytest.param("rm {t}", id="rm"),
    pytest.param("mv /tmp/a {t}", id="mv"),
    pytest.param("sed -i s/a/b/ {t}", id="sed-i"),
    pytest.param("echo x > {t}", id="redirect"),
]


@pytest.mark.parametrize("role", ["none", "reviewer", "verifier", "implementer-owning-the-test"])
@pytest.mark.parametrize("template", TEST_OVERWRITE_SHAPES)
def test_overwriting_a_test_is_refused_by_the_tests_rule(
    template: str, role: str, sandbox: Sandbox
) -> None:
    """T-0145 criterion 6 (docs/09 §3, "deleting or overwriting tests").

    Every role but the test-author is refused ``rm``/``mv``/``sed -i``/``>`` on a test file, by the
    tests rule. **Adversarial row**: ``implementer-owning-the-test``. Its task owns
    ``tests/unit/test_x.py``, so the ownership rule would *allow* the write; only the tests rule can
    refuse it. Without the owned premise, removing the tests rule would leave this row green, the
    ownership rule denying in its place -- the shape of the M1 escape in ``protect_paths.sh``.
    """
    command = template.format(t=TEST_FILE)
    if role == "implementer-owning-the-test":
        role = "implementer"
        sandbox.on_task("T-0099", [TEST_FILE])
    res = run_hook(command, sandbox, role=role)
    assert_blocked(res, command, f"{role} may not delete, move or overwrite a test")
    assert TESTS_RULE_REASON in res.stderr, (
        f"refused, but not by the tests rule.\nstderr: {res.stderr.strip()}"
    )
    assert OWNERSHIP_REASON not in res.stderr, (
        f"the ownership rule, not the tests rule, refused this.\nstderr: {res.stderr.strip()}"
    )


@pytest.mark.parametrize(
    ("template", "role"),
    [
        *[
            pytest.param(p.values[0], "test-author", id=f"test-author-{p.id}")
            for p in TEST_OVERWRITE_SHAPES
        ],
        pytest.param("cat {t}", "implementer", id="implementer-reads-a-test"),
    ],
)
def test_the_test_author_may_rewrite_tests(template: str, role: str, sandbox: Sandbox) -> None:
    """T-0145 criterion 6, the too-wide twins.

    The test-author's job is rewriting tests, so the same four shapes are allowed for it; and
    reading a test is not overwriting it, so ``cat`` is allowed for the implementer (whose task
    owns the file, so ownership has nothing to refuse either). A tests rule applied to every role,
    or keyed on any mention of ``tests/*.py``, turns one of these red.
    """
    command = template.format(t=TEST_FILE)
    if role == "implementer":
        sandbox.on_task("T-0099", [TEST_FILE])
    res = run_hook(command, sandbox, role=role)
    assert_allowed(res, command, f"{role} may do this to a test file")


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("git push origin T-0099-probe", id="git-push"),
        pytest.param("gh pr create --fill", id="gh-pr-create"),
        pytest.param("gh pr merge 3", id="gh-pr-merge"),
    ],
)
def test_an_implementer_may_not_push_or_open_a_pr(command: str, sandbox: Sandbox) -> None:
    """T-0145 criterion 7: the implementer does not push or open PRs; the verifier does, after
    review (docs/09). Asserted by the rule's own message, so no other refusal can stand in for it.
    """
    sandbox.on_task("T-0099", [OWNED_PATH])
    res = run_hook(command, sandbox, role="implementer")
    assert_blocked(res, command, "the implementer does not push or open PRs")
    assert PUSH_RULE_REASON in res.stderr, (
        f"refused, but not by the push/PR rule.\nstderr: {res.stderr.strip()}"
    )


@pytest.mark.parametrize(
    ("command", "role"),
    [
        pytest.param("git push origin T-0099-probe", "verifier", id="verifier-pushes"),
        pytest.param("git push origin T-0099-probe", "none", id="main-session-pushes"),
        pytest.param("gh pr view 3", "implementer", id="implementer-views-a-pr"),
    ],
)
def test_push_and_pr_reads_stay_open_to_the_others(
    command: str, role: str, sandbox: Sandbox
) -> None:
    """T-0145 criterion 7, the too-wide twins: the rule is about the implementer, and about
    ``create``/``merge``. A rule applied to every role refuses the verifier's push; one widened to
    every ``gh pr`` refuses the implementer reading a PR.
    """
    sandbox.on_task("T-0099", [OWNED_PATH])
    res = run_hook(command, sandbox, role=role)
    assert_allowed(res, command, f"{role} may do this")


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("git clean -fd", id="clean-fd"),
        pytest.param("git branch -D x", id="branch-D"),
        pytest.param("git rebase -i HEAD~2", id="rebase-i"),
        pytest.param("git commit -q --no-verify -m x", id="commit-no-verify"),
    ],
)
def test_the_remaining_destructive_git_forms_are_refused(command: str, sandbox: Sandbox) -> None:
    """T-0145 criterion 8: the four destructive-git forms group 3 left unpinned (it pins only
    ``reset --hard`` and ``push --force``). Each is the only row for its alternative of the regex,
    so dropping one alternative turns exactly its row red. Role ``none``: no other rule applies.
    """
    res = run_hook(command, sandbox, role="none")
    assert_blocked(res, command, "destructive or history-rewriting git")
    assert DESTRUCTIVE_GIT_REASON in res.stderr, (
        f"refused, but not by the destructive-git rule.\nstderr: {res.stderr.strip()}"
    )


@pytest.mark.parametrize(
    "command",
    [
        # A branch name containing ``-f`` is not a force flag (the hook's own comment).
        pytest.param("git push origin T-0145-fix-foo", id="push-branch-name-with-dash-f"),
        pytest.param("git branch -d x", id="branch-d-lowercase"),
        pytest.param("git commit -q -m x", id="commit-without-no-verify"),
    ],
)
def test_lookalikes_of_destructive_git_are_allowed(command: str, sandbox: Sandbox) -> None:
    """T-0145 criterion 8, the too-wide twins: a safe delete of a merged branch, an ordinary
    commit, and a push of a branch whose name contains ``-f``. A regex that matched ``-d`` as well
    as ``-D``, any ``commit``, or ``-f`` anywhere after ``push`` would refuse them.
    """
    res = run_hook(command, sandbox, role="none")
    assert_allowed(res, command, "not destructive")


# --------------------------------------------------------------------------------------------
# Group 14 -- T-0156 R4: the GPU gate covers the new console command ``diffusion_gearvis``.
#
# ``diffusion_gearvis`` and ``astra`` are the same function (R1), so ``diffusion_gearvis bench``
# spends exactly the GPU time ``astra bench`` does and ``diffusion_gearvis up`` starts exactly the
# services ``astra up`` does. R4: the hook treats them "exactly as" the old spellings. The oracle
# for "exactly" is the hook itself on the old spelling in the same sandbox: same verdict, same
# refusal text. The rows that failed before the hook change carried ``pending(task="T-0156")``
# until activation; the too-wide twins (quoted mentions, the read-only subcommands, a clean-tree
# ``up``) passed before it and must keep passing, so they never carried the marker.
# --------------------------------------------------------------------------------------------

NEW_BENCH_AND_OLD_TWIN = [
    pytest.param("diffusion_gearvis bench B1", "astra bench B1", id="bare"),
    pytest.param("uv run diffusion_gearvis bench B1", "uv run astra bench B1", id="uv-run"),
]


@pytest.mark.parametrize(("command", "twin"), NEW_BENCH_AND_OLD_TWIN)
def test_t0156_new_name_bench_without_a_token_is_refused_like_astra_bench(
    command: str, twin: str, sandbox: Sandbox
) -> None:
    """R4: ``diffusion_gearvis bench B1`` with no launch token is BLOCKED by the GPU gate.

    The reason is asserted (T-0145 rule): it must be the measurement refusal ``astra bench``
    gets, word for word, and not another rule's. The twin is run first so the oracle is taken
    from the same sandbox; it must itself name ``/launch`` or the comparison would be empty.
    """
    assert not sandbox.token.exists(), "sandbox must start without a launch approval"
    oracle = run_hook(twin, sandbox)
    assert_blocked(oracle, twin, "the old spelling is the measurement class (oracle)")
    assert "/launch" in oracle.stderr, f"oracle refusal lacks /launch: {oracle.stderr.strip()}"
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "the new command name spends the same GPU time, no token")
    assert res.stderr == oracle.stderr, (
        "R4: the new name must be refused exactly as the old one.\n"
        f"new: {res.stderr.strip()}\nold: {oracle.stderr.strip()}"
    )
    assert "differs from the one the launch-reviewer approved" not in res.stderr
    assert "tree is dirty" not in res.stderr


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("diffusion_gearvis bench B1", id="bare"),
        pytest.param("uv run diffusion_gearvis bench B1", id="uv-run"),
    ],
)
def test_t0156_new_name_bench_with_its_token_on_a_clean_tree_runs_and_spends_the_token(
    command: str, sandbox: Sandbox
) -> None:
    """R4: with a matching ``/launch`` token on a clean tree the run is allowed.

    "Allowed" alone would pass today, when the hook does not see the command at all; what shows
    the gate was passed rather than skipped is that the single-use token was consumed.
    """
    sandbox.approve(command)
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "approved by /launch, tree clean")
    assert not sandbox.token.exists(), (
        "allowed, but the launch token was not consumed: the hook never gated the command"
    )


@pytest.mark.parametrize(
    ("command", "twin"),
    [
        pytest.param("diffusion_gearvis up --profile lean", "astra up --profile lean", id="bare"),
        pytest.param(
            "uv run diffusion_gearvis up --profile lean",
            "uv run astra up --profile lean",
            id="uv-run",
        ),
    ],
)
def test_t0156_new_name_up_on_a_dirty_tree_is_refused_like_astra_up(
    command: str, twin: str, sandbox: Sandbox
) -> None:
    """R4: ``diffusion_gearvis up`` is a routine service start: refused on a dirty tree, naming
    the dirty path, without sending the agent to ``/launch`` (T-0016 I12), word for word as
    ``astra up`` is refused in the same sandbox."""
    dirty = sandbox.make_dirty(NEW_HARNESS)
    oracle = run_hook(twin, sandbox)
    assert_blocked(oracle, twin, "the old spelling on a dirty tree (oracle)")
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "a service started against a tree an agent is editing")
    assert dirty in res.stderr, f"the refusal must name {dirty}.\nstderr: {res.stderr.strip()}"
    assert "/launch" not in res.stderr, f"a routine start needs no review: {res.stderr.strip()}"
    assert "token" not in res.stderr.lower(), f"a routine start needs no token: {res.stderr}"
    assert res.stderr == oracle.stderr, (
        f"R4: refused differently from astra up.\nnew: {res.stderr}\nold: {oracle.stderr}"
    )


@pytest.mark.parametrize(
    ("approved", "command"),
    [
        pytest.param("uv run astra bench B1", "uv run diffusion_gearvis bench B1", id="uv-run"),
        pytest.param("astra bench B1", "diffusion_gearvis bench B1", id="bare"),
    ],
)
def test_t0156_an_astra_bench_token_does_not_authorise_diffusion_gearvis_bench(
    approved: str, command: str, sandbox: Sandbox
) -> None:
    """R4 negative: the token matches the exact command string, so an approval of the old
    spelling is not an approval of the new one (and is not spent by it)."""
    sandbox.approve(approved)
    res = run_hook(command, sandbox)
    assert_blocked(res, command, "the approval was for the astra spelling")
    assert "differs from the one the launch-reviewer approved" in res.stderr, (
        f"refused, but not by the token-match rule.\nstderr: {res.stderr.strip()}"
    )
    assert sandbox.token.exists(), "a refused command must not consume another command's token"


def test_t0156_a_diffusion_gearvis_bench_token_does_not_authorise_astra_bench(
    sandbox: Sandbox,
) -> None:
    """R4, the reverse direction (passes today and must keep passing): a hook that normalised
    both names to one before matching the token would let either approval run the other."""
    sandbox.approve("uv run diffusion_gearvis bench B1")
    res = run_hook("uv run astra bench B1", sandbox)
    assert_blocked(res, "uv run astra bench B1", "approval was for the new spelling")
    assert "differs from the one the launch-reviewer approved" in res.stderr, res.stderr


T0156_TEXT_ONLY_MENTIONS = [
    pytest.param(
        'git commit -q -m "T-0156: diffusion_gearvis bench B1 is gated like astra bench"',
        id="commit-m-quotes-bench",
    ),
    pytest.param(
        "git commit -q -F - <<'MSG'\nT-0156: the hook\n\n"
        "diffusion_gearvis bench B1 and diffusion_gearvis up --profile lean are gated.\nMSG",
        id="commit-heredoc-quotes-bench-and-up",
    ),
    pytest.param("grep -rn 'diffusion_gearvis bench' docs/ .claude/", id="grep-pattern"),
]


@pytest.mark.parametrize("command", T0156_TEXT_ONLY_MENTIONS)
def test_t0156_quoting_diffusion_gearvis_bench_is_not_running_it(
    command: str, sandbox: Sandbox
) -> None:
    """R4 adversarial (too-wide direction): writing about the new command is not running it,
    under the same quoting rules as today. Dirty tree and no token, so neither the measurement
    gate nor the service-start gate can be what lets it through."""
    sandbox.make_dirty()
    assert not sandbox.token.exists()
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "the command name is inert quoted text")


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("diffusion_gearvis doctor", id="doctor"),
        pytest.param("uv run diffusion_gearvis status", id="status"),
        pytest.param("uv run diffusion_gearvis down", id="down"),
        pytest.param("diffusion_gearvis --help", id="help"),
    ],
)
def test_t0156_the_new_names_read_only_subcommands_are_not_gated(
    command: str, sandbox: Sandbox
) -> None:
    """R4 too-wide twin: only ``bench`` and ``up`` are gated, as for ``astra``. A hook that keyed
    on the program name alone would refuse these on a dirty tree or demand a token."""
    sandbox.make_dirty()
    assert not sandbox.token.exists()
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "a read-only subcommand spends nothing")


def test_t0156_new_name_up_on_a_clean_tree_needs_no_token(sandbox: Sandbox) -> None:
    """R4 too-wide twin of the dirty-tree row: ``up`` is a routine start (I12), not an engine
    start, so the new spelling must not be put behind a ``/launch`` token either."""
    command = "uv run diffusion_gearvis up --profile lean"
    assert not sandbox.token.exists()
    res = run_hook(command, sandbox)
    assert_allowed(res, command, "clean tree, routine service start")
