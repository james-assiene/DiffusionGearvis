---
paths:
  - "tests/**"
---

# Tests

- Layers and markers: `tests/unit` (pure, no I/O), `tests/contract` (fakes by default; `-m gpu` selects the real service), `tests/integration`, `tests/e2e`, `tests/bench`, `tests/eval`. Markers: `gpu`, `audio`, `desktop`, `slow`. The fast suite is `uv run pytest tests/unit tests/contract -q -m "not gpu"` — serial, the canonical run, ceiling 120 s (two minutes); the Stop hook runs the identical selection with `-n 12 --dist loadfile` under a 60 s budget (T-0031, T-0143). Nothing is deselected for speed: a test that leaves the per-turn gate must be named by the verifier's command list, and no invocation uses the `slow` marker to narrow the fast suite.
- Every contract test is parametrised over `backend=["fake", "real"]`; the fake and the real service must produce the same assertions. A behaviour added to a fake needs a real-service test in the same PR.
- No tautologies (`assert sum(p) == 1` on a softmax), no re-implementing the function under test inside the test, no asserting current behaviour captured from the implementation. Expected values come from hand computation, fixtures, or an independent oracle (brute force on tiny inputs, `fractions.Fraction`, float64 numpy).
- Statistical tests use fixed seeds and Bonferroni-corrected thresholds; state n and the regime in the test name or docstring.
- Turn-machine tests use the virtual clock fixture and Hypothesis stateful testing; invariants from architecture §4 live in `tests/unit/test_turn_invariants.py` and are never weakened. Inside xdist workers (the Stop hook, `make fast-parallel`) properties run every phase but shrink and report the raw counterexample; the serial `make fast` run shrinks (T-0146). Never add `phases=` or a smaller `max_examples` to a test to make the gate fast; `--hypothesis-profile=default` restores shrinking in a parallel run.
- Benchmarks read `turns.jsonl` and compare to thresholds in `docs/03-test-plan.md` §7; a threshold change requires a `docs/tasks/` entry.
- Do not mark tests `xfail`/`skip` to get green. A justified `xfail(strict=True)` names the tracking task id in its reason.
- Tests written ahead of their implementation carry `@pytest.mark.pending(task="T-NNNN")` (or a module-level `pytestmark`). `tests/conftest.py` runs them as `xfail(strict=True)` until the test-author removes the marker at the start of `/implement` for that task; a pending test that passes fails the suite (XPASS strict), so stubs and tautologies cannot hide. `docs/tasks/T-NNNN.md` must exist; `scripts/check_pending.py` reports the backlog. Import the module under test at module level (stubs exist for every planned module) so a pending test never errors at collection.
- Fakes live in `tests/fakes/` and satisfy the `runtime_checkable` protocols in `local_astra/contracts/protocols.py` (`tests/contract/test_fakes_conform.py`); the `backend` fixture in `tests/contract/conftest.py` parametrises `fake` / `real` (`real` is marked `gpu`).
- **A guard needs a row for being too wide as well as one for being absent.** Most invariants people write about a guard are *one-sided*:
  they fire when it is too permissive and are trivially satisfied when it is too strict. On 2026-09-16 the endpointer's staleness guard could
  be keyed on the wrong clock — refusing every word spoken within a second or so of the previous turn's end, seconds on the CPU recogniser —
  and **all 37 rows stayed green**, because the only negative row used a 30 s gap that no plausible over-wide guard would fail. A guard that
  is too wide silences the user; a guard that is absent lets noise through. Write the discriminating case for both, and when you quote a
  mutation-sensitivity number, say which direction it was measured in.
- **A coarse mutation can make an unpinned behaviour look covered.** Mutate the shape of the *ruling*, not a blunt version of it. On
  2026-09-16 a reviewer checked whether "a failed start still produces rows for the later services" was pinned: `return 1` on any failed
  verdict reddened a row and would have been reported as covered, while the mutation shaped like the actual decision — abort only on the two
  failure verdicts — left the whole suite green. The behaviour was genuinely unpinned. If your mutation reddens a row for a reason other than
  the property you are testing, you have learned nothing.
- **To check whether a premise rewrite killed an assertion, diff the assertions.** When a change forces existing rows' *setup* to change,
  `git diff main...HEAD -- tests/ | grep -E "^-[^-]" | grep assert` lists every assertion the change removed, and the per-module assertion
  count says whether the suite got stricter or weaker. "I adjusted the fixtures" and "I deleted the check" look identical in a normal diff and
  are one command apart.
- **Activation is three things in one commit, not one.** Removing a row's `pending` marker also requires moving the task's `**Status:**` to
  `in progress` *and* naming the module under its `## Tests to write` — `classify_failures.py` reads both, and a red module the task does not
  name is reported as a regression against a finished task. This bit four times on 2026-09-16: twice a status left at `planned`, once a module
  the test-author moved for a good reason, once four modules whose pins a criterion legitimately required to move. Every time the hook was
  right and the task file was stale. The same applies in reverse when a criterion forces an *existing* module's pin to change: name it, or its
  legitimate red reads as collateral damage and the next person stops trusting the suite.
- **A test double closes by delegation, not by enumeration.** A double that hand-lists the names it exposes has two failure modes, both
  seen on 2026-09-16: an implementation naming a shared exception base the list forgot dies with an `AttributeError` *inside* the test
  instead of failing an assertion, which silently constrains how the implementation may be written; and anything the list did not think of
  escapes to the real thing — one double served two command runners by name, so an implementation reaching for a third would have run a
  command against this box and passed. Delegate every name you are not deliberately faking to the real module, and make the dangerous ones
  raise a named assertion. Enumerating is how both gaps happened.
- **Never export `GIT_DIR` or `GIT_WORK_TREE` to run the suite** (T-0186, 2026-10-28). With `GIT_DIR` exported at the main repo, a hook test's sandbox `git init` + commit landed on `main`, and every agent's path hook then refused its edits. The conftest now strips git's repo selectors and the spawn guard refuses a `git` child that still carries one, but don't rely on that: git-range rows skip in a `git archive` copy by design, so run them from a real worktree.
- **Mutation harnesses: name the copy uniquely, and prove where the import came from.** The session scratchpad is *shared between
  concurrent subagents*, so a generic directory (`mut`, `copy`, `tmp`) is a race: on 2026-09-16 one agent rsynced its worktree over
  another's mutation copy between two runs, and the victim's "14 failed" was measured against a tree that was no longer its own. Use a
  unique name (`$$`, the task id, the agent id). Second, the editable install puts the *worktree* on `sys.path` through a `.pth` file, so a
  run started inside a copy can silently import the module you think you mutated. Set `PYTHONPATH` to the copy, **`cd` into it**, and print
  `<module>.__file__` from inside the pytest process. The cwd is the third way this bites and the one that survives a correct
  `PYTHONPATH`: `sys.path[0]` is the empty string, which resolves to the working directory, and it is searched *before*
  `PYTHONPATH` — so a run launched from the worktree imports the worktree however carefully the copy was built. A mutation result
  without that origin proof is not evidence.
- **A test that discovers the implementation's attributes by name pattern dictates how the implementation may be written — and if it
  *calls* what it finds, it also dictates their arity and can fire their side effects.** On 2026-09-16 a helper collected every `dir(gate)`
  name containing "refus" and invoked each callable among them; a private `_refuse(self, source)` then killed **nine** rows with a `TypeError`
  raised *inside the helper*, not an assertion — so the suite reported a type error where it meant "that name is reserved". The live
  constraint, which nobody had written down, was that no attribute of that class whose name contains "refus" may take an argument. This is the
  sibling of "a test double closes by delegation, not by enumeration", aimed at the code under test instead of at the double. Discover by
  pattern if you must; **never invoke what you discovered**, and filter to non-callables or to zero-parameter signatures.
- **Pytest's assertion-rewrite cache keys on mtime and size, so a same-length mutation can survive `git checkout`.** Found 2026-09-17
  while mutation-testing a conftest guard: a mutation of the same byte length, followed by a checkout that restores the mtime, leaves the
  *mutated* bytecode in `__pycache__/conftest.cpython-*-pytest-*.pyc` and the next run silently uses it. A "reverted" arm that still shows
  the mutant's result is this. Purge `__pycache__` under `tests/` between arms (`find tests -name __pycache__ -exec rm -rf {} +`), and prefer
  mutations that change the length.
- **A source-grep guard matches prose as readily as code.** `tests/unit/test_prompt_budget.py` greps `session.py` for a slash-four shape
  to stop the four-characters-per-token guess returning. On 2026-09-17 a docstring citing "criteria 3/4/9" matched it and reddened a row
  with nothing to do with the change. When you write a guard that greps source text, either scope it to code (strip strings and comments
  first) or say in its failure message that prose can trip it; and when you write a docstring in a guarded file, know what it is guarded for.
- Fixtures live in `tests/fixtures/` with a `MANIFEST.md` stating provenance (recording conditions, generator script, licence).
- **A panel that clears state on an id change must be tested in the order the server actually publishes.** T-0054's page forgot a fence
  when a new surface id arrived; the end-state row delivered `surface → fence` and was green, but the server emits `fence → surface`, so the
  first surface of every turn erased that turn's own notice, and no row noticed. When a row pins "state X is cleared by message Y", write
  the companion in the wire order: the message that *sets* X, then the message that must *not* clear it. Take the order from a real session
  over the fakes, not from the row's convenience.
- **A reduced-motion neutraliser must set `transition-property: none`, not a near-zero duration.** Chrome 147: `*, *::before, *::after
  { transition-duration: 0.001s !important }` *creates* a transition on every property of every element, and `getComputedStyle` then
  returns the previous value for a frame — T-0056's six chip states all read as IDLE under emulated `reduce`. Use `transition-property:
  none !important; transition-duration: 0s !important; animation: none !important`. A headless row that reads computed style right after
  a state change must run under that block or wait a frame; `node --test` cannot see this class of failure, only the Chrome module can.
- **A scope or contract pin compares against a fixed commit, never a moving ref.** T-0089's "contracts change only by these additions"
  rows read `git show main:<path>`. They were green on the branch and went red the moment T-0089 merged, because `main` then already held
  the additions. The verifier's trial merge could not see this. `git merge-base HEAD main` fails more quietly: after the merge the diff is
  empty and the row passes vacuously.
  - Pin to the literal base SHA the task branched from, in a named constant with a comment.
  - Skip with a stated reason when that commit is absent.
  - Pin both ends (T-0094). A pinned base compared against the working tree still decays: four cd-group guards went red on main once
    later tasks legitimately edited their paths. After merge, a scope guard compares `BASE..TIP`, both literals from the task's merge
    commit M (`BASE = merge-base M^1 M^2`, `TIP = M^2`; not the declared fork point when the branch merged main mid-way). An invariant
    meant to hold forever pins content (a sha256 literal) instead. An exemption list for later tasks is a sign the range is open.
    `tests/unit/test_guards_use_fixed_refs.py` enforces the moving-ref half, and (T-0094c) flags a literal base older than its
    `CLOSED_AS_OF` compared with the working tree. Whoever merges a task that converts or closes scope guards bumps `CLOSED_AS_OF` to
    that merge commit in the same merge.
- **Hypothesis replays saved failures, so a mutant "kill" can be a replay.** Found 2026-10-17 (T-0112): with the `.hypothesis` database on, a property class reddened under a mutant only because an earlier mutant's failing example was stored and replayed. Run mutation checks of Hypothesis classes with `settings(database=None)` (and more than one seed) before counting a kill.
- **Assert a deny's reason whenever another rule could also deny** (T-0145, 2026-10-23). A hook test whose only assertion is the exit code passes when the wrong rule denies. On 2026-10-23 the implementer sandbox's owned list parsed empty, so every role-rule row was denied by "lists no path", and removing the implementer's `tests/` deny broke no test. Give the sandbox the grants that the rule under test overrides, assert the rule's own message, and assert the competing rules' messages are absent.
- **Linearity rows count traced work and time CPU** (T-0129, 2026-10-24). Never use a `perf_counter` median or a wall-time ratio for a "linear, not quadratic" claim: under the Stop hook's 12-worker load such rows were red in 44–177 of 360 trials. Count traced line events in the code under test (`tests/unit/_work_count.py`), which is exact, and pair that with `time.thread_time`, gc off, sizes interleaved, min of k, because C-level blow-ups are invisible to line counts. Ratio bars come from the linear/quadratic prediction, not from current output.
- **A fresh interpreter does not load conftest, so an in-process patch does not protect a child** (T-0175, 2026-10-27). `brave_key_nowhere` patched the key path in-process only. A6's BASE subprocess (an old tree run with `sys.executable -c`) read the real `~/.config/diffusion_gearvis/brave_api_key`. The fix is the session jail plus spawn guard in `tests/conftest.py` and `tests/_home_jail.py`:
  - From the conftest import on, `HOME` and `XDG_CONFIG_HOME` point at a per-session directory, and every child inherits it. Caches are mirrored into it by an allowlist of symlinks; `.config` is never mirrored.
  - An audit hook refuses any spawn whose env has no `HOME`, or names the real home.
  - Give a hand-built child env a `HOME`: a tmp dir, or `os.environ["HOME"]`. Never loosen the guard.
  - The only escape is the `real_home_env` fixture, which works only under `ASTRA_LIVE_SEARCH=1`.
- **A probe that calls a pure function twice must give each call its own copy of any stateful fixture behind an injected seam** (T-0148, 2026-10-24). `step_is_pure` called the turn machine twice through the test model's `SpeechGate`, so the first call moved the gate. A rare PAUSED-resume shape then failed the line-704 assertion inside the probe, which made a correct machine look broken. Snapshot the fixture, give each call a deepcopy, restore it, and assert it is unchanged.
