# Handwritten maths and physics attempts (T-0223)

Ten one-page student attempts for the education demo: five maths (M1-M5) and five physics (P1-P5),
first/second-year university level. Each is written out the way a student would, 6-7 lines, with
**exactly one** realistic mistake; every later line follows from that mistake, so the boxed answer
is wrong in a way a tutor can trace back to one line. The problems and solutions are our own
wording; nothing is copied from a textbook.

- `attempts.json`: the source of truth. Per page: `id`, `course`, `title`, `problem` (printed at the
  top), `lines` (each `{"text", "mistake"}`; exactly one has `"mistake": true`; text uses matplotlib
  mathtext between `$...$`), `mistake` (the ground truth in one sentence), `correct_result`,
  `judge_keywords`, `false_blame_keywords` and `diagram` (what is drawn, or `null`).
- `out/<id>.png`: one page each, 1700 x 2200 px (US letter at 200 dpi).
- `out/attempts.pdf`: all ten pages, M1..M5 then P1..P5.
- `fonts/`: the vendored handwriting font (`PatrickHand-Regular.ttf`) and its licence (`OFL.txt`).

## Ground truth

The mistake line is numbered within the attempt (line 1 is the first handwritten line).

| id | problem | mistake (line) | the student's answer | correct result |
|---|---|---|---|---|
| M1 | ∫ x·e^{2x} dx by parts | line 3: v = 2e^{2x} (differentiated dv = e^{2x} dx instead of integrating) | (2x − 1)e^{2x} + C | (x/2)e^{2x} − (1/4)e^{2x} + C |
| M2 | tangent to y = sin(x²) at x = √π | line 3: dy/dx = cos(x²), chain-rule factor 2x missing | y = −x + √π | slope −2√π, y = −2√π x + 2π |
| M3 | area between y = x and y = x² on [0, 1], with a sketch | line 2: integrates x² − x (lower minus upper) | −1/6 | 1/6 |
| M4 | eigenvalues of the matrix with rows (2, 1), (1, 2) | line 3: det = (2 − λ)² **+** (1)(1) (sign of the off-diagonal product) | λ = 2 ± i | λ = 1 and 3 |
| M5 | y′ = 2y, y(0) = 3, with a sketch | line 5: from ln y = 2x + C writes y = e^{2x} + C (added constant instead of a factor) | y = e^{2x} + 2 | y = 3e^{2x} |
| P1 | block on a 30° incline, μ_k = 0.2, m = 2 kg, with a free-body diagram | line 3: N = mg instead of mg cos θ | a = 2.94 m/s² | a = g(sin θ − μ cos θ) ≈ 3.20 m/s² |
| P2 | ball thrown horizontally at 5 m/s from a 20 m cliff, with a trajectory sketch | line 2: 20 = 5t + 4.9t² (horizontal speed put in the vertical equation) | t ≈ 1.57 s, x ≈ 7.9 m | t ≈ 2.02 s, x ≈ 10.1 m |
| P3 | 6 Ω in series with (3 Ω ∥ 6 Ω) on 12 V, with a circuit diagram | line 2: R23 = 3 + 6 = 9 Ω (parallel pair added like series) | R_eq = 15 Ω, I = 0.8 A | R_eq = 8 Ω, I = 1.5 A |
| P4 | Atwood machine, m1 = 3 kg, m2 = 5 kg, with a pulley diagram | line 3: a = (m2 − m1)g / m2 (divides by m2 only) | a ≈ 3.92 m/s², T ≈ 41.2 N | a = g/4 ≈ 2.45 m/s², T ≈ 36.8 N |
| P5 | pendulum L = 1 m released at **45°** from vertical, with a diagram | line 3: h = L cos θ instead of L(1 − cos θ) | v ≈ 3.72 m/s | v = √(2gL(1 − cos 45°)) ≈ 2.40 m/s |

P5 uses 45°, not the 60° of the first plan: at 60° cos θ = 1 − cos θ = 0.5, so the wrong height
gives the right answer (3.13 m/s either way) and the mistake would be invisible in the result.

## Judge keywords (for the T-0223 group B check)

`judge_keywords` are pre-registered before any GPU run. A diagnosis **locates** the mistake when,
case-insensitively, it contains every entry; an entry with `|` is satisfied by any one of its
alternatives (so `"half e|½e|1/2 e"` accepts a spoken "half e to the 2x"). Ruling R-2: every
alternative must be specific to the diagnosis, something a wrong diagnosis would not contain (no
bare "= 1"); words the page itself prints (e.g. "normal", "parallel", "horizontal") may appear only
beside a specific entry for the same page. `false_blame_keywords` are phrases that mean the answer
called a correct line wrong; any one of them appearing fails the turn.

## Re-render

```bash
uv run --with matplotlib==3.11.2 --with pillow==12.3.0 python scripts/demo/render_handwritten.py
```

No project dependency is added: matplotlib and pillow come from `uv run --with`, pinned to the
versions the committed `out/` was rendered with (`MATPLOTLIB_VERSION`, `PILLOW_VERSION` in the
script; it warns on stderr under other versions, whose bytes may differ). The output is
deterministic (fixed seed `20261101` for the per-line jitter, fixed PNG/PDF metadata,
`SOURCE_DATE_EPOCH` pinned): two runs give byte-identical PNGs and PDF. Flags: `--attempts`,
`--font-dir`, `--out`.

What the renderer draws: a printed worksheet header (course, "Name:", problem statement) in DejaVu
Sans; lined paper with a red margin; the attempt in the handwriting font in dark-blue ink, each line
with a small seeded offset (≤ ~20 px) and rotation (≤ 0.9°); the final answer boxed; diagrams with
matplotlib's sketch path effect (the xkcd look): incline free-body diagram (P1), cliff trajectory
(P2), circuit (P3), pulley (P4), pendulum with L, the 45° angle and the bottom point only (P5: no
height marker, since the true drop L(1 − cos θ) would hint the answer), shaded region between y = x and y = x² (M3), and the
student's own solution curve y = e^{2x} + 2 through (0, 3) (M5). Lines with fractions, limits or
roots take two ruled rows.

## Font and licence

Handwriting: **Patrick Hand** Regular by Patrick Wagesreiter, from the google/fonts repository
(`ofl/patrickhand/`), licensed under the SIL Open Font License 1.1; the licence text is
`fonts/OFL.txt`. It has good digits and covers ∫ √ Ω π ² − ± ≈ ½ μ λ. Glyphs it lacks (θ, ∥, →
and similar) fall back per glyph to DejaVu Sans in plain text and to STIX Sans in mathtext, both
shipped with matplotlib, so nothing renders as a tofu box.

SHA-256: `PatrickHand-Regular.ttf` 0f173b3e6cb6d1af25babf7f0057c5ac4ee11f9992b0469bb817e967ef4ad0fc,
`OFL.txt` 377f4f9c19e935228552478eb68cc2ed82910988a60ba60e2ac73b09f32d02d1.
