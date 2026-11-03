# tests/fixtures/t0177

## `integral_1568x590.jpg` (T-0177 A, the t14 case; D1's shared-screen fixture)

- **What it is:** a synthetic shared screen, 1568x590 (the size of the `astra.frame` `source: screen` messages in
  `logs/sessions/20261027-230926-966483.jsonl`, James's t14 case), JPEG quality 85, baseline, 4:2:0. A white page
  with a grey title bar (`worksheet-3.pdf`), the heading `Calculus II - worksheet 3, problem 2`, `Evaluate:`, and one
  large definite integral: **the integral from 0 to pi of x sin(x) dx** (integrand `x sin(x)`, lower limit `0`, upper
  limit `π`; its value is π). No real screen, person or document is depicted.
- **sha256** `345e4edc1df2b17da9944ba411a1befc89dcf1e62587064de896a51b143e1489`, 40,305 bytes.
- **Generator:** `make_integral.py` in this directory (sha256
  `a49b2b6e32efdbc884b9ae65853c491df41a8e6194a734ea8730890d3e604516`), run from the T-0177 worktree on 2026-10-28:

  ```
  uv run python tests/fixtures/t0177/make_integral.py tests/fixtures/t0177/integral_1568x590.jpg
  ```

  Two runs into the session scratchpad gave byte-identical output before the run above.
- **Build it was made with:** Pillow 12.3.0, libjpeg(-turbo) 6.2 API, FreeType 2.14.3; font
  `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf` from Debian/Ubuntu `fonts-dejavu-core` 2.37-8 (sha256
  `ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280`). Another build may encode different bytes;
  tests pin the committed file's sha256, never a regeneration.
- **Licence:** generated for this repository (Apache-2.0, as the repository). DejaVu fonts are under the Bitstream
  Vera / DejaVu free licence; the glyphs are rendered into pixels, the font file is not redistributed.
- **Used by:** `tests/eval/test_t0177_t14_case.py` (CPU, the t14 shape over fakes), and group D's
  `scripts/gpu/t0177_shared_smoke.py` (posted as `astra.frame` at 1 Hz).
