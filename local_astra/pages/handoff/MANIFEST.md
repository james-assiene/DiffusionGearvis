# Handoff shared images (T-0208 B2)

The pictures a user "shares" in the H1-H3 handoff scenarios (`local_astra/pages/handoff_tasks.py`). No start-page
server serves this directory; tests and the T-0208 C harness post the bytes as an `astra.frame`.

| file | handoff | content |
|---|---|---|
| `h1_steam_wand.jpg` | H1 | a line-drawn espresso machine, a label plate "Barista Express BES870", the steam wand drawn snapped inside a red circle; no caption (the label plate is its only text) |
| `h2_stove_box.jpg` | H2 | a drawn box front reading "Trailhead Duo Stove" and "2-person", with a stove outline |
| `h3_chat.jpg` | H3 | a drawn chat window with one message from "Sam" asking for a large pepperoni from Domino's, pickup at the W 44th St store |

Provenance: original work, drawn by `scripts/make_t0208_shared_images.py` with Pillow from shapes and plain text only
(Pillow's bundled default font); no photo, logo or copied markup. 1280x800 RGB, JPEG quality 85. The committed bytes
are the source of truth; `image_sha256` in `handoff_tasks.HANDOFFS` pins them.

Licence: Apache-2.0, as the rest of the repository.
