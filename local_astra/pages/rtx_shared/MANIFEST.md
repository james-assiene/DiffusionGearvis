# RTX shared images (T-0209 A1.8)

The pictures a user "shares" in the RTX 5090 audience scenarios (`local_astra/pages/rtx_tasks.py`). No start-page
server serves this directory; tests and the T-0209 harnesses post the bytes as an `astra.frame`.

| file | scenario | content |
|---|---|---|
| `r1_blender_error.jpg` | R1 | a drawn Blender-like main window titled "Blender 4.2.3 LTS" behind a "Blender Preferences" window with "System" highlighted in its left list; a "Cycles Render Devices" heading with the tabs None, CUDA, OptiX, HIP, oneAPI (CUDA selected) and the body "No compatible GPUs found for Cycles"; a red status bar reading "CUDA binary kernel for this graphics card compute capability (12.0) not found."; no GPU name and no logo |

Provenance: original work, drawn by `scripts/make_t0209_shared_images.py` with Pillow from shapes and plain text only
(Pillow's bundled default font); no screenshot, logo or copied markup. 1280x800 RGB, JPEG quality 85. The committed
bytes are the source of truth; `image_sha256` in `rtx_tasks.RTX` pins them.

Licence: Apache-2.0, as the rest of the repository.
