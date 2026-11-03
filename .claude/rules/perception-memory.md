---
paths:
  - "local_astra/perception/**"
  - "local_astra/memory/**"
  - "frontend/**"
---

# Perception, memory, frontend

- Pixel-first: the model sees screenshots and camera frames by default; DOM/page text, PDF text layer, cells and AT-SPI are *assists* behind HUD checkboxes, off by default, and the badge state is logged with every turn. Focus crops (screen + active-window/pointer crop) are the pixel-only mitigation for dense documents.
- v1 perception source is the HUD page (`getDisplayMedia`/`getUserMedia` → JPEG at 1 fps by default (`frame_interval_ms`, T-0227) → `astra.frame` over the WebSocket; `astra.frame.text` only when the page-text assist is on); the native daemon (portal ScreenCast → PipeWire → GStreamer `pipewiresrc target-object=`, CPU path, restore tokens, portal app-id) is M2+. Both implement the same `PerceptionClient` contract and share one contract-test module. Never use `mss`/X11 grabs on Wayland paths; never CPU-map NVIDIA DMA-BUFs (negotiate plain `video/x-raw`).
- Frames are downscaled to the resolution buckets in `perception/policy.py`; screen frames use the maximum-budget bucket by default (pixel-first, ≈ 1120 visual tokens per image per docs/12 §3; the estimate becomes a measurement in Q2), the focus crop is a second image; changing a bucket requires re-running B2.
- Ring buffers are 60 s; `get_frame(at=speech_onset)` is the default for queries, not `latest`.
- Global mute stops capture at the pipeline level (PipeWire node paused), not at the consumer; the test checks node state. Per-app exclusions match on app-id before a frame is stored.
- Memory-lite: SQLite + FTS5 in the Conductor process; episodes `ts, source, thumbnail, window_title?, caption?`; captions are written by DiffusionGemma when idle > 10 s (request aborted on speech onset; never pre-empts a live turn); retrieval is FTS5 over captions with a time filter, top-3 thumbnails attached to the prompt. No embeddings/SigLIP/Florence/LanceDB unless a recall eval fails on visual concepts.
- Nothing under `perception/` or `memory/` opens a non-loopback socket or writes outside `~/.local/share/local_astra/`.
- The frontend never handles audio (the Conductor owns PipeWire capture/playback and any AEC); in v1 it does capture video frames for perception, renders A2UI surfaces (`@a2ui/lit` + `edu` catalog, pinned to spec v0.9.1) and the denoising view. Protocol events are the OpenAI Realtime vocabulary plus `astra.*` extensions (`astra.frame`, `astra.frame.text`, `astra.assist.toggle`, `astra.ui.surface`, `astra.ui.action`, `astra.canvas.step`, `astra.canvas.commit`); add new events to `contracts/protocol.py` and the codec round-trip test. A2UI payloads are validated (fixer → schema → topology) before they reach the page; never render an unvalidated surface.
