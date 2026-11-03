# DiffusionGearvis

An on-device voice-and-vision assistant inspired by research demos like Google's Project Astra,
for a single RTX 5090 desktop: DiffusionGemma
(vLLM, NVFP4) thinks, Kyutai STT listens, Pocket TTS speaks on the CPU, the HUD page shows
A2UI surfaces and the live denoising view. Docs in `docs/` (start at `docs/00-README.md`);
how work happens in `CLAUDE.md` and `docs/09-agent-workflow.md`; status in `PROGRESS.md`.

**Status:** research prototype; needs Linux, one NVIDIA RTX 5090 (32 GB) and Docker; not
production software.

## Requirements

- Linux (developed on Debian/Ubuntu) with PipeWire audio.
- One NVIDIA RTX 5090 (32 GB). The NVIDIA driver must be 580 or newer, with the NVIDIA Container Toolkit installed, so Docker can use the GPU (`docker run --gpus all`).
- Docker. Your user must be able to run it, directly or through the `docker` group.
- [uv](https://docs.astral.sh/uv/) and git. `uv` fetches Python 3.11 itself if the machine has none.
- Google Chrome or Chromium, to open the HUD page. (The agent's own browser for `--tools live` is a separate, pinned Chromium; see below.)
- PortAudio for the microphone and speakers: `sudo apt install libportaudio2`.
- At least 4 CPU cores: text-to-speech is pinned to cores 2 and 3.
- For `--tools live` with a visible browser window, a desktop session (`DISPLAY` or `WAYLAND_DISPLAY`); otherwise use `--browser headless`.
- Node.js only if you run the HUD's own tests (`make hud`).
- A headset is recommended. There is no echo cancellation yet, so with speakers Gearvis can hear itself.
- About 60 GB of free disk:
  - the vLLM image, about 31 GB;
  - the language model weights, about 18 GB;
  - the speech-to-text image, about 6 GB, and its weights, about 2.3 GB.

  The weights download once, into your Hugging Face cache (`~/.cache/huggingface`).

## One-time setup

All of this is done once per machine, and it needs the network.

```bash
git clone https://github.com/james-assiene/DiffusionGearvis.git && cd DiffusionGearvis
bash scripts/bootstrap.sh        # checks the toolchain, runs uv sync and the test suite (several minutes); never uses the GPU

# Model weights. The language model is pinned to the revision in services/versions.lock;
# the speech-to-text model is downloaded at its current revision.
uvx --from huggingface_hub hf download nvidia/diffusiongemma-26B-A4B-it-NVFP4 \
    --revision 2ea837236295d617ac27f8c17d61228081932c40
uvx --from huggingface_hub hf download kyutai/stt-1b-en_fr-candle

# Container images and the vLLM plugin
docker pull vllm/vllm-openai:v0.29.0
make stt-image                   # builds the speech-to-text server image, about 15 minutes
make plugin-wheel                # the plugin that streams the denoising steps to the HUD

# Only for --tools live: the agent's own pinned Chromium
uv run playwright install chromium

# Download the text-to-speech server and its voice once (it is slow the first time):
# wait until http://127.0.0.1:8081/health answers (e.g. curl it from another terminal), then press Ctrl-C
uvx --from pocket-tts==3.1.0 pocket-tts serve --host 127.0.0.1 --port 8081
```

If you ever re-run `uv sync`, use `uv sync --all-extras`: a plain `uv sync` or `uv sync --extra X` removes the other extras (the browser, audio and search packages).

If a Hugging Face model page asks you to accept its licence or to log in, accept it on the website and run `uvx --from huggingface_hub hf auth login` first.

`diffusion_gearvis fetch-models` is not implemented in v0.1. Use the download commands above.

## Run Gearvis

Start the three services (the language model, speech-to-text and text-to-speech). They run in the background.

```bash
uv run diffusion_gearvis up --profile lean
uv run diffusion_gearvis status        # wait until vllm, stt and tts all say "ok" (the model takes 2–3 minutes)
```

The first `up` also downloads Pocket TTS (text-to-speech) and its voice, so it needs the network that one time. If you skipped the warm-up step above, that download can take longer than `up` waits, and `up` reports `tts FAILED`; see the first row of the troubleshooting table.

Then start a session and serve the page:

```bash
uv run diffusion_gearvis session
```

The session serves the page by default and prints `the HUD page is at http://127.0.0.1:1420/?ws=…`. Open that address in Chrome. `--no-serve-hud` starts the session without serving the page. If port 1420 is already in use, the session says so in one line, gives the address in case a Gearvis HUD is already serving there, and keeps running. Then:

- **Talk:** speak into your microphone. The session listens on your default PipeWire input, which it names on its first lines.
- **Type:** or type a question in the box at the bottom of the page and press Enter.
- **Share:** click **Share** and pick a tab, window or camera so Gearvis can see it. Mute (or the `m` key) stops sharing.

Gearvis answers aloud, and shows the conversation, with any cards it makes, on the left. On the right, **Thinking** shows the model's real denoising steps as they happen:
- each cell is the token at that position of the canvas, updating at every step;
- finished canvases fold away;
- the canvas that ends the answer folds as "end of sequence".

### What v0.1 can and cannot do

- **It can:**
  - see a tab, window or camera you share;
  - answer by voice;
  - show cards;
  - show its real denoising steps.
- **Tools are off by default** (`tools = "off"` in `astra.toml`), so out of the box it will tell you it cannot search or act.
- **`--tools live`** lets it click, type and scroll in **its own sandboxed browser**, which can open **public https websites**. It cannot control your desktop. `--no-web` keeps it on its built-in loopback start pages. Read [Real websites](#real-websites) before you use it.
- **Web search is on in live sessions.** Gearvis sends the queries it makes to DuckDuckGo (falling back to Brave, then Mojeek; no account or key) and writes them to your local session log; `--search off` turns it off, and `--tools dry-run --search ddgs` turns it on without the browser. If you create `~/.config/diffusion_gearvis/brave_api_key` yourself (one Brave Search API key, `chmod 600`), a search that all three keyless engines refuse is retried once through the Brave Search API (`api.search.brave.com`, which counts against your Brave quota); the key is sent only in its request header and never printed or logged, and a file with looser permissions, a symlink or a malformed key is ignored with one warning line. Results are treated as untrusted text, but a crafted page can still steer the model: on our loopback prompt-injection page (suite S14) the model followed the injected instruction in 4 of 20 trials (T-0138 baseline, 2026-10-25; lean, canvas 256, 48 steps, vLLM 0.29.0, NVFP4 rev 2ea8372, 1120 visual tokens, headless 1280×800, n = 20, order seed 0).
- **Nothing else leaves your machine.** The services bind to 127.0.0.1 only; search and the agent's browser are the two paths out, and both are off with `--search off --no-web`.

### Known limits in v0.1

- **Web tasks are hit and miss.** On our loopback demo scenarios it finished 7 of 15 (T-0211 C), on the three.js editor 1 of 15 (T-0210 C), and on a shared-screen handoff it opens its browser far more often than it finishes the job (T-0220: 9 of 9 opened, 2 of 9 finished). It may also say a task is done when it is not; the completion check catches some of these, not all.
- **Pages can steer it.** A crafted web page can make it follow instructions you did not give (see [Real websites](#real-websites)).
- **Settings panel:** a change applies at the next turn. If you press **Stop and apply** while Gearvis is still listening, later presses do nothing until that turn ends (the change still applies then). Two changes made in the same instant can make the second one come back "not changed: stale"; make it again. Changes are not saved between sessions yet.
- **Thinking out loud.** Occasionally Gearvis keeps talking after its answer to check its own work ("Let me be careful."). Once it starts repeating itself the rest of that reply is silenced, but a sentence or two of self-talk can still be heard. Maths it writes in LaTeX is read as words.
- **Checking homework.** On 39 real phone-scanned homework pages (French thermodynamics, electronics and probability), Gearvis with the Education tutor prompt called 14 of 15 correct pages correct and named the right line on 8 of 19 pages with a mistake. It can misread handwriting and then blame a correct line, and on some pages with a mistake it says the work is fine. Treat it as a second pair of eyes, not a marker.
- **A turn can end in silence.** Rarely (2 of 78 turns in those runs) Gearvis thinks for a long time and then stops without answering, or the model times out; you hear nothing. Ask again.
- **Brave fallback:** if you set up the optional Brave key and a result carries a broken character, that one turn can end in an error; the session carries on.

### How many screens Gearvis sees

Your shared screen or camera is captured about once a second. Each turn shows the model at most 3 of those screens: the one on screen when you started speaking, plus any that changed while you spoke. Earlier screens from the conversation fill the remaining image slots, up to 4 images per request; older ones become a one-line caption.

Both numbers are settings. Change them in the Settings panel (they apply from the next turn), or at launch with a flag or the profile key of the same name:

- **Screens per turn (max images)**: `--max-frames-per-turn 1|2|3|4`, profile key `max_frames_per_turn`, default 3. At 4 the turn's own screens take every image slot, so no earlier screen stays an image.
- **Screen capture interval (ms)**: `--frame-interval-ms 250|500|750|1000|1500|2000|3000|5000|10000`, profile key `frame_interval_ms`, default 1000. It covers the shared screen and camera only; the agent browser takes one screenshot per step regardless.

The ceiling of 4 is not a vLLM limit. It is a flag in our launcher, `services/vllm.sh`: `--limit-mm-per-prompt '{"image": 4, "video": 1}'`. The app repeats it as `max_images` in `local_astra/conductor/prompt_builder.py`, and a request with more images than the launcher allows is refused (HTTP 400). Raising it is possible, but v0.1 has not tested it:

- **Context.** At the default 1120 visual tokens, each screen costs about 1,100 prompt tokens. The prompt budget is about 14,300 tokens (16,384 minus room for the answer), so 13 images would be the absolute maximum with no conversation left. About 8 is the realistic ceiling.
- **GPU memory.** vLLM reserves memory for the image limit when it starts. The KV pool (4 GiB) is sized for 4 images plus history, and the engine already uses 86 % of a 32 GB card, so a higher limit can fail at start-up for lack of memory.
- **Latency.** Every extra image is read before Gearvis can start answering, so the first word comes later. This has not been measured.
- **Fewer images.** With fewer images (`--max-frames-per-turn 1|2`) or cheaper ones (`--visual-tokens 280|560`) Gearvis answers sooner and keeps more conversation, but it sees less: small text becomes hard to read.

To try a higher limit, change both numbers (the flag in `services/vllm.sh` and `max_images`), then run `down` and `up`. If `up` reports that the engine is short of memory, go back to 4. A launch option for this is planned after v0.1.

Capturing more often than once a second (a lower `frame_interval_ms`) makes Gearvis notice changes sooner but costs CPU in the page. Capturing less often makes the screen it answers about older.

### Stop

Press Ctrl-C in the session's terminal, then stop the services:

```bash
uv run diffusion_gearvis down
```

`down` stops all three services, including the speech-to-text container. Run `status` to confirm nothing is left.

### Useful options

| Option | What it does |
|---|---|
| `session --silent` | play through a null device (no speakers) |
| `session --no-mic` | open no microphone; type instead |
| `session --mic <node>` | pick a PipeWire microphone by name |
| `session --tools dry-run` | let the model request tools; they are described, never executed |
| `session --tools live` | let the model act in its own browser (headed by default, `--browser headless`), with web search and public https websites on |
| `--no-web` | with `--tools live`: keep the agent's browser on its built-in loopback start pages |
| `--search off` | with `--tools live`: no web search (it is on by default in live) |
| `diffusion_gearvis config` | print the resolved profile (`astra.toml`) |

### Settings

The gear button in the page header opens **Settings**. Every setting has a short note under it: what it does, when it helps, when it does not, and what we measured. They are grouped as Seeing the screen, Acting in the browser, Checking its work, Guidance and prompt, Search and web, Speech, and Engine.

- **Changed from the page, no restart:** ReGround and its crop size, action outcome, repeat guard, screen reminder, completion check, re-asking an announced call, the screen-now label, the overview frame and the automatic page overview, thought-call recovery, the screenshot guidance (v1 to v4), visual tokens per image, and **your instructions to Gearvis**. Most of these apply to `--tools live` sessions only; elsewhere the row says it does not apply. The context helpers keep their own checkboxes above.
- **When it applies:** at the start of the next turn. The conversation, memory, the agent's browser and its page are kept. A change made during a turn waits for the turn to end; **Stop and apply** stops the turn first. Changing the guidance, your instructions or the visual tokens makes the first reply after it a little slower (the model reads its prompt again once).
- **Your instructions to Gearvis:** a box of up to 1000 characters added at the end of the system message. Apply sends it, Reset to default removes it. **Show the full system message** shows what the model receives. The persona and the built-in guidance can be read there but not edited. Safety rules (no passwords or card numbers, ask before buying or sending, the web fence) are enforced by Gearvis's code, not by this text. Your text is not written to the session log, except inside the model requests it is part of.
- **Prompt library:** the **Prompt** list under the box has four built-in prompts: Default (no added instructions), Education tutor, Gaming companion and Browser specialist. Choosing one fills the box; nothing changes until you press Apply. To keep your own, type a name and press **Save as…**; **Delete** removes a saved one. Saved prompts are stored in this browser for this HUD address only, not on the server. **Export** downloads them as `gearvis-prompts.json`; **Import** adds the prompts from such a file (it asks before replacing one you already have).
- **Needs a restart:** tools mode, web search, real websites, the browser window and size, tool steps per turn, action echo units, filler clips, the wait tool, the page overview tool, and the engine (canvas, steps, model length). Those rows show the command to run, with a Copy button.
- **Not a setting:** thinking is always on (switching it per request would make every reply slow), and the honest-ending clips always play.
- Changes last for this session only. Saving them as your defaults is planned after v0.1.

#### Options at a glance

One row per setting, in the panel's order, with the same notes the panel shows. The numbers come from the runs named in each note; the panel shows the full evidence. The tables are generated from `local_astra/settings_catalog.py` by `uv run python scripts/gen_readme_options.py`; edit the catalog, not the tables.

<!-- options:begin -->
The numbers were measured under this regime: DiffusionGemma 26B-A4B NVFP4 rev 2ea8372, vLLM 0.29.0, lean, canvas 256, 48 steps, thinking on; headless Chromium 1280x800, live, typed turns, order seed 0, engine unseeded; n as given. Suite run-to-run noise is about 10 points per 100 (T-0137 C). Some runs differ; the panel's evidence names them.

**Seeing the screen**

| Setting | Default | Useful when | Not useful when | Changes |
|---|---|---|---|---|
| Visual tokens per image | 1120 in live (T-0132 A1) | small text, dense pages, precise clicks | fast chat about a simple screen; speed matters more than precision | live from the panel (at launch: `--visual-tokens`) |
| Screens per turn (max images) | 3 (T-0059) | the screen changes several times while you ask, and the steps in between matter | a mostly still screen; at 4 the turn's own screens take every image slot, so no earlier screen stays an image (each becomes its caption line) | live from the panel (at launch: `--max-frames-per-turn`) |
| Screen capture interval (ms) | 1000 ms, one frame per second per source (T-0227) | a fast-changing screen, where a shorter interval notices changes sooner | a loaded box or a still screen; a longer interval means the screen Gearvis answers about can be that old | live from the panel (at launch: `--frame-interval-ms`) |
| ReGround | off (T-0133) | small or dense targets: on the three.js editor click misses fell 35/96 → 10/74 (T-0212, n = 15 per arm); turn the overview frame off first (`--reground` at launch does it) | ordinary pages, where it adds a request per click for no gain (T-0132 C: off 65, crop 280 64/100); refused with the overview frame | live from the panel (at launch: `--reground`) |
| ReGround crop tokens | 280 with ReGround on at a 1120 main budget (T-0133 A5), else the main budget | a smaller crop costs half the pass-2 time for about the same result: T-0132 C n=100 crop 280 64/100 vs crop 1120 66/100 (ReGround off 65/100) | with ReGround off it does nothing, so it is refused | live from the panel (at launch: `--reground-visual-tokens`) |
| Screen-now label | label (T-0134c) | the model loses track of which screenshot is current | rarely; `off` only to reproduce old runs | live from the panel (at launch: `--live-feedback`) |
| Page overview tool | off | only for experiments: the model never called it (T-0140 B) | everyday use: no gain from offering it (T-0140 B: 66/100 with it, 71 without) | needs a restart: `--page-overview` |
| Automatic page overview | off | rarely: the default overview frame already shows long pages; T-0150 B no change overall (S4 0 → 20/20 but S5 13 → 0/20); turn the overview frame off first (`--auto-overview` at launch does it) | most pages: the model clicks the overview's points on the live screen (T-0150 B: S5 13 → 0/20) | live from the panel (at launch: `--auto-overview`) |
| Overview frame | on, on 1.05–2.0-screen pages (T-0152) | pages a little taller than the screen | pages over two screens tall or shorter than 1.05 screens, where it does not apply; refused with ReGround | live from the panel (at launch: `--overview-frame`) |
| Context helpers (page text, PDF text, cells, accessibility tree) | off (D12a) | text the pixels cannot carry | screens the model reads from the pixels, which is the default (every turn logs which were on) | live from its own checkboxes |

**Acting in the browser**

| Setting | Default | Useful when | Not useful when | Changes |
|---|---|---|---|---|
| Action echo units | model | pixels only to reproduce runs from before T-0134c (arm A) | everyday use: model units are the units the model clicks in | needs a restart: `--action-echo` |
| Thought-call recovery | on (T-0135b) | always in live | debugging the model's raw behaviour | live from the panel (at launch: `--thought-call-recovery`) |
| Wait tool | on (T-0202 D) | slow-loading pages (T-0202 C probe 1/10 → 8/10) | fast pages, where it is not called (T-0202 C: no wait in 30 suite turns) | needs a restart: `--wait-tool` |
| Tool steps per turn | 8, web 20 (T-0178) | raise it for long tasks that run out of steps | short tasks: a stuck turn takes longer to give its honest ending | needs a restart: `--max-tool-steps` |
| Tool steps per turn (web) | 20 in live with web on (T-0178) | raise it for long tasks on real websites that run out of steps | short tasks: a stuck turn takes longer to give its honest ending | needs a restart: `--max-tool-steps` |
| Agent browser window | headed | headed: watching it work at the desk | headless: a box with no display, or scripted runs | needs a restart: `--browser` |
| Agent browser size | 1280×800 | a page that needs a bigger or smaller window | comparing with the measured runs: every A/B ran at 1280×800 | needs a restart: `browser_viewport` in astra.toml |
| Tools mode | off (profile) | live: Gearvis uses its own browser; dry-run: see the calls it would make | off is enough for talking about your shared screen | needs a restart: `--tools` |

**Checking its work**

| Setting | Default | Useful when | Not useful when | Changes |
|---|---|---|---|---|
| Action outcome | off | experiments only | everyday use: no measured gain (T-0136 D, T-0212, T-0214); needs `live_feedback` ≠ off | live from the panel (at launch: `--action-outcome`) |
| Repeat guard | on (T-0179, James's ruling) | stops stuck loops early | when the page really needs a repeated click | live from the panel (at launch: `--repeat-guard`) |
| Screen reminder | on (T-0201 D) | always in live | rarely; `off` only to reproduce runs from before T-0201 D | live from the panel (at launch: `--screen-reminder`) |
| Completion check | on (T-0207 C) | browser tasks that end in a claim | pure chat (it never fires there) | live from the panel (at launch: `--completion-check`) |
| Re-ask an announced call | on (T-0172) | the model says it will act and does not | rarely; `off` only to see the model's raw behaviour | live from the panel (at launch: `--announced-call-reask`) |

**Guidance and prompt**

| Setting | Default | Useful when | Not useful when | Changes |
|---|---|---|---|---|
| Screenshot guidance | v3 (T-0191 C) | v3 for web; v2 is what `--no-web` gets anyway | v4: worse on the shared handoff (T-0211 C: 3/9 → 1/9) | live from the panel (at launch: `--live-guidance`) |
| Your instructions to Gearvis | empty | style or language preferences, domain hints | long or conflicting text (untested against the A/Bs) | live from the panel |
| Thinking | on | always on: every A/B ran with it on | cannot be turned off: switching it per request would make every reply slow | fixed: not a setting |

**Search and web**

| Setting | Default | Useful when | Not useful when | Changes |
|---|---|---|---|---|
| Web search | on in live (T-0125) | questions that need the web | private or offline work | needs a restart: `--search` |
| Real websites | web on | tasks on real websites | the loopback demo pages (unreachable in web mode), offline work | needs a restart: `--no-web` |

**Speech**

| Setting | Default | Useful when | Not useful when | Changes |
|---|---|---|---|---|
| Filler clips | off (James 2026-10-28) | voice chats with long thinking | typed use, demos | needs a restart: `--filler-clips` |
| Honest-ending clips (out_of_steps, not_finished, lost_thread, at_checkout) | on | always on: they say why a turn stopped early | cannot be turned off | fixed: not a setting |

**Engine (relaunch only)**

| Setting | Default | Useful when | Not useful when | Changes |
|---|---|---|---|---|
| Engine: canvas, steps, max_num_seqs, KV, max-model-len, --diffusion-config, STT/TTS backend, profile | per `services/versions.lock` and the profile | a different profile or engine budget for your GPU | mid-conversation: it restarts the services | needs a restart: `--profile` |

<!-- options:end -->

## Real websites

With `--tools live`, Gearvis's own browser can open public https websites (it says so when the session starts; `--no-web` turns it off). Treat it as a capable but gullible assistant.

In web mode the browser starts on an address page, and the built-in loopback demo pages (the catalogue, deals and shop) are not reachable from it. To run those demos, start the session with `--no-web`.

**The risk.** Web pages can contain instructions aimed at the model, and the model sometimes follows them: on our injection test page the model followed planted instructions in 4 of 20 turns (T-0138 S14). That was measured with the lean profile, canvas 256, 48 steps, NVFP4 rev 2ea8372, vLLM 0.29.0, n = 20, order seed 0. So a page may make it navigate to, or type into a form on, a site you did not ask for, and what it types can include things you said in the session.

**What it cannot do:**
- type into fields a page marks as password or card-number fields, or type text that looks like a card number;
- use your logins: it runs a fresh profile with no cookies, history or saved passwords, and keeps nothing between sessions;
- download or upload files;
- write to your clipboard;
- reach your local network or this machine's services: every request goes through an egress proxy that allows only public addresses on port 443.

**What it still asks first.** Before buying, sending, posting or deleting anything it asks you, and saying "no" stops it.

**What you should do:** do not sign in to anything in its window, and do not read out passwords, codes or card numbers. Password or code boxes that a page does not mark as such are not recognised.

### Real websites: the browser sandbox

Web mode needs Chromium's sandbox. On Ubuntu 24.04 and later, unprivileged user namespaces are restricted by AppArmor (`kernel.apparmor_restrict_unprivileged_userns = 1`), and Playwright's Chromium then fails with "Chromium sandboxing failed!". If the session says so, install this profile once (it lets only Playwright's Chromium binaries create user namespaces):

```bash
sudo tee /etc/apparmor.d/gearvis-playwright-chromium >/dev/null <<'EOF'
abi <abi/4.0>,
include <tunables/global>
profile gearvis-playwright-chromium @{HOME}/.cache/ms-playwright/chromium-*/chrome-linux64/chrome flags=(unconfined) { userns, }
profile gearvis-playwright-headless @{HOME}/.cache/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell flags=(unconfined) { userns, }
EOF
sudo apparmor_parser -r /etc/apparmor.d/gearvis-playwright-chromium
```

Or run without real websites: `--no-web`.

## When something is wrong

**Run `doctor` first.** `uv run diffusion_gearvis doctor` is a read-only check of this machine; it prints PASS, WARN or FAIL per line, with the fix. While the services are running its `vram.free` line fails; that is expected, because the model holds the GPU.

**Where to look.** Service logs are in `logs/` (`logs/vllm.log`, `logs/stt.log`, `logs/tts.log`); a FAILED row from `up` quotes the end of its log. Each session writes `logs/sessions/<date>-<time>-<pid>.jsonl` and prints the path when it starts. Console messages say `astra …`: that is the program's earlier name, kept as an alias, and means the same command.

| What you see | Likely cause | What to do |
|---|---|---|
| `tts FAILED` on the first `up` | Pocket TTS was still downloading; the download keeps running | When `uv run diffusion_gearvis status` shows tts `ok`, stop that leftover process with `pkill -f 'pocket-tts serve'`, then run `up` again |
| `cannot reach the docker daemon …` or `permission denied` in a FAILED row | Docker is not running, or your user cannot run it | `sudo systemctl start docker`; add yourself to the `docker` group, then log out and back in |
| `no astra-moshi:0.6.4 — build it first: make stt-image` | The speech-to-text image is not built | `make stt-image` (about 15 minutes) |
| `no plugin wheel; run 'make plugin-wheel'` | The vLLM plugin is not built | `make plugin-wheel` |
| `refusing to start the engine; it would die at its own startup check … short by N MiB` | Other programs hold GPU memory (often a browser using the GPU) | Close GPU-heavy programs, run `down`, then `up` |
| vllm `not healthy after 240 s` (or its launcher exited) | The model is not in your Hugging Face cache, or Docker cannot reach the GPU | Check `logs/vllm.log`; make sure the `hf download` finished; check the NVIDIA Container Toolkit with `docker run --rm --gpus all ubuntu nvidia-smi` |
| A Hugging Face download asks you to log in or accept a licence | The model page requires it | Accept it on the website, then `uvx --from huggingface_hub hf auth login` |
| `cannot serve the HUD page on 127.0.0.1:1420: the port is in use` | Another program, or another Gearvis, uses port 1420 | If a Gearvis HUD is already serving there, open the printed address; otherwise stop the other program |
| `no capture source found -- plug a microphone … or start the session with --no-mic` | No microphone is visible to PipeWire | Plug one in, pick it with `--mic <node>`, or use `--no-mic` and type |
| `PortAudio is not installed` (session) or `sounddevice is unusable` (doctor) | PortAudio is missing, or the audio package was removed by a narrow `uv sync` | `sudo apt install libportaudio2`; then `uv sync --all-extras` |
| Gearvis hears itself and answers itself | Speakers, and there is no echo cancellation yet | Use a headset, or `--silent` and type |
| `--tools live refused: … Chromium … is not installed` (or `playwright is not importable`) | The agent's browser is not installed | `uv sync --all-extras && uv run playwright install chromium` |
| Chromium says `Host system is missing dependencies` | A minimal system lacks libraries Chromium needs | `uv run playwright install-deps chromium` (uses sudo) |
| `--tools live refused: a headed browser needs a display` | No desktop session | Run from a desktop, or add `--browser headless` |
| `Chromium sandboxing failed!` | Ubuntu 24.04+ restricts the sandbox Chromium needs | Install the AppArmor profile in [Real websites: the browser sandbox](#real-websites-the-browser-sandbox), or use `--no-web` |
| The Thinking pane says "denoising text off" | The tokenizer files are missing from your Hugging Face cache | Re-run the language model's `hf download` at the pinned revision |
| `web search is off: …` at session start | A proxy variable is set, or the search package is missing | Unset `http_proxy`/`https_proxy`, or `uv sync --all-extras`; `--search off` silences the line |
| Searches come back empty | The keyless search engines refused (rate limits) | Try again later; optionally set up the Brave key (see "What v0.1 can and cannot do") |
| The engine answers HTTP 400 about too many images (e.g. "At most 4 image(s) may be provided") | `max_images` in the app was raised above the launcher's `--limit-mm-per-prompt` | Keep the two numbers equal (see [How many screens Gearvis sees](#how-many-screens-gearvis-sees)), then `down` and `up` |
| `up` fails with an out-of-memory error after you raised the image limit | vLLM reserves memory for the image limit at start-up | Go back to 4 images, or free GPU memory |
| `cannot bind the conductor endpoint on port N` | A session is already running | Stop the other session (Ctrl-C in its terminal) |
| A service is still running after `down` | Docker could not be reached, or `up` never recorded it (a failed start) | `uv run diffusion_gearvis status`; `docker ps` and `docker stop <id>` for the engine, `docker stop astra-stt` for speech-to-text, `pkill -f 'pocket-tts serve'` for text-to-speech |

**Network.** All services bind to loopback (127.0.0.1) only. The paths out are the one-time downloads, web search and the agent's browser in web mode; `--search off --no-web` turns the last two off. (`up` starts Pocket TTS through `uvx`, which may check PyPI for the pinned package.)

## Tests

    uv run pytest tests/unit tests/contract -q -m "not gpu"   # the canonical fast suite, CPU only

The fast suite needs no special hardware. The tests marked `gpu` need the RTX 5090 box with the
engine running, and the tests marked `audio` (under `tests/integration/`) need the PipeWire
null-sink rig (`bash scripts/audio_rig.sh up`); neither is part of the fast suite.

The `tests/cd/fixtures/perstep_golden_*.pt` fixtures are PyTorch files, which are Python pickles:
loading an untrusted one can run code. The constrained-decoding tests load them with
`torch.load(..., weights_only=True)`; do the same if you load them yourself.

## How this repository is developed

The agent workflow (planner, test-author, implementer, reviewer, verifier, with the hooks that
enforce it) lives in `.claude/`; `docs/09-agent-workflow.md` describes it.

## Licence

Apache-2.0: see `LICENSE`. Third-party material and its licences are listed in `NOTICE`.

## Contributing

Issues are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md) for what to include and how the tests run. Pull requests may not be reviewed quickly. Report security problems privately as described in [SECURITY.md](SECURITY.md).
