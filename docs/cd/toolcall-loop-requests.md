# Tool-loop request rows: form G vs A (Q4), an image in a continued message (Q5), and the compaction A/B (T-0115a)

**Status: pre-registration, 2026-10-17. Not run.** It runs inside T-0115's `/launch`, as step 3 of
`scripts/locked/t0115_loop_probe.sh`. Everything above *Results* is the pre-registration. None of it may be edited
after the run. Task: [`docs/tasks/T-0115a.md`](../tasks/T-0115a.md). Harness: `scripts/gpu/t0115a_loop_requests.py`.
Eval row: `tests/eval/test_loop_requests.py` (gpu, `T0115A_RUN_DIR`).

## Questions

- **Q4 (H-T111a-10).** Does the host-rendered continuation (form G) behave like the OpenAI `role: tool` body (form A)
  at step 2? Report-only.
- **Q5.** Does vLLM accept an image part in a continued assistant message? T-0111c needs the answer. Report-only.
- **E.** Does eliding older tool responses with the product's `loop.compact` hurt answers? What does one compaction
  cost in cache misses and latency? The E bars decide whether T-0120b is planned as is.

## Regime (pre-registered)

| | |
|---|---|
| engine | profile `lean`; vLLM `v0.29.0`; DiffusionGemma NVFP4 rev `2ea8372`; chat template `36e3a42e`; tokenizer `cc8d3a0c`; CL 256, 48 steps; thinking on; prefix caching on; `max_num_seqs 3` |
| device, boot | RTX 5090, driver 595.84, gpu-mem 0.86; the same boot as T-0115 (`scripts/locked/t0115_loop_probe.sh`) |
| tools | `native:bbfc53ccdd80` on every request (`tool_choice: "none"`, `skip_special_tokens: false`, `return_token_ids: true`); repair on (a kit with repair off is refused); the repair never runs here |
| cases | the dev split of `tests/eval/toolcall/cases.jsonl` (sha256 `460cd1c3…`). Ids read: `search-01`, `search-02`, `click-01`, `click-03`, `type-02`, `key-01` (Q4, Q5) and `scroll-01` (its screen, for E). The test split is never read |
| n | Q4: 6 cases × 2 arms × 5 = 60. E: 4 histories × 2 placements × 10 repeats × 2 arms = 160, plus 40 warm. Q5: 3. Total 263 requests |
| seed | none (the diffusion path refuses `seed`); runs are not reproducible |
| concurrency | 1; `LlmOptions()` as the session sends it; no prefill, no repair |
| prompts | step 1 is built as a fresh session builds one typed turn: `PromptBuilder(DEFAULT_SYSTEM_PROMPT, thinking=on, tools)`, no history, the case's screen as the one frame (none for the two search cases), no assists, no memory hits |
| gate files | the manifest records the sha256 of `local_astra/toolcall/loop.py` and `local_astra/conductor/llm_client.py`; the eval row skips if either has changed |

Estimated duration (not measured): about 4–5 minutes of requests. This assumes about 1 s per request at
concurrency 1, which is about twice the 0.47–0.51 s mean of the T-0103 and T-0108a pass-1 requests, because the E
prompts are longer. The mock-engine run of the same 263 requests takes about 3 s of harness overhead.

## Method

### Q4: form G vs form A

Six step-2 cases, T-0111a's set:

| case | step 1 text | call | response |
|---|---|---|---|
| `search-01` | "Let me look that up." | `google_search{query:"weather lisbon"}` | `search_payload` of 3 fixed fake results (below) |
| `search-02` | "Let me check the opening hours." | `google_search{query:"british museum"}` | `search_payload` of 3 fixed fake results (below) |
| `click-01` | "Clicking Sign in." | `click_at{x:500,y:499}` | `action_payload("would click at (640, 399)")` |
| `click-03` | "Adding the Quiet One headphones to your cart." | `click_at{x:789,y:380}` | `action_payload("would click at (1010, 304)")` |
| `type-02` | "Searching the shop for a USB-C hub." | `type_at{press_enter:true,text:"usb-c hub",x:500,y:36}` | `action_payload("would type 'usb-c hub' at (640, 29) and press Enter")` |
| `key-01` | "Undoing that." | `press_key{combo:"Ctrl+Z"}` | `action_payload("would press Ctrl+Z")` |

The calls are the cases' reference calls. The action texts are what `DryRunToolExecutor` writes for them on the
1280×800 screens (`to_pixels`).

Fake results for `search-01` (title, url, snippet):
1. "Lisbon weather tomorrow - hourly forecast", `https://weather.example/lisbon/tomorrow`, "Sunny spells in the morning, cloud building after 15:00. High 24 C, low 16 C. Light north-westerly wind."
2. "Lisbon 10-day forecast", `https://forecast.example/pt/lisbon`, "Dry for most of the week with temperatures a little above the seasonal average; a chance of showers on Friday."
3. "Portugal weather warnings", `https://meteo.example/warnings`, "No warnings are in force for the Lisbon district."

Fake results for `search-02`:
1. "Visit the British Museum - opening times", `https://museum.example/visit`, "Open daily 10:00-17:00, Fridays until 20:30. Last entry 30 minutes before closing. Closed 24-26 December."
2. "British Museum - plan your visit", `https://museum.example/plan`, "Free entry; booking a timed ticket is recommended at weekends and during school holidays."
3. "British Museum (Bloomsbury, London)", `https://maps.example/british-museum`, "Great Russell St, London. Nearest stations: Tottenham Court Road, Holborn, Russell Square."

**Arm G** is `loop.step_prompt(base, (step,))`, sent through `VllmClient`. **Arm A** is the raw OpenAI body:
- `base`'s body as `VllmClient.request_body` renders it;
- then an `assistant` message whose `content` is step 1's text and with one `tool_calls` entry (`id` `call_0`,
  `type` `function`, the step's name, `arguments` a JSON string). Ruling R6: this is the OpenAI equivalent of what
  form G shows the model (the text, then the call), so the start-repetition comparison compares like with like;
  an earlier draft sent no content;
- then a `role: tool` message with `tool_call_id` `call_0` and the response as a JSON string.

Arm A carries no `continue_final_message`. It goes over loopback httpx, because `Message` has no `tool_calls`. Its
stream is read by the client's own id reader, so both arms are read the same way.

**Order.** The 5 repeats are outermost; within each, the cases go in the order above. For each (case, repeat), G
goes first when the case index plus the repeat is even, and A goes first otherwise.

**Read (report-only), per arm, n = 30:**
- the outcome shares:
  - `call`: a tool commit outside an open thought;
  - `thought_open`: the request's last canvas ends inside a thought (the final state, as for E below);
  - `final`: a non-empty answer;
  - `empty`: otherwise;
- errored requests;
- the start-sentence repetition rate. The step-2 text is the answer commits outside an open thought, joined. It
  repeats when it begins with step 1's first sentence, with both texts normalised: case folded and runs of
  whitespace collapsed. An empty step-1 text never repeats.

### E: the compaction A/B

**Histories.** There are 4 invented-fact questions. Each history is 5 `google_search` steps built from the literals
below, asked over the screen of `scroll-01` (`screens/d05_article.jpg`).

Step *i* (1–5) has these parts:
- the text `STEP_TEXTS[i]`;
- the call `google_search{query: queries[i]}`;
- the response `search_payload` of 5 results. The first result is the **fact result** at the placement's fact
  step, and the step's **lead** otherwise. The other four are the history's pool items `(2(i−1) + k) mod 8`, for
  k = 0..3.

Each result's snippet is then **padded**, so the step is maximal:
- in result order, round robin, each snippet gets the next sentence of the history's filler list, then "See
  details.", then "More.", each with a leading space;
- a sentence is added only if the snippet stays within `MAX_SNIPPET` (300) and the rendered block stays within
  `MAX_RESPONSE` (1500);
- padding stops when no sentence fits anywhere.

So `render_response` never cuts a block, and the fact stays whole at the start of its snippet. The unit row checks
that every block is within 100 chars of 1500.

**Placements.** In **old**, the fact is in step 1's response, which `compact` elides at its defaults. In **kept**,
it is in step 5's.

**The step texts** are the same for every history. They are neutral, so they read as well after a result that
answers the question as after one that does not:
1. "Let me look that up."
2. "Let me check another source to be sure."
3. "I'll cross-check that with one more search."
4. "Let me look for an official page as well."
5. "One last search to confirm."

**Arms.**
- FULL is `step_prompt(base, steps)`.
- COMPACT is `step_prompt(base, loop.compact(steps, E_BUDGET))`, with `E_BUDGET = 7000` and `compact` at its
  defaults (`keep_last` 2, target 7000·2//3 = 4666).

Each 5-step continuation is about 8 000 chars, which is over 7000. After steps 1 and 2 are elided it is still over
4666. Steps 4 and 5 are the kept tail. So exactly steps 1–3 are elided, and each becomes the stub
`{note:"earlier result omitted",results:5}`. The unit row checks the elided set by hand, and `main` refuses to run
otherwise. The warm request is `step_prompt(base, steps[:4])`, the step-4 prompt.

**Order.** The histories go in the order below, and the placements old then kept. For each (history, placement):
- a warm request is sent before repeats 0, 2, 4, 6 and 8;
- each repeat sends FULL then COMPACT when the repeat is even, and COMPACT then FULL when it is odd.

**Recorded per request:**
- the outcome, or **errored** (below);
- prompt tokens, from `/tokenize` in chat form, before the run;
- the `/metrics` deltas of `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total` around the request;
- first-commit ms and done ms, both on the harness's monotonic clock.

**Outcome of one E request** (from its commits; the eval row re-derives it independently):
0. A request that errored is **errored**, whatever its commits hold (ruling R4). Errored means a status other than
   200, a timeout, a transport error, or a stream that ends without a `finish_reason`. A truncated stream can
   carry the fact before the client raises; it is still errored, never correct.
1. A tool commit outside an open thought decides it: `google_search` is **re-search**, and any other call is
   **other**.
2. Otherwise, a request whose **last canvas** ends inside an unclosed thought is **other** (ruling R1: the final
   state, read from the last commit's `thought_open`). It is not a final answer, so it is never correct, even when it
   quotes the fact. `VllmClient.generate` marks every commit of a canvas that ends inside a thought. So a thought
   that one canvas leaves open and a later canvas closes, followed by an answer, is a final answer.
3. Otherwise, the answer text is read, with special literals dropped and stripped:
   - empty is **other**;
   - text holding an accepted string is **correct**;
   - otherwise it is **unsupported final**. A fact that appears only inside a closed thought is not in the answer.

**The match** (ruling R2) is case-insensitive: both texts are case-folded. It also has number and word
boundaries: the accepted string must not be preceded or followed by a letter or digit, so `16:45` does not hold
`6:45`. The accepted strings are exactly the list below. The matcher derives no other forms.

### The histories, verbatim

**The accepted strings, the complete pre-registered list** (the fact first, then the stated equivalents; the
12-hour and unpadded forms are listed explicitly):

| history | accepted |
|---|---|
| `ferry` | `19:40`, `7:40 pm`, `7:40pm`, `7:40 p.m.`, `twenty to eight` |
| `market` | `06:45`, `6:45`, `6:45 am`, `6:45am`, `6:45 a.m.`, `quarter to seven` |
| `hall` | `1,187`, `1187`, `1 187`, `one thousand one hundred and eighty-seven`, `one thousand one hundred eighty-seven` |
| `lighthouse` | `41.6`, `forty-one point six` |

Bare `7:40` is not accepted for `ferry`. Without "pm" it names the morning time, while unpadded `6:45` is the same
24-hour time as `06:45`. No accepted string appears in any question, step text, call, lead, pool item or filler
sentence. The fact appears in exactly one rendered response of each placement (unit row).

#### `ferry`

- **Question:** "When does the last Sunday ferry leave Pier 9 for Halvard Island?"
- **Fact:** `19:40`. **Equivalents:** `7:40 pm`, `7:40pm`, `7:40 p.m.`, `twenty to eight`.
- **Queries:**
  1. "last sunday ferry pier 9 halvard island"
  2. "halvard island ferry timetable sunday"
  3. "pier 9 ferry departures halvard"
  4. "halvard ferries official timetable"
  5. "halvard island ferry sunday evening sailing"
- **Fact result:** "Halvard Island Ferries - Sunday timetable", `https://halvardferries.example/timetable/sunday`, "Sunday sailings from Pier 9: first departure 08:15, last departure 19:40. Return crossings from the island run hourly until 20:30."
- **Leads (steps 1–5):**
  1. "Halvard Island Ferries - timetables", `https://halvardferries.example/timetable`, "Timetables for weekday, Saturday and Sunday sailings from Pier 9. Choose a day to see the departures."
  2. "Getting to Halvard Island", `https://visithalvard.example/getting-here`, "The island is reached by ferry from Pier 9 in about 35 minutes. Sailings are less frequent at weekends."
  3. "Pier 9 passenger terminal", `https://harbour.example/pier-9`, "Pier 9 serves the Halvard and Moss Bay routes. The ticket office opens 30 minutes before the first sailing."
  4. "Halvard Island Ferries - about us", `https://halvardferries.example/about`, "We operate two vessels on the Pier 9 to Halvard route all year round, weather permitting."
  5. "Halvard ferry - reviews", `https://travelreviews.example/halvard-ferry`, "Friendly crew and a short crossing. Check the timetable before travelling on a Sunday."
- **Pool (0–7):**
  0. "Moss Bay ferry timetable", `https://mossbayferry.example/times`, "Moss Bay sailings leave Pier 9 every two hours on weekdays."
  1. "Halvard Island - things to do", `https://visithalvard.example/see-and-do`, "Coastal walks, the old lighthouse keeper's cottage and a small seafood market."
  2. "Harbour car parks", `https://harbour.example/parking`, "Long-stay parking is available next to Pier 9; pay at the machine or by app."
  3. "Halvard Island weather", `https://weather.example/halvard`, "Strong winds can cancel crossings; the operator posts cancellations on its website."
  4. "Ferry fares - Halvard route", `https://halvardferries.example/fares`, "Adult returns, child fares and a ten-journey card. Bicycles travel free."
  5. "Island ferries in the region", `https://coastaltravel.example/ferries`, "A guide to the passenger ferries that leave the city harbour, with links to each operator."
  6. "Halvard Island accommodation", `https://visithalvard.example/stay`, "Two guesthouses and a campsite; book ahead in summer."
  7. "Pier 9 accessibility", `https://harbour.example/pier-9/access`, "Step-free access to the pontoon; assistance on request at the ticket office."
- **Filler:**
  - "Timings can change on public holidays."
  - "Tickets can be bought on board or online."
  - "Allow ten minutes for boarding."
  - "Dogs on leads are welcome on deck."
  - "Check the operator's website before you travel."
  - "Updated regularly by the operator."

#### `market`

- **Question:** "What time does the Ostrava Lane farmers' market open on Saturdays?"
- **Fact:** `06:45`. **Equivalents:** `6:45`, `6:45 am`, `6:45am`, `6:45 a.m.`, `quarter to seven`.
- **Queries:**
  1. "ostrava lane farmers market saturday opening time"
  2. "ostrava lane market hours"
  3. "ostrava lane farmers market"
  4. "ostrava lane market council page"
  5. "ostrava lane saturday market opening"
- **Fact result:** "Ostrava Lane Farmers' Market - opening times", `https://ostravalanemarket.example/times`, "Saturday market: stalls open at 06:45 and trade until 13:00. The Wednesday evening market runs from 16:00 to 20:00."
- **Leads:**
  1. "Ostrava Lane Farmers' Market", `https://ostravalanemarket.example`, "Local growers, bakers and cheesemakers every Saturday and Wednesday evening. See our opening times page."
  2. "Markets in the city - council guide", `https://citycouncil.example/markets`, "Street markets, farmers' markets and night markets across the city, with links to each organiser."
  3. "Ostrava Lane market - stallholders", `https://ostravalanemarket.example/stalls`, "Around forty stalls, most of them within thirty miles of the city."
  4. "Ostrava Lane market on a map", `https://maps.example/ostrava-lane`, "Ostrava Lane, between the railway arches and the canal towpath."
  5. "Ostrava Lane market - visitor reviews", `https://localreviews.example/ostrava-lane`, "Arrive early for the bread, the best loaves sell out quickly."
- **Pool:**
  0. "Canal towpath walks", `https://canaltrust.example/walks`, "A flat walk along the canal passes Ostrava Lane and the old wharf."
  1. "Seasonal produce calendar", `https://growers.example/calendar`, "What is in season each month, from asparagus to squash."
  2. "Parking near Ostrava Lane", `https://citycouncil.example/parking`, "On-street parking is free before nine on weekend mornings."
  3. "Railway arches businesses", `https://archesquarter.example`, "Cafes, workshops and a brewery under the arches."
  4. "Become a stallholder", `https://ostravalanemarket.example/apply`, "Applications are open to producers who grow, rear or make what they sell."
  5. "Food markets guide", `https://foodguide.example/markets`, "Our pick of the region's food markets, reviewed by readers."
  6. "Bus routes to the canal", `https://citybuses.example/routes`, "Routes 4 and 12 stop at the canal bridge."
  7. "Ostrava Lane market - news", `https://ostravalanemarket.example/news`, "New stalls this season and a summer music programme."
- **Filler:**
  - "Cash and cards are accepted at most stalls."
  - "Bring a reusable bag."
  - "Opening times may vary in winter."
  - "Dogs are welcome on leads."
  - "Run by a community cooperative."
  - "Check the website for changes."

#### `hall`

- **Question:** "How many seats does the main auditorium of Corvane Hall have?"
- **Fact:** `1,187`. **Equivalents:** `1187`, `1 187`, `one thousand one hundred and eighty-seven`, `one thousand one hundred eighty-seven`.
- **Queries:**
  1. "corvane hall main auditorium seats"
  2. "corvane hall capacity"
  3. "corvane hall auditorium seating plan"
  4. "corvane hall venue information"
  5. "corvane hall main auditorium number of seats"
- **Fact result:** "Corvane Hall - venue facts", `https://corvanehall.example/venue`, "The main auditorium seats 1,187, with a further 240 seats in the studio theatre and standing room in the foyer bar."
- **Leads:**
  1. "Corvane Hall - seating plan", `https://corvanehall.example/seating`, "Stalls, circle and balcony. Choose a performance to see the seats that are still available."
  2. "Corvane Hall", `https://encyclopedia.example/corvane-hall`, "A concert hall opened in the 1960s and refurbished twice; home of the city's symphony orchestra."
  3. "Corvane Hall - hire the venue", `https://corvanehall.example/hire`, "The auditorium and studio theatre are available for conferences and private events."
  4. "Corvane Hall - accessibility", `https://corvanehall.example/access`, "Wheelchair spaces in the stalls and circle; hearing loops in both performance spaces."
  5. "Corvane Hall - visitor reviews", `https://localreviews.example/corvane-hall`, "Great acoustics; the balcony is steep, so book the circle if you can."
- **Pool:**
  0. "What's on at Corvane Hall", `https://corvanehall.example/whats-on`, "Orchestral concerts, comedy and family shows this season."
  1. "Corvane Hall refurbishment", `https://citynews.example/corvane-refurb`, "The latest refurbishment replaced the stage floor and the ventilation."
  2. "Getting to Corvane Hall", `https://corvanehall.example/directions`, "A short walk from the central station; the nearest car park is on Mill Street."
  3. "City symphony orchestra", `https://citysymphony.example`, "Our season at Corvane Hall runs from September to June."
  4. "Corvane Hall box office", `https://corvanehall.example/tickets`, "Book online, by phone or in person; the box office opens at noon."
  5. "Concert halls in the region", `https://culture.example/halls`, "A guide to the region's concert halls and their resident ensembles."
  6. "Corvane Hall bars and food", `https://corvanehall.example/food`, "Pre-show dining in the foyer restaurant; interval drinks can be ordered in advance."
  7. "Corvane Hall history", `https://corvanehall.example/history`, "From the opening night to the present day, in photographs."
- **Filler:**
  - "Seat availability changes as tickets are sold."
  - "Booking fees may apply."
  - "Doors open one hour before the performance."
  - "Children's tickets are half price."
  - "Large bags cannot be taken into the auditorium."
  - "Information supplied by the venue."

#### `lighthouse`

- **Question:** "How tall is the Tessaly Point lighthouse?"
- **Fact:** `41.6`. **Equivalents:** `forty-one point six`.
- **Queries:**
  1. "tessaly point lighthouse height"
  2. "tessaly point lighthouse"
  3. "how tall is tessaly point lighthouse"
  4. "tessaly point lighthouse heritage listing"
  5. "tessaly point lighthouse tower height metres"
- **Fact result:** "Tessaly Point lighthouse - heritage record", `https://heritage.example/tessaly-point`, "Tower height 41.6 metres from base to vane; the light stands 63 metres above high water and is visible for 22 nautical miles."
- **Leads:**
  1. "Tessaly Point lighthouse", `https://encyclopedia.example/tessaly-point`, "A white masonry tower on the headland, first lit in the nineteenth century and automated in the 1980s."
  2. "Visit Tessaly Point", `https://coastpath.example/tessaly-point`, "The lighthouse grounds are open in summer; the tower itself is not open to the public."
  3. "Tessaly Point lighthouse - photographs", `https://photos.example/tessaly`, "Pictures of the tower, the keepers' cottages and the fog signal house."
  4. "Lighthouses of the north coast", `https://lighthouses.example/north-coast`, "Tessaly Point is one of seven lights on the north coast, all now automated."
  5. "Tessaly Point walk", `https://walks.example/tessaly-point`, "A circular walk from the village to the headland and back along the cliffs."
- **Pool:**
  0. "Tessaly village", `https://tessaly.example`, "A fishing village with a harbour, two pubs and a small museum."
  1. "Coastal birdwatching", `https://birds.example/coast`, "Gannets and puffins can be seen from the headland in early summer."
  2. "Lighthouse keepers' cottages", `https://holidaylets.example/keepers-cottage`, "Self-catering cottages next to the light, sleeping up to six."
  3. "North coast weather", `https://weather.example/north-coast`, "Sea fog is common on the headland in spring."
  4. "Maritime museum", `https://maritime.example`, "Exhibits on shipwrecks, lifeboats and the keepers who ran the lights."
  5. "Coast path guide", `https://coastpath.example/guide`, "Stage-by-stage descriptions of the north coast path."
  6. "Lighthouse lenses explained", `https://lighthouses.example/lenses`, "How the large glass lenses of old lights focus a small lamp into a beam."
  7. "Tessaly harbour", `https://tessaly.example/harbour`, "Boat trips round the headland leave from the harbour wall."
- **Filler:**
  - "The light is maintained by the coastal authority."
  - "The grounds close at dusk."
  - "There is no parking on the headland."
  - "Information from the heritage record."
  - "Stay on the marked paths near the cliffs."
  - "Last updated this year."

The manifest records the sha256 of every warm, FULL and COMPACT continuation text (`e_texts_sha256`), so the bytes
that were sent can be checked against these literals.

### Q5: an image in a continued assistant message, run last

There are 3 requests, one for each action case `click-01`, `click-03` and `type-02`, in that order. Each takes the
case's step-1 body (as `VllmClient` renders it), then adds a final `assistant` message with two content parts:
- the step-1 continuation text, `text + call + response`;
- an `image_url` part holding the case's screen as a JPEG data URL. This is the screen "after" the dry-run action.

`continue_final_message` is on, `add_generation_prompt` is off, and the body goes over raw loopback httpx.

**Readings:**
- **accepted**: status 200, and the stream ends (`[DONE]` or a `finish_reason`);
- **rejected**: anything else while `GET /health` still answers 200;
- **engine-down**: `/health` stops answering after the request.

`/health` is probed after every Q5 request, up to 3 times, 1 s apart (ruling R8). One answer is enough to rule out
engine-down. Only 3 failures in a row read engine-down, so a single blip after a 400 reads rejected. The line records
`health_checks`. A rejected reading is not a stop: all three Q5 requests are sent, and the run exits 0.

Q5 runs last because a crash in the patched image path kills the EngineCore (`.claude/rules/gpu-services.md`). An
engine-down reading stops the harness. Every earlier row is already on disk and is kept. The report says
`engine-down`, and the exit code is 1.

## Pre-registered rules

### E bars (H-T115a-1, ruled by the human 2026-10-17), exact fractions of each cell's n (40)

| bar | rule |
|---|---|
| **EC** (control) | FULL correct ≥ 28/40 at **both** placements. Otherwise the E rows are **inconclusive**: the model cannot use 5-step histories at all |
| **E1** (kept placement, non-inferiority) | COMPACT correct ≥ FULL correct − 4/40 |
| **E2** (old placement, recovery) | COMPACT (correct + re-search) ≥ 32/40 |
| **E3** (old placement, harm) | COMPACT unsupported final ≤ 8/40 |

**Errored requests (ruling R4).** Errored E requests are counted inside their cell's n (the report's tally has an
`errored` column per placement and arm). They are never correct. If **3 or more of the 160** E arm requests errored,
spread over cells or in one, E is **inconclusive**, and the reason names the errored count. This mirrors T-0115's V
rule and can only make the outcome more cautious (flagged for the human). Errored requests do not make the run fail.
Only 10 errored requests in a row stop it. A complete run exits 0, errored requests or not.

**Outcome:**
- "**inconclusive: <reasons>**" when EC fails or 3 or more E requests errored (every reason that applies is named);
- otherwise "**elision viable**" iff EC, E1, E2 and E3 all hold;
- otherwise "**elision hurts: <every failing bar>**".

At n = 40, E3 can fail only together with E2: 9 unsupported leaves at most 31 for correct plus re-search. So E3 adds
a named reason, not an independent gate (the orchestrator's observation, 2026-10-17).

### Cost (report-only)

**Why only the first request of each cell measures anything.** The warm request makes the miss a one-compaction
cost only for a COMPACT whose tail is not already cached. Within a (history, placement) cell, every later request is
a byte-identical repeat of one already sent. With prefix caching on, the later warms, FULLs and COMPACTs are
therefore expected to hit almost entirely. Only the first COMPACT of each cell, after its warm, measures a cold tail.
A cost measured on every repeat would need a warm per repeat and per-repeat-distinct prompts, which is outside the
40-warm design (orchestrator ruling, 2026-10-17).

**The cost reading, per placement (ruling R3; `report["E"]["cost"]["by_placement"][placement]`).** Each cell
contributes its first COMPACT after its first warm, and its **first pair**: the lowest repeat after the warm with
both arms. Per placement, over its 4 cells:
- `compact_miss_tokens_first_per_cell_median` (and `_n`): the median of `prefix_query_tokens −
  prefix_hit_tokens` of those first COMPACTs;
- `predicted_miss_tokens_median`: the median of that placement's predicted misses (manifest
  `predicted_miss_tokens["<history>/<placement>"]`, measured cells only);
- `first_commit_ms_diff_median` and `done_ms_diff_median`: the median of each cell's first-pair COMPACT − FULL
  difference;
- `ms_per_miss_token`: that done median divided by that first-COMPACT miss median.

**The old placement's reading is the one-compaction cost** (R9). Its predicted miss is the COMPACT prompt's tokens
after the first elided block: the `/tokenize` count minus the count of the same body cut just before the first
stub (its warm, steps 1–4 in full, is cached up to there). The **kept** placement is reported separately and is not
the one-compaction cost. Its COMPACT shares the three stubs and step 4 with the same history's old COMPACT, sent
earlier in the run. So its miss starts where the two COMPACT texts diverge (step 5's response), and its predicted
value is the count after that common prefix. This assumes the old COMPACT is still in the prefix cache when the kept
cell runs (only the kept warm and the kept FULL r0 run in between, a few thousand tokens against a pool of about 17k);
if it was evicted, the kept reading is high and says nothing about compaction.

**Accepted forms, a known cost.** For the ferry history (19:40), a bare `7:40` is not accepted, because it names the
morning time; so "7:40 in the evening" and "7.40pm" read as unsupported. This affects only the ferry history (10 of
each 40) and is pre-registered as such.

**Cache sanity values (pooled, top level of `cost`).** These are not the reading:
- `compact_miss_tokens_median`: the median miss over all 80 COMPACT requests, expected near zero;
- `first_commit_ms_diff_median` and `done_ms_diff_median`: medians over all 80 pairs, paired by (history,
  placement, repeat), and their `ms_per_miss_token`;
- `compact_miss_tokens_first_per_cell_median`: over both placements (n = 8);
- `predicted_miss_tokens_median`: over all 8 cells.

The eval rows re-derive the pooled and the per-placement values from `requests.jsonl` alone.

### Q4 and Q5

Report-only, as above. No gate.

## Limitations (recorded)

- The E histories are synthetic: short step texts written by the harness, not loops written by the model. T-0120b
  may re-check on recorded S6 loops if E is borderline.
- The step texts are the same after a step-1 result that answers the question as after one that does not.
- Form A's `arguments` and tool `content` are JSON strings. Whether vLLM parses them before rendering is the
  engine's business; the row measures form A as an OpenAI client would send it.

## Results

Not run yet. After the run, this section gets:
- the regime as recorded (commit, versions, boot);
- every k/n of `report.json`;
- the E outcome;
- the Q4 table;
- the Q5 readings;
- the cost rows per placement (old as the reading), with the predicted value beside the measured one, and the pooled
  sanity values.

## Results (run at c398fdf, 2026-10-18)

The placeholder above is part of the pre-registration and is left as it was. This section is written after the one
run, which was step 3 of `scripts/locked/t0115_loop_probe.sh` in the T-0115 `/launch`. Sources, in the
`t0115-launch` worktree:
- `logs/t0115a-loop-requests-c398fdf/`: `manifest.json`, `report.json`, `report.md`, `requests.jsonl`;
- `logs/t0115-loop-probe-c398fdf-run.log`, from line 271 on.

### Regime

As `manifest.json` `regime` states it:

| key | value |
|---|---|
| commit | `c398fdf628d8691fc80d14fd5055af53efff99cb` |
| base_url | `http://127.0.0.1:8001/v1` (loopback) |
| served_model | `nvidia/diffusiongemma-26B-A4B-it-NVFP4` |
| engine_version | `{"version":"0.29.0"}` |
| versions_lock_sha256 | `5bab0570362d68c841f82f8d56cd58ddf1dd3253a2d1208f3ea2fe7109246f25` |
| cases_sha256 / split | `460cd1c337dddf0cacc64cca149ec6af21a90899c42450ea5791fc3409afef85` / `dev` |
| tools / repair | `native:bbfc53ccdd80` / `on` |
| profile / canvas_length / max_denoising_steps / thinking | `lean` / 256 / 48 / `true` |
| seed / concurrency | `null` (unseeded) / 1 |
| options | `LlmOptions() as the session sends it` |
| requests | 263 (Q4 60, E 200 = 160 arm + 40 warm, Q5 3; all written, `complete: true`, `stopped: null`) |
| e_budget / compact | 7000 / `loop.compact(steps, E_BUDGET) at its defaults (keep_last 2, target budget * 2 // 3)` |
| warm | `the step-4 prompt before every repeat r with r % 2 == 0, per (history, placement)` |
| e_screen_case / q5_cases | `scroll-01` / `click-01`, `click-03`, `type-02` |
| accepted_match | `the complete pre-registered list per history (e_histories), case-insensitive, not inside a longer number or word (R2)` |
| thought_open | `the last commit's (the final state, R1)` |
| errored | `non-200, timeout, transport error or a stream without a finish_reason; never correct; >= 3/160 make E inconclusive (R4)` |
| q5_health | `up to 3 /health probes 1 s apart (R8)` |
| form_a | `assistant content = step 1's text, plus one tool_calls entry (R6)` |
| predicted_miss | `old: tokens after the first stub; kept: tokens after the divergence from the same history's old COMPACT (R3)` |
| gate files | `loop.py` `1bfc50b4…`, `llm_client.py` `8010b258…` |
| elided | steps {0, 1, 2} (steps 1–3) in all 8 (history, placement) cells, as pre-registered |

This was the same boot as T-0115: RTX 5090, driver 595.84, vLLM 0.29.0, NVFP4 rev `2ea8372`, one fresh boot healthy
at 00:13:21. The harness ran from 00:25:29 to 00:26:53, about 1.4 min, against the 4–5 min estimate. The engine
stopped at 00:26:54 with 0 containers left.

Below, **[Ra]** stands for this whole regime: lean, vLLM 0.29.0, NVFP4 rev `2ea8372`, chat template `36e3a42e`,
tokenizer `cc8d3a0c`, CL 256, 48 steps, thinking on, prefix caching on, `max_num_seqs 3`, RTX 5090 / driver
595.84, gpu-mem 0.86, c398fdf, tools `bbfc53ccdd80`, dev split, concurrency 1,
unseeded, no audio, one run.

### The pre-registered outcome

**E outcome: `inconclusive: FULL correct below 28/40 at the old placement`** (`report.json` `E.outcome`, which
continues "(the model does not use 5-step histories)"). [Ra], n = 40 per cell. The eval row re-derived the
tally, Q5 and the cost medians from `requests.jsonl` and agrees (run log: 6 passed). `report.json` sha256
`abe7a945…`.

| placement | arm | correct | re-search | unsupported | other | errored |
|---|---|---|---|---|---|---|
| old | FULL | 21/40 | 0/40 | 19/40 | 0/40 | 0/40 |
| old | COMPACT | 0/40 | 9/40 | 30/40 | 1/40 | 0/40 |
| kept | FULL | 30/40 | 0/40 | 10/40 | 0/40 | 0/40 |
| kept | COMPACT | 32/40 | 0/40 | 8/40 | 0/40 | 0/40 |

| bar | reading [Ra] | threshold | holds |
|---|---|---|---|
| EC (control) | old FULL 21/40, kept FULL 30/40 | FULL correct ≥ 28/40 at both placements | **no** (old) |
| E1 (kept, non-inferiority) | COMPACT 32/40 vs FULL 30/40 | COMPACT ≥ FULL − 4/40 (≥ 26/40) | yes |
| E2 (old, recovery) | COMPACT correct + re-search 0/40 + 9/40 = 9/40 | ≥ 32/40 | no |
| E3 (old, harm) | COMPACT unsupported 30/40 | ≤ 8/40 | no |
| errored E arm requests | 0/160 | inconclusive at ≥ 3/160 | not triggered |

EC failing makes E inconclusive under the rule, so E1–E3 are reported but do not decide anything. T-0120b is not
planned "as is" on this reading.

### Report-only

**Q4: form G vs form A at step 2** ([Ra], n = 30 per arm, 6 cases × 5 repeats):

| arm | final | call | thought_open | empty | errored | repeats start | done ms median |
|---|---|---|---|---|---|---|---|
| G | 22/30 | 8/30 | 0/30 | 0/30 | 0/30 | 12/30 | 306.85 |
| A | 12/30 | 0/30 | 1/30 | 17/30 | 0/30 | 2/30 | 167.45 |

Per case (final / call / thought_open / empty / repeats start, each out of 5):

| case | G | A |
|---|---|---|
| `search-01` | 5 / 0 / 0 / 0 / 0 | 4 / 0 / 1 / 0 / 0 |
| `search-02` | 5 / 0 / 0 / 0 / 0 | 5 / 0 / 0 / 0 / 0 |
| `click-01` | 3 / 2 / 0 / 0 / 1 | 0 / 0 / 0 / 5 / 0 |
| `click-03` | 1 / 4 / 0 / 0 / 4 | 0 / 0 / 0 / 5 / 0 |
| `type-02` | 3 / 2 / 0 / 0 / 2 | 0 / 0 / 0 / 5 / 0 |
| `key-01` | 5 / 0 / 0 / 0 / 5 | 3 / 0 / 0 / 2 / 2 |

On the two search cases the forms behave alike: G 10/10 final, A 9/10 final. After an action response, form A
returns an empty answer in 17/20 requests: all 15 on the three click/type cases and 2/5 on `key-01`. Form G
answers or calls on every request. Those empty answers lower A's done median, but they do not explain the whole
gap. A's 13 non-empty answers have a done median of 227.4 ms, against G's 306.85 ms over all 30. On the two search
cases alone it is 218.4 ms for A against 277.1 ms for G (n 10 each; [Ra]). G's cost is a higher
start-sentence repetition: 12/30, all on the action cases, 5/5 on `key-01` ("Undoing that."). The repetition
belongs to the speaking policy, not to the request form.

**Q5: an image part in a continued assistant message** ([Ra], n = 3):

| request | reading | status | `/health` after | health checks |
|---|---|---|---|---|
| `click-01` | rejected | 500 | 200 | 1 |
| `click-03` | rejected | 500 | 200 | 1 |
| `type-02` | rejected | 500 | 200 | 1 |

**Rejected 3/3** ([Ra]) for the body as sent, with the image part placed **after** the text in the continued
assistant message. The engine stayed up. Nothing more general is shown.

**Probable mechanism (not measured by this row).** `continue_final_message` cuts the rendered prompt at the end of
the last text block. That drops the trailing `<|image|>` placeholder, so vLLM receives two images for one
placeholder and asserts. The evidence:
- The pinned template (`36e3a42e`, `chat_template.jinja:331-332`) does emit `<|image|>` for assistant image parts.
- transformers' `chat_template_utils.py:544-593` truncates the rendered text at the end of the final message's
  content. This was read in a local checkout; the version inside the vLLM container is unverified.
- `logs/vllm.log` shows the 500s raised in `_matched_updates_from_result` with `AssertionError: Failed to apply
  prompt replacement for mm_items['image'][1]`. The second image, the one in the continued message, got no
  placeholder.

**Untested:** an image part placed before the text in the continued message, and a template-side fix.

**Cost per placement** (report-only, R3; [Ra]). Each placement's row comes from its 4 cells' first COMPACT after
the first warm, and their first pair. Old is the one-compaction cost:

| placement | first-COMPACT miss tokens, median (n; values) | predicted miss, median (per cell) | first-pair COMPACT − FULL first-commit ms | first-pair done ms | ms per miss token |
|---|---|---|---|---|---|
| **old** | 929.5 (4; 923, 927, 932, 1 230) | 874.5 (884, 875, 873, 874) | +0.95 | +0.95 | 0.001 |
| kept | 349.5 (4; 344, 349, 350, 358) | 333.5 (340, 340, 327, 323) | −26.75 | −26.80 | −0.077 |

One compaction misses about 930 tokens, near the predicted 874.5. The latency cost is about 1 ms, within noise at
n = 4. The kept row is close to its prediction from the divergence point, so the old COMPACT was still cached when
the kept cell ran.

Cache sanity values, pooled and expected near zero ([Ra]):
- median COMPACT miss 27.0 tokens (n 80);
- first COMPACT per cell over both placements 640.5 (n 8), against predicted 606.5;
- paired COMPACT − FULL first-commit −4.85 ms and done −4.75 ms (n 80 pairs);
- −0.176 ms per miss token.

### Exploratory re-read (NOT the pre-registered result)

**This section changes nothing above. The pre-registered E outcome is `inconclusive`, and it stands.** This is
a post-hoc reading. It was not fixed before the run, and it was chosen after seeing the answers.

**Why.** Many `unsupported` answers carry the fact, but spelled out as words: "seven forty p.m.", "six forty-five".
The model answers in speakable form, presumably because the system prompt makes it a voice assistant. The
pre-registered list
has word forms only for `twenty to eight`, `quarter to seven`, the `hall` count and `forty-one point six`. It has
no word form of the clock times. So every spelled-out ferry or market time read `unsupported`. The `hall` and
`lighthouse` histories, whose word forms were listed, read correct in 20/20 and 20/20 FULL requests.

**Method.** The recount used `uv run --no-sync python -c` over `requests.jsonl` and wrote no files.
1. Answer text: the `answer`-segment commits whose `thought_open` is false, joined and stripped.
2. Check: the pre-registered matcher, re-implemented this way, reproduces the harness's correct/unsupported reading
   on all 160 arm requests (0 mismatches).
3. Re-read: only `unsupported` lines may become `correct`. `re_search`, `other` and `errored` are left as they are.
4. Matching: casefold, hyphens to spaces, whitespace collapsed, with the R2 boundaries.
5. Added forms, reading **A**: `ferry` gets `seven forty p.m.`, `seven forty p.m` and `seven forty pm`, so
   "seven-forty p.m." also matches. `market` gets `six forty five`, so "six forty-five", "six-forty-five" and "six
   forty-five am" match. Bare "six forty-five" is accepted because unpadded `6:45` was pre-registered. Bare "seven
   forty" is still not accepted, as before.
6. Two boundary answers fall outside A and are shown separately:
   - **B**: ferry/old/FULL/r0, "seven forty in the evening". The pre-registration named "7:40 in the evening" as a
     known unsupported cost.
   - **C**: ferry/kept/FULL/r6, "seven-fort p.m.", a garbled word.

**The re-read tally** ([Ra], n = 40 per cell; correct / re-search / unsupported / other / errored):

| placement | arm | registered | re-read A (spelled-out forms) | A + B ("in the evening") | A + B + C (garbled "seven-fort") |
|---|---|---|---|---|---|
| old | FULL | 21 / 0 / 19 / 0 / 0 | **39** / 0 / 1 / 0 / 0 | 40 / 0 / 0 / 0 / 0 | 40 / 0 / 0 / 0 / 0 |
| old | COMPACT | 0 / 9 / 30 / 1 / 0 | **0** / 9 / 30 / 1 / 0 | 0 / 9 / 30 / 1 / 0 | 0 / 9 / 30 / 1 / 0 |
| kept | FULL | 30 / 0 / 10 / 0 / 0 | **39** / 0 / 1 / 0 / 0 | 39 / 0 / 1 / 0 / 0 | 40 / 0 / 0 / 0 / 0 |
| kept | COMPACT | 32 / 0 / 8 / 0 / 0 | **40** / 0 / 0 / 0 / 0 | 40 / 0 / 0 / 0 / 0 | 40 / 0 / 0 / 0 / 0 |

**Against the orchestrator's quick count.** The quick count was old FULL 39/40, kept FULL 40/40, kept COMPACT 40/40,
and old COMPACT 0 correct / 9 re-search / 30 "couldn't find" / 1 other.
- Old COMPACT and kept COMPACT agree under every reading.
- Old FULL 39/40 agrees with reading A, which leaves out "in the evening".
- **Kept FULL disagrees.** It is 39/40 under reading A, and 40/40 only if the garbled "seven-fort p.m." counts.
  No single reading gives both old FULL 39 and kept FULL 40: that pair excludes B and includes C.

Under reading A, every request in the old COMPACT cell checks out:
- all 30/30 `unsupported` answers say the fact was not found ("couldn't find", "unable to find", "cannot find",
  "haven't found"), and none states a wrong time or number;
- the 9 re-searches are ferry 8 and hall 1;
- the 1 `other` (ferry/old/COMPACT/r4) drafted a `google_search` call inside a thought that never closed.

Under reading A, the E bars would read:
- EC holds (39/40 and 39/40 ≥ 28/40);
- E1 holds (40/40 ≥ 39/40 − 4/40);
- E2 fails (9/40 < 32/40);
- E3 fails (30/40 > 8/40).

That would be "elision hurts: E2, E3". The old COMPACT cell has no spelled-out fact to recover, so E2 and E3 fail
under every reading. This is exploratory and is not a substitute outcome.

**Example answers per cell** (answer text verbatim, [Ra]):
- old FULL:
  - r1 ferry "The last Sunday ferry leaves Pier 9 for Halvard Island at seven-forty p.m." (registered `unsupported`;
    A `correct`);
  - r0 market "The Ostrava Lane farmers' market opens at six forty-five on Saturdays and stays open until one in the
    afternoon." (registered `unsupported`; A `correct`);
  - r0 ferry "The last ferry leaves Pier 9 for Halvard Island on Sunday at seven forty in the evening." (registered
    `unsupported`; A `unsupported`; B `correct`).
- old COMPACT:
  - r0 hall "I'm unable to find the exact number of seats in the main auditorium of Corvane Hall from the
    information available." (`unsupported`, honest);
  - r1 market "I couldn't find the exact opening time for the Ostrava Lane farmers' market on Saturdays, but parking
    nearby is free before nine on weekend mornings." (`unsupported`, honest; the "nine" comes from a pool snippet
    about parking);
  - r3 ferry "I couldn't find the specific time for the last Sunday ferry. I'll try to find the official ferry
    website to see the full timetable." then `google_search{query:"official Halvard Island ferry timetabl…"}`
    (`re_search`).
- kept FULL:
  - r0 ferry "The last ferry from Pier 9 to Halvard Island on Sunday leaves at seven forty p.m." (registered
    `unsupported`; A `correct`);
  - r4 market "The Ostrava Lane farmers' market opens at 6:45 am on Saturdays and stays open until 1:00 pm.The Ostrava
    Lane farmers' market opens at 6:45 am on Saturdays." (`correct` under both; the answer commits really are
    joined with no space);
  - r6 ferry "The last Sunday ferry from Pier 9 to Halvard Island leaves at seven-fort p.m." (registered and A
    `unsupported`; C `correct`).
- kept COMPACT:
  - r4 ferry "The last Sunday ferry from Pier 9 to Halvard Island leaves at seven forty p.m." (registered
    `unsupported`; A `correct`);
  - r3 market "The Ostrava Lane farmers' market opens at six forty-five am on Saturdays." (registered `unsupported`;
    A `correct`);
  - r0 lighthouse "The Tessaly Point lighthouse is 41.6 meters tall from the base to the vane, and it stands 63 meters
    above high water." (`correct` under both).

**What the re-read suggests (exploratory).**
- The model does use 5-step histories when the fact is present: FULL is 39/40 at both placements under reading A.
- When `compact` elides the result that held the fact, the model never invents the fact: none of the 30 unsupported
  answers states a value.
- But it mostly gives up: 30/40 answers say "I couldn't find it". It searches again in only 9/40 requests, and 8 of
  those 9 are in one history (ferry).
- The stub `{note:"earlier result omitted",results:5}` reads to the model as "there was nothing", not as "look
  again".

**Lesson.** The accepted lists for a voice assistant must include spelled-out number forms: clock times in words,
with and without "a.m."/"p.m.", and with hyphens or spaces. The model is told it speaks, and it writes numbers the
way they are said. A list that holds only digit forms, for any fact the model will voice, measures its formatting,
not its answer. Every future pre-registered accepted list should carry this.

**Implication, ruled by the human (2026-10-18).** The compaction stub should invite re-searching, and T-0120b
carries it. When `compact` elides a step, the model needs to be told that the earlier result exists and can be
fetched again by searching, rather than being left with a note it reads as an empty result.

### What this means

- **T-0115's loop is `ready`** (see `docs/cd/toolcall-loop-probe.md`, *Results*). It is fit for dogfooding, but
  turning `tool_loop` on by default is the human's call (H-T111a-1, H-T115-9).
- **Q5 rejected (3/3, HTTP 500, engine stayed up)** for an image part after the text in a continued assistant
  message, as sent. The probable cause is that `continue_final_message` truncation drops the trailing `<|image|>`
  placeholder (see Q5 above for the evidence). An image placed before the text, or a template-side fix, is untested.
  T-0111c should treat both as open, not as ruled out.
- **Form G is confirmed over form A.** The two match on search results. After action results, form A returns an
  empty answer in 17/20 requests, while form G answered or called on every request. The loop keeps the
  host-rendered continuation.
- **E is inconclusive under the pre-registered rule.** The exploratory re-read says elision as built hurts at the
  old placement (E2 and E3 fail under every reading) by making the model give up. It does not make the model invent
  the fact. One compaction costs about 930 cache-miss tokens and about 1 ms. Hence the ruled change to the stub in
  T-0120b.
