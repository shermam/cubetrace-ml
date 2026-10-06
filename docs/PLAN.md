# cubetrace-ml — plan

The model side of the dissertation ("fine-grained manual action recognition in video with sensor
supervision: the case of speedcubing"). The capture app (phase 1–4 of
[shermam/cubetrace](https://github.com/shermam/cubetrace), `0.4.0`) records solves with one or more
cameras and the Bluetooth cube's move stream on one clock; this repository turns those recordings into
a dataset, features and models, and reports how well a model reproduces the move stream from the video.

## Decisions (from the research notes, 2026-09/10, restated)

- **The recipe**: a frozen pretrained frame encoder, its per-frame features cached once; a small
  temporal model on the features; two heads tried in this order: (a) per-frame classification with peak
  picking (timing comes out directly; no collapse rule), (b) CTC over the normalized alphabet with an
  onset token; a decoder with the cube-notation prior and the state-consistency check later.
- **The alphabet**: the 24-symbol normalization of the notes (quarter turns, doubles as one symbol,
  slices as one symbol); a per-frame target is "no onset" or the symbol whose onset is nearest within a
  tolerance.
- **Labels on the host clock**: a move's time is the attempt's clock fit `a·cubeMs + b` (its `hostMs`
  is the packet's arrival, 14–34 ms of jitter; `--time-base arrival` uses it instead) and a
  clip's frame times (`t0HostMs` + cumulative `dtMs`); a camera's frames lag the cube by the clip's
  `syncResidualMs` (its session's clapperboard; null when the camera had no check: the clip is then
  usable with an unknown lag, flagged).
- **Split by session**, never by attempt: whole recording days held out; at least one lighting and one
  camera held out when the data allow.
- **Metrics**: WER on the move sequence; F1 of onsets within ±25 and ±50 ms; solve-level exact match
  (the predicted sequence replays to solved); per-frame loss early on; everything bucketed by TPS.
- **Data**: ~500 solves gives a first working model in the owner's own setup, ~2,000 a robust one. On
  the evening of 2026-10-05 `cubetrace-ml report` on the bucket counted 9 sessions, 581 attempts (463
  with video), 1,221 clips, 6.2 hours of video and 2.9 of solving, 66,578 quarter turns (56,667
  symbols), TPS median 4.55; the owner records 60–190 attempts on a solving day. At 30 fps and 4.5
  moves a second a move spans about 7 frames.
- **Compute**: the bucket is in us-central1; features and training run on a GPU there (a T4 or an L4)
  once the billing account allows GPUs, or on a free notebook GPU with the features pulled once;
  the dataset tooling and the evaluation need no GPU.

## Phase M board

| Task | Scope | Depends on | Status |
|---|---|---|---|
| M0 | `cubetrace_ml`: the dataset over a local mirror or the bucket; records validated; the per-frame label track per clip (frame host times, the lag, the move onsets, the phase, the gyro); the alphabet normalization; the consistency filter; splits by session; the manifest and its report; a visual check | – | ✅ #1 (5180d37) |
| M1 | the frozen encoder's features per clip, cached (local or bucket), with the decode/crop/resize path and its throughput measured | M0 | ✅ #2 (186c047); the GPU run pending (g) |
| M2 | the first models on the cached features (per-frame + peak picking; CTC), the evaluation report by TPS bucket on a held-out session, the baseline numbers | M1 | ⬜ |

### M0 — the dataset tooling

**Goal.** From a dataset root, every attempt's clips with a per-frame label track, ready for M1, and a
manifest that says what the dataset holds.

**Scope.** The package `cubetrace_ml` (src layout, `uv`, `ruff`, `pytest`, the `cubetrace-ml` command,
CI). (1) `dataset`: a root that is a local folder mirroring the bucket's layout or a `gs://` prefix
(`google-cloud-storage`, Application Default Credentials; the local path is what the tests and the
agents use); listing sessions and attempts; reading `session.json`, `attempt.json`, a clip's
`frames.json` and `gyro.json` validated against `schemas/` (`jsonschema`), with a small on-disk cache
for the bucket's JSON. (2) `moves`: the alphabet normalization (quarter turns kept, a double turn =
two same-face quarter turns within 200 ms merged into one symbol, a slice = two opposite-face turns
reported together merged into one symbol; the reference is `normalizar()` in the owner's `cubo.py`,
which the coordinator provides), with the onset time of a merged symbol being its first turn's. (3)
`align`: for one clip, the frame host times (`t0HostMs` + cumulative `dtMs`), the lag applied
(`syncResidualMs`; a clip without a check keeps `lag = null` and is flagged `unsynced`), the window of
the clip (the segment's events: scramble `scrambleStart` to `scrambleDone`, solve `solveStart` to
`solveEnd`), the moves of the segment, and the per-frame track: for each frame, the nearest onset's
symbol and its signed distance in ms, the phase (`scramble`, `inspection`, `solve`, `after`), and the
gyro's quaternion interpolated to the frame when `gyro.json` is there. (4) `filter`: the consistency
filter from `result.replayOk` and the record's `status`, a clip's `truncatedStart`, and a check that the
frames file's count matches the video's frame count (PyAV: read the stream's frame count or decode the
tail; measured, not assumed). (5) `splits`: a deterministic split by session (train/val/test with a seed
and a held-out day), written to the manifest. (6) `manifest`: one row per clip (session, day, attempt,
camera, segment, fps, frames, seconds, width, height, lag, moves in the window, TPS, replayOk, gyro
rate, split) as parquet and CSV, and a `report` command that prints the counts (attempts, clips, hours,
moves, by camera and day, the TPS histogram) — the answer to "how much do we have". (7) `inspect`: a
contact sheet (PNG) of a clip's frames around a few onsets with the symbol and the distance drawn, for
the eye; and a `check-alignment` command that reports, per clip, the frames file against the decoded
video (count, duration) and the share of moves inside the window.

**Acceptance.** Unit tests on synthetic fixtures (records generated in the tests; a tiny video written
with PyAV) for the alignment math, the normalization (doubles, slices, the onset of a merged symbol),
the lag, the splits' determinism and the manifest; `cubetrace-ml report` and `cubetrace-ml inspect`
run on the coordinator's local mirror of a few real attempts (the coordinator runs them and pastes the
numbers into the Outcome note); CI green.

**Outcome (M0).** The package `cubetrace_ml` (`records`, `store`, `dataset`, `moves`, `align`, `video`,
`filter`, `checks`, `splits`, `manifest`, `contact_sheet`, `cli`) and the `cubetrace-ml` command
(`report`, `manifest`, `splits`, `inspect`, `check-alignment`, `validate`), with CI (ruff and pytest on
Python 3.11 and 3.12) and `docs/DATA.md`, which states every rule below; 69 tests on synthetic records
(a factory the schemas check) and 24-frame MP4s written with PyAV, the bucket through an in-memory
stand-in for its client.

*Decisions.* (1) **The time base**: a move's `hostMs` is its Bluetooth packet's arrival (its residuals
against the attempt's fit are exactly the record's `residualP95Ms`, 14–34 ms on the mirror), so a move's
time is by default the fit, `a·cubeMs + b` ("the cube's clock through the attempt's fit" of the
decisions above), and its `hostMs` when it is off the fit's clock or the fit is not a line through one
clock; `--time-base arrival` uses `hostMs` throughout (the onset `hostMs + syncResidualMs`).
The clapperboard's `offsetMs` was measured against `hostMs` (the app's `sync-run.ts` logs `event.hostMs`),
and the fit is the least-squares line through those arrivals, so the lag holds for both. The events that
are moves (`scrambleStart`, `scrambleDone`, `solveStart`, `solveEnd`) take their moves' times. (2) **The
lag**: a move's onset on the frames is its time plus `syncResidualMs`; a frame at `tMs` shows `tMs − lag`
(the phase, the window and the gyro are taken there, as the app's clip viewer does); an unsynced clip
keeps `lagMs` null and is computed at 0. (3) **The normalization** follows `normalizar()` with its
defaults, a slice under 20 ms and a double under **200 ms** (both are flags), and never merges across
the scramble and the solve. (4) The phase is the attempt's
(`before`, `scramble`, `inspection`, `solve`, `after`), not the clip's: a solve clip's lead-in is the
inspection, sometimes the scramble's end. (5) The nearest onset is taken among all the attempt's
symbols, the earlier on a tie. (6) The gyro is slerped along the shorter arc, NaN outside its samples.
(7) The filter's attempt reasons (`dnf`, `replay-failed`) exclude both clips of the attempt; an unsynced
clip is usable; `moves-outside-clip` is added. (8) The split's day is the UTC date of `createdMs`, or of
the earliest `scrambleShown` for a session without `session.json`. (9) The manifest is written with
polars (parquet and CSV), with columns beyond the contract (`video`, measured `fps`, the three frame
counts, the crop, `movesCovered`, `status`, `usable`, `reasons`), an attempt table and `manifest.json`.
(10) `check-alignment` decodes every frame (`--fast`: the container's header); the header probe skips
FFmpeg's stream probe (2 ms a clip instead of 40, the same counts), so `report` and `manifest` measure
the counts on a local root and not on a bucket root, where it would download every MP4 (`--video`).
(11) `google-cloud-storage` is the optional extra `gcs`: about fifteen packages only the coordinator
needs; one listing per command, files cached by generation, MP4s downloaded when a command needs them.

*The mirror* (9 attempts, 3 sessions, 24 clips): `validate` 44 records, 0 errors, 1 warning (a session
folder without `session.json`); `report` 3 sessions (1 without `session.json`), 9 attempts (9 solved, 9
with video), 24 clips, 0.12 h of video and 0.05 h of solving, 1,026 quarter turns (733 in solves) and
893 symbols (682 in solves), 24/24 usable, 12/24 unsynced (every phone clip), the phones at 30 fps
measured for 60 nominal, the laptop's lag 53.1 or 430.1 ms; split: 2026-10-05 test, the two sessions of
2026-10-03 train and val; `check-alignment` 24/24: the frames file, the record, the container and the
decode agree on every count, the presentation times match the frames files to 0.01 ms, every onset
falls inside its clip (margins 2.5–4.3 s before the window, 0.5–1.5 s after); no move off the fit; no
slice in 893 symbols, but one `R` `L'` pair arrived in one packet (0 ms apart by `hostMs`) 32 ms apart on
the cube's clock: the time base decides whether it is an `M`. `inspect`'s sheets show the turns around
the outlined frames. Counted rather than eyed (the frame differences in the crop, averaged around the
solves' onsets, three clips per camera and lag): the laptop at 430.1 ms peaks 20–40 ms before
`onset + lag`, so that lag is real and its sign right (the other sign would put the peak 860 ms away);
the laptop at 53.1 ms peaks 60–100 ms after it, so that session's clapperboard looks about 80 ms short;
the unsynced phones, at 0, peak 20–40 ms after the onset.

*The bucket* (the coordinator, 2026-10-05 23:00 UTC, `report` on `gs://cubetrace-data/users/<uid>`,
247 s cold with 1,811 JSON files cached, no MP4 downloaded): 9 sessions, 581 attempts (581 solved, 463
with video), 1,221 clips, 6.20 h of video, 2.92 h of solving, 66,578 quarter turns (47,380 in solves),
56,667 symbols (43,063 in solves); 1,220/1,221 usable (one `moves-outside-clip`); 673 unsynced (the
662 phone clips and 11 laptop clips); the laptop at 29.97 fps measured, the phones at 28.9–30.0 for 60
nominal, lags 53.1–430.1 ms; by day 09-27 116 attempts and 11 clips (0.1.0, no video), 09-30 191 and
366, 10-02 110 and 220, 10-03 64 and 224 (two sessions), 10-05 100 and 400; TPS median 4.55 (n=581,
2.5–6.5); 9,818 moves placed by their arrival, all of them in 86 attempts of 2026-09-27 (0.1.0, before
the attempt's fit; one of them has video): in the attempts with video 114 of 53,022 moves. The laptop's
lags by clip: 53.1 ms (200 clips), 61.4 (118), 109.8 (220), 430.1 (10), none (11); 148 attempts have two
cameras. The one unusable clip is a 165-move scramble whose clip ends after 125 of them. The default
split puts 2026-10-05 (400 clips, 2.0 h) in `test` and the two 2026-09-27 sessions without clips in
`val`: the rule has to weigh clips, follow-up (f).

*Limits.* One lag per clip, no drift within it; the day is a UTC date (a session after 21:00 in Brazil
falls on the next day); the reference's slice table had `E` and `E'` swapped against the usual notation
(`E` turns as `D`, so `U`+`D'` is `E`; its `M` and `S` agreed): `SLICES` follows the usual notation and
the reference was corrected (review, 2026-10-05); the bucket was not read by the agent (no credentials).

*Follow-ups.* The double threshold (200 ms, the reference's) and the `E` direction were settled at the
review; the rest is under "Phase M follow-ups" below, (a)–(e).

### M1 — the features

**Goal.** Per-frame features of every clip, cached, so that M2 trains in minutes.

**Scope.** Decode with PyAV, crop to the clip's `crop` rectangle when there is one (else the full frame),
resize to the encoder's input, run a frozen encoder (a DINOv2 ViT-S/14 and a ResNet-18 as the two
candidates; the choice recorded with its throughput), write one array per clip (frames × dim, float16)
with the frame host times beside it, to a local features root or the bucket; resumable; a `features`
command with a manifest filter; the throughput measured on CPU and on the GPU used.

**Decisions (the coordinator's brief, 2026-10-05).** (1) Encoders by name in `encoders.py`, each with
its input size, normalization and output dim: `stub` (no weights: a seeded random projection of the
resized gray frame, dim 64; the tests' encoder), `resnet18` (torchvision's ImageNet weights, the 512-dim
pool, input 224), `dinov2-vits14` (ViT-S/14, the CLS token and the patch tokens' mean, 768, input 224;
timm or torch.hub, whichever works, recorded); the weights fetched on first use only, the tests never
downloading. (2) `torch` and `torchvision` (and `timm`) in an optional extra `features`, CPU wheels by
default through uv's PyTorch pattern, a CUDA build by another extra; `uv.lock` committed; CI under about
five minutes. (3) The framing (follow-up (e)): the record's `crop` when there is one, else a square from
the motion of the segment's window (gray at about 160 pixels, the frame differences, a light blur, 15% of
the peak, the 5th–95th percentiles of the marginals, 15% padding, square, clamped), recorded in the
features' meta, drawn by `crop-preview`; `--crop record|auto|none`; a record crop that is not square
letterboxed. (4) One `.npz` per clip under `<out>/<encoder>/<sessionId>/<nnnn>/<camera>.<segment>.npz`
(`x` float16, `tMs`, `shownMs`, `inWindow`, `meta`), every frame encoded; resumable by encoder, crop mode
and frame count, `--force`, written under a temporary name. (5) The selection from a manifest or the
root, filtered by split, session, camera, segment, usable (on by default) and a limit; `--batch`,
`--workers`, `--device`. (6) The throughput measured stage by stage (`bench`) on this machine's CPU and on
the mirror; the GPU's is the coordinator's. (7) The split rule of follow-up (f). (8) The docs: `DATA.md`'s
"The features" and this Outcome note.

**Acceptance.** Unit tests on synthetic records and PyAV-written videos (the stub's determinism and shape,
the resize and letterbox, the auto crop of a moving blob, the npz layout and meta, resume, the manifest
filters, the new split rule); ruff, pytest and CI green with the features extra installed (CPU); a smoke
run on the mirror: `features --encoder stub` on every clip, the real encoders on two clips if their
weights download, `crop-preview` on a phone clip and a laptop clip, the throughput numbers.

**Outcome (M1).** The modules `encoders`, `framing` and `features` (and the filter-graph decoding in
`video`), the commands `features`, `bench` and `crop-preview`, the split rule of (f) in `splits`,
`scripts/check_encoders.py`, and `docs/DATA.md`'s "The features", which states every rule below; 103
tests (34 new) on synthetic records and PyAV-written videos (a blob going round a circle for the motion
crop), none of which downloads anything.

*Decisions.* (1) **The extras**: `features` (torch, torchvision, timm) takes torch and torchvision from
PyTorch's CPU index on Linux and Windows, `cu128` from its CUDA 12.8 index (uv's pattern: sources keyed
by extra, explicit indexes, the two extras declared conflicting). The lock has torch 2.14.1+cpu and
torchvision 0.29.1+cpu for `features`, torch 2.11.0+cu128 and torchvision 0.26.0+cu128 for `cu128` (2.11
is the last release with a CUDA 12.8 build; PyPI's own Linux torch is now the CUDA 13 build), timm
1.0.30. CI: the checks (3.11 and 3.12) sync `--extra gcs`, since `--all-extras` would ask for both
exclusive extras; a `features` job (3.12) installs the CPU build, kept in setup-uv's cache under its own
suffix and unpruned, and runs every test (22–30 s); a manual `weights` job runs
`scripts/check_encoders.py`. (2) **DINOv2 through timm** (`vit_small_patch14_dinov2.lvd142m`,
`img_size=224` with `dynamic_img_size=True`: the position embeddings resampled once, at load, to 16 × 16
patches), not torch.hub: a locked dependency instead of code fetched from GitHub at run time. Its vector is
the final norm's CLS token and the mean of the patch tokens; ResNet-18 is torchvision's `IMAGENET1K_V1`
with `fc` as the identity. Both take the ImageNet mean and std, which the tests check against timm's and
torchvision's own configurations. (3) **Precision**: fp32 on the CPU, fp16 autocast on CUDA by default
(`--precision`); the features are stored as float16 either way. (4) **The decode path**: the cut, the
scale (area) and the letterbox run in an FFmpeg filter graph on the decoder's YUV frames (`crop` with
`exact=1`, `scale`, `format=rgb24`, `pad`): 2.5 times numpy and PIL on full RGB frames, about as fast as
the decode itself. The laptop's record crop is cut as recorded, 816 × 703, scaled to 224 × 193 and
letterboxed (15 black rows above, 16 below). (5) **The motion crop** is the brief's, with a floor of a
quarter of the frame's shorter side, even pixels, and the whole frame when nothing moves; its squares are
cached under `<out>/crops/` so a second encoder does not decode a phone clip twice; `crop-preview` also
finds the motion's square on a clip with a record crop, to compare them. (6) **Resume** by encoder, crop
mode and frame count, as asked: the time base is in `meta` but not in the rule (it moves `inWindow` by
milliseconds; M2 aligns its labels from the records). (7) **The splits** (f): `val` takes whole sessions in
the seeded shuffle's order whenever one brings its clips closer to 15% of all the clips (at least one,
never all), rather than "until the share is reached", which on the bucket can take a 400-clip session at
once (33%); the test day's tie goes to the later day; the counts are the manifest's rows, usable or not,
so the split does not move with `--video`. On the bucket's day counts (M0's Outcome) the held-out day
becomes 2026-10-03 (224 of 1,221 clips, 18%) instead of 2026-10-05 (400, 33%); val's sessions depend on
their sizes, which `cubetrace-ml splits` on the bucket prints. On the mirror 2026-10-05 and 2026-10-03
(12 clips each) tie for the test day: the later one, as before. (8) **The features root is a folder**
(`--out`, `$CUBETRACE_FEATURES`); the bucket takes it with `gcloud storage rsync`.

*The mirror* (24 clips, 13,288 frames; 4 vCPUs, an Intel Xeon at 2.8 GHz; `--workers 1`, batch 64; the
PyTorch encoders with random weights, the same compute as their own):

| Stage | Frames per second |
|---|---|
| decode alone (PyAV, `thread_type` AUTO) | 306 (laptop, 1920 × 1080) to 370 (phone, 1080 × 1920) |
| motion crop pass (12 phone clips, 6,672 frames) | 393 |
| decode, cut and scale to 224 (`bench --encoders none`) | 296 (214 overall, with the motion pass) |
| the same with `--workers 2` | 294 overall |
| decode, cut and scale to 32, and `stub` | 392 overall (482 with `--workers 2`); the stub alone 11,410 |
| `resnet18`, CPU fp32 (the decode beside it: 182) | 56 (55 overall) |
| `dinov2-vits14`, CPU fp32 (the decode beside it: 178) | 20 (20 overall) |
| GPU | pending (the coordinator) |

`features --encoder stub` on the 24 clips wrote them in 52.3 s (254 fps overall: one worker runs the
motion pass, then the decode); the second run skipped all 24 in under a second. At these rates the
bucket's 6.2 h of video (about 670,000 frames) would take 3.4 h with ResNet-18 and 9.3 h with DINOv2 on
this CPU: the GPU is for the real run, where the decode (300 to 400 fps here, more with `--workers` and
cores) sets the pace.

*The weights.* This container's egress policy blocks download.pytorch.org (the CPU and CUDA wheels, and
torchvision's weights), huggingface.co (timm's) and dl.fbaipublicfiles.com (torch.hub's DINOv2): `uv sync
--extra features` fails here, and `features --encoder resnet18 --limit 2` (and `dinov2-vits14`) exits 2
with "could not load resnet18's weights (torchvision ResNet18_Weights.IMAGENET1K_V1 (download.pytorch.org)):
<urlopen error Tunnel connection failed: 403 Forbidden>" ("403 Forbidden" for DINOv2). The lock was
resolved by a temporary CI job that committed it (e7996fd); the local runs used PyPI's torch 2.14.1 (the
CUDA 13 build, 5.7 GB, on the CPU) in a scratch environment. The pretrained path is verified on CI's
runner instead: the manual `weights` job (run 37389930611, at 0ef8e85) downloaded both (ResNet-18's
44.7 MB, DINOv2's from Hugging Face) and wrote the synthetic test split's features through
`cubetrace-ml features`: (24, 512) and (24, 768) float16, finite, `loaded: pretrained`, frames that
differ.

*The crops on the mirror.* `auto` takes the laptop's record rectangle on its 12 clips. The 12 phone clips'
motion squares are all 1080 × 1080, the frame's whole width: before its padding the motion's box spans
817 to 958 of the 1,080 pixels across (the hands reach from edge to edge) and 676 to 1,136 down; padded,
it is wider than the frame, so the square is the full width placed on the motion's vertical centre, y 538
to 586 on the rear phone (the cube and both hands; the monitor above and the desk below cut off) and 278
to 444 on the front phone (the cube, the hands and the chest; the face above the square but for the
chin). In the frames looked at, the cube spans 85 to 105 of the encoder's 224 pixels there, against
about 140 in the laptop's rectangle. The motion squares on the laptop's clips (for comparison only) are
capped too, 1080 × 1080 at x 126 to 356; the record's rectangle is tighter, and in two of the frames
looked at it cut an edge of the cube (it moves during a solve).

*Limits.* The phones' squares are the whole width: a tighter framing needs to find the cube itself. The
motion crop costs each phone clip a second decode (once: the squares are cached). The decode is the
bottleneck for the stub and for any GPU encoder: 300 to 400 fps on these 4 vCPUs with one worker, and
the decode alone went from 214 to 294 fps overall with two, so the GPU machine's cores and `--workers`
set its pace. A clip's frames are held in memory at the input size while it waits for the encoder
(about 150 kB a frame at 224). The PyTorch packages in the lock cannot be re-resolved from the agents'
container; the weights were never loaded on this machine, only on CI's.

*Follow-ups.* (g) to (j) below.

### M2 — the first models

**Goal.** The first numbers: WER, F1@±25/±50 ms and exact replay by TPS bucket on a held-out session,
for the per-frame + peak-picking head and the CTC head, against a trivial baseline.

**Scope.** A temporal model (1D conv + BiLSTM or a small transformer) on the cached features; the two
heads; training with a config file and a seed; the evaluation report as Markdown with the plots; the
decode with the state-consistency check as a flag; the fps ablation (30 → 15 fps by dropping frames)
as a first curve.

## Phase M follow-ups

(a) A per-clip lag estimated from the video (the motion around the onsets, as M0's review counted it),
to audit the clapperboard's lags (one session's 53.1 ms looked about 80 ms short) and to give the
unsynced clips one. (b) The slice threshold checked on the cube's clock once the data have slices (one
`R` `L'` pair arrived in one packet 32 ms apart on the cube's clock: two turns at 20 ms). (c) The
scramble clips of a DNF or a failed replay recovered from the resyncs' states. (d) A held-out camera and
lighting in the splits when the data allow. (e) The phones' clips have no `crop` (the whole frame, the
cube small in it): M1 needs a framing for them (a fixed rectangle per camera, or a detector) — done in M1
(a square from the motion; see (h)). (f) The splits weigh clips, not sessions: a session without clips
joins no split's pool (the bucket's two 2026-09-27 sessions made an empty `val`), `val` targets a share of
the clips, and the held-out day is chosen so that `test` is about a fifth of the hours unless
`--held-out-day` says otherwise — done in M1 (by clips, not hours). (g) The GPU's throughput and the
bucket's features (the coordinator): `uv sync --extra cu128`, `scripts/check_encoders.py --device cuda`,
then `features --encoder dinov2-vits14` and `--encoder resnet18` on the bucket's root, the folder synced
to the bucket. (h) A tighter framing of the phones (a cube or hand detector, or the app's `crop` on the
phones too), and whether the laptop's record rectangle, which can cut the cube's edge, should be widened
to the motion's square. (i) A `cu130` extra (torch 2.14 on CUDA 13, a driver of 580 or later) if the GPU
machine's driver allows it: `cu128` stops at torch 2.11. (j) A change to the PyTorch packages of the lock
is resolved on a runner (or a machine that reaches download.pytorch.org), as e7996fd was.
