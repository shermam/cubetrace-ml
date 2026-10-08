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
| M2 | the first models on the cached features (per-frame + peak picking; CTC), the evaluation report by TPS bucket on a held-out session, the baseline numbers | M1 | ✅ #3 (24df2e7); the first real numbers below (2026-10-07) |
| M3 | the cube's orientation as an input: the gyro's quaternion (and its change) per frame beside the features, a controlled comparison on the clips that have a gyro, the confusion analysis in the report | M2, the features of (g) | ✅ #4 (c12c16e); the real comparison runs after the first chain |
| M4 | the orientation in the camera's frame: a rotation per attempt (or session) between the gyro's frame and the camera's, learnt on the training attempts and estimated for a test attempt from its scramble's known moves; the oracle bound; the gravity diagnostic | M3 | ✅ #5 (69bba2b); the real runs follow |

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
| GPU: one L4 (`g2-standard-8`, 8 vCPUs, us-west1-a, 2026-10-07), the whole bucket, 6 workers | DINOv2 843 fps and ResNet-18 1,606 fps on the GPU in fp16; the decode 96–109 fps per worker; the motion pass 100 fps per worker over the 682 phone clips; overall 352 fps for DINOv2 (1,240 clips, 649,792 frames, 31 min, the motion pass included) and 544 fps for ResNet-18 (20 min, the crops cached) |

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

**Goal.** The first numbers: WER, F1@±25/±50 ms and exact replay by TPS bucket on the held-out split,
for the per-frame + peak-picking head and the CTC head, against a trivial baseline, from the cached
features of M1.

**Scope.** (1) `labels`: for one clip, from the records through M0's `align_clip`, the frames kept (the
segment's window plus a margin, 15 frames by default, on each side), the reference symbol sequence of
the window with each onset's frame time (`onset + lag`), and the per-frame target: the frame nearest to
each onset carries the symbol's class (1–24; 0 is "no onset"), with an optional soft target on its
neighbours (`--label-frames`); clips whose moves fall outside the frames, and unusable clips, are
skipped; loaded once into memory from a features root plus a manifest (`--split`). (2) `models`: the
features (T × D, float16 read as float32) through a linear projection (256), two 1D convolutions
(kernel 5), a two-layer BiGRU (128 per direction) and a head; a small transformer encoder (4 layers,
256, 4 heads) as the alternative body (`--body`); two heads: `perframe` (25 classes, cross-entropy with a
weight on the onset classes, peak picking on 1 − P(no onset) with a minimum distance of 2 frames and a
threshold chosen on `val` to maximize F1@50) and `ctc` (24 symbols + blank, greedy decoding, the spike
frames as its onset times). (3) `train`: `cubetrace-ml train --config <toml> --features <root>
--encoder <name> --root <dataset> --manifest <parquet> --out runs/<name>`, with AdamW, a cosine schedule,
dropout, time masking as augmentation, batches of whole clips padded with masks, a seed, early stopping
on `val`, a CSV log and checkpoints; CPU and CUDA. (4) `evaluate`: `cubetrace-ml evaluate --run runs/
<name> --split test`: WER (edit distance over the symbol sequence, per clip, aggregated by split, by
segment and by TPS bucket of 0.5), onset F1 at ±25 and ±50 ms (one-to-one greedy matching within the
tolerance; `timing` ignores the symbol, `symbol` requires it), exact match (WER 0) and **replay** (the
predicted solve sequence applied to the record's `scrambledFacelets` leaves every face one colour; a
facelet simulator with the owner's `cubo.py` as the reference, slices expanded to their primitive pair),
each against the baseline (onsets at the peaks of the feature-difference norm, the training set's most
frequent symbol); `report.md` with the tables and PNG plots (the loss curves, F1 against the tolerance,
WER by TPS bucket, one clip's P(onset) over its reference onsets). (5) The fps ablation: 30 → 15 fps by
dropping every other frame at training and test, the same report. (6) A `--consistency` decode flag that
merges adjacent same-face predictions by the normalization rules and drops immediate cancellations, with
its effect in the report. (7) `docs/DATA.md` gains "The labels and the runs"; the Outcome note carries the
numbers of whatever features exist when the task runs (the mirror's stub features at least) and the
commands for the real run.

**Acceptance.** Unit tests: the labels (the kept frames, the nearest-frame target, the margin, the soft
target), WER and the F1 matching on known cases, the replay on `cubo.py`'s own cases (a scramble and its
inverse, the T-perm twice), peak picking, CTC greedy decoding, the baseline, and a training smoke test on
synthetic features with a planted onset signal where the model beats the baseline in a few CPU epochs;
ruff, pytest and CI green (the training tests in the `features` CI job); `train` and `evaluate` run end
to end on the mirror's stub features (the coordinator's local run).

**Outcome (M2).** The modules `cube` (a facelet simulator), `labels`, `metrics`, `decode`, `config`,
`models`, `train`, `evaluate` and `report`, the commands `train` and `evaluate`, the example runs in
`configs/` (`perframe-bigru`, `ctc-bigru`, `perframe-transformer`), matplotlib among the base
dependencies, and `docs/DATA.md`'s "The labels and the runs", which states every rule below; 158 tests
(55 new) on synthetic records, the models' (`test_train.py`, 11) on synthetic features with a planted
onset signal: they need PyTorch and run in the `features` CI job, whose whole pytest step took 29 s on
the runner.

*Decisions.* (1) **The kept frames** are the segment's window and the frames nearest its first and last
onsets (a window shorter than a frame interval has no frame of its own: two turns 9 ms apart), plus the
margin, counted in the clip's frames before the stride of `--fps`. **The reference** is every onset
inside the kept frames, the other segment's included (counted as `foreign`; none on the mirror), so that
the targets, CTC's sequence and the scores see the same moves. (2) **Collisions**: an onset whose frame is
taken goes to the free neighbour nearer its time, else to no frame (counted; it stays in the sequence):
none on the mirror at 30 or at 15 fps. (3) **The soft target** (`--label-frames`, 0 by default) puts
`soft_decay ** d` on the onset's class and the rest on "no onset"; the loss divides by the frames' total
weight, which for hard targets is PyTorch's weighted mean (tested). (4) **The network**: the two
convolutions are residual with a layer norm; the BiGRU runs packed to the clips' lengths; the transformer
is pre-norm with sinusoidal positions and the padding masked; the input's mean and deviation are buffers
of the state dict, so a checkpoint carries its normalization; the padding changes no output (tested for
both bodies and heads). (5) **Selection on val** uses the pooled F1@50 (symbol), steadier than the mean
over a few clips, and CTC the pooled WER; the per-frame threshold is chosen after every epoch (the one
nearest 0.5 among equal F1s), and each checkpoint keeps its own epoch's. (6) **The baseline** also needs a
threshold: chosen on val by F1@50 on timing (its one symbol cannot choose it); and a shift, the training
clips' median offset from a reference onset to the nearest motion peak within 100 ms (+15.0 ms on the
mirror's stub features, +25.6 at 15 fps), so that the motion's delay alone does not fail it at ±25 ms.
(7) **The matching** takes, for each prediction in time order, the earliest unmatched reference onset
within reach: a maximum matching when every onset has the same reach (the nearest-first choice is not),
per symbol for `symbol`. (8) **The consistency pass** merges first, then drops cancellations as a stack
(nested ones too; `R R R'` within 200 ms is `R2 R'`); its slice rule cannot fire at 30 fps (two peaks are
at least 67 ms apart) but can for CTC. (9) **The run folder** adds `metrics.json` (every aggregate) and,
when the run builds its manifest, `manifest/`, so that `evaluate` reads the same splits; a split other than
test writes `report-<split>.md` and the like; `--force` clears the old run's files. The plotted clip is
the test solve clip of the median WER (its first 8 s), named by attempt, camera and segment, not by
session. (10) **matplotlib** is a base dependency, not the `features` extra: the report needs no PyTorch,
so its tests run in every job. The lock was resolved on a runner by a temporary workflow (388e669, removed
in 8366f32): it added matplotlib 3.11.2 and its seven dependencies, nothing else. (11) **The seeds**:
PyTorch's, `random`'s, and a NumPy generator of the seed for the clips' order and the time masks (the
legacy global NumPy seed is not set: ruff's NPY002); two CPU runs with one seed give identical weights
(tested).

*The synthetic result* (the tests' data: 8 sessions of 2 attempts, each a solve of 16 random quarter turns
with 20% doubles and its inverse as the scramble, one laptop camera; features of dimension 16, noise of
deviation 0.3 plus each onset's symbol embedding of norm 2 fading over 4 frames from the frame nearest
it; 24 train, 4 val and 4 test clips, 3,484 training frames; this machine's 4 vCPUs; the test split,
pooled):

| run | epochs (best) | seconds | WER | F1@25 symbol | F1@50 symbol | F1@50 timing | exact | replay |
|---|---|---|---|---|---|---|---|---|
| baseline (every run's) | – | – | 1.213 | 0.057 | 0.057 | 0.757 | 0% | 0% |
| per-frame BiGRU, the test's (width 64, GRU 32, batch 4, lr 3e-3) | 8 (7) | 11.6 | 0.180 | 0.835 | 0.882 | 0.945 | 0% | 0% |
| per-frame BiGRU, default size, batch 4 | 60 (11) | 117.5 | 0.016 | 0.992 | 0.992 | 0.992 | 75% | 50% |
| CTC BiGRU, default size, batch 4 | 60 (34) | 119.8 | 0.016 | 0.959 | 0.992 | 0.992 | 75% | 100% |
| per-frame transformer (width 64, ff 128) | 8 (6) | 3.9 | 0.148 | 0.917 | 0.917 | 0.933 | 0% | 0% |
| per-frame BiGRU, the test's, at 15 fps | 30 (21) | 19.7 | 0.197 | 0.516 | 0.871 | 0.935 | 0% | 0% |

Both heads learn the planted symbols. On val, the per-frame head reaches F1@50 0.97 at epoch 3 (18
steps) and 1.0 at epoch 11; CTC first sits 21 epochs (126 steps) on a plateau where it emits about one
symbol per clip, then climbs to 0.96 by epoch 25 and 1.0 at epoch 34. With the test's small model (width
64, GRU 32) CTC does not leave the plateau in 600 steps, with or without dropout, time masks and weight
decay (test F1@50 0 to 0.30): CTC wants the default width and a few hundred steps. At 15 fps the
per-frame head needs more epochs (F1@50 0.258 after 8) and its F1@25 is bounded by the 67-ms frame; in
this synthetic signal, which starts on the 30-fps frame nearest the onset, a quarter of the onsets are
labelled on the kept frame before their signal appears.

*The mirror* (9 attempts, 3 sessions, 24 clips; M1's stub features: dimension 64; the manifest's split,
2026-10-05 held out: 6 train clips with 3,027 kept frames and 318 symbols, 6 val with 3,163 and 298, 12
test with 4,693 and 554; 10 epochs each, batch 8, so one optimizer step per epoch; the test split,
pooled, with `--consistency`):

| run (best epoch, training seconds) | system | WER mean / pooled | F1@25 symbol | F1@50 symbol | F1@50 timing | exact | replay |
|---|---|---|---|---|---|---|---|
| `perframe-bigru` (10, 29 s) | model | 1.111 / 0.957 | 0.025 | 0.051 | 0.325 | 0% | 0% |
|  | + consistency | 0.846 / 0.812 | 0.012 | 0.022 | 0.218 | 0% | 0% |
|  | baseline | 1.832 / 1.532 | 0.040 | 0.081 | 0.454 | 0% | 0% |
| `ctc-bigru` (1, 27 s) | model | 0.962 / 0.962 | 0.000 | 0.003 | 0.090 | 0% | 0% |
|  | + consistency | 0.960 / 0.962 | 0.000 | 0.003 | 0.077 | 0% | 0% |
|  | baseline | 1.832 / 1.532 | 0.040 | 0.081 | 0.454 | 0% | 0% |
| `perframe-transformer` (3, 34 s) | model | 1.739 / 1.451 | 0.035 | 0.053 | 0.379 | 0% | 0% |
|  | + consistency | 0.851 / 0.823 | 0.008 | 0.011 | 0.133 | 0% | 0% |
|  | baseline | 1.832 / 1.532 | 0.040 | 0.081 | 0.454 | 0% | 0% |
| `perframe-bigru-15fps` (8, 16 s) | model | 0.895 / 0.861 | 0.020 | 0.041 | 0.190 | 0% | 0% |
|  | + consistency | 0.886 / 0.854 | 0.013 | 0.027 | 0.167 | 0% | 0% |
|  | baseline | 1.174 / 1.000 | 0.040 | 0.071 | 0.382 | 0% | 0% |

The baseline's motion peaks find more onsets than any model (F1@50 on timing 0.454 against at most
0.379) and over-predict (934 onsets for 554 in the reference); the per-frame BiGRU predicts about as many
as there are (585, threshold 0.5), the transformer more (918, at 0.6), the BiGRU at 15 fps fewer (227,
at 0.65), and CTC 109, one in five. The consistency pass halves or more the per-frame models' predictions
(585 to 273, 918 to 200: the stub models' adjacent predictions cancel), which lowers their WER and their
F1 alike. No collision and no onset of the other segment in any split, at 30 or 15 fps.

The commands (from the checkout, `CUBETRACE_DATA` the mirror, `<out>` a scratch folder, `<stub>` M1's
features root):

```
cubetrace-ml manifest --out <out>/manifest
cubetrace-ml train --config configs/perframe-bigru.toml --features <stub> --encoder stub \
    --manifest <out>/manifest/manifest.parquet --out <out>/runs/perframe-bigru --set train.epochs=10
cubetrace-ml evaluate --run <out>/runs/perframe-bigru --consistency
# the same with configs/ctc-bigru.toml, configs/perframe-transformer.toml, and perframe-bigru --fps 15
```

The six test solve clips' references replay to solved (6 of 6), and all 9 attempts' normalized solves
replay from their `scrambledFacelets` (the facelet model also matches `cubo.py` on 2,000 random
sequences, and the records' `scramble` gives their `scrambledFacelets` on all 9).

*Limits.* The stub features are a random projection of a 32-pixel gray frame and the mirror's train split
is six clips: the mirror's numbers check the pipeline end to end, not a model, and the baseline (the
motion alone) beats the models there, as it may. CUDA was not exercised (no GPU in this container): the
code puts the batches and the model on the device and packs the lengths on the CPU, as PyTorch asks; the
first GPU run is its test. An onset's predicted time is its peak's frame time (no interpolation), which
bounds F1@25 at 15 fps. The replay is strict: one wrong symbol fails a solve. CTC on few steps stays in
its plateau (above); the bucket gives about 100 steps an epoch at batch 8. The real-encoder features do
not exist yet.

*The real run.* Once the GPU run (follow-up (g)) has written `gs://cubetrace-data/features/<run>/`, on
the GPU machine (`uv sync --locked --extra cu128 --extra gcs`) or a CPU one (`--extra features` instead of
`--extra cu128`), with the bucket's records read through Application Default Credentials (or `ROOT` a
local mirror of `users/<uid>`):

```
ROOT=gs://cubetrace-data/users/<uid>
gcloud storage rsync -r gs://cubetrace-data/features/<run> features/<run>
uv run --no-sync cubetrace-ml manifest --root $ROOT --out runs/manifest
for config in perframe-bigru ctc-bigru perframe-transformer; do
  uv run --no-sync cubetrace-ml train --config configs/$config.toml --root $ROOT --features features/<run> \
      --encoder dinov2-vits14 --manifest runs/manifest/manifest.parquet --out runs/$config-dinov2
  uv run --no-sync cubetrace-ml evaluate --run runs/$config-dinov2 --split test --consistency
done
uv run --no-sync cubetrace-ml train --config configs/perframe-bigru.toml --root $ROOT \
    --features features/<run> --encoder dinov2-vits14 --manifest runs/manifest/manifest.parquet \
    --fps 15 --out runs/perframe-bigru-dinov2-15fps
uv run --no-sync cubetrace-ml evaluate --run runs/perframe-bigru-dinov2-15fps --split test --consistency
# and --encoder resnet18 for the second encoder; evaluate --split val for the val reports
```

*Follow-ups.* (k) to (o) below.

### The first real numbers (2026-10-07, the coordinator)

DINOv2 ViT-S/14 features of the whole bucket (1,240 clips; the GPU run of `docs/GPU.md`), the split of
M1's rule (train: the sessions of 2026-09-30 and 2026-10-05 and one of 2026-09-27, 796 clips, 337,168
frames, 38,677 symbols; val: 2026-10-02, 220 clips; test: the two sessions of 2026-10-03, 224 clips,
98,113 frames, 10,777 symbols), the configs as committed, trained on this container's 4 CPU cores
(about 200 s an epoch).

| Run (DINOv2, test split, pooled) | WER | F1@50 timing | F1@50 symbol | F1@25 timing | F1@25 symbol | exact | replay |
|---|---|---|---|---|---|---|---|
| `perframe-bigru` (best epoch 15 of 23, early stop) | 0.489 | 0.818 | 0.543 | 0.659 | 0.442 | 0% | 0% |
| the same with `--consistency` | 0.528 | 0.745 | 0.471 | 0.596 | 0.380 | 0% | 0% |
| the motion baseline | 1.622 | 0.523 | 0.092 | 0.305 | 0.051 | 0% | 0% |
| `ctc-bigru` (10 epochs: the blank plateau, nothing emitted) | 1.000 | – | – | – | – | 0% | 0% |

Means over the clips: WER 0.456 (scramble 0.39, solve 0.52), F1@50 timing 0.813, symbol 0.596; flat
across the TPS buckets (WER 0.43–0.49 from 3.5 to 6.5 TPS). The val loss rises from the first epoch
while the val F1@50 improves until epoch 15 (overfitting on 796 clips). CTC's train loss fell from 5.9 to
2.6 in 10 epochs with the greedy decode still all blank: a longer run (40 epochs) is queued.

**The confusions** (`perframe-bigru`, the onsets matched within ±50 ms: 8,897 of 10,777): the symbol is
right at 64% of them; the errors are another face (17%), the opposite face (12%) and the same face
turned the other way or doubled (6%). Per symbol: `U` 81%, `U2` 91%, `D` 90%, `D'` 86%, `D2` 83%, `F2`
85%, `R2` 73%, `L2` 72%, against `R` 42%, `R'` 40%, `F` 48%, `F'` 48%, `L` 26%, `L'` 32%, `B'` 46%; the
top confusions `F`→`B` 150, `F'`→`B'` 147, `L`→`R` 131, `B'`→`F'` 128, `L'`→`R'` 118, `R'`→`F'` 78. The
two cameras are alike (the laptop 63% right at 81% recall, the phone 66% at 85%, the phone without a
lag). The reading: the labels name the face in the cube's own frame, the camera sees a layer turn in the
world; the owner holds the cube with U or D up most of the time, so the top and bottom layers are
recognizable, while every rotation of the cube in the hands permutes which side face the camera sees:
the model has no way to tell `R` from `F` from `L` from `B` without the orientation. That is M3.

### M3 — the orientation

**Goal.** The side faces: the cube's orientation per frame given to the model, and a controlled
measurement of what it buys.

**Scope.** (1) `labels`: beside the kept frames' features, the gyro's orientation at each kept frame
from M0's track (`qx qy qz qw` slerped at `shownMs`), the change of orientation between consecutive kept
frames (the relative quaternion, `q_t · conj(q_{t−1})`, as four numbers, identity where a frame has
none), and a presence flag; a clip without `gyro.json` or whose gyro does not cover the kept frames
gets zeros and the flag 0. (2) `config`: `data.inputs` = `features` (as now) or `features+gyro` (the
9 extra channels concatenated to the features before the projection, standardized like them); and
`data.require_gyro` (true: clips without a gyro are skipped, counted as `no-gyro`), so that a baseline
and the gyro run train on exactly the same clips. (3) `evaluate`: a **Confusions** section in the report
(the matched onsets within ±50 ms; right / same face other turn / opposite face / other face; the per-
symbol accuracy table; the twelve top confusions; by camera), from the predictions table: the analysis
of the coordinator's `confusions.py`, which this task absorbs. (4) The experiment, run by the
coordinator on the cached DINOv2 features: `perframe-bigru` with `require_gyro` and `inputs=features`
against `inputs=features+gyro`, same seed; the Outcome note carries both reports' numbers and the
per-symbol accuracies side by side. (5) `docs/DATA.md`: the inputs and the gyro's channels.

**Acceptance.** Unit tests: the gyro channels at the kept frames (exact at the samples, slerped
between, zeros and flag 0 without a gyro, the relative quaternion of a known rotation), the config, the
model's input width, the report's Confusions section on known predictions; a synthetic training test
where the symbol depends on an orientation channel (the same planted onset signal in the features, the
face permuted by a synthetic orientation) and the `features+gyro` model beats the `features` one;
ruff, pytest and CI green; `train` and `evaluate` run on the mirror's stub features with the mirror's
gyro files (9 attempts have one).

**Outcome (M3).** The gyro's orientation as a model input and the confusion analysis in every report:
`align`'s quaternion product, conjugate and change of orientation (`relative_rotations`); `labels`' gyro
channels (`GYRO_CHANNELS`, `gyro_channels`), the `no-gyro` skip, the splits' gyro counts and
`model_input`; `config`'s `data.inputs` (`features`, `features+gyro`) and `data.require_gyro`; `train`'s
input normalization (the features' as before, then the gyro channels'); `metrics`' `Confusions`;
`report`'s Confusions section, its inputs row and gyro column, and `confusions_of` over a predictions
table; a confusion line in `evaluate`'s output; and `docs/DATA.md`'s "The labels and the runs", which
states every rule below. 186 tests (28 new), the factory's synthetic orientation among them.

*Decisions.* (1) **The channels**, as the contract: per kept frame the orientation at `shownMs` from M0's
track (x, y, z, w as the app records them, no hemisphere chosen), its change since the previous kept frame
`q_t · conj(q_{t−1})` in the hemisphere w ≥ 0 (between kept frames at `--fps 15`), and the flag; a frame
without an orientation has q = 0, the change the identity (there and at the next frame) and the flag 0.
Per frame, not per clip: a clip whose gyro covers part of its kept frames keeps its orientations there.
(2) **`require_gyro`** skips a clip only when none of its kept frames has an orientation (`no-gyro`); a
partly covered clip is kept, and the splits' counts say the coverage (the clips and the kept frames with
an orientation: the report's data table, `gyroClips` and `gyroFrames`). (3) **The normalization**: the
features' as before; the gyro channels' by the same rule (mean and deviation over the training frames, the
deviation floored at 1e-6), but the flag as it is (mean 0, deviation 1): with `require_gyro` it is 1 on
every training frame, and the rule would turn a 0 at test time into −10⁶. The baseline keeps the features'
statistics, so its numbers are the same in both runs of a comparison. (4) **`inputs = features` is M2 bit
for bit**: no `gyro.json` read (no new way to skip a clip), the same input arrays, initialization and
random draws; checked by training one small run with `main`'s code and the branch's on the tests' data
(the same weights, log and metrics). (5) A `gyro.json` that cannot be read (invalid, or not 4 numbers of q
per sample) skips its clips as `records` when the run reads the gyro. (6) **The confusions** come from the
predictions table (`report.confusions_of`, which reads a run's `predictions.parquet` as well): the model's
decoding only, its onsets matched on timing within ±50 ms (F1@50 timing's matches); the kinds of the
coordinator's `confusions.py`, the slices a family of their own (a slice for another slice is the opposite
face, a slice for a face turn another face); the per-symbol table a grid of the faces by `X`, `X'`, `X2`;
the top confusions' ties in the alphabet's order; `metrics.json` adds the whole matrix. (7) Booleans in the
configuration: `--set data.require_gyro=true` (TOML; `1` or `yes` refused). (8) **The synthetic test**: the
factory turns the cube about U (one state per segment and another from a pause of the solve on, a few
degrees of wobble about x, `gyro.json` at 14 Hz from 2 s before the scramble to 1 s after the solve) and
plants the symbol the camera sees; the test turns the cube half way or not (`F` and `B`, `R` and `L` alike
to the features: the first real run's top confusions) and trains the small transformer without the
regularizers (the inputs reach both bodies through the same projection): the BiGRU of the tests' size needs
about 200 optimizer steps for it (below), more than the suite affords.

*The synthetic result* (the test's data: 10 sessions of 3 attempts, 48 train, 6 val and 6 test clips, 86
test symbols; features of dimension 16; the transformer of width 64 and feed-forward 128, batch 4, lr
3e-3, 15 epochs, no time masks, no dropout, `require_gyro`; seed 0; the test split, pooled; this
container's 4 vCPUs shared with a training run, 2 threads):

| inputs | best epoch | seconds | WER | F1@50 symbol | F1@50 timing | right at the matched onsets | exact | replay |
|---|---|---|---|---|---|---|---|---|
| `features` | 8 | 11.3 | 0.349 | 0.659 | 0.988 | 56 of 84 (67%) | 0% | 0% |
| `features+gyro` | 7 | 9.5 | 0.081 | 0.930 | 0.988 | 80 of 85 (94%) | 50% | 33% |

Seeds 1 to 3: 0.639, 0.648 and 0.624 against 0.908, 0.924 and 0.953 (+0.27 to +0.33; the test asserts
+0.2, both runs and their evaluations in 25 s here). The features alone get the U and D turns and half of
the side ones (about 0.65 is their ceiling here): 27 of their 28 wrong symbols are the opposite face. On 10
sessions of 2 attempts: the cube at any quarter (four states) with the same transformer, 12 epochs, 0.468
against 0.383, the pairs (seen symbol, orientation) being too many for the steps; the BiGRU of the tests'
size (width 64, GRU 32) with two states, 0.754 against 0.513 after 25 epochs at batch 4 (the gyro run
ahead from epoch 11 on val), and 0.817 against 0.464 after 12 at batch 2 without the regularizers.

*The mirror* (9 attempts, each with `gyro.json`: 4,604 samples at 12.6–14.8 Hz, continuous, with no sign
flip between consecutive samples, w < 0 on 38% of them; M1's stub features, dimension 64; the manifest's
split; `perframe-bigru` with `require_gyro`, 2 epochs of one step): 6 train, 6 val and 12 test clips, none
skipped, every kept frame with an orientation (3,027, 3,163 and 4,693). The gyro run's input is 64 + 9 =
73 wide; on test it scores WER 1.273 (pooled 1.096) and F1@50 symbol 0.023, the features' run 1.086
(0.982) and 0.019: the stub's noise. Its Confusions section: 227 of 554 onsets matched, 6% right, 12% the
same face, 17% the opposite face, 65% another face; by camera, `laptop (lag)` 10 right, 117 wrong and 150
unmatched, `phone-rear (no lag)` 3, 97 and 177.

*Limits.* The orientation is the cube's in the gyro's own reference, not the camera's: if that reference
moves between sessions (or the cameras do), one q is not one face toward the camera, and the model has to
learn where the camera is from the data (follow-up (p)). The quaternion's two signs: the app's stream is
continuous within an attempt but not kept in one hemisphere, so one orientation comes as q or −q and the
model has to learn both (follow-up (q)). On the bucket `gyro.json` begins with the app's T3.7
(2026-10-02): 2026-09-30's 366 clips should have none, so that with `require_gyro` the comparison trains
on about half of the first real run's training clips (the reports' data tables give the counts), and its
`features` run is the reference, not the first real numbers. The synthetic test shows that the channels
reach the model and can be used (two states, the transformer); the four-state case needs more steps than
the suite's. A partly covered clip's frames without an orientation are flagged, not filled in.

*The real comparison* (the coordinator; the commands of M2's real run, `ROOT` the bucket's root or its
mirror, `F` the features of (g), `M` the first real run's manifest), the two runs differing only in their
inputs, the same seed:

```
for inputs in features features+gyro; do
  name=perframe-bigru-dinov2-gyroclips-${inputs/+/-}
  uv run --no-sync cubetrace-ml train --config configs/perframe-bigru.toml --root $ROOT --features $F \
      --encoder dinov2-vits14 --manifest $M --set data.require_gyro=true --set data.inputs=$inputs \
      --set name=$name --out runs/$name
  uv run --no-sync cubetrace-ml evaluate --run runs/$name --split test --consistency
done
# side by side: each run's metrics.json, systems.model.all and confusions.perSymbol
```

**The real comparison** (the coordinator, 2026-10-08, on this container's CPU): `perframe-bigru` on the DINOv2
features with `require_gyro` (729 of the 796 training clips have an orientation, 172 of the 220 val clips; the
test split's 224 clips all do), `inputs=features` against `inputs=features+gyro`, seed 0, both best at epoch 10
of 18:

| test split, pooled | WER | F1@50 timing / symbol | F1@25 timing / symbol | matched onsets | right at matched |
|---|---|---|---|---|---|
| `features` | 0.503 | 0.804 / 0.531 | 0.631 / 0.425 | 8,513 of 10,777 | 64% |
| `features+gyro` | 0.505 | 0.808 / 0.534 | 0.642 / 0.427 | 8,771 of 10,777 | 64% |

**No gain.** The raw orientation reshuffles the side faces instead of resolving them: `B` 45% → 59%, `B'` 43% →
60%, `R` 41% → 51%, `R'` 35% → 43% and `F'` 44% → 51% improve, while `L` 31% → 22%, `L'` 31% → 27%, `U` 86% → 82%
and `D` 90% → 86% get worse; the kinds of error stay (opposite face 11% → 9%, other face 20% → 20%). The reading,
as follow-up (p) feared: q lives in the gyro's own frame, whose yaw (at least) is arbitrary at each power-on and
whose relation to the camera changes with every session and camera placement, so the mapping from q to the face
the camera sees is not one function the model can learn from a handful of sessions. The orientation has to be
expressed relative to the camera, which needs a calibration per session or per attempt: M4.

*Follow-ups.* (p) and (q) below.

### M4 — the orientation in the camera's frame

**Goal.** Turn the gyro's orientation into one the camera can use: a rotation between the gyro's frame and the
camera's, per attempt, so that "which cube face the camera sees on its right" becomes a function the model can
learn across sessions; and a measurement of what the side faces gain, with an oracle bound.

**Scope.** (1) **The diagnostic** (`cubetrace-ml gyro-frames`, a report): per session and attempt, from
`gyro.json`, the gyro-frame direction of the cube's three axes over the solve (their distributions on the sphere:
the modes), whether one axis stays aligned with a fixed gyro-frame direction across sessions (gravity: then only a
yaw is arbitrary) or not (then the whole rotation is), and how much the frame drifts within an attempt and within
a session (the modes' spread). (2) **The calibration**: a unit quaternion `c_a` per attempt (a 3-DoF rotation; a
yaw-only variant `--calibration yaw` when the diagnostic allows) applied to the gyro's orientation before the
channels, `q_cam = c_a · q`; the channels become the rotated orientation (as a rotation matrix's 9 entries, sign-
free, or the quaternion in w ≥ 0: a flag) and the relative rotation as before. (3) **Learning it**: on the training
attempts `c_a` is a learnable parameter of the model (one per attempt, initialized at the identity, trained jointly
with the network; a small penalty toward the attempts of the same session agreeing, `--session-tie`, as an option).
(4) **Estimating it on an unseen attempt**: the scramble's moves are known in advance (the app prescribes them),
so on a test attempt `c_a` is fit by maximizing the trained model's per-frame likelihood of the scramble clip's
reference labels (a coarse grid over the rotations, 24 cube symmetries × a finer yaw grid, then a few gradient
steps), and the solve clip is evaluated with that `c_a`: this is the **honest** number. The **oracle** fits `c_a`
on the attempt's whole labels, scramble and solve: the upper bound. A third number: `c_a` fixed at the identity
(what M3 measured). (5) `evaluate` reports the three on the same clips, with the Confusions section each; the
Outcome note carries the per-symbol accuracies of the side faces side by side. (6) `docs/DATA.md`: the
calibration and the channels; the run folder keeps the fitted `c_a` per attempt (`calibration.parquet`).

**Acceptance.** Unit tests: the rotation of the channels by a known calibration (a yaw of 90° maps `F`'s
direction to `R`'s), the learnable per-attempt parameter (gradients reach it; it stays unit), the estimation on a
synthetic attempt whose camera frame is rotated by a known yaw recovers it to within a few degrees from the
scramble's labels alone, the diagnostic on synthetic gyro files; the synthetic training test of M3 extended: the
test attempts' frames rotated by yaws the training never saw, where `features+gyro` without calibration fails and
the calibrated model (honest, from the scramble) recovers most of the oracle's F1; ruff, pytest and CI green; a
smoke run on the mirror.

**Outcome (M4).** The orientation in a camera's frame, `q_cam = c · q`, one rotation per attempt: the modules
`orientation` (NumPy: rotation matrices and quaternions, the cube's 24 symmetries and the estimation's grids,
the exponential map, headings, chordal means, the scramble's pose, the calibrated channels), `gyroframes` (the
diagnostic) and `calibrate` (PyTorch: the calibrated inputs, the learnt rotations, the session tie, the
label-free guess, the fits); `labels`' change in the cube's frame, scramble pose and `calibration_key`;
`config`'s `data.calibration`, `calibration_init`, `calibration_dof`, `gravity_axis` and `orientation`, and
`train.calibration_lr` and `session_tie`; `train`'s calibrated runs and `evaluate_run`'s four modes;
`report`'s Calibration section and `calibration.parquet`; the commands `gyro-frames` and `evaluate
--calibrate` and `--refine`; and `docs/DATA.md`'s "The orientation in the camera's frame", which states every
rule below. 220 tests (34 new), on held cubes in synthetic gyro frames and on M3's turned fixture with its
gyro frames turned.

*The diagnostic on the real records* (`cubetrace-ml gyro-frames` on the coordinator's JSON mirror of
2026-10-07: 586 attempts, 410 with gyro.json in 5 sessions, 191,915 samples at 12.6 Hz; 23 s with the
schemas' validation, 4 s without):

1. **The convention.** The angular velocity `v` follows the change in the cube's frame, `conj(q_t) · q_{t+1}`
   (Spearman 0.14, 0.18 and 0.34 on the x, y and z diagonals, at most 0.03 off them), and not the change in
   the gyro's frame (−0.08 to 0.00): q takes the cube's axes into the gyro's frame, so a change of reference
   acts on the left, `c · q`, as the brief has it.
2. **Gravity is the gyro's z axis.** The cube's z axis (the white face's) keeps one direction in every
   attempt and session: pooled concentration 0.90 (x 0.50 and y 0.48: every heading), 1.5° from the gyro's
   z, the median attempt's within 14° of it (90%: 30°), the five sessions' directions within 16° of one
   another. Only a yaw about z is arbitrary.
3. **The hold.** White up through all 410 scrambles and down through all 410 solves (turned over for the
   solve: the cross on the bottom), 15° off the vertical (10–90%: 8–32°); a scramble's mean orientation 174°
   from its solve's (median); the scramble's samples 18° from their mean (median), the solve's 51°; the
   vertical moves 9° (median) between the scramble and the solve, and between the solve's halves.
4. **The yaw.** The sessions' mean scramble headings sit up to 154° apart (median 95°); within a session the
   heading drifts 5 to 43° an hour (the four sessions of 4 to 44 hours) with residuals of 34 to 66° about
   that line (jumps as well as drift), over ranges of 145 to 725°; between consecutive attempts it moves
   4.6° (median; 90%: 12°).

So the calibration is a yaw about z (the defaults `data.calibration_dof = yaw`, `data.gravity_axis = z`), one
per attempt (a session's yaw is not one), and the scramble's pose is a natural start: the app prescribes the
scramble in the cube's frame, the solver holds the cube one way to apply it, so its heading there is the
frame's yaw plus how the solver faces the cameras. On the agents' mirror (9 attempts, 3 sessions) the same:
cube frame (0.22 against 0.04), z (0.94), white up in 9 scrambles and down in 9 solves, 5.6° between
consecutive attempts, sessions up to 135° apart.

*Decisions.* (1) **The channels**: the calibrated orientation as its rotation matrix's 9 entries (sign-free,
the default) or as the quaternion in w ≥ 0 (`data.orientation = quat`), the change, the flag: 14 channels
(or 9). The change is the cube's own, `conj(q_{t−1}) · q_t`, and not M3's `q_t · conj(q_{t−1})`: the brief
took the relative rotation to be unchanged by a fixed c, which holds for the change in the cube's frame only
(M3's, in the gyro's frame, turns with the calibration: c Δ c*). `data.calibration = none` (the default)
keeps M3's channels and runs bit for bit: one small run trained with `main`'s code and with the branch's gives
the same weights, log and metrics, and no new checkpoint key. (2) **The kinds**: `attempt` (one learnt
rotation per training attempt, keyed `sessionId/attemptIndex`, as the brief), and `identity` (the
calibrated channels with c the identity), `pose` (c fixed at each attempt's scramble pose undone, nothing
learnt) and `camera` (one per attempt and camera, `sessionId/attemptIndex/camera`: an attempt's two
cameras see the cube from two places, which one rotation can serve only if the network tells the cameras
apart). (3) **The start**: the learnt rotations start at their attempt's scramble pose undone
(`calibration_init = pose`, the default), not at the contract's identity (`calibration_init = identity`):
from the identity each training attempt's frame has to be found from scratch, which the synthetic test
manages only in part (below), and the pose is never a worse start than the raw frame. (4) **The guess** for
the keys the training never saw (val at every epoch; the `pose` mode of `evaluate`): the scramble pose
undone, then the training's mean correction (the network absorbs any rotation common to the learnt ones, so
the learnt rotations drift as a group), when the poses predict the learnt rotations (80% of the training
keys' corrections within 45° of their mean), else the learnt rotations' mean. (5) **The fits** score the
grid (24 yaws in 15° steps; for `rotation`, the brief's 24 symmetries × 24 yaws, which are 144 distinct
rotations) and the key's guess, by the training's loss with the checkpoint's class weights, and refine the
best three by a **compass search** (forward passes only: the step halving from 7.5°, or 22.5° for a
rotation, to under 1°), not by Adam: on this machine's CPU a forward and backward pass through the default
BiGRU runs at 1,200 to 2,000 frames a second against 11,000 to 20,000 for the forward alone, so the mirror's
evaluation took 3 min 49 s with a dozen Adam steps per candidate, 2 min 13 s with the three batched, and 25 s
with the search; and on random whole rotations the search lands within 0.4–0.8° where twelve Adam steps
fall 1 to 14° short of the rotation grid's coarse tilt. `evaluate --refine adam` keeps the Adam steps. (6)
**One `evaluate`, four modes** on the same clips (`none`, `pose`, `scramble`, `all`: the pose added to the
brief's three); `--calibrate` picks whose numbers the report's other sections give (`scramble` by default).
(7) **The session tie** is there (`train.session_tie`, a weight on the mean squared distance of each key's
rotation matrix from its session's mean) and off: the diagnostic finds a session's yaw moving tens of degrees
an hour. (8) The learnt rotations take their own learning rate (`train.calibration_lr` 0.01: an attempt's
parameter gets one or two steps an epoch) and no weight decay.

*The synthetic result* (the tests' fixtures: M3's turned cube, 10 sessions of 3 attempts, 48 train, 6 val and
6 test clips, 86 test symbols; M3's small transformer, 15 epochs, seed 0, 2 threads, 12 to 16 s a run;
pooled over the test split, the solve clips' side faces apart):

| fixture | run | calibration in `evaluate` | WER | F1@50 symbol | side faces right (solve) |
|---|---|---|---|---|---|
| test frames turned 180°, train and val at 0° | M3's channels | – | 0.686 | 0.304 | 4% |
| | `attempt` from the identity | `none` | 0.674 | 0.335 | 8% |
| | | `pose` (poses unused: agreement 0.58) | 0.674 | 0.335 | 8% |
| | | `scramble` (honest) | 0.174 | 0.860 | 68% |
| | | `all` (oracle) | 0.174 | 0.854 | 68% |
| every session's frame at its own yaw, the cube at home through the scrambles | M3's channels | – | 0.512 | 0.479 | 20% |
| | `pose` | `none` | 0.674 | 0.371 | 20% |
| | | `pose` | 0.070 | 0.943 | 85% |
| | | `scramble` | 0.070 | 0.936 | 85% |
| | | `all` | 0.070 | 0.936 | 85% |
| | `attempt` from the pose | `none` | 0.651 | 0.411 | 24% |
| | | `pose` | 0.105 | 0.920 | 81% |
| | | `scramble` | 0.081 | 0.931 | 85% |
| | | `all` | 0.081 | 0.931 | 81% |
| | `attempt` from the identity | `none` | 0.419 | 0.601 | 24% |
| | | `scramble` | 0.221 | 0.807 | 50% |
| | | `all` | 0.233 | 0.786 | 42% |

A frame the training never saw defeats M3's channels and the identity (F1@50 0.30 to 0.34: the side faces
read as their opposites); the fit on the scramble's labels recovers it (180° found within 4°, 1.9° from the
oracle's rotation, median) and with it the oracle's F1 (0.860 against 0.854). With every session's frame
arbitrary, the scramble pose alone calibrates without labels (0.943 fixed, 0.920 learnt from it), and the
learnt rotations from the identity reach only 0.81 (their corrections agree for 63% of the keys). The tests
assert these margins: F1 under 0.5 at the identity and for M3, the honest fit 0.3 above the identity and at
least 90% of the oracle's (the first fixture, 27 s); the pose 0.85 or more and the identity under 0.6 (the
second, through the commands, 16 s). The unit tests recover a yaw of 100° and a random whole rotation within
3° from labels alone, with a model that knows where the faces point.

*The mirror* (9 attempts; M1's stub features, dimension 64; `perframe-bigru` with `require_gyro` and
`calibration = attempt`, 3 epochs of one step; 3 training keys, a yaw about z from the pose; the input 64 + 14
= 78 wide): on test (12 clips, 3 keys) the four modes give WER 0.889 to 0.915 and F1@50 symbol 0.025 to
0.033, the stub's noise (the fits gain 0.013 and 0.001 of loss over the identity: a flat likelihood);
`evaluate` (the four modes, two fits) in 25 s with the search, 2 min 13 s with Adam.

*Limits.* The pose's premise, that the solver holds the cube the same way toward the cameras through every
scramble, is only measured through its consequences: the real run's `calibration.parquet` gives each test
attempt's honest rotation and its angle to the guess. A yaw per attempt leaves a camera's pitch to the
network (fixed per camera), and with `attempt` keys an attempt's two cameras share one rotation. A key whose
scramble clip is unusable keeps its guess (`guess (no clip)`). The convention rests on weak correlations (v is
4-bit and the solves' motion is mostly face turns), though their pattern is unambiguous. The yaw's drift
within an attempt is not measured (no reference besides the cube); the vertical moves 9° between an attempt's
scramble and solve. A fit scores its key's clips at the grid's 25 candidates (145 for a rotation) and at up to
36 (180) more in the search, forward passes of the frozen network: minutes of this CPU, not seconds, on the
real test split's 224 clips.

*The real comparison: pending (the coordinator).* The commands of M3's comparison with a calibration
(`ROOT` the bucket's root or its mirror, `F` the features of (g), `M` the first real run's manifest), each
`evaluate` giving the four modes side by side in its report's Calibration section and `calibration.parquet`:

```
uv run --no-sync cubetrace-ml gyro-frames --root $ROOT --out runs/gyro-frames
for calibration in attempt pose; do
  name=perframe-bigru-dinov2-gyroclips-$calibration
  uv run --no-sync cubetrace-ml train --config configs/perframe-bigru.toml --root $ROOT --features $F \
      --encoder dinov2-vits14 --manifest $M --set data.require_gyro=true --set data.inputs=features+gyro \
      --set data.calibration=$calibration --set name=$name --out runs/$name
  uv run --no-sync cubetrace-ml evaluate --run runs/$name --split test --consistency
done
# the contract's start: --set data.calibration=attempt --set data.calibration_init=identity
# the oracle's or the identity's numbers in the report's other sections: evaluate --calibrate all (or none)
```

*Follow-ups.* (r) to (u) below.

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
to the bucket — done on 2026-10-07 by the batch machine of `docs/GPU.md` (status 0 in 52 minutes end to
end, 1.6 GB under `gs://cubetrace-data/features/2026-10-07/`; the numbers in the table above). (h) A tighter framing of the phones (a cube or hand detector, or the app's `crop` on the
phones too), and whether the laptop's record rectangle, which can cut the cube's edge, should be widened
to the motion's square. (i) A `cu130` extra (torch 2.14 on CUDA 13, a driver of 580 or later) if the GPU
machine's driver allows it: `cu128` stops at torch 2.11. (j) A change to the PyTorch packages of the lock
is resolved on a runner (or a machine that reaches download.pytorch.org), as e7996fd was (and 388e669
for M2's matplotlib). (k) The real run of M2's Outcome note: the bucket's DINOv2 and ResNet-18 features
through the three configs and the 15-fps ablation, their numbers in that note; the GPU machine's startup
script could run `train` and `evaluate` after `features` and sync `runs/` to the bucket. (l) The plan's
decoder: the cube-notation prior and the state-consistency check, for instance a beam search over the
per-frame posteriors that keeps the sequences that replay (the replay metric is in place). (m) Sub-frame
onset times (a parabola through a peak and its neighbours) for F1@25 at 15 fps, and for the per-clip lag
of (a). (n) CTC's plateau, if it is long on the bucket: a head initialized toward the blank, or a short
warmup; and CTC's early stopping on val WER can keep a plateau epoch while the WER barely moves (the val
loss could break such ties); since the review, `train.min_epochs` (10) keeps early stopping from firing
inside the plateau. (o) One weight for every onset class: per-class weights (or a focal loss) if
the rare symbols (the slices, the doubles of B and D) lag on the bucket. (p) The orientation in the camera's
frame: a reference per session or per camera (the app's viewer calibration, its Re-zero with the cube's
front toward the camera, or one estimated from the labels: the rotation that best explains the side faces),
if the real comparison shows that the gyro's own frame does not carry over between sessions. (q) A
sign-free orientation input (the rotation matrix, or its first two columns: continuous and unique where
the quaternion has two signs), and the angular velocity `v` of `gyro.json` (not used yet) — the matrix done
in M4 (`data.orientation = matrix`, with a calibration); `v` still unused. (r) The scramble pose's premise on
the real records: the honest fits' rotations against the guesses (`calibration.parquet`, `angleToGuessDeg`);
if they agree, `data.calibration = pose` is the cheap path, also where no labels exist. (s) Per-camera
rotations (`data.calibration = camera`) if the per-attempt fits leave one camera's side faces behind (the
report's confusions by camera). (t) The yaw's drift within an attempt and the camera's pitch (a whole
rotation per camera and session beside the yaw per attempt). (u) The fits on the GPU machine: its startup
script could run `evaluate` after `train`; on the CPU the real test split's fits take minutes.
