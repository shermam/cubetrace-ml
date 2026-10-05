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
  2026-10-05 the bucket held 515 attempts (397 with video, 80 of them from two cameras), 949 clips,
  11 GB, 2.6 hours of solving, 42,020 moves; the owner records 60–170 attempts on a solving day. At
  30 fps and the owner's 4.5 moves a second a move spans about 7 frames.
- **Compute**: the bucket is in us-central1; features and training run on a GPU there (a T4 or an L4)
  once the billing account allows GPUs, or on a free notebook GPU with the features pulled once;
  the dataset tooling and the evaluation need no GPU.

## Phase M board

| Task | Scope | Depends on | Status |
|---|---|---|---|
| M0 | `cubetrace_ml`: the dataset over a local mirror or the bucket; records validated; the per-frame label track per clip (frame host times, the lag, the move onsets, the phase, the gyro); the alphabet normalization; the consistency filter; splits by session; the manifest and its report; a visual check | – | 🔄 PR |
| M1 | the frozen encoder's features per clip, cached (local or bucket), with the decode/crop/resize path and its throughput measured | M0 | ⬜ |
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
cube small in it): M1 needs a framing for them (a fixed rectangle per camera, or a detector).
