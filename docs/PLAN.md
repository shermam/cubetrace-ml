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
- **Labels on the host clock**: a move's `hostMs` (the cube's clock through the attempt's fit) and a
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
| M0 | `cubetrace_ml`: the dataset over a local mirror or the bucket; records validated; the per-frame label track per clip (frame host times, the lag, the move onsets, the phase, the gyro); the alphabet normalization; the consistency filter; splits by session; the manifest and its report; a visual check | – | ⬜ |
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
two same-face quarter turns within 150 ms merged into one symbol, a slice = two opposite-face turns
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

(none yet)
