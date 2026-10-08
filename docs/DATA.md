# The data, as cubetrace-ml reads it

The records are the capture app's ([shermam/cubetrace](https://github.com/shermam/cubetrace)
`docs/DATA-MODEL.md`); this page says how this repository consumes them: the timelines, the lag, the
alphabet, the per-frame track, the filter, the splits, the manifest, the features, the labels and the
runs of the models, and the cube's orientation in a camera's frame. The package is `src/cubetrace_ml`, the
command `cubetrace-ml`.

## A dataset root

```
<root>/sessions/<sessionId>/session.json
<root>/sessions/<sessionId>/attempts/<nnnn>/attempt.json
                                            gyro.json
                                            <camera>.<segment>.mp4
                                            <camera>.<segment>.frames.json
```

A root is a local folder or a `gs://` prefix (the bucket keeps the same layout under `users/<uid>/`, so
a root there is `gs://cubetrace-data/users/<uid>`). Every command takes `--root`, by default
`$CUBETRACE_DATA`. A bucket root needs the `gcs` extra (`uv sync --extra gcs`) and Application Default
Credentials: its prefix is listed once per command, and its files are read through
`~/.cache/cubetrace-ml/gcs/<bucket>/<object name>` (or `--cache`), each kept while its generation is the
listed one; an MP4 is downloaded only when a command needs its video.

Every record read is checked against `schemas/` (`--no-validate` skips it). `cubetrace-ml validate`
checks every record under the root, and each attempt's folder against its record: the folder's session
and index, each clip's MP4 (present, `bytes`) and frames file (`camera`, `segment`, `frames`,
`firstFrameHostMs`), `gyro.json` (`samples`), and the files no record names. A session folder without
`session.json` is a warning: its attempts are read, and its day comes from them (below).

## The timelines

Every time is in host milliseconds (`performance.timeOrigin + performance.now()` on the host).

- **A move's time.** A move's `hostMs` is when its Bluetooth packet arrived, with the packets' jitter
  (the attempt's `clock.residualP95Ms`: 14 to 34 ms on the first real attempts); the attempt's clock fit,
  `a·cubeMs + b`, is the cube's own timing on the host clock without it. With the time base `fit` (the
  default) a move's time is the fit's when the fit is a line through one cube clock (`a` within 0.9–1.1,
  `residualP95Ms` at most 250 ms) and the move is on that clock (the fit within 250 ms of its `hostMs`),
  and its `hostMs` otherwise; with `arrival` (`--time-base arrival`) it is `hostMs`.
- **The events.** `scrambleStart`, `scrambleDone`, `solveStart` and `solveEnd` are the arrivals of the
  first and last scramble and solve moves; each takes its move's time on the time base. Any other event
  time (a resync's report, the pickup) is kept as recorded.
- **A frame's time.** Frame `k` of a clip is at `tMs[k] = t0HostMs + dtMs[0] + … + dtMs[k]` (its frames
  file; `dtMs[0]` is 0). On the first real clips the MP4's presentation times match these to 0.01 ms.
- **The lag.** A clip's `syncResidualMs` is how far its camera's frames lag the cube: the clapperboard's
  median of (the middle of a turn's motion in the frames) − (the turn's `hostMs`). A move's onset on the
  clip's frame timeline is therefore **`onset + lag`**, and the frame at `tMs` shows the cube as it was
  at **`shownMs = tMs − lag`**: a positive lag means late frames, as they always are. A clip without a
  sync check (`syncResidualMs` null, the phones' clips until their own check) has no lag: it is
  `unsynced`, its `lagMs` is null, and its track is computed with a lag of 0.
- **The window.** A clip's segment window is, on the cube's timeline, `scrambleStart` to `scrambleDone`
  for a scramble clip and `solveStart` to `solveEnd` for a solve clip (no end for a DNF). The clip's
  frames begin before it (2–4 s) and end after it (about 1 s).

## The alphabet

24 symbols, each with a fixed index (its class id):

| Index | Symbols |
|---|---|
| 0–2 | `U` `U'` `U2` |
| 3–5 | `R` `R'` `R2` |
| 6–8 | `F` `F'` `F2` |
| 9–11 | `D` `D'` `D2` |
| 12–14 | `L` `L'` `L2` |
| 15–17 | `B` `B'` `B2` |
| 18–23 | `M` `M'` `S` `S'` `E` `E'` |

The cube reports quarter turns; `normalize` merges them as the owner's simulator does (`normalizar()` in
`ferramentas/cubo.py`), scanning greedily from the left:

1. **A slice**: two opposite faces, quarter turns of opposite notation direction (the same physical
   way), less than `--slice-ms` (20) apart: `R`+`L'` is `M`, `L`+`R'` is `M'`, `F'`+`B` is `S`, `B'`+`F` is
   `S'`, `U`+`D'` is `E`, `U'`+`D` is `E'`, in either order (the usual notation: `M` turns as `L`, `S` as
   `F`, `E` as `D`).
2. **A double**: two equal quarter turns of one face less than `--double-ms` (200) apart: `X X` or
   `X' X'` is `X2`.
3. Anything else is the quarter turn itself; a half turn in the input stays itself.

Two turns of different phases (scramble, solve) never merge. A merged symbol's onset is its first
turn's time and its end its second's.

## The per-frame track

`align_clip(attempt, frames, gyro)` gives a clip's track without decoding its video: a NumPy structured
array with one row per frame.

| Field | Meaning |
|---|---|
| `frame` | the frame's index in the clip |
| `tMs` | its host time: `t0HostMs + dtMs[0] + … + dtMs[frame]` |
| `shownMs` | the time it shows: `tMs − lag` (`tMs` when unsynced) |
| `symbol` | the alphabet index of the nearest onset (`onset + lag` nearest to `tMs`, the earlier on a tie), among all the attempt's symbols; −1 when the attempt has none |
| `onset` | that symbol's index in the attempt's normalized moves; −1 when none |
| `distanceMs` | `tMs − (onset + lag)`: negative before the onset; NaN when none |
| `phase` | the attempt's phase at `shownMs`: 0 `before` (`scrambleStart`), 1 `scramble` (to `scrambleDone`), 2 `inspection` (to `solveStart`), 3 `solve` (to `solveEnd`), 4 `after`; an event that did not happen never begins its phase |
| `inWindow` | `shownMs` is inside the segment's window |
| `qx` `qy` `qz` `qw` | the cube's orientation at `shownMs` from `gyro.json`: the sample there, or the slerp of the two samples around it along the shorter arc; NaN without `gyro.json` and outside its samples' span |

A per-frame target ("no onset" or the symbol whose onset is nearest within a tolerance) is
`symbol` where `|distanceMs|` is within the tolerance.

## The filter

A clip is usable when it has none of these reasons (the manifest's `reasons`, in this order):

| Reason | When |
|---|---|
| `dnf` | the attempt's `result.status` is `dnf` |
| `replay-failed` | `result.replayOk` is false: the solve's moves do not replay (moves went unseen) |
| `truncated-start` | the clip's `truncatedStart`: it begins later than asked |
| `missing-video` | the MP4 is not under the root |
| `bytes-mismatch` | the MP4's size is not the record's `bytes` |
| `missing-frames` | the frames file is missing or invalid |
| `frames-count-mismatch` | the frames file's `len(dtMs)` is not the record's `frames` |
| `video-frames-mismatch` | the video's frame count, when measured, is not the frames file's |
| `video-unreadable` | the MP4 could not be read |
| `moves-outside-clip` | an onset of the segment (plus the lag) falls outside the clip's frames |

The attempt's reasons apply to both its clips. An unsynced clip is usable; the manifest flags it.

## The splits

By session, never by attempt, weighed by clips (the manifest's rows, usable or not). A session's day is
the UTC date of its `session.json`'s `createdMs` (of its attempts' earliest `scrambleShown` without one).

- A session without clips joins no split: its split is `none`, and it counts for nothing below.
- **`test`** is the held-out day's sessions, whole. The held-out day is, by default, the day whose clips
  are closest to a fifth of all the clips (the later day on a tie); `--held-out-day` names another, or
  `latest` (the latest day with clips). The manifest's `manifest.json` records it as `testDay`.
- **`val`** targets a share of all the clips (`--val-fraction`, 0.15) in whole sessions: the other
  sessions with clips, sorted by id and shuffled with `random.Random(seed)` (`--seed`, 0 by default), are
  taken in that order whenever one brings val's clips closer to the share. When there are two or more,
  val has at least one (the closest to the share, if none comes closer) and never all of them.
- **`train`** is the rest.

The same sessions, clip counts and seed give the same assignment.

## The manifest

`cubetrace-ml manifest --out <folder>` writes `manifest.parquet` and `manifest.csv` (one row per clip),
`attempts.parquet` and `attempts.csv` (one row per attempt), and `manifest.json` (the settings and the
records that could not be read). The clips' columns:

| Column | Meaning |
|---|---|
| `sessionId`, `day`, `attemptIndex`, `camera`, `segment` | the clip |
| `video` | its MP4, relative to the root |
| `fpsNominal`, `fps` | the track's frame rate, and the measured one: (frames − 1) / span |
| `frames`, `framesFileCount`, `videoFrames` | the record's count, the frames file's, and the video's when measured (`--video`: `none`, `fast` the container's header, `full` a decode; by default `fast` for a folder, `none` for a bucket) |
| `seconds` | the span of the frame times |
| `width`, `height`, `cropX`, `cropY`, `cropW`, `cropH` | the encoded size and the framing rectangle (null for the whole frame) |
| `lagMs`, `unsynced` | `syncResidualMs` (null without a sync check) and whether it is null |
| `movesInWindow`, `movesCovered` | the segment's normalized symbols, and how many of their onsets (plus the lag) fall within the clip's frames |
| `tps`, `status`, `replayOk` | the attempt's `result` |
| `gyroRateHz` | the attempt's `gyro.rateHz`; NaN without a gyro file |
| `truncatedStart` | the clip's flag (false when absent) |
| `usable`, `reasons` | the filter's verdict and its reasons, `;`-separated |
| `split` | `train`, `val` or `test` (`none`, in the attempts' table, for a session without clips) |

`cubetrace-ml report` prints the counts: sessions, attempts, clips, hours of video and of solving,
moves before and after the normalization, the usable and unsynced shares and the filter's reasons, the
counts by camera, day and split, and the TPS histogram. `cubetrace-ml splits` prints each session's
split.

## Checking by eye and by count

- `cubetrace-ml inspect --session <id or prefix> --attempt <n> --camera <label> --segment <segment>` (or
  `--row <n>` of a manifest) writes a contact sheet: a row per onset, spread over the segment (6 by
  default), the frames around it (5), cropped to the clip's framing, each with the symbol and
  `tMs − (onset + lag)`, the nearest frame outlined. It goes to `out/` by default, which git ignores.
- `cubetrace-ml check-alignment` (all clips, or those `--session`, `--attempt`, `--camera` and `--segment`
  select) compares each clip's frames file with its record and its video: the counts (the decode's and
  the container's; `--fast` reads only the container's header), the largest difference between a
  frame's presentation time and its frames-file time, the segment's onsets within the clip, and the
  clip's margins around its window.

## The features

`cubetrace-ml features --encoder <name> --out <folder>` (the folder by default `$CUBETRACE_FEATURES`)
runs a frozen encoder on every frame of every selected clip and caches the result, one file per clip, so
that a model trains on the features without decoding video.

**The encoders** (`cubetrace_ml/encoders.py`). Each takes RGB frames at its square input and gives one
vector per frame.

| Name | Input | Dim | Weights | The vector |
|---|---|---|---|---|
| `stub` | 32 | 64 | none: a Gaussian projection, seed 0 | the gray frame (BT.601 luma, 0–1, less 0.5) times the 1024 × 64 projection; no PyTorch: the tests' encoder |
| `resnet18` | 224 | 512 | torchvision `ResNet18_Weights.IMAGENET1K_V1` | the global average pool before the classifier |
| `dinov2-vits14` | 224 | 768 | timm `vit_small_patch14_dinov2.lvd142m` (`img_size=224`, `dynamic_img_size`) | the final norm's CLS token (384) and the mean of its 256 patch tokens (384), concatenated |

The PyTorch encoders take the ImageNet mean and standard deviation (their weights' own) and need the
`features` extra: `uv sync --extra features` installs PyTorch's CPU build (from PyTorch's CPU index on
Linux and Windows), `uv sync --extra cu128` its CUDA 12.8 build on a GPU machine; the two exclude each
other. Their weights download on first use into torch's and Hugging Face's caches. `--device`
(`auto`: CUDA when there is one) and `--precision` (`auto`: fp16 autocast on CUDA, fp32 on the CPU) say
where and how they run. `scripts/check_encoders.py` runs them with their weights on the tests' synthetic
dataset: CI's manual `weights` job and the GPU machine use it.

**The selection.** The clips of a manifest (`--manifest`, parquet or CSV) or of the root (the manifest is
built first), filtered by `--split`, `--session` (an id or a unique prefix), `--camera`, `--segment` and
`--usable-only` (on by default: the filter's usable clips; `--no-usable-only` takes them all), in session,
attempt, segment and camera order, the first `--limit` of them.

**The framing** (`--crop`, `cubetrace_ml/framing.py`). Every frame of a clip is cut to one rectangle,
scaled (area averaging) so that its longer side is the encoder's input and letterboxed, centred on black,
to a square: a square rectangle is scaled without bars, so nothing is distorted. The cut, the scale and
the bars run in an FFmpeg filter graph on the decoder's frames.

- `auto` (the default): the record's `crop` when the clip has one (the laptop's), else the motion's.
- `record`: the record's `crop`, else the whole frame. `none`: the whole frame.
- **The motion's square.** The clip decoded in gray, scaled so that its shorter side is 160 pixels; the
  absolute differences of consecutive frames summed over the frames of the segment's window (`inWindow`;
  all the frames when fewer than two are in it); a 5-tap binomial blur; the energy at or above 15% of its
  peak; the bounding box of that mass clamped to the 5th and 95th percentiles of its column and row sums
  (a speck far from the hands carries too little of the mass to move them); each side moved out by 15% of
  the box's size; the shorter side grown to the longer about the centre, at least a quarter of the
  frame's shorter side and at most all of it; shifted into the frame; whole, even pixels. Nothing moving
  gives the whole frame. The parameters are recorded with the crop.

`cubetrace-ml crop-preview` (the clip by `--session --attempt --camera --segment`, or `--row` of a
manifest) draws the record's rectangle (blue) and the motion's (orange) on a frame, by default the middle
of the segment's window, beside the motion's energy map, and says which one `auto` takes.

**The files.** `<out>/<encoder>/<sessionId>/<nnnn>/<camera>.<segment>.npz`, written with NumPy (load with
`np.load(path)`; no pickles):

| Array | Type | Meaning |
|---|---|---|
| `x` | float16, frames × dim | the encoder's vector of every frame of the clip, in order |
| `tMs` | float64, frames | the frames' host times, `t0HostMs` + cumulative `dtMs` (the track's `tMs`) |
| `shownMs` | float64, frames | `tMs − lag`, `tMs` for an unsynced clip (the track's `shownMs`) |
| `inWindow` | bool, frames | `shownMs` inside the segment's window, on the run's time base |
| `meta` | a JSON string | below |

`meta`: `format` (1); `encoder` (`name`, `inputSize`, `dim`, `mean`, `std`, `weights`, `output`, and
`loaded`: `pretrained`, or the stub's `seed 0`); `cropMode` (the `--crop` asked for); `crop` (`x`, `y`,
`w`, `h` in the video's pixels, `source` `record`, `motion` or `none`, `letterboxed`, `frameWidth`,
`frameHeight`, `cached`, and the `motion` parameters for a motion crop); `clip` (`sessionId`,
`attemptIndex`, `camera`, `segment`, `frames`, `video`, `width`, `height`, `fpsNominal`, `windowMs`);
`timeBase`; `lagMs` and `unsynced`; `app` (the record's build: `version`, `commit`); `cubetraceMl`
(`version`, and `commit` and `dirty` when it runs from its checkout); `host` (`device` `cpu` or `cuda`,
`deviceName`, `precision`, `torch`); `timing` (the clip's `cropSeconds`, `decodeSeconds`,
`encodeSeconds`, `decodeFps`, `encodeFps`, and `wallSeconds` from the start of its decoding to its file,
which includes its wait when the decoding runs ahead); `file` (the path under the root); `writtenAt` (the
wall-clock time, UTC).

**Resuming.** A clip whose file is there, readable, with the same encoder, crop mode and frame count is
skipped (`--force` rewrites it); a file is written under a temporary name (`….npz.<id>.tmp`, never read)
and renamed, so an interrupted run leaves no partial `.npz`. The motion squares are kept in
`<out>/crops/<sessionId>/<nnnn>/<camera>.<segment>.json` (with the frame count, the time base and the
parameters), so a second encoder does not decode a clip twice; `--force` finds them again. A clip whose
decoded frame count is not its frames file's fails without stopping the run (the command then exits 1).

**Throughput.** `--workers` clips are decoded at once in threads ahead of the encoder (`--batch` frames
per encoder call). The run ends with each stage's frames per second: the motion crop, the decode with
the cut and the scale, the encoder, and the whole. `cubetrace-ml bench --encoders none,stub,…` measures
the same stages without writing anything (`none`: the decode, cut and scale alone at `--size`;
`--random-weights`: the PyTorch encoders' architectures without their weights, the same compute) and
prints a Markdown table.

## The labels and the runs

`cubetrace-ml train` trains a temporal model on the cached features and `cubetrace-ml evaluate` scores it
on a split, against a trivial baseline (both need the `features` extra: PyTorch, CPU or CUDA).

**The labels** (`cubetrace_ml/labels.py`), for each usable clip of a split of the manifest, from the
records through `align_clip` (on the run's time base):

- **The kept frames**: the segment's window (the frames whose `shownMs` is in it) and the frames nearest
  its first and last onsets (a window shorter than a frame interval has none of its own), plus `margin`
  frames (15) on each side, within the clip. With `--fps f`, every k-th of them from the first, k =
  round(the clip's measured rate / f): 2 for 15 on the 30-fps clips; the margin counts the clip's frames.
- **The reference**: every symbol of the attempt whose onset on the frames (`onset + lag`, the lag 0
  for an unsynced clip) falls within the kept frames (and half a kept interval beyond the first and the
  last), in time order: the segment's symbols, and any onset of the other segment that the margin reaches
  (counted as `foreign`; a very short inspection).
- **The per-frame target**: class 0 is "no onset" and class s + 1 the alphabet's symbol s (the order of
  "The alphabet" above). Each onset, in time order, puts its class on the kept frame nearest it; an onset
  whose frame is taken goes to the free neighbour nearer its time, and an onset with neither (three
  onsets within one frame) gets no frame: it stays in the reference and is counted as a collision. With
  `label_frames` k (`--label-frames`), the frames within k of an onset's frame take its class with the
  weight `soft_decay ** d` (0.5 to the power of the distance) and "no onset" with the rest; the nearer
  onset's frame wins.
- **The gyro's channels**: per kept frame, 9 numbers (`GYRO_CHANNELS`): the cube's orientation `qx qy qz
  qw` at the frame (the track's slerp of `gyro.json` at `shownMs`, x, y, z, w as the app records them;
  zeros where the frame has none: before the first sample, after the last, or without `gyro.json`), its
  change since the previous kept frame `dqx dqy dqz dqw` = `q_t · conj(q_{t−1})` (the Hamilton product: the
  rotation that takes the previous kept frame's orientation to this one, in the hemisphere w ≥ 0; the
  identity (0, 0, 0, 1) at the first kept frame and wherever either frame has none; with `--fps 15`,
  between kept frames, two frames apart) and the presence flag `gyro` (1 where the frame has an
  orientation, else 0). `gyro.json` is read only when the run asks for it (`data.inputs` = `features+gyro`
  or `data.require_gyro`); otherwise the channels say "none" and nothing changes. The app's quaternions run
  continuously (no sign flip between two samples on the mirror) but are not kept in one hemisphere: one
  orientation can come as q in one attempt and as −q in another. A calibrated run (`data.calibration`,
  below) takes the change in the cube's own frame instead, `conj(q_{t−1}) · q_t`, which no change of the
  reference frame alters, and the attempt's scramble pose; its model turns q into the camera's frame
  first.
- **The skips**, counted by reason: `records` (its records cannot be read, `gyro.json` included when it is
  read), `no-window` (its segment has no move and no frame in its window), `moves-outside-frames` (an
  onset of its segment falls outside the clip's frames), `no-gyro` (with `data.require_gyro`: no
  `gyro.json`, or its samples' span reaches none of the kept frames; a clip partly covered is kept, its
  frames without an orientation flagged 0), `no-features` (no readable features file of the encoder) and
  `features-mismatch` (the file's frame count is not the frames file's, or a kept frame's `tMs` differs by
  more than 0.01 ms). A split's counts also say how many clips and kept frames have an orientation, when
  the gyro was read.
- **The features**: the kept frames' rows of the clip's `.npz`, held in memory as float16 for the whole
  run and cast to float32 batch by batch.
- **The inputs** (`data.inputs`): `features` (the default: the features alone, as before M3, bit for bit)
  or `features+gyro` (each frame's features and then its 9 gyro channels, 768 + 9 for DINOv2; with a
  calibration, the 14 calibrated channels instead, 768 + 14).

**The model** (`models.py`): the inputs standardized with the training clips' mean and standard deviation
per channel (kept in the model's state; the gyro's presence flag as it is, mean 0 and deviation 1, since
it can be 1 on every training frame), a linear projection to `width` (256), `conv_layers` (2) residual
1D convolutions of kernel `conv_kernel` (5) with GELU and a layer norm, then the body, `bigru` (a BiGRU of
`gru_layers` 2 and `gru_hidden` 128 per direction, over the clips packed to their lengths) or
`transformer` (an encoder of `transformer_layers` 4, `transformer_heads` 4, `transformer_ff` 1024, pre-norm,
sinusoidal positions, the padding masked out), dropout `dropout` (0.2), and a linear layer of 25 outputs
per frame. The head decides what they mean:

- `perframe`: "no onset" and the 24 classes. The loss is the cross-entropy against the (soft) target with
  the class weights 1 for "no onset" and N0/N1 for every onset class (N0 and N1 the training clips' frames
  without and with an onset), so that both weigh the same in all. The decoding: P(onset) = 1 − P(no
  onset); its peaks are the local maxima at or above the threshold (a plateau's first frame), at least
  `min_distance` frames (2) apart (the higher kept); a peak's symbol is the onset class with the highest
  probability summed over the peak and `neighbours` (1) frames on each side; its onset time is the peak's
  frame time. The threshold is chosen on val among 0.1, 0.15, …, 0.9 by the pooled F1@50 (symbol), the one
  nearest 0.5 among equals, after every epoch.
- `ctc`: the blank (0) and the 24 symbols, PyTorch's CTC loss against each clip's reference (each clip's
  loss over its reference's length, a clip that cannot be aligned counting 0). The decoding is greedy:
  each frame's most probable class, repeats collapsed and blanks dropped; a symbol's onset time is the
  first frame of its run.

**The training** (`train.py`): AdamW (`lr` 1e-3, `weight_decay` 0.01), the learning rate on a cosine
from `lr` to 0 over `epochs` (30) epochs, stepped per batch; batches of `batch` (8) whole clips padded at
the end with a mask; `time_masks` (3) spans of 1 to `time_mask_frames` (10) frames of each training clip
zeroed (the standardized features) as the augmentation; the gradient's norm clipped at `clip_grad` (1);
the clips' order and the masks drawn from a NumPy generator of the seed (`seed`, which also seeds
PyTorch and `random`), so one seed gives one run on the CPU. After every epoch the val split is decoded:
the run keeps the epoch with the best val F1@50 (symbol, pooled) for the per-frame head, the best val WER
(pooled) for CTC, and stops after `patience` (8) epochs without a better one, never before `min_epochs`
(10): CTC emits nothing for its first hundred-odd steps, and a stop inside that plateau would keep an
empty model.

**The configuration** is a TOML file (`configs/perframe-bigru.toml`, `configs/ctc-bigru.toml`,
`configs/perframe-transformer.toml`) of the sections `data` (`margin`, `fps`, `label_frames`,
`soft_decay`, `time_base`, `inputs`, `require_gyro`), `model`, `train`, `decode` (`min_distance`,
`neighbours`, `threshold`) and `paths` (`features`, `encoder`, `root`, `manifest`), every key optional;
`--set section.key=value` overrides one (the value read as TOML: `--set train.epochs=5`,
`--set data.require_gyro=true`; a bare word is a string: `--set data.inputs=features+gyro`), and `--fps`,
`--label-frames`, `--device`, `--features`, `--encoder`, `--root` and `--manifest` say the same as their
keys. The run's name is the file's stem, or `--set name=…`.

```
cubetrace-ml train --config configs/perframe-bigru.toml --root <dataset> --features <features root> \
    --encoder dinov2-vits14 --manifest <manifest.parquet> --out runs/perframe-bigru
cubetrace-ml evaluate --run runs/perframe-bigru --split test --consistency
```

What the orientation buys, on the same clips: two runs that differ only in their inputs, both with
`--set data.require_gyro=true` (the clips without a gyro skipped by both), one of them with
`--set data.inputs=features+gyro`, and each one's `evaluate`.

**The run folder** (`--out`, by default `runs/<name>`, which git ignores; `--force` clears a run that is
there):

| File | What |
|---|---|
| `config.json` | the resolved configuration, the paths included |
| `log.csv` | per epoch: `epoch`, `train_loss`, `val_loss`, `val_f1_50` (pooled, symbol), `val_wer` (pooled), `lr` (at the epoch's end), `seconds`, `threshold` (the per-frame head's choice on val) |
| `best.pt`, `last.pt` | the best and the last epoch: `state` (the state dict), `config`, `dim` (the input's width: the features', plus 9 with the gyro, 14 with the calibrated matrix), `epoch`, `metrics`, `threshold` (the epoch's), `baseline` (its settings) and `data` (the train and val splits' counts, `gyroClips` and `gyroFrames` among them: null when the gyro was not read), and a calibrated run's `calibration` (below); `torch.load(path, weights_only=True)` reads them |
| `manifest/` | the manifest, when the run built it from the root (no `--manifest`) |
| `report.md` | `evaluate`'s tables: the run (its inputs among them), the counts of each split (and the gyro's coverage when the run read it), the model against the baseline (means and pooled), by segment, by TPS bucket, the consistency pass, the confusions, a calibrated run's calibration (below), F1 against the tolerance |
| `plots/*.png` | `loss.png` (the losses and the val metric by epoch), `f1-tolerance.png` (pooled F1, timing and symbol, at ±10 to ±100 ms), `wer-tps.png` (the mean WER per TPS bucket), `onsets.png` (one clip's P(onset) over its first 8 s, the reference onsets as lines, the predicted ones as dots: the solve clip of the median WER) |
| `predictions.parquet` | one row per clip: the clip, its TPS and bucket, the kept frames and stride, the reference's and each system's symbols and onset times (ms on the frames' timeline), and each system's WER, F1, exact match and replay |
| `metrics.json` | the aggregates of `report.md`, by system, segment and bucket, every tolerance from 10 to 100 ms; the run's `inputs` and `requireGyro`; the model's `confusions` (below): `matched`, `reference`, `kinds`, `perSymbol` (`right`, `matched`, `accuracy`), `top` (the twelve), `byCamera` (`right`, `wrong`, `unmatched`, `accuracy`, `recall`) and the whole `matrix` (reference → predicted → onsets); and a calibrated run's `calibration` (below) |
| `calibration.parquet` | a calibrated run's rotation of every key of the split by mode (below) |

`evaluate --split val` writes `report-val.md`, `plots-val/`, `predictions-val.parquet`, `metrics-val.json`
(and `calibration-val.parquet`) instead (`--split train` the `-train` ones); `--checkpoint last` takes
`last.pt`.

**The metrics** (`metrics.py`), per clip, of each system's sequence against the reference:

- **WER**: the edit distance between the two symbol sequences (each substitution, insertion and deletion
  1) over the reference's length; a clip with an empty reference has none.
- **Onset F1** at ±25 and ±50 ms (and every 5 ms from 10 to 100 for the curve): the predicted onsets
  matched one to one to the reference onsets within the tolerance (inclusive), greedily in time order:
  each prediction takes the earliest unmatched reference onset within reach, which matches as many as
  any one-to-one matching can when every onset has the same reach. `timing` ignores the symbols;
  `symbol` matches equal symbols only. F1 = 2·TP / (2·TP + FP + FN).
- **Exact**: the edit distance is 0. **Replay** (a solve clip): the predicted sequence, applied to the
  attempt's `scrambledFacelets` (`cube.py`: the 54 facelets in Kociemba order, a double a half turn, a
  slice its pair of face turns: `M` is `R` then `L'`), leaves every face one colour. The report also
  counts the solve clips whose reference replays: all of them, unless the labels or the simulator are
  wrong.
- **The aggregates** of a split, of a segment and of a TPS bucket (the attempt's `result.tps` in bins of
  0.5, `4.0–4.5`; a scramble clip takes its attempt's): the means over the clips (a clip without a value
  left out) and the pooled counts (the edits over all the reference symbols; the F1, precision and recall
  of the summed matches).
- **The confusions** (the model's decoding only, from the predictions table: `report.confusions_of` reads
  a run's `predictions.parquet` too): the model's onsets matched one to one to the reference's within ±50
  ms on timing (the matches of F1@50 timing), and for each match the kind of the predicted symbol: `right`;
  `same face, other turn` (`R'` or `R2` for `R`, `M'` for `M`); `opposite face` (`L` for `R`; the slices
  are a family of their own: `S` for `M` counts here); `other face` (the rest, a slice for a face turn
  among them). Counted by kind, by reference symbol (the share right at its matched onsets), by pair (the
  twelve most frequent confusions in the report) and by camera, a camera's clips told apart by whether
  they have a lag (`laptop (lag)`, `phone-rear (no lag)`): the matched onsets right and wrong, and the
  reference onsets unmatched.

**The systems** each clip is scored for:

- **model**: the head's decoding above.
- **model + consistency** (`evaluate --consistency`): the model's sequence with adjacent predictions
  merged by the normalization's rules (two equal quarter turns of a face under 200 ms apart are its
  double; two opposite faces turning the same way under 20 ms apart a slice) and then adjacent cancelling
  pairs dropped (`R R'`, `R2 R2`, `M M'`) until none is left; the merged symbol keeps the first one's
  time.
- **baseline**: the motion alone. Each frame's distance from the previous one in the standardized
  features (the features alone, whatever the inputs), over the clip's 99th percentile and clipped to 1;
  its peaks as above (at least 2 frames apart) at or above a threshold chosen on val by the pooled F1@50
  (timing), moved back by the training clips' median offset from a reference onset to the nearest peak
  within 100 ms, and every one of them the training clips' most frequent symbol.

## The orientation in the camera's frame

The gyro's quaternion q gives the cube's orientation in the gyro's own frame, whose relation to a camera is
unknown; M3's runs gave the model q as it is, and the model could not learn which side face the camera sees
from it. A calibrated run turns it into a camera's frame first: `q_cam = c · q`, one rotation c per attempt
(or per attempt and camera).

**The gyro's frame** (`cubetrace-ml gyro-frames`, `gyroframes.py`; `--out <folder>` writes `gyro-frames.md`,
`gyro-frames.json` and `gyro-frames.parquet`, one row per attempt and segment). For every attempt with
`gyro.json`, over each segment's window: every sample's rotation matrix (column k: the cube's axis k in the
gyro's frame); per cube axis its principal direction over the samples, sign-free (the top eigenvector of the
orientation tensor of its columns: an axis held up and held down count alike), and its concentration (the
top eigenvalue: 1 for one direction throughout, 1/3 for every direction alike); the segment's mean
orientation (the chordal mean: the top eigenvector of Σ q qᵀ) and the samples' angles from it. Then:

- **The convention**: Spearman's correlation of the cube's angular velocity `v` (sample t + 1's) with the
  change from sample t to t + 1 as a rotation vector per second, taken in the cube's frame
  (`conj(q_t) · q_{t+1}`) and in the gyro's (`q_{t+1} · conj(q_t)`). A gyroscope measures in its own body:
  the frame `v` follows says that q takes the cube's axes into the gyro's frame (`v = cube frame`), so that a
  change of reference acts on the left, `c · q`.
- **Gravity**: the per-attempt principal directions of each cube axis pooled over the attempts (their own
  orientation tensor). The cube axis whose directions stay together across attempts and sessions (pooled
  concentration at least 0.8) is the one held vertical, and its direction is gravity's in the gyro's frame;
  within 15° of a gyro axis, that axis is `gravity_axis`, and only a yaw about it is arbitrary. When no axis
  stays together, the gyro's frame is arbitrary in three degrees of freedom.
- **The hold**: the vertical cube axis along gravity's direction or against it, per segment, and its tilt.
- **The yaw**: each attempt's scramble pose (the mean orientation over its scramble) and its heading about
  the gravity axis (`orientation.heading`: the angle of the rotation about the axis that, undone, brings the
  pose nearest the identity); per session (in the order of their first attempt) its circular mean and
  spread, its range, its drift per hour (a line through the unwrapped headings, for a session of half an
  hour or more) and the residual, the median change between consecutive attempts; between sessions, how far
  their means sit apart.

**The scramble's pose.** The app prescribes the scramble in the cube's own frame, so the solver holds the
cube one way while applying it: the attempt's scramble pose (`orientation.scramble_pose`: the chordal mean
of the gyro's samples inside `scrambleStart`–`scrambleDone` on the run's time base, at least 3 of them) is
its reference, and undoing it (`conj(pose)` for a rotation; the rotation about the gravity axis by minus its
heading for a yaw) calibrates the attempt without labels, as far as the solver holds the cube the same way
toward the cameras each time.

**The calibrated channels** (`orientation.calibrated_channels`, `calibrate.calibrated_inputs`), per kept
frame after the features: the orientation in the camera's frame `c · q`, as its rotation matrix's 9
entries row by row (`data.orientation` = `matrix`, the default: continuous and the same for q and −q) or as
the quaternion in the hemisphere w ≥ 0 (`quat`), zeros where the frame has no orientation; the cube's own
change `conj(q_{t−1}) · q_t` (4); the flag. 14 channels with the matrix, 9 with the quaternion. They are
standardized like the features, with the training clips' channels at the training's starting rotations,
the flag as it is.

**The configuration** (`data`, with `inputs = features+gyro`):

| Key | Values |
|---|---|
| `calibration` | `none` (the default: M3's channels, the gyro's own frame, bit for bit); `identity` (the calibrated channels, c the identity); `pose` (c each attempt's scramble pose undone, the identity without one); `attempt` (one rotation learnt per training attempt, keyed `sessionId/attemptIndex`); `camera` (one per attempt and camera, `sessionId/attemptIndex/camera`) |
| `calibration_init` | the learnt rotations' start: `pose` (the default: each key's scramble pose undone, the identity without one) or `identity` |
| `calibration_dof` | `yaw` (the default: one angle about `gravity_axis`) or `rotation` (a rotation vector through the exponential map) |
| `gravity_axis` | `x`, `y` or `z` (the default: the records' gravity, `gyro-frames`) |
| `orientation` | `matrix` (the default) or `quat` |

and in `train`, `calibration_lr` (0.01: the learnt rotations' learning rate, with no weight decay, on the same
cosine schedule) and `session_tie` (0: off; a weight times the mean squared distance of each key's rotation
matrix from its session's mean, added to the loss).

**Learning** (`calibration = attempt` or `camera`): each training key has its own parameter (an angle, or a
rotation vector), so every rotation stays a unit quaternion; it is trained with the network, by the
gradients of its clips' batches. The checkpoint's `calibration` record keeps the keys, their learnt
rotations and starts, the label-free guess (below), the class weights and the features' width.

**The clips the training never saw** (val at every epoch, the evaluated split) take the run's label-free
guess (`calibrate.Guess`): the identity for `identity`; the scramble pose undone for `pose`; for a learnt
calibration, the scramble pose undone and then the training's mean correction (the chordal mean of each
learnt rotation times its own start undone), when the poses predict the learnt rotations (at least 80% of the
training keys' corrections within 45° of that mean), else the learnt rotations' mean.

**The evaluation** of a calibrated run gives the split's clips their rotations four ways, on the same clips
(`evaluate --calibrate` chooses whose numbers the report's other sections give: `scramble` by default):

- `none`: the identity, the gyro's frame as it is (what M3 measured);
- `pose`: the label-free guess;
- `scramble`: each key's rotation fit on its scramble clips' labels, the network frozen: honest for the solve
  clips, whose labels it never reads (the app prescribes the scramble before the solve);
- `all`: fit on all its clips' labels: the oracle.

A fit (`calibrate.fit_calibrations`) scores the key's clips' loss (the training's: the per-frame head's
weighted cross-entropy with the training's class weights, CTC's loss for CTC) at every candidate of a grid,
the yaws about the gravity axis in 15° steps (24; for `rotation`, those yaws composed with the cube's 24
symmetries, 144 distinct rotations), and at the key's guess; then refines the best three (`--refine
search`, the default: a compass search, each moving to its best neighbour at ± the step about the gravity
axis, or about each axis for a rotation, up to twice a step, the step halving from 7.5° (22.5° for a
rotation) until under 1°, forward passes only; `--refine adam`: twelve Adam steps on the angle or the
rotation vector, the learning rate decayed tenfold, all three at once). A key without a clip to fit on keeps
its guess. On this machine's CPU a forward and backward pass through the default BiGRU runs at a tenth of the
forward's rate, so the search is the default; a dozen Adam steps also fall short of the rotation grid's
coarse tilt.

The report's **Calibration** section: the four side by side on the solve clips and on the scramble clips
(where `scramble` is fit: its numbers there are optimistic), WER, F1@50 timing and symbol, F1@25 symbol, the
share right at the matched onsets and the replay; the side faces' per-symbol accuracies on the solve clips
(`R` … `B2`, the side faces pooled, U and D for comparison); each mode's rotations (where they came from, their
angle to the guess, the loss gained over the identity) and the honest rotation's angle to the oracle's.
`metrics.json` adds `calibration` (the settings, each mode's aggregates over all, the solve and the scramble
clips and its confusions on the solve clips, the fits' summaries) and the run folder
`calibration.parquet`: one row per key and mode (`pose`, `scramble`, `all`) with `key`, `sessionId`,
`attemptIndex`, `camera` (for `camera` keys), `mode`, the rotation `qx qy qz qw`, its heading `yawDeg`, its
`angleToGuessDeg`, the fit's `loss`, `identityLoss` and `guessLoss`, the `clips`, `frames` and `onsets` it was
fit on, and its `source` (`grid`, `guess`, `guess (no clip)`, or the guess's `pose`, `mean`, `identity`).

```
cubetrace-ml gyro-frames --root <dataset> --out <folder>
cubetrace-ml train --config configs/perframe-bigru.toml --root <dataset> --features <features root> \
    --encoder dinov2-vits14 --manifest <manifest.parquet> --set data.inputs=features+gyro \
    --set data.require_gyro=true --set data.calibration=attempt --out runs/perframe-bigru-calibrated
cubetrace-ml evaluate --run runs/perframe-bigru-calibrated --split test --consistency
```
