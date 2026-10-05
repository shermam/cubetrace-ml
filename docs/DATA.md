# The data, as cubetrace-ml reads it

The records are the capture app's ([shermam/cubetrace](https://github.com/shermam/cubetrace)
`docs/DATA-MODEL.md`); this page says how this repository consumes them: the timelines, the lag, the
alphabet, the per-frame track, the filter, the splits and the manifest. The package is
`src/cubetrace_ml`, the command `cubetrace-ml`.

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
