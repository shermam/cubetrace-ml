# cubetrace-ml

The model side of the speedcubing dissertation: from the recordings the capture app
([shermam/cubetrace](https://github.com/shermam/cubetrace)) makes (video of a solve from one or more
cameras, with the move stream the Bluetooth cube reports, on one clock) to a model that reproduces the
move stream from the video alone.

- `docs/PLAN.md`: the phase M board (M0 dataset tooling, M1 features, M2 first models, M3 the cube's
  orientation as an input, M4 the orientation in the camera's frame) with each task's contract and outcome.
- `docs/DATA.md`: the records as this repository consumes them (written by M0), the per-frame
  features `cubetrace-ml features` caches (M1; `uv sync --extra features`, or `--extra cu128` on a GPU),
  and the labels, models, metrics and run folders of `cubetrace-ml train` and `cubetrace-ml evaluate`
  (M2; the same extras; `configs/` holds the example runs), and the gyro's frame (`cubetrace-ml
  gyro-frames`) and the orientation's calibration into a camera's frame (M4).
- `docs/GPU.md`: the GPU machine that extracts the features of the whole dataset (a batch job the
  coordinator runs; the owner's one-time grants).
- `schemas/`: the app's JSON Schemas of the records, copied with their commit.
- The research notes behind the plan live in the owner's private repository (`random-research/
  speedcubing-video-to-moves.md` and the dissertation proposal); the decisions they settled are
  restated in `docs/PLAN.md`.

The dataset itself is never in this repository: it lives in the owner's bucket (`gs://cubetrace-data`,
us-central1) and in local mirrors of it; the tests run on synthetic fixtures.
