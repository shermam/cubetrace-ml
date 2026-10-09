# CLAUDE.md — cubetrace-ml

Python code that turns the capture app's recordings into a dataset, features and models. The coordinator
session plans and reviews; implementing agents do one task each on a branch `task/<id>-<slug>`, one PR
per task, squash-merged by the coordinator. Read `docs/PLAN.md` (the board, the contracts, the Outcome
notes) before anything else.
`docs/pipeline/` is the explanatory series for the owner (the pipeline top down, one page per stage);
keep it in step with the code when a stage changes.

## The data

- The records are those of [shermam/cubetrace](https://github.com/shermam/cubetrace) `docs/DATA-MODEL.md`:
  one folder per attempt, `sessions/<sessionId>/attempts/<nnnn>/` with `attempt.json` (the moves with
  `cubeMs` on the cube's clock and `hostMs` the arrival of their packet on the host clock, the attempt's
  cube-clock fit `clock {a, b}` that puts a move on the host clock without the packets' jitter, the
  events, the result, the
  `video[]` entries with each clip's `firstFrameHostMs`, `framesFile` and `syncResidualMs`, the
  camera's lag), one `<camera>.<segment>.mp4` and `<camera>.<segment>.frames.json` per clip (per-frame
  host times: `t0HostMs` plus the cumulative `dtMs`), `gyro.json` (the cube's orientation samples), and
  `sessions/<sessionId>/session.json` (the cameras, their sync checks in `clock.cameras`). The JSON
  Schemas are vendored in `schemas/`.
- A dataset root is a folder (or a `gs://` prefix) with that layout: locally `CUBETRACE_DATA=<root>`.
  The real recordings are the owner's and are **never committed, never copied into the repository, never
  printed** (no frames, no file listings with the owner's ids beyond what a test needs); the tests use
  synthetic fixtures generated in the test suite. The bucket is read with Application Default
  Credentials by the coordinator only: agents work on the local mirror the coordinator provides and
  never handle keys (never print the environment).
- The moves: the cube reports quarter turns (`R`, `R'`, …); a double turn is two quarter turns 70–110 ms
  apart and a slice move two opposite-face turns reported together. The 24-symbol alphabet of the
  research notes merges them; `ferramentas/cubo.py`'s `normalizar()` (the owner's simulator, provided to
  the task that needs it) is the reference for that merge.

## Conventions

- Python 3.11+, `uv` (`uv sync`, `uv run …`), `src/` layout (`cubetrace_ml`), `ruff` (check and format),
  `pytest`; a `cubetrace-ml` console script for the commands. Video is decoded with PyAV (`av`), which
  bundles FFmpeg: no system `ffmpeg` is assumed. Heavy work is batch and resumable; nothing here needs a
  GPU before M1.
- The encoders (M1) need the `features` extra (PyTorch's CPU build; `cu128` on a GPU machine), which
  comes from download.pytorch.org; the agents' container cannot reach it (nor huggingface.co, where the
  weights are): there `uv sync` takes no PyTorch extra, the PyTorch tests skip, and a lock change to the
  PyTorch packages is resolved on a runner. CI installs the CPU build and runs every test.
- Every PR runs `uv run ruff check`, `uv run ruff format --check` and `uv run pytest`, locally and in CI
  (`.github/workflows/ci.yml`). Tests never touch the network or real data.
- Commit messages: a sentence as the title, a body that says why; push after every commit; the commit's
  trailers as the coordinator's brief gives them. Docs in English, in the present tense; each task ends
  with its Outcome note in `docs/PLAN.md`.
