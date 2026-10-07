# The GPU run

The features of the whole dataset (M1's `cubetrace-ml features`) are extracted on one GPU machine in
us-central1, next to the bucket, by the coordinator; nothing else in this repository needs a GPU before
M2's training, which reuses the same recipe. The machine is a batch job: it boots, does its work, syncs
the results to the bucket and shuts itself down. No one logs into it.

## What runs where

- `scripts/gpu/startup.sh` is the machine's startup script. It waits for the NVIDIA driver, installs
  `uv` and this repository at the requested ref with the `cu128` and `gcs` extras, runs
  `scripts/check_encoders.py --device cuda`, mirrors `<data root>/sessions` to the local disk, runs
  `cubetrace-ml features` for each requested encoder with `/mnt/features` as the features root, syncs
  that folder to the run's bucket prefix after each encoder, uploads its logs, prints
  `CUBETRACE-ML-STATUS <n>` on the serial console (0: everything written) and shuts the machine down.
  A machine that boots again finds its marker file and does nothing but shut down.
- `scripts/gpu/vm.py` is the coordinator's driver (`check`, `create`, `status`, `serial`, `delete`): it
  creates the instance from the Deep Learning VM image family `common-cu129-ubuntu-2404-nvidia-580`
  (Ubuntu 24.04, CUDA 12.9, driver 580, which runs torch's CUDA 12.8 build) with the startup script and
  the run's settings as instance metadata, follows the serial console until the status line, and deletes
  the instance. It authenticates with Application Default Credentials; the coordinator points
  `GOOGLE_APPLICATION_CREDENTIALS` at the deploy service account's key for the duration of a command,
  as it does for the bucket.
- The results land under `gs://cubetrace-data/features/<run>/`: `<encoder>/<sessionId>/<nnnn>/
  <camera>.<segment>.npz` as `docs/DATA.md` describes, `crops/` (the motion squares) and `logs/`. M2
  reads them from a local mirror of that prefix.

## The owner's one-time setup

Three grants in the Cloud console of the project `cubetrace-cacd9`, all once:

1. **The deploy service account can run machines.** IAM & Admin → IAM → the principal
   `firebase-adminsdk-fbsvc@cubetrace-cacd9.iam.gserviceaccount.com` (the one whose key the coordinator
   holds) gets the roles **Compute Instance Admin (v1)** and **Service Account User**. The first creates,
   reads and deletes instances and their serial console; the second lets it start a machine that runs as
   the project's default compute service account.
2. **The machine can read and write the bucket.** Cloud Storage → bucket `cubetrace-data` → Permissions →
   grant **Storage Object Admin** to the project's default compute service account,
   `<project number>-compute@developer.gserviceaccount.com` (IAM & Admin → Service Accounts lists it;
   `vm.py check` prints it once step 1 is done). It reads `users/<uid>/sessions/` and writes
   `features/`.
3. **The quotas** (done on 2026-10-05): NVIDIA L4 GPUs and NVIDIA T4 GPUs at 1 in us-central1 on an
   activated billing account. If the first `create` fails with a message naming `GPUS_ALL_REGIONS`,
   request 1 on "GPUs (all regions)" in IAM & Admin → Quotas.

## A run, step by step (the coordinator)

```
uv sync --locked --extra gcs
uv run --no-sync python scripts/gpu/vm.py check                       # the quotas and the default SA
uv run --no-sync python scripts/gpu/vm.py create --name ml-features-1 \
    --data-root gs://cubetrace-data/users/<uid> \
    --out-prefix gs://cubetrace-data/features/2026-10-06 \
    --encoders dinov2-vits14,resnet18                                 # --args "--limit 20" for a trial
uv run --no-sync python scripts/gpu/vm.py serial --name ml-features-1 --follow
uv run --no-sync python scripts/gpu/vm.py delete --name ml-features-1
```

The machine is a `g2-standard-8` (one L4, 8 vCPUs, 32 GB) with a 100 GB balanced disk, on demand
(`--spot` for Spot pricing, which can stop the machine at any time; the run resumes on the next machine,
since the features root is synced after each encoder and `features` skips what is there). `--machine
n1-standard-8 --gpu nvidia-tesla-t4` is the fallback when no L4 is available in the zone (`--zone
us-central1-b`, `-c` and `-f` are the other zones with GPUs). The decode sets the pace, not the GPU:
M1 measured 300 to 400 frames a second per 4 vCPUs on the mirror; the bucket's 670,000 frames
(2026-10-05) should take about 20 minutes per encoder on 8 vCPUs plus the one-time motion pass and the
mirroring, under an hour for both encoders, about one dollar on demand (L4 machines cost about $0.85
an hour in us-central1).

## What the first runs showed (2026-10-07)

- The trial (`--args "--limit 20"`, a `g2-standard-8` in us-west1-a, since every zone of us-central1 and
  us-east1 was out of L4s and T4s that evening and `us-east4-b` has no `g2` machines at all): the whole
  job took 3 minutes 47 seconds from the startup script's first line to the logs' upload, the dataset
  mirror of the 1,241 clips included; `uv sync` installed 87 packages in under two seconds after the
  download; the weights downloaded without a token. The 20 laptop clips (10,836 frames) took 20 s per
  encoder: DINOv2 ViT-S/14 at 994 frames a second and ResNet-18 at 1,419 on the L4 in fp16, while the
  decode ran at about 110 frames a second per worker, 532 overall with 6 workers: **the decode sets the
  pace, the GPU is mostly idle**, as M1 predicted. The machine cost about 6 cents.
- The serial console lags the script: the driver saw the machine stopping before the status line, which
  only the bucket's `logs/` showed. The script now waits 20 s after the status line before shutting down.
- Zone stockouts are the norm for single GPUs: `create` tries the zones in turn; a run can land in any US
  region, the bucket's reads from another region costing about a cent a gigabyte.

## When something goes wrong

- `ZONE_RESOURCE_POOL_EXHAUSTED` on `create`: no L4 free in that zone at that moment; try another zone
  or the T4 fallback.
- A quota message on `create`: the row it names, in IAM & Admin → Quotas.
- `CUBETRACE-ML-STATUS 3`: no driver after ten minutes (the image's first boot installs it; a second
  boot would, too). `4`: uv, git or `uv sync` failed (the console says which). `5`: the encoders could
  not load on CUDA. `6`: the mirror from the bucket failed (the machine's service account lacks the
  bucket role of step 2). `7`: a `features` run exited with an error (its log is under `logs/`). `8`: the
  sync to the bucket failed.
- The machine is always deleted by the coordinator after the status line; `vm.py status` and the
  console's VM instances page show whether one is still running.
