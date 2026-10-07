#!/bin/bash
# cubetrace-ml: the GPU machine's startup script (Compute Engine metadata `startup-script`; see docs/GPU.md).
#
# It runs as root when the machine boots: waits for the GPU driver, installs uv and this repository,
# mirrors the dataset to the local disk, extracts the features of the requested encoders on the GPU,
# syncs them and the logs to the bucket, and shuts the machine down, so that a forgotten machine stops
# costing. Its settings are instance metadata (scripts/gpu/vm.py sets them):
#   ml-data-root    gs://cubetrace-data/users/<uid>        required: the dataset root
#   ml-out-prefix   gs://cubetrace-data/features/<run>     required: where the features and logs go
#   ml-encoders     dinov2-vits14,resnet18                 the encoders, in order (default)
#   ml-ref          main                                   the branch, tag or commit of shermam/cubetrace-ml
#   ml-args         extra arguments of `cubetrace-ml features`, e.g. "--split train --limit 20"
#   ml-workers      decoding threads (default: the vCPUs minus 2, at least 2)
#   ml-shutdown     1 (default) shuts the machine down at the end; 0 leaves it up
# The line `CUBETRACE-ML-STATUS <n>` on the serial console is the driver's signal (0: everything written).
set -uo pipefail
LOGDIR=/var/log/cubetrace-ml
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/startup.log") 2>&1
echo "=== cubetrace-ml startup $(date -u +%FT%TZ) on $(hostname) ==="

MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
meta() { curl -sf -H 'Metadata-Flavor: Google' "$MD/$1" 2>/dev/null || echo "${2:-}"; }
DATA_ROOT=$(meta ml-data-root)
OUT_PREFIX=$(meta ml-out-prefix)
ENCODERS=$(meta ml-encoders dinov2-vits14,resnet18)
REF=$(meta ml-ref main)
ARGS=$(meta ml-args)
WORKERS=$(meta ml-workers)
SHUTDOWN=$(meta ml-shutdown 1)
MARK=/var/lib/cubetrace-ml-done

finish() {
  local status=$1
  echo "=== cubetrace-ml finished with status $status at $(date -u +%FT%TZ) ==="
  if [ -n "$OUT_PREFIX" ]; then
    gcloud storage cp -r "$LOGDIR" "$OUT_PREFIX/logs/$(hostname)-$(date -u +%Y%m%dT%H%M%SZ)" || true
  fi
  echo "CUBETRACE-ML-STATUS $status"
  # The guest agent forwards this output to the serial console with a lag: give it a moment, or the
  # driver following the console sees the machine stop before the status line (the trial of 2026-10-07).
  sleep 20
  if [ "$SHUTDOWN" = "1" ]; then shutdown -h now; fi
  exit "$status"
}

if [ -f "$MARK" ]; then echo "this machine already ran its job ($MARK): nothing to do"; finish 0; fi
if [ -z "$DATA_ROOT" ] || [ -z "$OUT_PREFIX" ]; then echo "ml-data-root and ml-out-prefix are required"; finish 2; fi
echo "data root $DATA_ROOT; out prefix $OUT_PREFIX; encoders $ENCODERS; ref $REF; args '$ARGS'"

# The GPU: the Deep Learning VM image installs the NVIDIA driver on its first boot; wait for it.
for _ in $(seq 1 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi || { echo "no NVIDIA driver after 10 minutes"; finish 3; }
echo "vCPUs $(nproc); memory $(free -g | awk '/Mem:/ {print $2}') GB; disk $(df -h / | awk 'NR==2 {print $4}') free"

# uv, the code and its environment (torch's CUDA 12.8 build runs on the image's 580 driver).
export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh || finish 4
export PATH="/root/.local/bin:$PATH"
rm -rf /opt/cubetrace-ml
git clone --quiet https://github.com/shermam/cubetrace-ml /opt/cubetrace-ml || finish 4
cd /opt/cubetrace-ml || finish 4
git checkout --quiet "$REF" || finish 4
git log --oneline -1
uv sync --locked --extra cu128 --extra gcs || finish 4
uv run --no-sync python scripts/check_encoders.py --device cuda || finish 5

# The dataset, mirrored to the local disk: the same region, so minutes; a bucket root would pay a
# round trip per file.
mkdir -p /mnt/data /mnt/features
gcloud storage rsync -r "$DATA_ROOT/sessions" /mnt/data/sessions || finish 6
du -sh /mnt/data
export CUBETRACE_DATA=/mnt/data
uv run --no-sync cubetrace-ml report | tee "$LOGDIR/report.txt" || true

# The features, one encoder at a time; the motion crops are found once and reused from /mnt/features/crops.
if [ -z "$WORKERS" ]; then WORKERS=$(( $(nproc) - 2 )); fi
[ "$WORKERS" -lt 2 ] && WORKERS=2
status=0
for enc in ${ENCODERS//,/ }; do
  echo "=== features $enc (workers $WORKERS) $(date -u +%FT%TZ) ==="
  # shellcheck disable=SC2086  # ml-args is a list of arguments
  uv run --no-sync cubetrace-ml features --root /mnt/data --out /mnt/features --encoder "$enc" \
    --device cuda --workers "$WORKERS" $ARGS 2>&1 | tee "$LOGDIR/features-$enc.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then echo "features $enc exited $rc"; status=7; fi
  gcloud storage rsync -r /mnt/features "$OUT_PREFIX" || status=8
done
du -sh /mnt/features
touch "$MARK"
finish "$status"
