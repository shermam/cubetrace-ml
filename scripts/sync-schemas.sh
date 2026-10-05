#!/bin/sh
# Copies the record schemas from a cubetrace checkout: scripts/sync-schemas.sh ../cubetrace
set -eu
src="${1:?path to a cubetrace checkout}/packages/core/schema"
for name in attempt frames gyro session; do cp "$src/$name.schema.json" "$(dirname "$0")/../schemas/$name.schema.json"; done
echo "schemas copied from $(git -C "$1" rev-parse --short HEAD)"
