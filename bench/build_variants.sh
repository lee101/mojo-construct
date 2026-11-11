#!/usr/bin/env bash
# Builds experimental variants of the codec library for the parallel sweep.
# Writes /tmp/parwork/lib-<workers>-<tag>.so. Not used by the normal build.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p /tmp/parwork
for spec in "$@"; do
  workers="${spec%%:*}"
  threshold="${spec##*:}"
  out="/tmp/parwork/lib-w${workers}-t${threshold}.so"
  sed -e "s/^comptime PARALLEL_THRESHOLD = .*/comptime PARALLEL_THRESHOLD = ${threshold}/" \
      -e "s/^comptime PARALLEL_WORKERS = .*/comptime PARALLEL_WORKERS = ${workers}/" \
      "$repo_dir/src/construct.mojo" > /tmp/parwork/construct-w${workers}-t${threshold}.mojo
  mojo build --emit shared-lib "/tmp/parwork/construct-w${workers}-t${threshold}.mojo" -o "$out"
  echo "built $out"
done
