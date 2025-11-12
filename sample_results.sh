
#!/usr/bin/env bash
# ============================================================
# Sample a subset of .pkl results by experiment type
# Usage: ./sample_results.sh <results_dir> <target_dir>
# ============================================================

set -euo pipefail

# ---- Input validation ----
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <results_dir> <target_dir>"
  exit 1
fi

RESULTS="$1"
TARGET="$2"

if [[ ! -d "$RESULTS" ]]; then
  echo "Error: results directory '$RESULTS' not found."
  exit 1
fi

mkdir -p "$TARGET"

# ---- Collect experiment types ----
experiment_types=$(find "$RESULTS" -maxdepth 1 -type f -name "*.pkl" \
  | sed -E 's#.*/(.+_[0-9])_.*#\1#' | sort | uniq)

# ---- Process each experiment type ----
for etype in $experiment_types; do
  echo "Processing experiment type: $etype"

  # Select earliest 40 files by modification time
  mapfile -t selected < <(
    find "$RESULTS" -maxdepth 1 -type f -name "${etype}_*.pkl" -printf "%T@ %p\n" \
      | sort -n | head -n 40 | cut -d' ' -f2-
  )

  count=${#selected[@]}
  echo "Selected $count files for $etype"

  # Copy preserving timestamps
  for file in "${selected[@]}"; do
    cp -p "$file" "$TARGET"
  done
done

echo "Done. Copied files to '$TARGET'."
