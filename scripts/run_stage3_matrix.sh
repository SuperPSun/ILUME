#!/usr/bin/env bash
set -u

run_id="$(date +%Y%m%d_%H%M%S)"
log_dir="artifacts/stage3_v2/training_optimized/logs/${run_id}"
status_file="${log_dir}/status.tsv"
mkdir -p "$log_dir"
printf 'name\tstatus\texit_code\tlog\n' >"$status_file"

run_and_continue() {
  local name="$1"
  shift
  local command_string
  local command_status=0
  printf -v command_string '%q ' "$@"
  script --quiet --flush --return \
    --command "$command_string" "$log_dir/${name}.log" </dev/null \
    || command_status="$?"
  if [ "$command_status" -eq 0 ]; then
    printf '%s\tOK\t0\t%s\n' \
      "$name" "$log_dir/${name}.log" | tee -a "$status_file"
  else
    printf '%s\tFAILED\t%s\t%s\n' \
      "$name" "$command_status" "$log_dir/${name}.log" \
      | tee -a "$status_file" >&2
  fi
  return 0
}

run_gpu() {
  local name="$1"
  shift
  run_and_continue "$name" \
    env CUDA_VISIBLE_DEVICES=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    PYTORCH_ALLOC_CONF=expandable_segments:True "$@"
}

run_and_continue install \
  python -m pip install -e ".[dev,tokenizers]"

run_gpu prepare \
  ilume-stage3-prepare --config configs/stage3_home.yaml

for fold in 1 2 3 4 5; do
  run_gpu "home_fold${fold}" \
    ilume-stage3-train \
    --config configs/stage3_home.yaml \
    --fold "$fold" \
    --output-dir \
    "artifacts/stage3_v2/training_optimized/home/fold${fold}"
done

run_gpu home_valid_summary \
  ilume-stage3-evaluate \
  --config configs/stage3_home.yaml \
  --checkpoint-dir artifacts/stage3_v2/training_optimized/home \
  --split valid \
  --output \
  artifacts/stage3_v2/training_optimized/home/five_fold_summary.json

for experiment in \
  shared_bottom \
  mmoe \
  early_solute \
  without_feature_gate \
  without_self_gate
do
  config="configs/stage3_${experiment}.yaml"
  for fold in 1 2 3 4 5; do
    run_gpu "${experiment}_fold${fold}" \
      ilume-stage3-train \
      --config "$config" \
      --fold "$fold" \
      --output-dir \
      "artifacts/stage3_v2/training_optimized/${experiment}/fold${fold}"
  done
  run_gpu "${experiment}_valid_summary" \
    ilume-stage3-evaluate \
    --config "$config" \
    --checkpoint-dir \
    "artifacts/stage3_v2/training_optimized/${experiment}" \
    --split valid \
    --output \
    "artifacts/stage3_v2/training_optimized/${experiment}/five_fold_summary.json"
done

run_gpu home_test_ensemble \
  ilume-stage3-evaluate \
  --config configs/stage3_home.yaml \
  --checkpoint-dir artifacts/stage3_v2/training_optimized/home \
  --split test \
  --ensemble-folds \
  --output \
  artifacts/stage3_v2/training_optimized/home/test_ensemble_metrics.json

printf '\nStage 3 v2 status: %s\n' "$status_file"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$status_file"
else
  cat "$status_file"
fi
