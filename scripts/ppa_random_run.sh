#!/bin/sh
GPU_1_MODELS="gat mlp ppr gcn peg sage nbfnet ncnc heuristics"
GPU_3_MODELS="gae concat mf n2v neo-gnn ncn buddy seal lpformer"
RUNS="1 2 3 4 5"
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
PYTHON_BIN="/data/miniconda3/envs/lp/bin/python"
RESULT_DIR="$PROJECT_ROOT/results/ogbl/random/ogbl-ppa"
SUMMARY_FILE="$RESULT_DIR/ppa_random_results.txt"
mkdir -p "$RESULT_DIR" || exit 1
: > "$SUMMARY_FILE"
checkpoint_set_is_complete()
{
  checkpoint_model="$1"
  for checkpoint_run in $RUNS
  do
    checkpoint_path="$PROJECT_ROOT/checkpoints/heart/ogbl-ppa/$checkpoint_model/model_checkpoint$checkpoint_run"
    [ -s "$checkpoint_path" ] || return 1
  done
  return 0
}
for output_model in $GPU_1_MODELS $GPU_3_MODELS
do
  : > "$RESULT_DIR/$output_model.txt"
done
run_queue()
{
  queue_gpu="$1"
  shift
  for queue_model in "$@"
  do
    queue_output="$RESULT_DIR/$queue_model.txt"
    if [ "$queue_model" != "heuristics" ] && ! checkpoint_set_is_complete "$queue_model"
    then
      echo "SKIPPED: incomplete checkpoint set; required runs: $RUNS" > "$queue_output"
      echo "GPU $queue_gpu SKIP $queue_model (incomplete checkpoint set)"
      continue
    fi
    echo "GPU $queue_gpu START $queue_model"
    if (
      cd "$PROJECT_ROOT" || exit 1
      CUDA_VISIBLE_DEVICES="$queue_gpu" "$PYTHON_BIN" -u \
        -m eval_modes.ppa_random_eval \
        --model "$queue_model" \
        --runs $RUNS \
        --device cuda:0 \
        --legality observed-history \
        --data-seed 0 \
        --negative-seed 3001 \
        --test-cap 100000 \
        --negatives 500 \
        --no-save
    ) > "$queue_output" 2>&1
    then
      echo "COMPLETED" >> "$queue_output"
      echo "GPU $queue_gpu DONE $queue_model"
    else
      queue_status=$?
      echo "FAILED exit=$queue_status" >> "$queue_output"
      echo "GPU $queue_gpu FAILED $queue_model exit=$queue_status"
      return "$queue_status"
    fi
  done
  return 0
}
run_queue 1 $GPU_1_MODELS &
worker_1=$!
run_queue 3 $GPU_3_MODELS &
worker_3=$!
run_status=0
wait "$worker_1" || run_status=1
wait "$worker_3" || run_status=1
{
  echo "ogbl-ppa random-negative evaluation"
  echo "500 random legal negatives per selected test positive"
  echo "runs: $RUNS"
  echo
  for summary_model in $GPU_1_MODELS $GPU_3_MODELS
  do
    echo "================================================================"
    echo "model: $summary_model"
    echo "================================================================"
    if [ "$(tail -n 1 "$RESULT_DIR/$summary_model.txt")" = "COMPLETED" ]
    then
      cat "$RESULT_DIR/$summary_model.txt"
    elif grep -q '^SKIPPED:' "$RESULT_DIR/$summary_model.txt"
    then
      cat "$RESULT_DIR/$summary_model.txt"
    elif [ -s "$RESULT_DIR/$summary_model.txt" ]
    then
      cat "$RESULT_DIR/$summary_model.txt"
      echo "INCOMPLETE: this model did not exit successfully."
    else
      echo "NOT RUN: an earlier queue task failed."
    fi
    echo
  done
} > "$SUMMARY_FILE"
echo "Combined report: $SUMMARY_FILE"
exit "$run_status"
