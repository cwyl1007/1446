#!/bin/sh
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
PYTHON_BIN="/data/miniconda3/envs/lp/bin/python"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
[ -n "$LOG_ROOT" ] || LOG_ROOT="$PROJECT_ROOT/logs/full_graph_4gpu/$STAMP"
mkdir -p "$LOG_ROOT"
GPU_COUNT="0"
SEEN_GPUS=""
for RUN_GPU in $GPUS
do
  case "$RUN_GPU" in
    ""|*[!0-9]*)
      echo "Invalid GPU id in GPUS: $RUN_GPU" >&2
      exit 2
      ;;
  esac
  case " $SEEN_GPUS " in
    *" $RUN_GPU "*)
      echo "Duplicate GPU id in GPUS: $RUN_GPU" >&2
      exit 2
      ;;
  esac
  SEEN_GPUS="$SEEN_GPUS $RUN_GPU"
  GPU_COUNT=$((GPU_COUNT + 1))
done
if [ "$GPU_COUNT" -eq 0 ]
then
  echo "No GPUs configured. Set GPUS, for example GPUS=\"0 1 2 3\"." >&2
  exit 2
fi
case "$EDGE_BATCH_SIZE:$NODE_CHUNK_SIZE:$COMPARISON_BATCH_SIZE" in
  *[!0-9:]*|*::*|:*|*:)
    echo "Batch, chunk, and comparison settings must be non-negative integers." >&2
    exit 2
    ;;
esac
case "$TEST_POSITIVE_CAP" in
  ""|*[!0-9]*)
    echo "TEST_POSITIVE_CAP must be a non-negative integer." >&2
    exit 2
    ;;
esac
[ -n "$COMPUTE_AUC" ] || COMPUTE_AUC="no"
case "$COMPUTE_AUC" in
  yes|no) ;;
  *)
    echo "COMPUTE_AUC must be yes or no." >&2
    exit 2
    ;;
esac
CHECKPOINT_ROOT_PATH="$CHECKPOINT_ROOT"
case "$CHECKPOINT_ROOT_PATH" in
  /*) ;;
  *) CHECKPOINT_ROOT_PATH="$PROJECT_ROOT/$CHECKPOINT_ROOT_PATH" ;;
esac
SELECTOR_CHECKPOINT_ROOT_PATH="$SELECTOR_CHECKPOINT_ROOT"
if [ -n "$SELECTOR_CHECKPOINT_ROOT_PATH" ]
then
  case "$SELECTOR_CHECKPOINT_ROOT_PATH" in
    /*) ;;
    *) SELECTOR_CHECKPOINT_ROOT_PATH="$PROJECT_ROOT/$SELECTOR_CHECKPOINT_ROOT_PATH" ;;
  esac
fi
[ -n "$TARGET_CHECKPOINT_MODE" ] || TARGET_CHECKPOINT_MODE="heart"
[ -n "$SELECTOR_CHECKPOINT_MODE" ] || SELECTOR_CHECKPOINT_MODE="heart"
RUN_STATE="$(mktemp -d /tmp/full_graph_run.XXXXXX)"
if [ -z "$RUN_STATE" ] || [ ! -d "$RUN_STATE" ]
then
  echo "Unable to create the temporary scheduler directory." >&2
  exit 2
fi
TASK_DIR="$RUN_STATE/tasks"
SLOT_DIR="$RUN_STATE/slots"
PRIMED_DIR="$RUN_STATE/primed"
FAILED_FILE="$RUN_STATE/failed"
SKIPPED_FILE="$RUN_STATE/skipped"
mkdir -p "$TASK_DIR" "$SLOT_DIR" "$PRIMED_DIR"
: > "$FAILED_FILE"
: > "$SKIPPED_FILE"
cleanup()
{
  case "$RUN_STATE" in
    /tmp/full_graph_run.*)
      [ ! -d "$RUN_STATE" ] || rm -rf -- "$RUN_STATE"
      ;;
  esac
}
stop_children()
{
  trap - INT TERM
  echo
  echo "Stopping active jobs..." >&2
  for STOP_GPU in $GPUS
  do
    STOP_PID="$(sed -n '1p' "$SLOT_DIR/$STOP_GPU.pid" 2>/dev/null)"
    [ -z "$STOP_PID" ] || kill "$STOP_PID" 2>/dev/null || true
  done
  for STOP_GPU in $GPUS
  do
    STOP_PID="$(sed -n '1p' "$SLOT_DIR/$STOP_GPU.pid" 2>/dev/null)"
    [ -z "$STOP_PID" ] || wait "$STOP_PID" 2>/dev/null || true
  done
  exit 130
}
trap cleanup EXIT
trap stop_children INT TERM
for RUN_GPU in $GPUS
do
  : > "$SLOT_DIR/$RUN_GPU.pid"
  : > "$SLOT_DIR/$RUN_GPU.label"
  : > "$SLOT_DIR/$RUN_GPU.log"
  : > "$SLOT_DIR/$RUN_GPU.start"
  : > "$SLOT_DIR/$RUN_GPU.key"
done
checkpoint_run_selected()
{
  SELECTED_CHECKPOINT_FILE="$1"
  if [ -z "$RUNS" ]
  then
    return 0
  fi
  SELECTED_CHECKPOINT_NAME="$(basename -- "$SELECTED_CHECKPOINT_FILE")"
  SELECTED_CHECKPOINT_RUN="$(printf '%s\n' "$SELECTED_CHECKPOINT_NAME" | sed 's/^model_checkpoint//')"
  for REQUESTED_RUN in $RUNS
  do
    [ "$REQUESTED_RUN" != "$SELECTED_CHECKPOINT_RUN" ] || return 0
  done
  return 1
}
target_checkpoints_exist()
{
  TARGET_CHECKPOINT_DIR="$1"
  for TARGET_CHECKPOINT_FILE in "$TARGET_CHECKPOINT_DIR"/model_checkpoint*
  do
    if [ -f "$TARGET_CHECKPOINT_FILE" ] && checkpoint_run_selected "$TARGET_CHECKPOINT_FILE"
    then
      return 0
    fi
  done
  return 1
}
target_checkpoint_set_is_complete()
{
  TARGET_CHECKPOINT_DIR="$1"
  if [ -z "$RUNS" ]
  then
    target_checkpoints_exist "$TARGET_CHECKPOINT_DIR"
    return
  fi
  for TARGET_RUN in $RUNS
  do
    case "$TARGET_RUN" in
      ""|*[!0-9]*)
        echo "Invalid run id in RUNS: $TARGET_RUN" >&2
        return 1
        ;;
    esac
    [ -s "$TARGET_CHECKPOINT_DIR/model_checkpoint$TARGET_RUN" ] || return 1
  done
  return 0
}
selector_checkpoint_for_dataset()
{
  printf '%s\n' "$SELECTOR_CHECKPOINT_ROOT_PATH/$SELECTOR_CHECKPOINT_MODE/$1/$SELECTOR_CHECKPOINT_MODEL/model_checkpoint$SELECTOR_CHECKPOINT_RUN"
}
selector_negatives_for_dataset()
{
  NEGATIVE_CONFIGURED="$SELECTOR_NEGATIVES"
  if [ -z "$NEGATIVE_CONFIGURED" ] || [ "$NEGATIVE_CONFIGURED" = "auto" ]
  then
    printf '%s\n' "500"
  else
    printf '%s\n' "$NEGATIVE_CONFIGURED"
  fi
}
TASK_TOTAL="0"
queue_task()
{
  TASK_TOTAL=$((TASK_TOTAL + 1))
  printf '%s|%s|%s\n' "$1" "$2" "$3" > "$TASK_DIR/task_$TASK_TOTAL"
}
record_skip()
{
  printf '%s\n' "$1" >> "$SKIPPED_FILE"
}
for RUN_DATASET in $DATASETS
do
  if [ -n "$SELECTOR_CHECKPOINT_MODEL" ]
  then
    SELECTOR_CHECKPOINT="$(selector_checkpoint_for_dataset "$RUN_DATASET")"
    if [ ! -f "$SELECTOR_CHECKPOINT" ]
    then
      for RUN_MODEL in $MODELS
      do
        record_skip "dataset=$RUN_DATASET model=$RUN_MODEL | no $SELECTOR_POLICY selector checkpoint: $SELECTOR_CHECKPOINT"
      done
      continue
    fi
  fi
  for RUN_MODEL in $MODELS
  do
    TARGET_CHECKPOINT_DIR="$CHECKPOINT_ROOT_PATH/$TARGET_CHECKPOINT_MODE/$RUN_DATASET/$RUN_MODEL"
    if target_checkpoint_set_is_complete "$TARGET_CHECKPOINT_DIR"
    then
      queue_task "checkpoint" "$RUN_DATASET" "$RUN_MODEL"
    else
      record_skip "dataset=$RUN_DATASET model=$RUN_MODEL | no selected $TARGET_CHECKPOINT_MODE checkpoint"
    fi
  done
done
for RUN_DATASET in $DATASETS
do
  for RUN_HEURISTIC in $HEURISTICS
  do
    if [ -n "$SELECTOR_CHECKPOINT_MODEL" ]
    then
      SELECTOR_CHECKPOINT="$(selector_checkpoint_for_dataset "$RUN_DATASET")"
      if [ ! -f "$SELECTOR_CHECKPOINT" ]
      then
        record_skip "dataset=$RUN_DATASET heuristic=$RUN_HEURISTIC | no $SELECTOR_POLICY selector checkpoint: $SELECTOR_CHECKPOINT"
        continue
      fi
    fi
    queue_task "heuristic" "$RUN_DATASET" "$RUN_HEURISTIC"
  done
done
if [ -s "$SKIPPED_FILE" ]
then
  echo "Skipping jobs without complete checkpoint sets:" >&2
  sed 's/^/  /' "$SKIPPED_FILE" >&2
fi
if [ "$TASK_TOTAL" -eq 0 ]
then
  if [ -s "$SKIPPED_FILE" ]
  then
    SKIPPED_COUNT="$(awk 'NF { count += 1 } END { print count + 0 }' "$SKIPPED_FILE")"
    {
      echo "==================== Summary ===================="
      echo "Total:     0"
      echo "Succeeded: 0"
      echo "Failed:    0"
      echo "Skipped:   $SKIPPED_COUNT"
      echo "Logs:      $LOG_ROOT"
      echo
      echo "Skipped jobs:"
      sed 's/^/  /' "$SKIPPED_FILE"
    } | tee "$LOG_ROOT/summary.txt"
    exit 0
  fi
  echo "No jobs were created. Check DATASETS, MODELS, RUNS, and HEURISTICS." >&2
  exit 2
fi
selector_is_active()
{
  ACTIVE_WANTED_KEY="$1"
  for ACTIVE_GPU in $GPUS
  do
    ACTIVE_KEY="$(sed -n '1p' "$SLOT_DIR/$ACTIVE_GPU.key" 2>/dev/null)"
    [ "$ACTIVE_KEY" != "$ACTIVE_WANTED_KEY" ] || return 0
  done
  return 1
}
SELECTED_TASK="0"
select_next_task()
{
  SELECTED_TASK="0"
  SELECT_CANDIDATE="1"
  while [ "$SELECT_CANDIDATE" -le "$TASK_TOTAL" ]
  do
    if [ ! -e "$TASK_DIR/claim_$SELECT_CANDIDATE" ]
    then
      IFS='|' read -r SELECT_KIND SELECT_DATASET SELECT_NAME < "$TASK_DIR/task_$SELECT_CANDIDATE"
      SELECT_KEY=""
      if [ -n "$SELECTOR_POLICY" ]
      then
        SELECT_KEY="$SELECTOR_POLICY-$SELECT_DATASET"
      fi
      SELECT_ELIGIBLE="yes"
      if [ -n "$SELECT_KEY" ] && [ ! -e "$PRIMED_DIR/$SELECT_KEY" ] && selector_is_active "$SELECT_KEY"
      then
        SELECT_ELIGIBLE="no"
      fi
      if [ "$SELECT_ELIGIBLE" = "yes" ]
      then
        mkdir "$TASK_DIR/claim_$SELECT_CANDIDATE"
        SELECTED_TASK="$SELECT_CANDIDATE"
        return
      fi
    fi
    SELECT_CANDIDATE=$((SELECT_CANDIDATE + 1))
  done
}
launch_task()
{
  LAUNCH_GPU="$1"
  LAUNCH_INDEX="$2"
  IFS='|' read -r LAUNCH_KIND LAUNCH_DATASET LAUNCH_NAME < "$TASK_DIR/task_$LAUNCH_INDEX"
  LAUNCH_LABEL="$LAUNCH_KIND evaluation=$EVALUATION_LABEL dataset=$LAUNCH_DATASET name=$LAUNCH_NAME"
  LAUNCH_LOG="$LOG_ROOT/$LAUNCH_KIND/$EVALUATION_LABEL/$LAUNCH_DATASET/$LAUNCH_NAME.log"
  LAUNCH_KEY=""
  [ -z "$SELECTOR_POLICY" ] || LAUNCH_KEY="$SELECTOR_POLICY-$LAUNCH_DATASET"
  if [ "$TEST_POSITIVE_CAP" -eq 0 ]
  then
    LAUNCH_POSITIVE_SCOPE="complete-test-positive-split"
    LAUNCH_POSITIVE_CAP="all"
  else
    LAUNCH_POSITIVE_SCOPE="deterministic-test-positive-cap-$TEST_POSITIVE_CAP"
    LAUNCH_POSITIVE_CAP="$TEST_POSITIVE_CAP"
  fi
  if [ -z "$SELECTOR_POLICY" ]
  then
    LAUNCH_EVAL_CAP="$EVAL_CAP"
  fi
  mkdir -p "$(dirname -- "$LAUNCH_LOG")"
  echo "[$LAUNCH_INDEX/$TASK_TOTAL] GPU $LAUNCH_GPU START $LAUNCH_LABEL"
  echo "    log: $LAUNCH_LOG"
  (
    case "$LAUNCH_DATASET" in
      ogbl-*) LAUNCH_FRAMEWORK="ogb" ;;
      *) LAUNCH_FRAMEWORK="pyg" ;;
    esac
    set -- "$PYTHON_BIN" -m "$EVALUATOR_MODULE" \
      --framework "$LAUNCH_FRAMEWORK" \
      --device cuda:0 \
      --split "$SPLIT" \
      --candidate-policy "$CANDIDATE_POLICY" \
      --test-positive-cap "$TEST_POSITIVE_CAP" \
      --root "$DATA_ROOT" \
      --comparison-batch-size "$COMPARISON_BATCH_SIZE" \
      --compute-auc "$COMPUTE_AUC"
    if [ -n "$SELECTOR_POLICY" ]
    then
      SELECTOR_NEGATIVES="$(selector_negatives_for_dataset "$LAUNCH_DATASET")"
      if [ -n "$SELECTOR_CHECKPOINT_MODEL" ]
      then
        set -- "$@" \
          "--$SELECTOR_POLICY-negatives" "$SELECTOR_NEGATIVES" \
          "--$SELECTOR_POLICY-checkpoint-root" "$SELECTOR_CHECKPOINT_ROOT" \
          "--$SELECTOR_POLICY-checkpoint-run" "$SELECTOR_CHECKPOINT_RUN" \
          "--$SELECTOR_POLICY-cache-dir" "$SELECTOR_CACHE_DIR" \
          "--$SELECTOR_POLICY-score-batch-size" "$SELECTOR_SCORE_BATCH_SIZE"
      else
        set -- "$@" \
          "--$SELECTOR_POLICY-negatives" "$SELECTOR_NEGATIVES" \
          "--$SELECTOR_POLICY-cache-dir" "$SELECTOR_CACHE_DIR" \
          "--$SELECTOR_POLICY-score-batch-size" "$SELECTOR_SCORE_BATCH_SIZE" \
          "--$SELECTOR_POLICY-selector-seed" "$SELECTOR_SEED" \
          "--$SELECTOR_POLICY-selector-epochs" "$SELECTOR_EPOCHS" \
          "--$SELECTOR_POLICY-selector-eval-steps" "$SELECTOR_EVAL_STEPS" \
          "--$SELECTOR_POLICY-selector-patience" "$SELECTOR_PATIENCE" \
          "--$SELECTOR_POLICY-selector-batch-size" "$SELECTOR_BATCH_SIZE"
        [ -z "$SELECTOR_METRIC" ] || set -- "$@" "--$SELECTOR_POLICY-selector-metric" "$SELECTOR_METRIC"
      fi
    else
      set -- "$@" --eval-cap "$LAUNCH_EVAL_CAP"
      [ -z "$PLANETOID_INPUT_ROOT" ] || set -- "$@" --planetoid-input-root "$PLANETOID_INPUT_ROOT"
    fi
    if [ "$LAUNCH_KIND" = "checkpoint" ]
    then
      TARGET_CHECKPOINT_DIR="$CHECKPOINT_ROOT_PATH/$TARGET_CHECKPOINT_MODE/$LAUNCH_DATASET/$LAUNCH_NAME"
      set -- "$@" --mode "$TARGET_CHECKPOINT_MODE"
      for TARGET_CHECKPOINT_FILE in "$TARGET_CHECKPOINT_DIR"/model_checkpoint*
      do
        if [ -f "$TARGET_CHECKPOINT_FILE" ] && checkpoint_run_selected "$TARGET_CHECKPOINT_FILE"
        then
          set -- "$@" --checkpoint "$TARGET_CHECKPOINT_FILE"
        fi
      done
      [ "$EDGE_BATCH_SIZE" -le 0 ] || set -- "$@" --edge-batch-size "$EDGE_BATCH_SIZE"
      [ "$NODE_CHUNK_SIZE" -le 0 ] || set -- "$@" --node-chunk-size "$NODE_CHUNK_SIZE"
    else
      HEURISTIC_MODE="heart"
      [ -z "$SELECTOR_POLICY" ] || HEURISTIC_MODE="$SELECTOR_POLICY"
      set -- "$@" --mode "$HEURISTIC_MODE" --dataset "$LAUNCH_DATASET" --heuristic "$LAUNCH_NAME"
      [ -z "$SOURCE_BATCH_SIZE" ] || set -- "$@" --source-batch-size "$SOURCE_BATCH_SIZE"
    fi
    [ "$QUIET" != "1" ] || set -- "$@" --quiet
    if [ -n "$EXTRA_ARGS" ]
    then
      for EXTRA_ARG in $EXTRA_ARGS
      do
        set -- "$@" "$EXTRA_ARG"
      done
    fi
    echo "GPU=$LAUNCH_GPU"
    echo "TASK=$LAUNCH_LABEL"
    echo "POSITIVE_QUERY_SCOPE=$LAUNCH_POSITIVE_SCOPE"
    echo "POSITIVE_EVAL_CAP=$LAUNCH_POSITIVE_CAP"
    [ -n "$SELECTOR_POLICY" ] || echo "AUXILIARY_LOADER_EVAL_CAP=$LAUNCH_EVAL_CAP"
    printf 'COMMAND='
    for COMMAND_ARG in env "CUDA_VISIBLE_DEVICES=$LAUNCH_GPU" "$@"
    do
      printf ' %s' "$COMMAND_ARG"
    done
    printf '\n'
    echo "STARTED=$(date --iso-8601=seconds)"
    echo
    if [ "$DRY_RUN" = "1" ]
    then
      echo "DRY_RUN=1; command not executed."
      exit 0
    fi
    exec env CUDA_VISIBLE_DEVICES="$LAUNCH_GPU" "$@"
  ) > "$LAUNCH_LOG" 2>&1 &
  printf '%s\n' "$!" > "$SLOT_DIR/$LAUNCH_GPU.pid"
  printf '%s\n' "$LAUNCH_LABEL" > "$SLOT_DIR/$LAUNCH_GPU.label"
  printf '%s\n' "$LAUNCH_LOG" > "$SLOT_DIR/$LAUNCH_GPU.log"
  date +%s > "$SLOT_DIR/$LAUNCH_GPU.start"
  printf '%s\n' "$LAUNCH_KEY" > "$SLOT_DIR/$LAUNCH_GPU.key"
}
COMPLETED="0"
SUCCEEDED="0"
while [ "$COMPLETED" -lt "$TASK_TOTAL" ]
do
  for RUN_GPU in $GPUS
  do
    RUN_PID="$(sed -n '1p' "$SLOT_DIR/$RUN_GPU.pid" 2>/dev/null)"
    if [ -n "$RUN_PID" ] && ! kill -0 "$RUN_PID" 2>/dev/null
    then
      RUN_RC="0"
      wait "$RUN_PID" || RUN_RC="$?"
      RUN_START="$(sed -n '1p' "$SLOT_DIR/$RUN_GPU.start")"
      RUN_LABEL="$(sed -n '1p' "$SLOT_DIR/$RUN_GPU.label")"
      RUN_LOG="$(sed -n '1p' "$SLOT_DIR/$RUN_GPU.log")"
      RUN_KEY="$(sed -n '1p' "$SLOT_DIR/$RUN_GPU.key")"
      RUN_ELAPSED=$(( $(date +%s) - RUN_START ))
      if [ "$RUN_RC" -eq 0 ]
      then
        echo "GPU $RUN_GPU DONE  $RUN_LABEL ($RUN_ELAPSED seconds)"
        SUCCEEDED=$((SUCCEEDED + 1))
        [ -z "$RUN_KEY" ] || : > "$PRIMED_DIR/$RUN_KEY"
      else
        echo "GPU $RUN_GPU FAIL  $RUN_LABEL rc=$RUN_RC ($RUN_ELAPSED seconds)"
        echo "    log: $RUN_LOG"
        printf '%s\n' "$RUN_LABEL | rc=$RUN_RC | $RUN_LOG" >> "$FAILED_FILE"
      fi
      : > "$SLOT_DIR/$RUN_GPU.pid"
      : > "$SLOT_DIR/$RUN_GPU.label"
      : > "$SLOT_DIR/$RUN_GPU.log"
      : > "$SLOT_DIR/$RUN_GPU.start"
      : > "$SLOT_DIR/$RUN_GPU.key"
      COMPLETED=$((COMPLETED + 1))
      RUN_PID=""
    fi
    if [ -z "$RUN_PID" ] && [ "$COMPLETED" -lt "$TASK_TOTAL" ]
    then
      select_next_task
      if [ "$SELECTED_TASK" -gt 0 ]
      then
        launch_task "$RUN_GPU" "$SELECTED_TASK"
      fi
    fi
  done
  [ "$COMPLETED" -ge "$TASK_TOTAL" ] || sleep "$POLL_SECONDS"
done
FAILED_COUNT="$(awk 'NF { count += 1 } END { print count + 0 }' "$FAILED_FILE")"
SKIPPED_COUNT="$(awk 'NF { count += 1 } END { print count + 0 }' "$SKIPPED_FILE")"
{
  echo "==================== Summary ===================="
  echo "Total:     $TASK_TOTAL"
  echo "Succeeded: $SUCCEEDED"
  echo "Failed:    $FAILED_COUNT"
  echo "Skipped:   $SKIPPED_COUNT"
  echo "Logs:      $LOG_ROOT"
  if [ "$FAILED_COUNT" -gt 0 ]
  then
    echo
    echo "Failed jobs:"
    sed 's/^/  /' "$FAILED_FILE"
  fi
  if [ "$SKIPPED_COUNT" -gt 0 ]
  then
    echo
    echo "Skipped jobs:"
    sed 's/^/  /' "$SKIPPED_FILE"
  fi
} | tee "$LOG_ROOT/summary.txt"
[ "$FAILED_COUNT" -eq 0 ]
