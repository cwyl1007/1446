#!/bin/sh
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

MODES="heart"
POOL="all"
EVAL_CAP="reference"
PPA_QUERY_PANEL="local-seeded"
NUM_RUNS="5"
BASE_SEED="0"
SELECTION_METRIC="MRR"
DATASETS="ogbl-ppa"
MODELS="buddy neo-gnn ncn ncnc nbfnet peg lpformer n2v heuristics"
GPUS="0 1 2 3"

NUMBA_THREADS_PER_WORKER="38"
RANKED_NEGATIVES_BACKEND="auto"
COMPUTE_AUC="yes"
TRAIN_NEGATIVE_SAMPLER="fast"
TRAINING_PATH="auto"
CACHE_EVAL_EDGES="auto"
TEST_EVAL_POLICY="final"
TRAIN_SAMPLES_PER_EPOCH=""
EVAL_BATCH_SIZE=""
MODEL_DECODE_BATCH_SIZE=""

unset CUDA_VISIBLE_DEVICES
unset LINKPREDICTION_COMPACT_HEART_TERMINAL
OGBL_PPA_QUERY_PANEL="$PPA_QUERY_PANEL"
NUMBA_NUM_THREADS="$NUMBA_THREADS_PER_WORKER"
export OGBL_PPA_QUERY_PANEL NUMBA_NUM_THREADS

SEEN_GPUS=""
for GPU in $GPUS
do
  case " $SEEN_GPUS " in
    *" $GPU "*) echo "Duplicate GPU id in GPUS: $GPU"; exit 1 ;;
  esac
  SEEN_GPUS="$SEEN_GPUS $GPU"
done
[ -n "$SEEN_GPUS" ] || { echo "GPUS must list at least one CUDA device."; exit 1; }
command -v setsid >/dev/null 2>&1 || { echo "setsid is required for orphan-safe worker cleanup."; exit 1; }

resolve_eval_cap()
{
  if [ "$EVAL_CAP" = "reference" ]
  then
    case "$1" in heart) printf '%s\n' "100000" ;; *) printf '%s\n' "500" ;; esac
  else
    printf '%s\n' "$EVAL_CAP"
  fi
}

RUN_TEMP="$(mktemp -d /tmp/ogbl_run.XXXXXX)" || exit 1
TASK_DIR="$RUN_TEMP/tasks"
FAILURE_DIR="$RUN_TEMP/failures"
mkdir -p "$TASK_DIR" "$FAILURE_DIR" || exit 1
for GPU in $GPUS
do
  : > "$FAILURE_DIR/gpu_$GPU"
done
WORKER_PIDS=""

cleanup()
{
  case "$RUN_TEMP" in /tmp/ogbl_run.*) [ ! -d "$RUN_TEMP" ] || rm -rf "$RUN_TEMP" ;; esac
}

stop_workers()
{
  trap '' HUP INT TERM
  for PID in $WORKER_PIDS
  do
    kill -TERM "$PID" 2>/dev/null || true
  done
  for PID in $WORKER_PIDS
  do
    wait "$PID" 2>/dev/null || true
  done
  exit 130
}

trap cleanup 0
trap stop_workers HUP INT TERM

record_failure()
{
  printf '%s\n' "$1" >> "$WORKER_FAILURE_FILE"
}

run_python()
{
  NUMBA_NUM_THREADS="$NUMBA_THREADS_PER_WORKER" \
    CUDA_VISIBLE_DEVICES="$WORKER_GPU" \
    LINKPREDICTION_COMPACT_HEART_TERMINAL=0 \
    PYTHONUNBUFFERED=1 \
    setsid "$@" < /dev/null &
  WORKER_CHILD_PID="$!"
  wait "$WORKER_CHILD_PID"
  STATUS="$?"
  WORKER_CHILD_PID=""
  return "$STATUS"
}

run_model_job()
{
  set -- python -m ogbl.main \
    --device cuda \
    --mode "$RUN_MODE" \
    --dataset "$RUN_DATASET" \
    --model "$RUN_MODEL" \
    --metric "$SELECTION_METRIC" \
    --num-runs "$NUM_RUNS" \
    --base-seed "$BASE_SEED" \
    --eval-cap "$RUN_EVAL_CAP" \
    --pool "$POOL" \
    --heart-negatives "$RUN_HEART_NEGATIVES" \
    --ranked-negatives-backend "$RANKED_NEGATIVES_BACKEND" \
    --compute-auc "$COMPUTE_AUC" \
    --train-negative-sampler "$TRAIN_NEGATIVE_SAMPLER" \
    --training-path "$TRAINING_PATH" \
    --cache-eval-edges "$CACHE_EVAL_EDGES" \
    --test-eval-policy "$TEST_EVAL_POLICY"
  [ -z "$TRAIN_SAMPLES_PER_EPOCH" ] || set -- "$@" --train-samples-per-epoch "$TRAIN_SAMPLES_PER_EPOCH"
  [ -z "$EVAL_BATCH_SIZE" ] || set -- "$@" --eval-batch-size "$EVAL_BATCH_SIZE"
  [ -z "$MODEL_DECODE_BATCH_SIZE" ] || set -- "$@" --model-decode-batch-size "$MODEL_DECODE_BATCH_SIZE"
  run_python "$@"
}

run_n2v_job()
{
  run_python python -m ogbl.n2v_main \
    --device cuda \
    --mode "$RUN_MODE" \
    --dataset "$RUN_DATASET" \
    --metric "$SELECTION_METRIC" \
    --num-runs "$NUM_RUNS" \
    --base-seed "$BASE_SEED" \
    --eval-cap "$RUN_EVAL_CAP" \
    --pool "$POOL" \
    --heart-negatives "$RUN_HEART_NEGATIVES" \
    --ranked-negatives-backend "$RANKED_NEGATIVES_BACKEND" \
    --compute-auc "$COMPUTE_AUC"
}

run_heuristics_job()
{
  run_python python -m ogbl.heuristics_main \
    --device cuda \
    --mode "$RUN_MODE" \
    --dataset "$RUN_DATASET" \
    --metric "$SELECTION_METRIC" \
    --seed "$BASE_SEED" \
    --eval-cap "$RUN_EVAL_CAP" \
    --pool "$POOL" \
    --heart-negatives "$RUN_HEART_NEGATIVES" \
    --ranked-negatives-backend "$RANKED_NEGATIVES_BACKEND" \
    --compute-auc "$COMPUTE_AUC"
}

run_current_job_once()
{
  case "$RUN_KIND" in
    main) run_model_job ;;
    n2v) run_n2v_job ;;
    heuristics) run_heuristics_job ;;
    *) return 2 ;;
  esac
}

run_current_job()
{
  JOB_LABEL="gpu=$WORKER_GPU mode=$RUN_MODE dataset=$RUN_DATASET model=$RUN_MODEL"
  echo "=================================================="
  echo "Running: $JOB_LABEL"
  echo "=================================================="
  if run_current_job_once
  then
    echo "Finished: $JOB_LABEL"
    echo
    return 0
  fi
  echo "Failed: $JOB_LABEL"
  echo
  record_failure "$JOB_LABEL"
  return 1
}

queue_task()
{
  TASK_NUMBER=$((TASK_NUMBER + 1))
  printf '%s|%s|%s|%s|%s|%s\n' "$1" "$2" "$3" "$4" "$5" "$6" > "$CURRENT_TASK_DIR/task_$TASK_NUMBER"
}

queue_dataset_task()
{
  queue_task "$1" "$RUN_MODE" "$2" "$3" "$(resolve_eval_cap "$RUN_MODE")" "500"
}

queue_current_dataset_tasks()
{
  TASK_NUMBER="0"
  QUEUED_GROUP_MODELS=""
  for RUN_MODEL in $MODELS
  do
    case " $QUEUED_GROUP_MODELS " in *" $RUN_MODEL "*) continue ;; esac
    QUEUED_GROUP_MODELS="$QUEUED_GROUP_MODELS $RUN_MODEL"
    case "$RUN_MODEL" in
      n2v) queue_dataset_task n2v "$RUN_DATASET" n2v ;;
      heuristics) queue_dataset_task heuristics "$RUN_DATASET" heuristics ;;
      *) queue_dataset_task main "$RUN_DATASET" "$RUN_MODEL" ;;
    esac
  done
  TOTAL_TASKS="$TASK_NUMBER"
}

dataset_group_is_new()
{
  DATASET_GROUP_KEY="|$RUN_MODE:$RUN_DATASET|"
  case "$QUEUED_DATASET_GROUPS" in *"$DATASET_GROUP_KEY"*) return 1 ;; esac
  QUEUED_DATASET_GROUPS="$QUEUED_DATASET_GROUPS$DATASET_GROUP_KEY"
}

stop_worker_job()
{
  trap '' HUP INT TERM
  if [ -n "$WORKER_CHILD_PID" ]
  then
    /bin/kill -TERM -- "-$WORKER_CHILD_PID" 2>/dev/null
    wait "$WORKER_CHILD_PID" 2>/dev/null
  fi
  exit 130
}

run_worker()
{
  WORKER_GPU="$1"
  WORKER_FAILURE_FILE="$FAILURE_DIR/gpu_$WORKER_GPU"
  WORKER_CHILD_PID=""
  TASK_NUMBER="1"
  trap stop_worker_job HUP INT TERM
  while [ "$TASK_NUMBER" -le "$TOTAL_TASKS" ]
  do
    if mkdir "$CURRENT_TASK_DIR/claim_$TASK_NUMBER" 2>/dev/null
    then
      if IFS='|' read -r RUN_KIND RUN_MODE RUN_DATASET RUN_MODEL RUN_EVAL_CAP RUN_HEART_NEGATIVES < "$CURRENT_TASK_DIR/task_$TASK_NUMBER"
      then
        run_current_job
      else
        record_failure "gpu=$WORKER_GPU scheduler could not read task=$TASK_NUMBER"
      fi
    fi
    TASK_NUMBER=$((TASK_NUMBER + 1))
  done
}

run_gpu_worker()
{
  run_worker "$1"
}

SCHEDULER_FAILURES="$FAILURE_DIR/scheduler"
ALL_FAILURES="$FAILURE_DIR/all"
: > "$SCHEDULER_FAILURES"
: > "$ALL_FAILURES"
GROUP_NUMBER="0"
QUEUED_DATASET_GROUPS=""

for RUN_MODE in $MODES
do
  for RUN_DATASET in $DATASETS
  do
    dataset_group_is_new || continue
    GROUP_NUMBER=$((GROUP_NUMBER + 1))
    CURRENT_TASK_DIR="$TASK_DIR/group_$GROUP_NUMBER"
    mkdir "$CURRENT_TASK_DIR" || exit 1
    queue_current_dataset_tasks
    [ "$TOTAL_TASKS" -gt 0 ] || continue
    echo "Starting dataset group: mode=$RUN_MODE dataset=$RUN_DATASET tasks=$TOTAL_TASKS"
    WORKER_PIDS=""
    for WORKER_GPU in $GPUS
    do
      run_gpu_worker "$WORKER_GPU" &
      WORKER_PIDS="$WORKER_PIDS $!"
    done
    for WORKER_PID in $WORKER_PIDS
    do
      wait "$WORKER_PID" || printf '%s\n' "worker pid=$WORKER_PID exited unexpectedly in mode=$RUN_MODE dataset=$RUN_DATASET" >> "$SCHEDULER_FAILURES"
    done
    WORKER_PIDS=""
    echo "Finished dataset group: mode=$RUN_MODE dataset=$RUN_DATASET"
  done
done

for GPU in $GPUS
do
  cat "$FAILURE_DIR/gpu_$GPU" >> "$ALL_FAILURES"
done
cat "$SCHEDULER_FAILURES" >> "$ALL_FAILURES"
FAILED_COUNT="$(awk 'NF { count += 1 } END { print count + 0 }' "$ALL_FAILURES")"
echo "==================== Summary ===================="
if [ "$FAILED_COUNT" -eq 0 ]
then
  echo "All runs completed successfully."
  exit 0
fi
echo "The following runs failed:"
cat "$ALL_FAILURES"
exit 1
