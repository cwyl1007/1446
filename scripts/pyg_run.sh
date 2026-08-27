#!/bin/sh
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

MODES="heart"
DATASETS="facebook"
MODELS="mf mlp ppr concat gcn gat sage gae seal buddy neo-gnn ncn ncnc nbfnet peg lpformer n2v heuristics"
GPUS="0 1 2"

POOL="all"
HEART_NEGATIVES="500"
HEART_BACKEND="gpu"
HEART_BATCH_SIZE="2048"
NUM_RUNS="5"
BASE_SEED="0"
NUMBA_THREADS_PER_WORKER="24"

SEEN_GPUS=""
for gpu in $GPUS; do
  case "$gpu" in
    0|1|2|3) ;;
    ""|*[!0-9]*) echo "Invalid GPU id in GPUS: $gpu"; exit 1 ;;
    *) echo "Unsupported GPU id in GPUS: $gpu (supported ids: 0 1 2 3)"; exit 1 ;;
  esac
  case " $SEEN_GPUS " in
    *" $gpu "*) echo "Duplicate GPU id in GPUS: $gpu"; exit 1 ;;
  esac
  SEEN_GPUS="$SEEN_GPUS $gpu"
done
[ -n "$SEEN_GPUS" ] || { echo "GPUS must list at least one CUDA device."; exit 1; }

for command in setsid mkfifo; do
  command -v "$command" >/dev/null 2>&1 || { echo "$command is required by the PyG runner."; exit 1; }
done
NUMBA_NUM_THREADS="$NUMBA_THREADS_PER_WORKER"
export NUMBA_NUM_THREADS
unset CUDA_VISIBLE_DEVICES

RUN_STATE="$(mktemp -d "${TMPDIR:-/tmp}/pyg_run.XXXXXX")" || { echo "Could not create scheduler state directory."; exit 1; }
TASK_FILE="$RUN_STATE/tasks"
DONE_FIFO="$RUN_STATE/done"
FAILED_FILE="$RUN_STATE/failed"
WORKER_PIDS=""
WORKER_ERROR=0

cleanup() {
  case "$RUN_STATE" in
    */pyg_run.*) [ ! -d "$RUN_STATE" ] || rm -rf -- "$RUN_STATE" ;;
  esac
}

stop_workers() {
  trap '' HUP INT TERM
  for pid in $WORKER_PIDS; do kill -TERM "$pid" 2>/dev/null; done
  for pid in $WORKER_PIDS; do wait "$pid" 2>/dev/null; done
  exit 130
}

trap cleanup EXIT
trap stop_workers HUP INT TERM
: > "$TASK_FILE"
: > "$FAILED_FILE"
for gpu in $GPUS; do : > "$RUN_STATE/failed.$gpu"; done

TASK_KEYS=""
for mode in $MODES; do
  for dataset in $DATASETS; do
    for model in $MODELS; do
      key="|$mode:$dataset:$model|"
      case "$TASK_KEYS" in *"$key"*) continue ;; esac
      TASK_KEYS="$TASK_KEYS$key"
      case "$model" in
        n2v|heuristics) kind="$model" ;;
        *) kind="main" ;;
      esac
      printf '%s|%s|%s|%s\n' "$kind" "$mode" "$dataset" "$model" >> "$TASK_FILE"
    done
  done
done
TOTAL_TASKS="$(wc -l < "$TASK_FILE")"
[ "$TOTAL_TASKS" -gt 0 ] || { echo "No PyG tasks were selected."; exit 1; }

run_child() {
  setsid "$@" < /dev/null &
  WORKER_CHILD_PID="$!"
  wait "$WORKER_CHILD_PID"
  status="$?"
  WORKER_CHILD_PID=""
  return "$status"
}

run_job() {
  job_kind="$1"
  job_mode="$2"
  job_dataset="$3"
  job_model="$4"
  JOB_LABEL="mode=$job_mode dataset=$job_dataset model=$job_model"
  echo "=================================================="
  echo "GPU $CUDA_VISIBLE_DEVICES running: dataset=$job_dataset model=$job_model mode=$job_mode"
  echo "=================================================="
  case "$job_kind" in
    main)
      run_child python -m pyg.main --device cuda --mode "$job_mode" \
        --dataset "$job_dataset" --model "$job_model" --num-runs "$NUM_RUNS" \
        --base-seed "$BASE_SEED" --pool "$POOL" --heart-negatives "$HEART_NEGATIVES" \
        --heart-backend "$HEART_BACKEND" \
        --heart-batch-size "$HEART_BATCH_SIZE"
      ;;
    n2v)
      run_child python -m pyg.n2v_main --device cuda --mode "$job_mode" \
        --dataset "$job_dataset" --num-runs "$NUM_RUNS" --base-seed "$BASE_SEED" \
        --pool "$POOL" --heart-negatives "$HEART_NEGATIVES" \
        --heart-backend "$HEART_BACKEND" \
        --heart-batch-size "$HEART_BATCH_SIZE"
      ;;
    heuristics)
      run_child python -m pyg.heuristics_main --device cuda --mode "$job_mode" \
        --dataset "$job_dataset" --seed "$BASE_SEED" --pool "$POOL" \
        --heart-negatives "$HEART_NEGATIVES" \
        --heart-backend "$HEART_BACKEND" --heart-batch-size "$HEART_BATCH_SIZE"
      ;;
    *) echo "Unknown PyG job kind: $job_kind" >&2; return 1 ;;
  esac
  status="$?"
  if [ "$status" -eq 0 ]; then
    echo "GPU $CUDA_VISIBLE_DEVICES finished: $JOB_LABEL"
  else
    echo "GPU $CUDA_VISIBLE_DEVICES failed ($status): $JOB_LABEL" >&2
  fi
  return "$status"
}

stop_worker_job() {
  trap '' HUP INT TERM
  if [ -n "$WORKER_CHILD_PID" ]; then
    /bin/kill -TERM -- "-$WORKER_CHILD_PID" 2>/dev/null
    wait "$WORKER_CHILD_PID" 2>/dev/null
  fi
  exit 130
}

worker() {
  worker_gpu="$1"
  worker_fifo="$2"
  export CUDA_VISIBLE_DEVICES="$worker_gpu"
  WORKER_CHILD_PID=""
  trap stop_worker_job HUP INT TERM
  while IFS='|' read -r job_kind job_mode job_dataset job_model; do
    if run_job "$job_kind" "$job_mode" "$job_dataset" "$job_model"; then status=0; else status="$?"; fi
    printf '%s|%s|%s\n' "$worker_gpu" "$status" "$JOB_LABEL" > "$DONE_FIFO"
  done < "$worker_fifo"
  trap - HUP INT TERM
}

open_worker_input() {
  case "$1" in
    0) exec 3> "$RUN_STATE/gpu.0" ;;
    1) exec 4> "$RUN_STATE/gpu.1" ;;
    2) exec 5> "$RUN_STATE/gpu.2" ;;
    3) exec 6> "$RUN_STATE/gpu.3" ;;
  esac
}

send_task() {
  case "$1" in
    0) printf '%s|%s|%s|%s\n' "$TASK_KIND" "$TASK_MODE" "$TASK_DATASET" "$TASK_MODEL" >&3 ;;
    1) printf '%s|%s|%s|%s\n' "$TASK_KIND" "$TASK_MODE" "$TASK_DATASET" "$TASK_MODEL" >&4 ;;
    2) printf '%s|%s|%s|%s\n' "$TASK_KIND" "$TASK_MODE" "$TASK_DATASET" "$TASK_MODEL" >&5 ;;
    3) printf '%s|%s|%s|%s\n' "$TASK_KIND" "$TASK_MODE" "$TASK_DATASET" "$TASK_MODEL" >&6 ;;
  esac
}

close_worker_input() {
  case "$1" in
    0) exec 3>&- ;;
    1) exec 4>&- ;;
    2) exec 5>&- ;;
    3) exec 6>&- ;;
  esac
}

pop_idle_gpu() {
  set -- $IDLE_GPUS
  IDLE_GPU="$1"
  shift
  IDLE_GPUS="$*"
}

mkfifo "$DONE_FIFO" || exit 1
for gpu in $GPUS; do mkfifo "$RUN_STATE/gpu.$gpu" || exit 1; done
exec 7<> "$DONE_FIFO"
for gpu in $GPUS; do
  worker "$gpu" "$RUN_STATE/gpu.$gpu" &
  WORKER_PIDS="$WORKER_PIDS $!"
done
for gpu in $GPUS; do open_worker_input "$gpu"; done
exec 8< "$TASK_FILE"

ACTIVE_COUNT=0
IDLE_GPUS="$GPUS"
CURRENT_GROUP=""
HAVE_TASK=0
TASKS_EXHAUSTED=0
SCHEDULER_ERROR=0
while :; do
  while [ -n "$IDLE_GPUS" ] && [ "$TASKS_EXHAUSTED" -eq 0 ]; do
    if [ "$HAVE_TASK" -eq 0 ]; then
      if IFS='|' read -r TASK_KIND TASK_MODE TASK_DATASET TASK_MODEL <&8; then
        HAVE_TASK=1
      else
        TASKS_EXHAUSTED=1
        break
      fi
    fi
    task_group="$TASK_MODE|$TASK_DATASET"
    if [ -z "$CURRENT_GROUP" ]; then
      CURRENT_GROUP="$task_group"
      echo "Starting dataset group: mode=$TASK_MODE dataset=$TASK_DATASET"
    elif [ "$task_group" != "$CURRENT_GROUP" ]; then
      break
    fi
    pop_idle_gpu
    send_task "$IDLE_GPU"
    ACTIVE_COUNT=$((ACTIVE_COUNT + 1))
    HAVE_TASK=0
  done

  if [ "$ACTIVE_COUNT" -eq 0 ]; then
    if [ "$HAVE_TASK" -eq 1 ]; then CURRENT_GROUP=""; continue; fi
    [ "$TASKS_EXHAUSTED" -eq 0 ] || break
  fi
  if ! IFS='|' read -r finished_gpu finished_status finished_label <&7; then
    echo "Scheduler completion channel closed unexpectedly." >&2
    SCHEDULER_ERROR=1
    break
  fi
  ACTIVE_COUNT=$((ACTIVE_COUNT - 1))
  IDLE_GPUS="$IDLE_GPUS $finished_gpu"
  [ "$finished_status" -eq 0 ] || printf '%s\n' "$finished_label" >> "$RUN_STATE/failed.$finished_gpu"
done

for gpu in $GPUS; do close_worker_input "$gpu"; done
exec 8<&-
for pid in $WORKER_PIDS; do wait "$pid" || WORKER_ERROR=1; done
exec 7>&-
for gpu in $GPUS; do cat "$RUN_STATE/failed.$gpu" >> "$FAILED_FILE"; done
FAILED_COUNT="$(wc -l < "$FAILED_FILE")"
COMPLETED_COUNT=$((TOTAL_TASKS - FAILED_COUNT))

echo "==================== Summary ===================="
echo "Completed $COMPLETED_COUNT/$TOTAL_TASKS runs."
if [ "$FAILED_COUNT" -eq 0 ] && [ "$SCHEDULER_ERROR" -eq 0 ] && [ "$WORKER_ERROR" -eq 0 ]; then
  echo "All runs completed successfully."
  exit 0
fi
[ "$FAILED_COUNT" -eq 0 ] || { echo "The following runs failed:"; cat "$FAILED_FILE"; }
exit 1
