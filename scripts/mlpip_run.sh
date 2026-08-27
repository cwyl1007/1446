#!/bin/sh
set -e
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
PYTHON_BIN="/data/miniconda3/envs/lp/bin/python"

GPUS="0 1"
DATASETS="cora citeseer pubmed amazon-c amazon-p wiki-chameleon wiki-squirrel reddit github facebook ogbl-collab ogbl-ddi ogbl-ppa ogbl-citation2"
MODELS="mf mlp ppr concat gcn gat sage gae seal buddy neo-gnn ncn ncnc nbfnet peg lpformer n2v"
RUNS="1 2 3"
MLPIP_SELECTOR_DEPTHS="2"
MLPIP_SELECTOR_HIDDEN_CHANNELS="256"
MLPIP_SELECTOR_DROPOUTS="0.1"
MLPIP_SELECTOR_LEARNING_RATES="0.01"
MLPIP_SELECTOR_WEIGHT_DECAYS="0"
HEURISTICS="all"
SPLIT="test"
TEST_POSITIVE_CAP="10000"
MLPIP_NEGATIVES="500"

MLPIP_CHECKPOINT_RUN="1"
MLPIP_SCORE_BATCH_SIZE="65536"
EVALUATOR_MODULE="eval_modes.mlpip_mode"
SELECTOR_POLICY="mlpip"
CANDIDATE_POLICY="$SELECTOR_POLICY"
EVALUATION_LABEL="$SELECTOR_POLICY"
SELECTOR_CHECKPOINT_MODEL="mlpip"
SELECTOR_CHECKPOINT_RUN="$MLPIP_CHECKPOINT_RUN"
SELECTOR_NEGATIVES="$MLPIP_NEGATIVES"
SELECTOR_SCORE_BATCH_SIZE="$MLPIP_SCORE_BATCH_SIZE"
SELECTOR_SEED=""
SELECTOR_EPOCHS=""
SELECTOR_EVAL_STEPS=""
SELECTOR_PATIENCE=""
SELECTOR_BATCH_SIZE=""
SELECTOR_METRIC=""
SELECTOR_TRAIN_GPU="0"
SELECTOR_EVAL_CAP="100000"
SELECTOR_BASE_SEED="0"
SELECTOR_MAX_EPOCHS="500"
SELECTOR_EVAL_STEPS="5"
SELECTOR_PATIENCE="10"
SELECTOR_NUM_RUNS="1"
IP_TRAINING_CONTRACT_TAG="shared_ip_strict_bce_v1"
SELECTOR_HEART_NEGATIVES="500"
SELECTOR_HEART_BATCH_SIZE="2048"
DATA_ROOT="dataset"
CHECKPOINT_ROOT="checkpoints"
EDGE_BATCH_SIZE="0"
NODE_CHUNK_SIZE="0"
COMPARISON_BATCH_SIZE="256"
SOURCE_BATCH_SIZE=""
QUIET="0"
DRY_RUN="0"
POLL_SECONDS="1"

case "$SELECTOR_TRAIN_GPU" in
  ""|*[!0-9]*)
    echo "SELECTOR_TRAIN_GPU must be a non-negative integer." >&2
    exit 2
    ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
for SELECTOR_HIDDEN_CHANNELS in $MLPIP_SELECTOR_HIDDEN_CHANNELS
do
  case "$SELECTOR_HIDDEN_CHANNELS" in
    ""|*[!0-9]*|0)
      echo "MLPIP_SELECTOR_HIDDEN_CHANNELS must contain positive integers." >&2
      exit 2
      ;;
  esac
  for SELECTOR_DEPTH in $MLPIP_SELECTOR_DEPTHS
  do
    case "$SELECTOR_DEPTH" in
      ""|*[!0-9]*|0)
        echo "MLPIP_SELECTOR_DEPTHS must contain positive integers." >&2
        exit 2
        ;;
    esac
    for SELECTOR_DROPOUT in $MLPIP_SELECTOR_DROPOUTS
    do
      for SELECTOR_LEARNING_RATE in $MLPIP_SELECTOR_LEARNING_RATES
      do
        for SELECTOR_WEIGHT_DECAY in $MLPIP_SELECTOR_WEIGHT_DECAYS
        do
          DEPTH_TAG="selector_depth_$SELECTOR_DEPTH"
          HIDDEN_TAG="hidden_$SELECTOR_HIDDEN_CHANNELS"
          DESIGN_TAG="$DEPTH_TAG"_"$HIDDEN_TAG"_"dropout_$SELECTOR_DROPOUT"_"lr_$SELECTOR_LEARNING_RATE"_"weight_decay_$SELECTOR_WEIGHT_DECAY"
          MLPIP_TRAIN_EXTRA_ARGS="--selector-dropout $SELECTOR_DROPOUT --selector-lr $SELECTOR_LEARNING_RATE --selector-weight-decay $SELECTOR_WEIGHT_DECAY"
          EXTRA_ARGS="--mlpip-selector-depth $SELECTOR_DEPTH --mlpip-selector-hidden-channels $SELECTOR_HIDDEN_CHANNELS --mlpip-selector-dropout $SELECTOR_DROPOUT --mlpip-selector-lr $SELECTOR_LEARNING_RATE --mlpip-selector-weight-decay $SELECTOR_WEIGHT_DECAY"
          MLPIP_CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints/mlpip_$DESIGN_TAG/$IP_TRAINING_CONTRACT_TAG"
          MLPIP_CACHE_DIR="mlpip_sets/$DESIGN_TAG/$IP_TRAINING_CONTRACT_TAG"
          SELECTOR_CHECKPOINT_ROOT="$MLPIP_CHECKPOINT_ROOT"
          SELECTOR_CACHE_DIR="$MLPIP_CACHE_DIR"
          SELECTOR_RESULTS_ROOT="$PROJECT_ROOT/results/mlpip_selector_training/$DESIGN_TAG/$IP_TRAINING_CONTRACT_TAG"
          LOG_ROOT="$PROJECT_ROOT/logs/mlpip_4gpu/$STAMP/$DESIGN_TAG/$IP_TRAINING_CONTRACT_TAG"
          mkdir -p "$LOG_ROOT/selector_training"
          for SELECTOR_DATASET in $DATASETS
          do
            MLPIP_CHECKPOINT="$MLPIP_CHECKPOINT_ROOT/heart/$SELECTOR_DATASET/mlpip/model_checkpoint1"
            SELECTOR_LOG="$LOG_ROOT/selector_training/$SELECTOR_DATASET.log"
            if [ ! -s "$MLPIP_CHECKPOINT" ]
            then
              echo "Training $SELECTOR_DATASET MLPIP depth $SELECTOR_DEPTH, hidden $SELECTOR_HIDDEN_CHANNELS selector on GPU $SELECTOR_TRAIN_GPU"
              if [ "$DRY_RUN" != "1" ]
              then
                case "$SELECTOR_DATASET" in
                  ogbl-*)
                    env CUDA_VISIBLE_DEVICES="$SELECTOR_TRAIN_GPU" \
                      "$PYTHON_BIN" -m ogbl.main \
                      --device cuda:0 --mode heart --dataset "$SELECTOR_DATASET" --model mlpip \
                      --selector-depth "$SELECTOR_DEPTH" --selector-hidden-channels "$SELECTOR_HIDDEN_CHANNELS" \
                      $MLPIP_TRAIN_EXTRA_ARGS \
                      --epochs "$SELECTOR_MAX_EPOCHS" --eval-steps "$SELECTOR_EVAL_STEPS" --patience "$SELECTOR_PATIENCE" \
                      --checkpoint-root "$MLPIP_CHECKPOINT_ROOT" --results-root "$SELECTOR_RESULTS_ROOT" \
                      --num-runs "$SELECTOR_NUM_RUNS" --base-seed "$SELECTOR_BASE_SEED" --eval-cap "$SELECTOR_EVAL_CAP" \
                      --root "$DATA_ROOT" --pool all --heart-negatives "$SELECTOR_HEART_NEGATIVES" \
                      --train-negative-sampler fast \
                      > "$SELECTOR_LOG" 2>&1
                    ;;
                  *)
                    env CUDA_VISIBLE_DEVICES="$SELECTOR_TRAIN_GPU" \
                      "$PYTHON_BIN" -m pyg.main \
                      --device cuda:0 --mode heart --dataset "$SELECTOR_DATASET" --model mlpip \
                      --selector-depth "$SELECTOR_DEPTH" --selector-hidden-channels "$SELECTOR_HIDDEN_CHANNELS" \
                      $MLPIP_TRAIN_EXTRA_ARGS \
                      --epochs "$SELECTOR_MAX_EPOCHS" --eval-steps "$SELECTOR_EVAL_STEPS" --patience "$SELECTOR_PATIENCE" \
                      --checkpoint-root "$MLPIP_CHECKPOINT_ROOT" --results-root "$SELECTOR_RESULTS_ROOT" \
                      --num-runs "$SELECTOR_NUM_RUNS" --base-seed "$SELECTOR_BASE_SEED" --eval-cap "$SELECTOR_EVAL_CAP" \
                      --pool all --heart-negatives "$SELECTOR_HEART_NEGATIVES" \
                      --heart-backend gpu \
                      --heart-batch-size "$SELECTOR_HEART_BATCH_SIZE" \
                      > "$SELECTOR_LOG" 2>&1
                    ;;
                esac
                [ -s "$MLPIP_CHECKPOINT" ] || { echo "Missing selector checkpoint: $MLPIP_CHECKPOINT" >&2; exit 1; }
              fi
            else
              echo "Reusing $SELECTOR_DATASET MLPIP depth $SELECTOR_DEPTH, hidden $SELECTOR_HIDDEN_CHANNELS selector: $MLPIP_CHECKPOINT"
            fi
          done
          export GPUS DATASETS MODELS RUNS HEURISTICS SPLIT TEST_POSITIVE_CAP
          export EVALUATOR_MODULE CANDIDATE_POLICY EVALUATION_LABEL SELECTOR_POLICY SELECTOR_CHECKPOINT_MODEL SELECTOR_CHECKPOINT_ROOT SELECTOR_CHECKPOINT_RUN
          export SELECTOR_NEGATIVES SELECTOR_CACHE_DIR SELECTOR_SCORE_BATCH_SIZE SELECTOR_SEED SELECTOR_EPOCHS SELECTOR_EVAL_STEPS SELECTOR_PATIENCE SELECTOR_BATCH_SIZE SELECTOR_METRIC
          export DATA_ROOT CHECKPOINT_ROOT EDGE_BATCH_SIZE NODE_CHUNK_SIZE COMPARISON_BATCH_SIZE SOURCE_BATCH_SIZE EXTRA_ARGS QUIET DRY_RUN POLL_SECONDS LOG_ROOT
          bash "$PROJECT_ROOT/scripts/evaluator_batch.sh" "$@"
        done
      done
    done
  done
done
