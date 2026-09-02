#!/bin/sh
set -e
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
PYTHON_BIN="/data/miniconda3/envs/lp/bin/python"

GPUS="0 1 2 3"
DATASETS="cora citeseer pubmed amazon-c amazon-p wiki-chameleon wiki-squirrel facebook ogbl-collab ogbl-ddi ogbl-ppa"
MODELS="mf mlp ppr concat gcn gat sage gae seal buddy neo-gnn ncn ncnc nbfnet peg lpformer n2v"
RUNS="1"
CONCAT_SELECTOR_DEPTHS="2"
CONCAT_SELECTOR_HIDDEN_CHANNELS="256"
CONCAT_SELECTOR_DROPOUTS="0.1"
CONCAT_SELECTOR_LEARNING_RATES="0.01"
CONCAT_SELECTOR_WEIGHT_DECAYS="0"
HEURISTICS="all"
SPLIT="test"
TEST_POSITIVE_CAP="100000"
COMPUTE_AUC="no"
CONCAT_NEGATIVES="500"

CONCAT_CHECKPOINT_RUN="1"
CONCAT_SCORE_BATCH_SIZE="65536"
EVALUATOR_MODULE="eval_modes.concat_mode"
SELECTOR_POLICY="concat"
CANDIDATE_POLICY="$SELECTOR_POLICY"
EVALUATION_LABEL="$SELECTOR_POLICY"
SELECTOR_CHECKPOINT_MODEL="concat"
SELECTOR_CHECKPOINT_RUN="$CONCAT_CHECKPOINT_RUN"
SELECTOR_CHECKPOINT_MODE="ranked-selector"
TARGET_CHECKPOINT_MODE="heart"
SELECTOR_NEGATIVES="$CONCAT_NEGATIVES"
SELECTOR_SCORE_BATCH_SIZE="$CONCAT_SCORE_BATCH_SIZE"
SELECTOR_SEED=""
SELECTOR_EPOCHS=""
SELECTOR_EVAL_STEPS=""
SELECTOR_PATIENCE=""
SELECTOR_BATCH_SIZE=""
SELECTOR_METRIC=""
SELECTOR_TRAIN_GPU="0"
SELECTOR_EVAL_CAP="100000"
SELECTOR_BASE_SEED="0"
SELECTOR_EPOCHS="500"
SELECTOR_EVAL_STEPS="5"
SELECTOR_PATIENCE="10"
SELECTOR_TRAINING_TAG="neutral_validation_strict_bce_v1"
DATA_ROOT="dataset"
CHECKPOINT_ROOT="checkpoints"
EDGE_BATCH_SIZE="0"
NODE_CHUNK_SIZE="0"
COMPARISON_BATCH_SIZE="256"
SOURCE_BATCH_SIZE=""
QUIET="0"
DRY_RUN="0"
POLL_SECONDS="1"

STAMP="$(date +%Y%m%d_%H%M%S)_$$"
for SELECTOR_DEPTH in $CONCAT_SELECTOR_DEPTHS
do
  case "$SELECTOR_DEPTH" in
    ""|*[!0-9]*|0)
      echo "CONCAT_SELECTOR_DEPTHS must contain positive integers." >&2
      exit 2
      ;;
  esac
  for SELECTOR_HIDDEN_CHANNELS in $CONCAT_SELECTOR_HIDDEN_CHANNELS
  do
    case "$SELECTOR_HIDDEN_CHANNELS" in
      ""|*[!0-9]*|0)
        echo "CONCAT_SELECTOR_HIDDEN_CHANNELS must contain positive integers." >&2
        exit 2
        ;;
    esac
    for SELECTOR_DROPOUT in $CONCAT_SELECTOR_DROPOUTS
    do
      for SELECTOR_LEARNING_RATE in $CONCAT_SELECTOR_LEARNING_RATES
      do
        for SELECTOR_WEIGHT_DECAY in $CONCAT_SELECTOR_WEIGHT_DECAYS
        do
          DEPTH_TAG="selector_depth_$SELECTOR_DEPTH"
          HIDDEN_TAG="hidden_$SELECTOR_HIDDEN_CHANNELS"
          DESIGN_TAG="$DEPTH_TAG"_"$HIDDEN_TAG"_"dropout_$SELECTOR_DROPOUT"_"lr_$SELECTOR_LEARNING_RATE"_"weight_decay_$SELECTOR_WEIGHT_DECAY"_"$SELECTOR_TRAINING_TAG"
          CONCAT_TRAIN_EXTRA_ARGS="--selector-dropout $SELECTOR_DROPOUT --selector-lr $SELECTOR_LEARNING_RATE --selector-weight-decay $SELECTOR_WEIGHT_DECAY"
          EXTRA_ARGS="--concat-selector-depth $SELECTOR_DEPTH --concat-selector-hidden-channels $SELECTOR_HIDDEN_CHANNELS --concat-selector-dropout $SELECTOR_DROPOUT --concat-selector-lr $SELECTOR_LEARNING_RATE --concat-selector-weight-decay $SELECTOR_WEIGHT_DECAY"
          CONCAT_CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints/concat_$DESIGN_TAG"
          CONCAT_CACHE_DIR="concat_sets/$DESIGN_TAG"
          SELECTOR_CHECKPOINT_ROOT="$CONCAT_CHECKPOINT_ROOT"
          SELECTOR_CACHE_DIR="$CONCAT_CACHE_DIR"
          SELECTOR_RESULTS_ROOT="$PROJECT_ROOT/results/concat_selector_training/$DESIGN_TAG"
          LOG_ROOT="$PROJECT_ROOT/logs/concat_4gpu/$STAMP/$DESIGN_TAG"
          mkdir -p "$LOG_ROOT/selector_training"
          for SELECTOR_DATASET in $DATASETS
          do
            CONCAT_CHECKPOINT="$CONCAT_CHECKPOINT_ROOT/$SELECTOR_CHECKPOINT_MODE/$SELECTOR_DATASET/concat/model_checkpoint1"
            SELECTOR_LOG="$LOG_ROOT/selector_training/$SELECTOR_DATASET.log"
            if [ ! -s "$CONCAT_CHECKPOINT" ]
            then
              echo "Training $SELECTOR_DATASET Concat depth $SELECTOR_DEPTH, hidden $SELECTOR_HIDDEN_CHANNELS selector on GPU $SELECTOR_TRAIN_GPU"
              if [ "$DRY_RUN" != "1" ]
              then
                case "$SELECTOR_DATASET" in
                  ogbl-*)
                    env CUDA_VISIBLE_DEVICES="$SELECTOR_TRAIN_GPU" \
                      "$PYTHON_BIN" -m ogbl.main \
                      --device cuda:0 --mode "$SELECTOR_CHECKPOINT_MODE" --dataset "$SELECTOR_DATASET" --model concat \
                      --selector-depth "$SELECTOR_DEPTH" --selector-hidden-channels "$SELECTOR_HIDDEN_CHANNELS" \
                      $CONCAT_TRAIN_EXTRA_ARGS \
                      --checkpoint-root "$CONCAT_CHECKPOINT_ROOT" --results-root "$SELECTOR_RESULTS_ROOT" \
                      --num-runs "1" --base-seed "$SELECTOR_BASE_SEED" --eval-cap "$SELECTOR_EVAL_CAP" \
                      --epochs "$SELECTOR_EPOCHS" --eval-steps "$SELECTOR_EVAL_STEPS" --patience "$SELECTOR_PATIENCE" \
                      --root "$DATA_ROOT" --train-negative-sampler fast \
                      > "$SELECTOR_LOG" 2>&1
                    ;;
                  *)
                    env CUDA_VISIBLE_DEVICES="$SELECTOR_TRAIN_GPU" \
                      "$PYTHON_BIN" -m pyg.main \
                      --device cuda:0 --mode "$SELECTOR_CHECKPOINT_MODE" --dataset "$SELECTOR_DATASET" --model concat \
                      --selector-depth "$SELECTOR_DEPTH" --selector-hidden-channels "$SELECTOR_HIDDEN_CHANNELS" \
                      $CONCAT_TRAIN_EXTRA_ARGS \
                      --checkpoint-root "$CONCAT_CHECKPOINT_ROOT" --results-root "$SELECTOR_RESULTS_ROOT" \
                      --num-runs "1" --base-seed "$SELECTOR_BASE_SEED" --eval-cap "$SELECTOR_EVAL_CAP" \
                      --epochs "$SELECTOR_EPOCHS" --eval-steps "$SELECTOR_EVAL_STEPS" --patience "$SELECTOR_PATIENCE" \
                      --root "$DATA_ROOT" \
                      > "$SELECTOR_LOG" 2>&1
                    ;;
                esac
                [ -s "$CONCAT_CHECKPOINT" ] || { echo "Missing selector checkpoint: $CONCAT_CHECKPOINT" >&2; exit 1; }
              fi
            else
              echo "Reusing $SELECTOR_DATASET Concat depth $SELECTOR_DEPTH, hidden $SELECTOR_HIDDEN_CHANNELS selector: $CONCAT_CHECKPOINT"
            fi
          done
          export GPUS DATASETS MODELS RUNS HEURISTICS SPLIT TEST_POSITIVE_CAP COMPUTE_AUC
          export EVALUATOR_MODULE CANDIDATE_POLICY EVALUATION_LABEL SELECTOR_POLICY SELECTOR_CHECKPOINT_MODEL SELECTOR_CHECKPOINT_ROOT SELECTOR_CHECKPOINT_RUN SELECTOR_CHECKPOINT_MODE TARGET_CHECKPOINT_MODE
          export SELECTOR_NEGATIVES SELECTOR_CACHE_DIR SELECTOR_SCORE_BATCH_SIZE SELECTOR_SEED SELECTOR_EPOCHS SELECTOR_EVAL_STEPS SELECTOR_PATIENCE SELECTOR_BATCH_SIZE SELECTOR_METRIC
          export DATA_ROOT CHECKPOINT_ROOT EDGE_BATCH_SIZE NODE_CHUNK_SIZE COMPARISON_BATCH_SIZE SOURCE_BATCH_SIZE EXTRA_ARGS QUIET DRY_RUN POLL_SECONDS LOG_ROOT
          bash "$PROJECT_ROOT/scripts/evaluator_batch.sh" "$@"
        done
      done
    done
  done
done
