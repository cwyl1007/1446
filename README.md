# Link Prediction

This repository trains and evaluates link-prediction models on PyTorch
Geometric (PyG) and Open Graph Benchmark (OGB) datasets. It supports generated
HeaRT, full-graph evaluation, and learned negative-selector evaluators. Run all
commands from the repository root.

## Install

The tested environment uses Python 3.10, PyTorch 2.5.1 with CUDA 12.1, and
PyG 2.7.0.

```bash
conda create -n lp python=3.10.20 -y
conda activate lp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c 'import torch, torch_geometric, ogb; print(torch.__version__, torch.cuda.is_available())'
```

Base datasets are stored under `dataset/` and download on first use when the
source supports it.

Supported PyG datasets are `cora`, `citeseer`, `pubmed`, `amazon-c`,
`amazon-p`, `wiki-chameleon`, `wiki-squirrel`, `reddit`,  and
`facebook`. Supported OGB datasets are `ogbl-collab`, `ogbl-ddi`, `ogbl-ppa`,
and `ogbl-citation2`.

Model runner names are `mf`, `mlp`, `ppr`, `concat`, `gcn`, `gat`, `sage`,
`gae`, `seal`, `buddy`, `neo-gnn`, `ncn`, `ncnc`, `nbfnet`, `peg`, `lpformer`,
and `n2v`. Use `heuristics` for the non-learned heuristic suite.

## Create configuration files

Grid search creates the hyperparameter file loaded by training:

```text
configs/<model>_<dataset>_config.json
```

Run the appropriate search:

```bash
python -m pyg.grid_search --device cuda --mode heart --dataset cora --model gcn
python -m ogbl.grid_search --device cuda --mode heart --dataset ogbl-collab --model gcn
```

The winning validation configuration is written to the canonical path, such as
`configs/gcn_cora_config.json`, and existing model-specific fields outside the
search space are preserved. Useful options are `--metric`, `--epochs`,
`--max-configs`, and `--output-config PATH`. Use `--output-config` to inspect a
result without replacing the active configuration.

## Run experiments

Run one configured model directly:

```bash
python -m pyg.main --device cuda --mode heart --dataset cora --model gcn --num-runs 5 --base-seed 0
python -m ogbl.main --device cuda --mode heart --dataset ogbl-collab --model gcn --num-runs 5 --base-seed 0
```

Node2Vec and heuristics use `pyg.n2v_main`, `ogbl.n2v_main`,
`pyg.heuristics_main`, and `ogbl.heuristics_main`.

For a batch, edit the quoted variables at the top of `scripts/pyg_run.sh` or
`scripts/ogbl_run.sh`:

```sh
MODES="heart"
DATASETS="cora citeseer pubmed"
MODELS="gcn sage n2v heuristics"
GPUS="0 1 2 3"
NUM_RUNS="5"
BASE_SEED="0"
```

Then run `sh scripts/pyg_run.sh` or `sh scripts/ogbl_run.sh`.

The evaluator runners score checkpoints created by those main runs. Edit their
headers in the same way:

```sh
GPUS="0 1 2 3"
DATASETS="cora citeseer pubmed"
MODELS="gcn sage lpformer n2v"
RUNS="1 2 3 4 5"
HEURISTICS="all"
SPLIT="test"
TEST_POSITIVE_CAP="100000"
```

Selector runners additionally expose `<NAME>_SELECTOR_DEPTHS`,
`<NAME>_SELECTOR_HIDDEN_CHANNELS`, `<NAME>_SELECTOR_DROPOUTS`,
`<NAME>_SELECTOR_LEARNING_RATES`, and `<NAME>_SELECTOR_WEIGHT_DECAYS`. Lists
produce a Cartesian product of selector configurations.

| Runner | Evaluation |
| --- | --- |
| `scripts/full_graph_run.sh` | Every legal candidate |
| `scripts/concat_run.sh` | Concat-selected negatives |
| `scripts/mlp_run.sh` | MLP-selected negatives |
| `scripts/learnedfeat_run.sh` | Learned-feature Concat-selected negatives |
| `scripts/concatip_run.sh` | Concat encoder with inner-product scoring |
| `scripts/mlpip_run.sh` | MLP encoder with inner-product scoring |

`learnedfeat` uses Concat with learned node features instead of the dataset's supplied features.

Launch a runner with `sh scripts/<name>_run.sh`. Where `PYTHON_BIN` is declared,
set it to the Python executable from your environment; evaluator batches also
declare it in `scripts/evaluator_batch.sh`. Use disjoint `GPUS` lists for
concurrent batches. Keep candidate caches unless you intend to rebuild them.

## Outputs

```text
configs/                         selected hyperparameters
checkpoints/heart/               trained target checkpoints
results/pyg/heart/               PyG HeaRT summaries
results/ogbl/heart/              OGB HeaRT summaries
results/full_graph/              full-graph results
results/concat/                  Concat results
results/mlp/                     MLP results
results/learnedfeat/             learnedfeat results
results/concatip/                Concat inner-product results
results/mlpip/                   MLP inner-product results
logs/                            batch logs
```

Evaluator runners skip missing target checkpoints. Preserve configurations,
seeds, result metadata, and candidate caches when reproducing an experiment.
