# UniER

UniER is a benchmark for item-level and path-level exercise recommendation.

## Environment

```bash
conda create -n unirec python=3.9
conda activate unirec
pip install -r requirements.txt
```

## Common pyKT data preprocessing

Use pyKT once to create `train_valid.csv`, `test.csv`, `train_valid_sequences.csv`, and `test_sequences.csv`:

```bash
git clone https://github.com/pykt-team/pykt-toolkit.git
cd pykt-toolkit/examples
python data_preprocess.py --dataset_name=assist2017
```

Following the instructions in Data Process, obtain `pkm.pth`, `pkc.pth`, and `qid_test_questions.txt` for subsequent exercise recommendation processing.

Weights & Biases logging in the supplied pyKT helper scripts is disabled by default. To enable it explicitly, set `WANDB_API_KEY` in the environment and pass `--use_wandb 1`. Do not store API keys in repository files.

## Unified commands

Run an entire model pipeline with one command:

```bash
python unier.py run --model AC --dataset assist2017 --device cuda:0
```

If model-local preprocessing has already been completed, training and evaluation require two commands:

```bash
python unier.py train --model AC --dataset assist2017 --device cuda:0
python unier.py evaluate --model AC --dataset assist2017 --device cuda:0
```

Every stage can also be run separately:

```bash
python unier.py prepare  --model AC --dataset assist2017
python unier.py train    --model AC --dataset assist2017
python unier.py evaluate --model AC --dataset assist2017
python unier.py report   --model AC --dataset assist2017
```

Only the model value changes between models:

```bash
python unier.py run --model KG4EX --dataset assist2017
python unier.py run --model AC --dataset assist2017
```

List supported models:

```bash
python unier.py list
```

Inspect every resolved command without executing training:

```bash
python unier.py run --model all --dataset assist2017 --device cpu --dry-run
```

## Supported models

Item-level:

- DRE
- AKT
- SimpleKT
- MMER
- KCP-ER
- MulOER-SAN
- KG4EX
- ER-TGA
- NR4DER

Path-level:

- SRC
- AC
- DQN
- RLTutor
- CSEAL
- GEHRL
- DLPR
- PKSD
- KnowLP

Aliases are case-insensitive. For example, `kcp_er`, `KCP-ER`, `simplekt`, and `SimpleKT` are accepted.

## Common arguments

```text
--model              Model name or "all"
--dataset            Dataset name; assist17 and assist2017 are normalized
--data-root          Unified dataset root, default: Datasets
--data-mode          auto, hardlink, or copy; default: auto
--refresh-data       Replace conflicting legacy model-local inputs
--device             cuda:0, cuda:1, or cpu
--seed               Random seed
--epochs             Training epochs for models exposing an epoch argument
--episodes           Episode count for path-level/RL models
--max-steps          Maximum recommendation-path length
--batch-size         Batch size; model defaults are used when omitted
--learning-rate      Learning rate; model defaults are used when omitted
--target-type        all or portion
--dkt-sort           Item-level only: reorder recommendation lists dynamically by DKT mastery during Ep evaluation
--top-k              Comma-separated evaluation cutoffs
--output-dir         Result root, default: outputs
--run-id             Run directory name, default: latest
--dry-run            Print commands and working directories only
--keep-going         Continue to later commands after a failure
--fresh              Ignore metrics from an existing run directory
```

Use command-specific help for the complete list:

```bash
python unier.py run --help
```

## Metrics and outputs

`evaluate` executes every metric applicable to the selected model and prints one combined report. The existing metric formulas are unchanged.

Item-level reports can contain ACC, novelty, diversity, volatility, coverage, proximity, NDCG, MAP, MRR, precision, recall, F1, EP-all, and EP-portion. Metrics not implemented by a model are omitted rather than synthesized.

To enable DKT-mastery-based dynamic sorting for an item-level model, add `--dkt-sort` to `evaluate` or `run`:

```bash
python unier.py evaluate --model KG4EX --dataset assist2017 --device cuda:0 --dkt-sort
```

Without this flag, the original Ep scripts and recommendation order are used. With it, Ep-all and Ep-portion use the dynamic DKT sorting implementation. The original recommendation file is kept unchanged, and the sorted list is written beside it with an `_update.txt` suffix.

Path-level models retain their original integrated training/evaluation behavior. Reward summaries printed by their runners are collected into the same result files.

Artifacts are stored under:

```text
outputs/<item-or-path>/<model>/<dataset>/<run-id>/
  config.json
  command_records.json
  metrics.json
  metrics.csv
  run.log
```

## Model-specific inputs inside the unified directory

- **AKT:** provide `models/akt/qid_test_predictions.txt`.
- **SimpleKT:** provide `models/simplekt/qid_test_predictions.txt`.
- **MulOER-SAN:** provide `models/muloer-san/qid_test_question_predictions.txt`.
- **Path-level models:** provide the common PLER artifacts listed in the unified tree; the CLI maps `assist2017` to the original `assist17` folder and skips duplicate model-local preprocessing.
- **AC:** its environment and agent DKT checkpoints are stored under `models/ac` and reused by the unified runner.
- **DLPR:** provide the DIMKT checkpoint produced by pyKT before running model-local preparation.
- **KnowLP:** requires a DIMKT checkpoint and GraphRAG outputs. The included `EDU-graphRAG/generate_concepts_textgrad.py` generates concept text for GraphRAG. Set `OPENAI_API_KEY` in the environment and run it with `--dataset assist17`.
- **SRC:** its original training script performs its test pass after training.
