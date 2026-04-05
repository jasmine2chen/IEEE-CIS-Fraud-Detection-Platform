# Contributing

Thank you for taking the time to contribute. This document explains how to get the development environment running and what to expect from the review process.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Project Layout](#project-layout)
- [Code Style](#code-style)
- [Testing](#testing)
- [Branch and PR Conventions](#branch-and-pr-conventions)
- [Adding a New Model](#adding-a-new-model)

---

## Development Setup

```bash
# 1. Clone and create a virtual environment
git clone <repo-url>
cd fraud_detection
python -m venv .venv && source .venv/bin/activate

# 2. Install the package in editable mode with dev extras
pip install -e ".[dev]"

# 3. Install and activate pre-commit hooks
pip install pre-commit
pre-commit install
```

After `pre-commit install`, every `git commit` will automatically run black,
ruff, isort, and mypy before the commit is accepted.

---

## Project Layout

```
src/
  config.py               YAML config loader
  train.py                Training orchestrator — one entry point for all models
  tune.py                 Optuna HPO — writes best params back to YAML
  data_prep/              Data loading and memory optimisation
  features/               Feature engineering (Magic UID, D-cols, uid aggs)
  models/                 One file per model family
  evaluation/             Metrics (FPR sweep, partial AUC) and plots
api/                      FastAPI serving layer
tests/                    pytest suite — mirrors src/ layout
configs/                  Single YAML for all training hyperparameters
docs/                     Architecture diagrams and production guide
```

Changes to `src/` should be mirrored by corresponding changes or additions
to `tests/`.

---

## Code Style

This project uses:

| Tool | Purpose | Config |
|---|---|---|
| **black** | Formatter | `pyproject.toml` |
| **ruff** | Linter (replaces flake8/pylint) | `pyproject.toml` |
| **isort** | Import sorter | `.pre-commit-config.yaml` (profile: black) |
| **mypy** | Static type checker | `pyproject.toml [tool.mypy]` |

All four run automatically on commit via pre-commit hooks. You can also run
them manually:

```bash
pre-commit run --all-files
```

### Docstrings

Follow **Google style**:

```python
def fpr_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: Optional[np.ndarray] = None,
) -> List[dict]:
    """Evaluate model performance across a range of FPR operating points.

    Args:
        y_true:  Ground truth binary labels.
        y_prob:  Predicted probabilities for the positive class.
        amounts: Transaction amounts for dollar-recall computation.

    Returns:
        List of dicts, one per FPR target, each containing threshold,
        actual_fpr, recall, precision, TP, FP, FN, TN, and dollar_recall.
    """
```

### Type Hints

All public functions must have fully annotated signatures. Internal helpers
(`_` prefix) should be annotated where non-obvious.

---

## Testing

```bash
make test             # run full suite
pytest tests/ -k api  # run only API tests
pytest -x             # stop on first failure
```

Test layout mirrors `src/`:

| Test file | Covers |
|---|---|
| `tests/test_data_prep.py` | `src/data_prep/`, `src/features/` |
| `tests/test_models.py` | `src/models/` |
| `tests/test_api.py` | `api/` |

**Rules:**
- Use `conftest.py` fixtures for shared sample data — avoid duplicating fixture
  construction in individual test files.
- Mock disk I/O (`joblib.load`, `torch.load`) in unit tests; integration tests
  that require the full pipeline should be marked `@pytest.mark.slow` and
  skipped in CI with `-m "not slow"`.
- New model code must include at minimum a smoke test that verifies the forward
  pass shape and that the two-stage pipeline (encoder → XGBoost) returns
  predictions without error.

---

## Branch and PR Conventions

### Branch names

```
feat/<short-description>    new functionality
fix/<short-description>     bug fix
refactor/<description>      no behaviour change
docs/<description>          documentation only
chore/<description>         tooling, CI, dependencies
```

### Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add transformer_xgboost hybrid model
fix: prevent temporal leakage in GNN edge construction
docs: add production cost analysis
chore: bump xgboost to 2.1.4
```

### PR checklist

Before opening a pull request, confirm:

- [ ] `pre-commit run --all-files` passes with no errors
- [ ] `make test` passes
- [ ] New or changed behaviour is covered by tests
- [ ] Public functions have Google-style docstrings
- [ ] `configs/model_config.yaml` is updated if new hyperparameters are introduced
- [ ] `CHANGELOG.md` entry added (if applicable)

---

## Adding a New Model

1. Create `src/models/<model_name>.py` following the two-stage pattern in
   `mlp_tree.py` or `transformer_tree.py`:
   - Encoder pre-training function
   - Embedding extraction function
   - `train_<model>_xgboost()` entry point that returns `(encoder, xgb_model)`

2. Register the model in `src/train.py` under the `elif model_type == ...` block.

3. Add search spaces in `src/tune.py` under the `_suggest_<model>_params()`
   section.

4. Add a smoke test in `tests/test_models.py`.

5. Document the architecture in [docs/architecture.md](docs/architecture.md).
