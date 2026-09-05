# feature-engineering-toolkit

Reusable feature-engineering transformers, an end-to-end ML workflow, and Markdown
reporting for reproducible data-science projects.

The toolkit ships scikit-learn-style transformers that operate on `pandas.DataFrame`
objects (imputation, encoding, scaling, datetime extraction, polynomial/interaction
features, and feature selection), a small workflow that loads a synthetic churn
dataset, engineers features, trains a classifier, evaluates it, and writes a Markdown
report, and a console-script entry point.

## Install

```bash
pip install -e .
```

## CLI quick start

```bash
feature-engineering-toolkit run --n-samples 2000 --seed 42 -o examples/output/demo_report.md
feature-engineering-toolkit schema
```

`run` loads the synthetic churn dataset, builds a feature-engineering + logistic
regression pipeline, evaluates it on a held-out stratified test split, and prints
(or writes with `-o`) a Markdown report with accuracy, ROC AUC, precision/recall/F1,
the confusion matrix, and the top features. `schema` prints the dataset's column
schema.

## Library quick start

```python
from feature_engineering_kit import (
    load_synthetic_churn_dataset,
    column_schema,
    run_churn_workflow,
)

df = load_synthetic_churn_dataset(n_samples=1000, seed=0)
print(column_schema(df))

result = run_churn_workflow(n_samples=1000, seed=0)
print(f"accuracy={result.accuracy:.3f} roc_auc={result.roc_auc:.3f}")
```

See `examples/run_demo.py` for a complete end-to-end demo and `tests/` for the
unit-test contract.
