# feature-engineering-toolkit

Reusable feature-engineering transformers, an end-to-end ML workflow, and Markdown
reporting for reproducible data-science projects.

The toolkit ships scikit-learn-style transformers that operate on `pandas.DataFrame`
objects (imputation, encoding including rare-category grouping, frequency encoding,
out-of-fold and leave-one-out target encoding, and Weight of Evidence /
Information Value, quantile and uniform binning, scaling,
datetime extraction, polynomial/interaction features, and feature selection), a small
workflow that loads a synthetic churn
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

## Out-of-fold target encoding

`TargetEncoder` replaces categorical values with a smoothed mean of the target.
Fitting those means on every training row and then using them as features for the
*same* rows leaks the label — a unique or rare category encodes to (almost) that
row's own `y`.

By default (`cv=5`), `fit_transform` uses K-fold **out-of-fold** encodings: each
training row is encoded from the folds it does not belong to. `transform` (test /
held-out data) always uses the global mapping learned from the full training set.

```python
from feature_engineering_kit import TargetEncoder

enc = TargetEncoder(columns=["plan"], target="churn", cv=5, random_state=0)
X_train_enc = enc.fit_transform(X_train, y_train)  # out-of-fold encodings
X_test_enc = enc.transform(X_test)                 # global training mapping
```

Pass `cv=None` to disable out-of-fold fitting (fit-on-all; not recommended for
model training features). Bayesian `smoothing` still shrinks rare categories
toward the global mean in both modes.

The churn workflow target-encodes `plan` this way: `run_churn_workflow` and
`build_preprocessing_pipeline` pass the run seed into `TargetEncoder` so fold
assignments are reproducible, then call `fit_transform` on the train split and
`transform` on the test split.

For a strict leave-one-out mean — every other row with the same category,
including binary targets — use `LeaveOneOutEncoder` below. Setting
`TargetEncoder(cv=n_rows, shuffle=False)` is leave-one-out only when the
target is continuous. Binary targets use stratified folds, which cannot hold
out one row at a time.

## Leave-one-out target encoding

`LeaveOneOutEncoder` replaces each training row with a smoothed mean of the
target on the other rows that share its category:

`LOO_i = (S_c - y_i + smoothing * m) / (n_c - 1 + smoothing)`

`S_c` and `n_c` are the target sum and count of category `c`, and `m` is the
global target mean. `smoothing=0` is the mean of the other rows in the
category. A level that appears only once has no other row, so it encodes to
`m`. A column of unique ids therefore becomes the constant global mean. A
fit-on-all `TargetEncoder(cv=None, smoothing=0)` on that same column copies
`y`.

`transform` does not leave a row out. Test rows use the smoothed means fit on
the full training set, the same mapping as `TargetEncoder` with `cv=None` and
the same `smoothing`. Unseen categories map to `m`.

```python
from feature_engineering_kit import LeaveOneOutEncoder

enc = LeaveOneOutEncoder(columns=["plan"], target="churn", smoothing=10.0)
X_train_loo = enc.fit_transform(X_train, y_train)  # other rows in the category
X_test_loo = enc.transform(X_test)                 # full-training smoothed means
```

With `smoothing=0`, a two-row category swaps the two labels: each row is
encoded as the other row's target. Use `TargetEncoder` with `cv >= 2` when
groups are that small and the column will be used to train a model.
`LeaveOneOutEncoder` is deterministic (no fold seed). It is a normal
`FeatureEngineeringPipeline` step: `fit_transform` writes leave-one-out
values, and a later `transform` uses the global training mapping.

## Weight of Evidence and Information Value

`WoEEncoder` replaces categorical values with Weight of Evidence for a binary
target:

`WoE_i = ln( P(X=i | y=0) / P(X=i | y=1) )`

Laplace `smoothing` (default `0.5`) keeps the log defined when a category has
zero events or non-events. Unseen categories map to `0.0`. After `fit`, each
column's Information Value is available on `iv_` and as a ranked table from
`iv_report()` (Siddiqi strength labels: unpredictive / weak / medium / strong /
suspicious). `information_value(X, y, columns)` computes IV without encoding.

The same leakage rule as target encoding applies: a rare category's raw WoE
reconstructs that row's label. Default `cv=5` makes `fit_transform` out-of-fold;
`transform` always uses the full-training mapping and IV tables.

```python
from feature_engineering_kit import WoEEncoder, information_value

enc = WoEEncoder(columns=["plan", "region"], target="churn", cv=5, random_state=0)
X_train_woe = enc.fit_transform(X_train, y_train)  # out-of-fold WoE
X_test_woe = enc.transform(X_test)                 # global training mapping
print(enc.iv_report())
print(enc.iv_table_)
print(information_value(X_train, y_train, columns=["plan", "region"]))
```

Optional churn-workflow hook: pass `woe_encode=["region"]` to
`run_churn_workflow` / `build_preprocessing_pipeline`, or
`feature-engineering-toolkit run --woe-encode region`. Those columns are
WoE-encoded (not one-hot or target-encoded). The Markdown report then includes
an Information Value section. A column cannot be both target-encoded and
WoE-encoded. Default runs still target-encode `plan` and one-hot `region`.

## Rare-category grouping

`RareCategoryGrouper` collapses infrequent categorical levels before a later
encoder (frequency, one-hot, target, or WoE). Any level whose training count is
below `min_count` (default `2`, so singletons are grouped) is replaced with
`other`, or a custom `other_label`. Levels at or above `min_count` are kept.
Categories that never appeared in training are treated as rare and mapped to
the same bucket. Missing values are left missing so a later imputer can still
see them.

Grouping is fit on the training rows only. A city that was common in training
stays unchanged on a test row even if it appears once there.

```python
from feature_engineering_kit import RareCategoryGrouper

grouper = RareCategoryGrouper(columns=["city"], min_count=20)
X_train_g = grouper.fit_transform(X_train)
X_test_g = grouper.transform(X_test)  # unseen cities -> "other"
```

Kept levels and training counts are stored on `kept_` and `counts_`. Use this
before target or WoE encoding when rare levels should share one smoothed mean
or one Weight of Evidence instead of an unstable per-level estimate.

## Frequency encoding

`FrequencyEncoder` replaces each category with its training-set frequency.
By default that is the relative frequency `count / n_rows`. Pass
`normalize=False` for raw counts. Unseen categories map to `0`. Missing values
are their own level and encode to the missing rate of the fitted sample (`0`
when that sample had no missing rows).

`min_count` optionally pools levels below that count into `other` and assigns
every pooled level, plus any unseen level, the pooled bucket's frequency. With
no missing values and `cv=None`, that is the same result as
`RareCategoryGrouper(min_count=...)` followed by `FrequencyEncoder()`.

Counting a row toward its own frequency slightly inflates rare levels: a unique
id encodes to `1/n` only because of itself. Pass `cv` (for example `cv=5`) to
make `fit_transform` out-of-fold. Each training row is encoded from the folds
it does not belong to, and a `min_count` bucket is recomputed inside each fold.
`transform` always uses the global training frequencies. Folds are a plain
`KFold` because frequency encoding does not use `y`. `cv=None` (the default)
encodes with full-sample frequencies. Setting `cv` to the number of rows and
`shuffle=False` is leave-one-out.

```python
from feature_engineering_kit import FrequencyEncoder

enc = FrequencyEncoder(columns=["city"], min_count=20, cv=5, random_state=0)
X_train_f = enc.fit_transform(X_train)  # out-of-fold frequencies
X_test_f = enc.transform(X_test)         # global training frequencies
```

After `fit`, per-level training counts are on `counts_`, the global encoding
map is on `maps_`, and the fallback frequency for unseen levels is on
`other_frequency_` (`0` when `min_count` is not set).

Both classes are normal `FeatureEngineeringPipeline` steps, so they can run
before one-hot, target, or WoE encoding:

```python
from feature_engineering_kit import (
    FeatureEngineeringPipeline,
    FrequencyEncoder,
    RareCategoryGrouper,
)

pipe = FeatureEngineeringPipeline(
    steps=[
        ("rare_city", RareCategoryGrouper(columns=["city"], min_count=20)),
        ("freq_city", FrequencyEncoder(columns=["city"])),
    ]
)
```

## Quantile binning

`QuantileBinning` discretizes numeric columns. The default
`strategy="quantile"` cuts at empirical quantiles so each bin has about the
same number of training rows (equal frequency). `strategy="uniform"` uses
equal-width cuts between the training minimum and maximum. Edges are learned
on `fit` and reused by `transform`, including on a held-out frame. Values
outside the training range clip to the nearest edge bin.

Duplicate quantiles (ties, sparse bins) are dropped. `n_bins_` is the number
of bins that remain, and `bin_edges_` stores the cuts. A constant column
becomes a single bin. `n_bins` (an integer `>= 2`) is the number of bins
requested, not a promise that every cut survives.

`encode="ordinal"` (the default) replaces each column with codes
`0 .. n_bins_-1`. Missing values stay missing. `encode="onehot"` replaces the
column with `{column}__bin_{i}` indicators; a missing row is all zeros.
`inverse_transform` maps codes back to bin midpoints. An all-zero one-hot row
inverts to missing. The inverse is the bin center, not the original value.

```python
from feature_engineering_kit import QuantileBinning

binner = QuantileBinning(columns=["age", "income"], n_bins=5, strategy="quantile")
X_train_b = binner.fit_transform(X_train)
X_test_b = binner.transform(X_test)          # training edges
X_mid = binner.inverse_transform(X_train_b)  # bin midpoints

wide = QuantileBinning(columns=["income"], n_bins=4, strategy="uniform", encode="onehot")
X_onehot = wide.fit_transform(X_train)
```

Pass `quantile_bin=["age"]` to `run_churn_workflow` /
`build_preprocessing_pipeline`, or
`feature-engineering-toolkit run --quantile-bin age --n-bins 5`. Binning runs
after numeric imputation. Binned columns are not standardized, so ordinal
codes stay `0 .. k-1`. One-hot binning drops the source column; the churn
workflow then skips polynomial and interaction terms that still name it.
Default runs do not bin.

`QuantileBinning` is a normal `FeatureEngineeringPipeline` step:

```python
from feature_engineering_kit import FeatureEngineeringPipeline, QuantileBinning

pipe = FeatureEngineeringPipeline(
    steps=[("bins", QuantileBinning(columns=["age"], n_bins=4))]
)
```
