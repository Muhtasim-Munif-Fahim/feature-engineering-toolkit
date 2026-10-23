# feature-engineering-toolkit

Reusable feature-engineering transformers, an end-to-end ML workflow, and Markdown
reporting for reproducible data-science projects.

The toolkit ships scikit-learn-style transformers that operate on `pandas.DataFrame`
objects (imputation including k-nearest-neighbour imputation, encoding including rare-category grouping, frequency encoding,
out-of-fold, leave-one-out, CatBoost-style ordered, and James-Stein target encoding, and Weight of Evidence /
Information Value, quantile and uniform binning, supervised MDLP (entropy) binning, scaling (including MaxAbs, Normalizer, Winsorizer, QuantileTransformer, Yeo-Johnson and Box-Cox),
datetime extraction (including CyclicalEncoder sin/cos), polynomial/interaction and B-spline features, and feature selection), a small
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




## Cyclical encoding

`CyclicalEncoder` maps a periodic numeric column onto the unit circle via
`sin(2πx/P)` and `cos(2πx/P)`, so hour 23 sits next to hour 0:

```python
from feature_engineering_kit import CyclicalEncoder

enc = CyclicalEncoder(columns=["hour", "month"], period={"hour": 24, "month": 12})
X = enc.fit_transform(X)  # adds hour__sin/cos, month__sin/cos; drops originals
```

## Yeo-Johnson power transform

`YeoJohnsonScaler` applies the Yeo-Johnson power transform (defined for
non-positive values, unlike Box-Cox) and optionally standardises each column.
λ is estimated per column by maximising the Gaussian log-likelihood of the
transformed values; pass `lmbda=` to fix it instead.

```python
from feature_engineering_kit import YeoJohnsonScaler

scaler = YeoJohnsonScaler(columns=["tenure", "monthly_charges"])
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # reuses fit-time λ / mean / scale
```


## Box-Cox power transform

`BoxCoxScaler` applies the classic Box-Cox power transform to **strictly
positive** columns and optionally standardises each column. λ is estimated per
column by maximising the Gaussian log-likelihood of the transformed values;
pass `lmbda=` to fix it instead.

Non-positive handling: `fit` raises `ValueError` if any selected column has a
non-positive finite value. At `transform` time non-positive or non-finite
entries become `NaN` (they are not clipped).

```python
from feature_engineering_kit import BoxCoxScaler

scaler = BoxCoxScaler(columns=["tenure", "monthly_charges"])
X_train = scaler.fit_transform(X_train)   # requires x > 0
X_test = scaler.transform(X_test)         # reuses fit-time λ / mean / scale
```

## MaxAbs scaling

`MaxAbsScaler` divides each column by its maximum absolute value so training
values land in `[-1, 1]` without shifting the origin (handy for sparse data).

```python
from feature_engineering_kit import MaxAbsScaler

scaler = MaxAbsScaler(columns=["tenure", "monthly_charges"])
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # reuses fit-time max abs
```



## Row-wise Normalizer

`Normalizer` scales each row to unit norm over the selected columns (`l1`,
`l2`, or `max`). Fit stores the column list only; transform divides by the
row norm (zero-norm rows are left unchanged).

```python
from feature_engineering_kit import Normalizer

scaler = Normalizer(columns=["tenure", "monthly_charges"], norm="l2")
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Winsorizer

`Winsorizer` clips each selected numeric column at fitted lower/upper
percentiles (default 5th and 95th). Fit stores per-column thresholds;
transform clips values outside those bounds (already-clipped data is
idempotent).

```python
from feature_engineering_kit import Winsorizer

scaler = Winsorizer(columns=["tenure", "monthly_charges"], limits=(0.05, 0.95))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # reuses fit-time percentiles
```

## QuantileTransformer scaling

`QuantileTransformer` maps each column through its empirical CDF to a uniform
`[0, 1]` distribution (or further to a standard normal). Fit-time quantile
references are reused at transform time, and `inverse_transform` recovers the
original scale.

```python
from feature_engineering_kit import QuantileTransformer

scaler = QuantileTransformer(
    columns=["tenure", "monthly_charges"],
    n_quantiles=100,
    output_distribution="uniform",
)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)          # reuses fit-time references
X_back = scaler.inverse_transform(X_train) # approximate round-trip
```

## KNN imputation

`KNNImputer` fills each missing numeric cell with the mean of that column over
the `n_neighbors` nearest training rows that have it observed. Distances use
`nan_euclidean_distances`: the Euclidean distance over shared coordinates,
rescaled by `p / m` for missing ones (the same convention as scikit-learn,
whose output it reproduces). With `weights="distance"`, donors are weighted
by `1 / d`. A row that shares no coordinate with any donor falls back to the
training mean. `add_indicator=True` appends `{column}__was_missing` flags.

```python
from feature_engineering_kit import KNNImputer, build_preprocessing_pipeline

imp = KNNImputer(columns=["tenure", "monthly_charges"], n_neighbors=5, weights="distance")
train_filled = imp.fit_transform(train_df)
test_filled = imp.transform(test_df)        # donors come from the training rows

pipe = build_preprocessing_pipeline(X, numeric_impute="knn", knn_neighbors=5)
```

## Iterative (MICE) imputation

`IterativeImputer` imputes numeric columns with chained regressions (van Buuren & Groothuis-Oudshoorn 2011). Missing cells start at the column mean. Each round then regresses every column on all the other columns and overwrites that column's missing cells with the predictions. Rounds stop at `max_iter`, or when the change in matrix infinity norm drops below `tol * max|observed|`.

- The default estimator is `"bayesian_ridge"`, which uses evidence-maximized ridge regression (`fit_bayesian_ridge`). `"linear"` uses ordinary least squares.
- `imputation_order` can be `ascending`, `descending`, `roman` or `arabic`. `min_value` and `max_value` clip imputed values, and `add_indicator` appends missing-value indicator columns.
- `transform` replays the fitted regression sequence on new data.
- Results match scikit-learn's `IterativeImputer` (`sample_posterior=False`) to floating-point precision, using pure NumPy.
- Pipeline option: `build_preprocessing_pipeline(..., numeric_impute="iterative")`.

```python
from feature_engineering_kit import IterativeImputer
imp = IterativeImputer(["age", "income", "tenure"], max_iter=10, add_indicator=True)
train_filled = imp.fit_transform(train)
test_filled = imp.transform(test)
```

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


## CatBoost-style ordered target encoding

`CatBoostEncoder` replaces each training row with a smoothed mean of the
target on *preceding* rows that share its category in a random order
(Prokhorenkova et al., NeurIPS 2018). With running sum `S_prev`, count
`n_prev`, global mean `m`, and smoothing weight `a`:

`enc_i = (S_prev + a * m) / (n_prev + a)`

When `n_prev = 0` the value is `m`, so a column of unique ids becomes the
constant global mean instead of copying `y`. Averaging over
`n_permutations` independent shuffles reduces the variance of the online
estimate. `transform` uses the full-training smoothed means (the same
mapping as `TargetEncoder(cv=None)` with the same `smoothing`); unseen
categories map to `m`.

```python
from feature_engineering_kit import CatBoostEncoder

enc = CatBoostEncoder(
    columns=["plan"],
    target="churn",
    smoothing=10.0,
    n_permutations=5,
    random_state=0,
)
X_train_cb = enc.fit_transform(X_train, y_train)  # ordered / permuted encodings
X_test_cb = enc.transform(X_test)                 # full-training smoothed means
```

`CatBoostEncoder` is a normal `Transformer`, so it slots into
`FeatureEngineeringPipeline` the same way as `LeaveOneOutEncoder` or
`JamesSteinEncoder`.

## James-Stein target encoding

`JamesSteinEncoder` replaces each category with an empirical-Bayes / James-Stein
shrinkage of its target mean toward the global mean. Rare levels shrink harder
than abundant ones:

```text
B_c = sigma2 / (sigma2 + n_c * tau2)
JS_c = (1 - B_c) * m_c + B_c * m
```

`sigma2` is the pooled within-category variance of `y`; `tau2` is the positive
part of the method-of-moments between-category variance. Unseen categories map
to the global mean.

```python
from feature_engineering_kit import JamesSteinEncoder

enc = JamesSteinEncoder(columns=["plan"], target="churn")
X_train_js = enc.fit_transform(X_train, y_train)
X_test_js = enc.transform(X_test)
```

`JamesSteinEncoder` is a normal `Transformer`, so it slots into
`FeatureEngineeringPipeline` the same way as `LeaveOneOutEncoder` or
`FrequencyEncoder`.

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

## Feature hashing

`HashingEncoder` applies the signed hashing trick (Weinberger et al., 2009):
each category maps to one of `n_components` buckets, so the output width does
not grow with cardinality, unseen levels need no special handling, and no
per-level state is stored. A second hash bit assigns a `+1`/`-1` sign so
colliding levels cancel in expectation (hashed inner products are unbiased
estimates of one-hot inner products).

```python
from feature_engineering_kit import HashingEncoder

enc = HashingEncoder(columns=["user_agent", "merchant_id"], n_components=32)
out = enc.fit_transform(df)          # user_agent__hash0..31, merchant_id__hash0..31

# one shared block for all columns (hashes "col=value" tokens)
shared = HashingEncoder(columns=["user_agent", "merchant_id"], n_components=64, combine=True)
```

Hashes use keyed BLAKE2b (`seed` is the key) rather than Python's salted
`hash()`, so encodings are identical across processes and machines. Missing
values hash as their own token (`handle_missing="value"`) or leave the block
all-zero (`"zero"`); `alternate_sign=False` gives plain bucket counts.

## B-spline features

`SplineTransformer` replaces each selected numeric column with
`n_knots + degree - 1` B-spline basis columns (`x__spline_0`, ...). Knots are
placed on `fit` (`knots="uniform"` between the training min/max, `"quantile"`
at empirical quantiles, or an explicit increasing array) and reused by
`transform`. Inside the training range the basis is non-negative, sums to one
per row, and has at most `degree + 1` non-zero entries, so a linear model on
these columns fits smooth non-monotone effects without high-degree polynomial
blow-up.

```python
from feature_engineering_kit import SplineTransformer

spl = SplineTransformer(columns=["tenure"], n_knots=6, degree=3, knots="quantile")
X_train = spl.fit_transform(X_train)
X_test = spl.transform(X_test)  # out-of-range values clip to the boundary
```

`extrapolation="constant"` (default) clips out-of-range values,
`"continue"` evaluates the padded basis, and `"error"` raises. Pass
`include_bias=False` to drop one column per feature (removes collinearity with
an intercept) and `keep_original=True` to keep the source column. Missing
values give `NaN` in every spline column of that row. `bspline_basis(x, knots,
degree)` exposes the underlying Cox-de Boor evaluator.

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

## Supervised MDLP binning

`MDLPBinning` is the Fayyad–Irani (1993) entropy discretizer. Each numeric
column is split recursively at the boundary that best separates the target
classes. A split is kept only if its information gain beats the Minimum
Description Length threshold, so the number of bins is chosen by the data:
a feature unrelated to the target stays a single bin, and a feature whose
class mix changes sharply gets a cut at each change. It works with binary
or multiclass targets (`fit` needs `y`). `max_bins` caps the number of bins
and keeps the most informative cuts (segments are expanded best-first by
entropy reduction). `min_samples_leaf` sets the smallest allowed bin.
`cut_points_`, `bin_edges_`, `encode="ordinal"|"onehot"` and
`inverse_transform` work as in `QuantileBinning`.

```python
from feature_engineering_kit import MDLPBinning

mdlp = MDLPBinning(columns=["age", "income"], max_bins=6)
X_train_b = mdlp.fit_transform(X_train, y_train)
X_test_b = mdlp.transform(X_test)      # training cuts
print(mdlp.cut_points_["age"])
```

## mRMR feature selection

`MRMRSelection` implements minimum-redundancy maximum-relevance selection
(Peng, Long & Ding, IEEE TPAMI 2005). It starts with the feature that has the
highest mutual information with the target, then greedily adds the feature
maximizing `I(f; y) - mean_{s in S} I(f; s)` (`scheme="difference"`, MID) or
`I(f; y) / mean_{s in S} I(f; s)` (`scheme="quotient"`, MIQ). Unlike
`MutualInfoSelection`, which ranks columns independently and happily keeps
three copies of the same signal, mRMR penalizes columns that repeat what has
already been chosen.

Mutual information is the plug-in estimate on equal-frequency bins
(`n_bins=10`, NaN gets its own bin), so selection is deterministic. Use
`task="regression"` to bin a continuous target.

```python
from feature_engineering_kit import MRMRSelection

sel = MRMRSelection(target="churn", k=8).fit(X_train, y_train)
sel.selected_     # pick order
sel.relevance_    # I(f; y) per numeric column (nats)
sel.scores_       # mRMR criterion when each feature was picked
X_small = sel.transform(X_test)   # non-numeric columns pass through
```

In the pipeline, `build_preprocessing_pipeline(..., select_k=8,
select_method="mrmr")` (or `"mutual_info"`) appends a final selection step,
and the CLI exposes it as `run --select-k 8 [--select-method mutual_info]`;
the report lists the selected features in pick order.

## ReliefF feature selection

`ReliefFSelection` ranks numeric columns by their ReliefF weight (Kononenko, 1994).
For each row it finds the `n_neighbors` nearest *hits* (same class) and the
`n_neighbors` nearest *misses* in each other class. A feature gains weight when
it differs on misses and loses weight when it differs on hits. Neighbours are
found in the full feature space, so ReliefF picks up features that only matter
*jointly*. In a `y = a XOR b` target, `a` and `b` each look like noise to
correlation or mutual-information ranking, but ReliefF ranks them first.

The scoring follows scikit-rebate:

- continuous diffs are range-scaled;
- columns with at most 10 distinct values are treated as categorical (0/1 mismatch);
- mixed data uses ReBATE's std-ramp.

For complete data, `relieff_scores` reproduces `skrebate.ReliefF` to floating-point
precision whenever `n_neighbors` is smaller than the smallest class. Missing values
are skipped per feature.

```python
from feature_engineering_kit import ReliefFSelection, relieff_scores

sel = ReliefFSelection(target="churn", k=8, n_neighbors=10).fit(X_train, y_train)
sel.selected_      # best first
sel.scores_        # ReliefF weight per numeric column, in [-1, 1]
X_small = sel.transform(X_test)
```

In the pipeline use `select_method="relieff"`; in the CLI use `run --select-k 8 --select-method relieff`.
