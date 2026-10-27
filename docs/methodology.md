# Methodology and validation

## Question and unit of analysis

Classify the **recorded injury outcome of an individual already involved in an injury accident**. The positive class comprises killed or hospitalized road users. There are no non-accident journeys in BAAC, so this dataset does not support estimating an arbitrary driver's probability of having an accident.

Features include collision, manoeuvre and safety-equipment information recorded after the event. This is a retrospective classification study, not a deployed pre-crash warning system or a causal evaluation of road-safety measures.

## Data integrity

- Load all four tables for each year as strings; normalize accident and vehicle keys without converting them to floating-point numbers.
- Reject missing identifiers, duplicate user keys, duplicated vehicle/characteristics keys, or unmatched references.
- Consolidate locations to one row per accident. For each selected attribute (`surf`, `catr`, `vma`), keep the sole non-null value when the road rows agree; otherwise mark it missing. Other road attributes are not used.
- Join with `validate="many_to_one"` and require the merged row count to equal the user count.
- Retain severity codes 1-4; positive codes are **2 and 3**. Unknown labels are excluded, never imputed.
- Parse both `HH:MM` and `HHMM`; reject invalid hours/minutes. Derive weekday from the recorded date.
- Treat coded categories as nominal. Category `-1` and missing categories become `unknown`; other codes retain their documented identities.
- Replace out-of-range coordinates, `(0, 0)`, and speed limits outside 1-130 km/h with missing values. Missing numeric predictors are imputed using training medians.

## Evaluation protocol

| Component | Decision |
|---|---|
| Training | All retained users in 2021 and 2022 |
| Final test | All retained users in 2023 |
| Group integrity | No accident identifier shared across the split |
| Preprocessing | Fitted only on the training partition |
| Model | Random Forest: 120 trees, depth 18, minimum leaf size 5, square-root feature sampling |
| Imbalance handling | Balanced class weights within each bootstrap sample |
| Random seed | 42 |
| Binary threshold | Fixed at 0.50 before evaluation |
| Tuning | None in the corrected benchmark |
| Baseline | Training class-prior probability; majority prediction at threshold 0.50 |

Accuracy alone can be misleading because the target is imbalanced. The results include severe-class precision, recall and F1, balanced accuracy, ROC-AUC and average precision. The PR plot shows the held-out positive rate as a reference. The fixed threshold is not presented as an operationally optimal cutoff.

## What was checked

The complete corrected workflow was executed on the supplied 12 CSV files. Data-contract unit tests cover severity codes, valid/invalid time formats, and prevention of duplicated people when locations disagree. Every annual merge checks referential integrity; the split checks accident overlap. The manifest stores input hashes, feature names, parameters, yearly audit counts and the confusion matrix.

## Limits

The 2023 test set is one temporal holdout, not an external validation dataset. Users within an accident remain statistically dependent, and no confidence intervals are claimed. Changes in recording practices or geography can alter performance. No subgroup/fairness analysis, probability calibration, causal analysis, new SHAP explanations, or live deployment validation has been performed. The hospitalisation label carries the publisher's data-quality caveat. Original historical scores use a different target and protocol and must not be compared as a before/after improvement.
