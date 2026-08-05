# MIT-BIH Record 119 R-Peak Validation

| Metric | Value |
|---|---:|
| Expert annotated beats | 1987 |
| Detected R peaks | 1988 |
| True positives | 1987 |
| False positives | 1 |
| False negatives | 0 |
| Precision | 0.9995 |
| Recall | 1.0000 |
| F1 score | 0.9997 |
| Mean timing error | 1.73 ms |
| Mean absolute timing error | 1.77 ms |
| Maximum absolute timing error | 27.78 ms |

## Interpretation

The NeuroKit R-peak detector identified all expert-annotated beats with one additional detection over the entire recording. The resulting precision, recall and F1 score demonstrate near-perfect agreement with the MIT-BIH expert annotations.
