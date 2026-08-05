# Detailed MIT-BIH arrhythmia baseline

This package is research-only and is not imported by the production API.

It constructs 216-sample beats around expert MIT-BIH `atr` annotation
positions, evaluates a configurable detailed ontology, calculates weights from
the training split only, and trains two otherwise identical central baselines:

1. weighted cross entropy;
2. weighted cross entropy with conservative training-only augmentation.

Download the original records and run the experiment:

```bash
python -m research.detailed_arrhythmia.scripts.download_mit_bih /path/to/mitdb
python -m research.detailed_arrhythmia.scripts.run_baseline /path/to/mitdb
```

The runner persists the annotation audit, patient split, split distributions,
training-only class weights, augmentation examples, separate checkpoints,
checkpoint metadata, evaluation results, confusion matrices, and ablation.

Passing numerical augmentation checks is not evidence of clinical or
physiological validation. All augmented examples require expert visual review.
