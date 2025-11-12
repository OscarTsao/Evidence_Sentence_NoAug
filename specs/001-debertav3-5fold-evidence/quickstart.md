# Quickstart — 5-Fold DeBERTaV3 Evidence Binding

## Prereqs

- Python 3.10+
- Install deps: `pip install -e '.[dev]'`
- Start MLflow UI (local):

```
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

## Training (planned CLI)

Once the training script is added (see tasks), run 5-fold CV:

```
python scripts/train_cv.py \
  model.name=microsoft/deberta-v3-base \
  data.posts=data/redsm5/posts.csv \
  data.criteria_dir=data/DSM5 \
  data.neg_strategy=stratified data.neg_ratio=3 \
  train.max_epochs=3 \
  train.batch_size=16 \
  trainer.optim=adamw_torch_fused \
  trainer.metric_for_best_model=f1_macro trainer.greater_is_better=true \
  trainer.bf16=true \
  trainer.lr_scheduler_type=linear trainer.warmup_ratio=0.06 \
  cv.folds=5 cv.group=post_id \
  loss.name=weighted_ce
```

Notes:
- If fused AdamW unsupported, set `trainer.optim=adamw_torch`.
- Precision policy: prefer BF16 (`trainer.bf16=true` on Ampere+); else use
  FP16 (`trainer.fp16=true`); else omit both for FP32.
- To enable focal loss: `loss.name=focal loss.gamma=2.0`.

## Inference (planned CLI)

```
python -m src.Project.SubProject.engine.infer \
  "<criterion text>" \
  "<sentence text>" \
  model.uri=mlruns/<run>/<artifact>/model
```

Outputs: label {0,1} and probability.
