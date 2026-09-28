# Training

Run commands from the repository root after installation. `python -m
scripts.train` and the installed `ns-train` entry point use the same training
CLI. On Windows, use UTF-8 for console output:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## Sampler settings

The README example explicitly sets the paper's default sampler settings:
Sobel, edge exponent 1.3, uniform mixture 0.6, EMA decay 0.9, hardness enabled,
block size 8, and uniform sampling within allocated cells.

Keep the model, seed, ray budget, preprocessing, split and optimisation schedule
fixed when comparing samplers. In the same training command, change only:

| Variant | Option before `sdfstudio-data` |
|---|---|
| Uniform baseline | `--pipeline.datamanager.sampler-type default` |
| Canny prior | `--pipeline.datamanager.adaptive-edge-type canny` |
| No edge prior | `--pipeline.datamanager.adaptive-edge-weight 0` |
| No hardness feedback | `--pipeline.datamanager.adaptive-hardness-enabled False` |
| No EMA smoothing | `--pipeline.datamanager.adaptive-ema-decay 0` |
| Pixel-level sampling | `--pipeline.datamanager.adaptive-block-size 1` |

No EMA smoothing is not the same as disabling hardness. The Canny option uses
Kornia 0.6.12's normalised non-maximum-suppressed magnitude, not its binary
hysteresis output. It is installed with the training dependencies.

## Scenes and backbones

Replace `--data` with the extracted scene directory containing `meta_data.json`.
Scene1, Scene2 and Scene3 include monocular depth/normal priors; use
`--include-mono-prior True` only with a model/loss configuration that uses them.
Scene1-HR and Scene4 have no such priors: use `--include-mono-prior False`.
Keep `--load-pairs False` for these examples.

Available SDF backbones include `neus`, `neus-facto`, `neus-facto-angelo`,
`monosdf`, `volsdf` and `bakedsdf`. Replace `neus-facto` in the README command
and inspect the corresponding `--help`; changing backbone also changes its
default model and optimisation settings. Scene4 uses `neus-facto-angelo` in
the paper. Ray counts and iteration budgets can be set before the data parser:

```powershell
--pipeline.datamanager.train-num-rays-per-batch 2048 `
--trainer.max-num-iterations 100000
```

These values illustrate CLI usage; use the paper's budget for the particular
experiment rather than applying one budget to all scenes and backbones.

## Evaluation views

The inherited parser defaults to `train_val_no_overlap=False`. Evaluation
images can therefore also be training images. For a new disjoint split, append:

```powershell
--skip-every-for-val-split 8 --train-val-no-overlap True
```

These are parser options and must follow `sdfstudio-data`. This selects every
eighth frame for evaluation and excludes those frames from training. This
example does not establish the split used by an existing paper run. W&B records
metrics produced by the trainer; a last logged value is not automatically a
whole-held-out-set average.

## Implementation scope

EGSA updates visited-pixel EMA state after each training step. Model checkpoints
do not save the sampler's EMA/history dictionaries, so resuming model weights
does not restore the exact sampling history. Use fixed-resolution image batches;
the recommended block size 8 divides all released image dimensions. Other block
sizes can produce clamped-border sampling bias when dimensions are not divisible.
Foreground masks are not enforced in every adaptive sampling branch.
