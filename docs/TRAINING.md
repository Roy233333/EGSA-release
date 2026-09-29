# Training

Run commands from the repository root after installation. `python -m
scripts.train` and the installed `ns-train` entry point use the same training
CLI. On Windows, use UTF-8 for console output:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
```

## Sampler settings

The EGSA options listed in the README set the paper's default sampler settings:
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
| Latest-error hardness (no temporal smoothing) | `--pipeline.datamanager.adaptive-ema-decay 0` |
| Pixel-level sampling | `--pipeline.datamanager.adaptive-block-size 1` |

## Scenes and backbones

Keep `--data` pointed at the actual extracted scene directory containing
`meta_data.json`, wherever it is stored.
Scene1, Scene2 and Scene3 include monocular depth/normal priors; use
`--include-mono-prior True` only with a model/loss configuration that uses them.
Scene1-HR and Scene4 have no such priors: use `--include-mono-prior False`.
Use `--load-pairs False` for the released datasets.

Available SDF backbones include `neus`, `neus-facto`, `neus-facto-angelo`,
`monosdf`, `volsdf` and `bakedsdf`.

Training/evaluation splits follow the SDFStudio parser settings; its default
`train_val_no_overlap=False` permits overlapping views.

## Implementation scope

EGSA updates visited-pixel EMA state after each training step. Model checkpoints
do not save the sampler's EMA/history dictionaries, so resuming model weights
does not restore the exact sampling history. Use fixed-resolution image batches;
the recommended block size 8 divides all released image dimensions. Other block
sizes can produce clamped-border sampling bias when dimensions are not divisible.
Foreground masks are not enforced in every adaptive sampling branch.
