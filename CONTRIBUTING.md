# Contributing

Please keep sampler changes separate from renderer or loss changes. Explain
whether a change affects sampling probabilities, state updates or only runtime.

Before proposing a change, run:

```powershell
python tests/test_ema_hardness.py
python tests/test_edge_priors.py
python tests/test_verify_dataset.py
python tests/test_sampler_defaults.py
```

The first two tests require PyTorch and the dependencies in `requirements.txt`.
Integration changes also require
a short SDFStudio training test; CPU unit tests do not establish GPU support.

For bug reports include Python/PyTorch/Kornia versions, upstream source version,
sampler settings, image dimensions and a minimal error log. Remove credentials,
private paths and personal information. Do not commit datasets, checkpoints,
training logs or proprietary images. Numerical changes should include regression
tests for repeated pixels, mass consistency and disabled-hardness behavior.
