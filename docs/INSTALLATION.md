# Installation

The complete training package is based on SDFStudio revision
`370902a10dbef08cb3fe4391bd3ed1e227b5c165`, with EGSA integrated. Install this
checkout instead of installing a second SDFStudio/Nerfstudio package into the
same environment. Use a separate environment to preserve existing experiments.

## Environment

The existing test environment uses Windows, Python 3.9.20, an NVIDIA RTX 4080,
PyTorch 1.12.1+cu113, torchvision 0.13.1+cu113, nerfacc 0.3.5,
tinycudann 1.7 and Kornia 0.6.12. `constraints-tested.txt` records selected
dependency versions, not a complete lockfile. A clean-machine environment build
has not been validated.

```powershell
conda create -n egsa python=3.9.20 pip -y
conda activate egsa
python -m pip install pip==24.3.1 setuptools==75.5.0 wheel==0.44.0
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

Install the CUDA toolkit and compatible C++ build tools, then install
[tiny-cuda-nn's PyTorch bindings](https://github.com/NVlabs/tiny-cuda-nn#pytorch-extension).
The CUDA runtime bundled in a PyTorch wheel is not a CUDA compiler. The tested
tinycudann package reports version 1.7; its exact native build is not pinned.
Do not assume the latest upstream branch is compatible with this older stack.

From this repository root:

```powershell
python -m pip install -c constraints-tested.txt -e .
python -m pip check
python scripts/check_environment.py --require-cuda
python -m scripts.train neus-facto --help
```

Inspect the reported import path to confirm this checkout is being used.
Native build requirements depend on platform, GPU and compiler. See the upstream
[SDFStudio installation guidance](https://github.com/autonomousvision/sdfstudio#installation)
for dependency build details; no separate SDFStudio checkout is needed here.

## Tests and first run

```powershell
python tests/test_edge_priors.py
python tests/test_ema_hardness.py
python tests/test_sampler_defaults.py
python tests/test_verify_dataset.py
```

Extract data outside the code directory, then follow the README training example.
Use TensorBoard to avoid a viewer dependency for a first run. Numerical tests
and short training-path checks are not full convergence or paper-score tests.
