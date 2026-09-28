"""Report dependencies and exercise the installed edge-prior branch. No writes."""
import argparse
import importlib
import importlib.metadata as metadata
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-cuda', action='store_true')
    # Backwards compatibility: both edge options are now always checked.
    parser.add_argument('--canny', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    print('Python:', sys.version)
    print('Executable:', sys.executable)
    errors = []
    for name in ['torch', 'torchvision', 'numpy', 'nerfacc', 'tinycudann', 'tyro', 'kornia']:
        try:
            module = importlib.import_module(name)
            print(name, metadata.version(name), getattr(module, '__file__', ''))
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    try:
        import torch
        print('Torch CUDA runtime:', torch.version.cuda)
        print('CUDA available:', torch.cuda.is_available())
        if args.require_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required but unavailable')
            print('GPU:', torch.cuda.get_device_name(0))
            assert (torch.ones(1, device='cuda') + 1).item() == 2
        from nerfstudio.data.edge_priors import compute_edge_map_cpu
        print('Installed EGSA edge module:', importlib.import_module('nerfstudio.data.edge_priors').__file__)
        image = torch.rand(32, 32, 3, generator=torch.Generator().manual_seed(42))
        sobel = compute_edge_map_cpu(image, 'sobel')
        assert torch.isfinite(sobel).all()
        canny = compute_edge_map_cpu(image, 'canny')
        assert torch.isfinite(canny).all()
        assert not torch.equal(sobel, canny), 'Canny unexpectedly matches Sobel'
    except Exception as exc:
        errors.append(f'Runtime check: {exc}')
    for error in errors:
        print('FAIL:', error)
    if errors:
        raise SystemExit(1)
    print('PASS: imports/edge branch checks only; full training is a separate test.')


if __name__ == '__main__':
    main()
