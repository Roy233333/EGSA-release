#!/usr/bin/env python3
"""Validate files referenced by an SDFStudio meta_data.json dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def safe_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe dataset path: {relative}')
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f'Path escapes dataset: {relative}')
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--manifest", type=Path, help="Per-file SHA-256 manifest (not archives.sha256)")
    args = parser.parse_args()

    root = args.dataset.resolve()
    metadata_path = root / "meta_data.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    frames = metadata.get("frames", [])
    if not frames:
        raise ValueError('Dataset must contain frames')
    if metadata.get('width', 0) <= 0 or metadata.get('height', 0) <= 0:
        raise ValueError('Dataset dimensions must be positive')
    missing: list[str] = []
    referenced = 0
    for frame in frames:
        required = ['rgb_path']
        if metadata.get('has_mono_prior'):
            required += ['mono_depth_path', 'mono_normal_path']
        for key in required:
            if not frame.get(key):
                raise ValueError(f'Missing required frame field: {key}')
        for key in (k for k in frame if k.endswith('_path')):
            relative = frame.get(key)
            if not relative:
                continue
            referenced += 1
            if not safe_path(root, relative).is_file():
                missing.append(relative)

    print(
        f"dataset={root}\n"
        f"resolution={metadata.get('width')}x{metadata.get('height')}\n"
        f"frames={len(frames)}\n"
        f"referenced_files={referenced}\n"
        f"missing_files={len(missing)}"
    )
    if missing:
        for relative in missing[:20]:
            print(f"MISSING {relative}")
        raise SystemExit(1)
    if args.manifest:
        verified = set()
        for line in args.manifest.read_text(encoding='utf-8').splitlines():
            expected, relative = line.split('  ', 1)
            h = hashlib.sha256()
            with safe_path(root, relative).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError(f'SHA-256 mismatch: {relative}')
            verified.add(relative)
        needed = {'meta_data.json'} | {v for f in frames for k, v in f.items()
                                      if k.endswith('_path') and isinstance(v, str) and v}
        if not needed.issubset(verified):
            raise ValueError('Manifest does not cover all metadata references')
        print(f'sha256_verified_files={len(verified)}')


if __name__ == "__main__":
    main()
