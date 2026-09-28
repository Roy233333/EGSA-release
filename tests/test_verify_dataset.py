"""Dependency-free regression tests for the public integrity checker."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_dataset.py'


class VerificationTests(unittest.TestCase):
    def test_integrity_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / 'rgb.png'
            image.write_bytes(b'test bytes; this test checks integrity, not decoding')
            metadata = {'width': 1, 'height': 1, 'frames': [{'rgb_path': 'rgb.png'}]}
            meta = root / 'meta_data.json'
            meta.write_text(json.dumps(metadata))
            manifest = root / 'files.sha256'
            manifest.write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.name + '\n'
                                        for p in (meta, image)))

            def run(*extra):
                return subprocess.run([sys.executable, str(SCRIPT), str(root), *extra],
                                      capture_output=True, text=True).returncode

            self.assertEqual(run('--manifest', str(manifest)), 0)
            image.write_bytes(b'corrupted')
            self.assertNotEqual(run('--manifest', str(manifest)), 0)
            image.unlink()
            self.assertNotEqual(run(), 0)
            metadata['frames'][0]['rgb_path'] = '../outside.png'
            meta.write_text(json.dumps(metadata))
            self.assertNotEqual(run(), 0)
            metadata['frames'] = []
            meta.write_text(json.dumps(metadata))
            self.assertNotEqual(run(), 0)
            metadata['frames'] = [{}]
            meta.write_text(json.dumps(metadata))
            self.assertNotEqual(run(), 0)


if __name__ == '__main__':
    unittest.main()
