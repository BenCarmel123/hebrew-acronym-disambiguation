"""Installed imports and top-to-bottom notebook execution on invented inputs."""
from pathlib import Path
import subprocess
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]


class NotebookTests(unittest.TestCase):
    def test_source_imports_do_not_write_or_use_network(self):
        code = """
import os, sys
from pathlib import Path
root = Path.cwd()

def guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
        raise RuntimeError('Network during import')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(root / name) for name in ('data', 'results', 'weights')):
            raise RuntimeError('Research read during import')
    if event == 'open' and args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
        raise RuntimeError('File write during import')

sys.addaudithook(guard)
import hebrew_acronyms.models.dictabert_similarity.model
import hebrew_acronyms.models.dictabert_cross_encoder.model
import hebrew_acronyms.models.dictabert_cross_encoder.encoding
import hebrew_acronyms.models.dictabert_cross_encoder.training
import hebrew_acronyms.models.dictabert_cross_encoder.workflow
import hebrew_acronyms.models.dictabert_cross_encoder.eval
import hebrew_acronyms.pipelines.check_environment
"""
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_cells_run_on_tiny_fixtures_without_cache(self):
        result = subprocess.run([sys.executable, "-B", "-m", "tests.run_notebook"],
                                cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout[-2000:])

    def test_offline_configuration_does_not_install_socket_hooks(self):
        from unittest.mock import patch
        from hebrew_acronyms.models.dictabert_cross_encoder.workflow import enable_offline
        with patch("sys.addaudithook") as hook:
            enable_offline()
        hook.assert_not_called()

    def test_explicit_training_paths_and_snapshot_boundaries(self):
        import tempfile
        from hebrew_acronyms.models.dictabert_cross_encoder.workflow import validate_training_paths, find_snapshot
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev, checkpoint = [root / name for name in ("train.csv", "dev.csv", "best.pt")]
            with self.assertRaises(ValueError):
                validate_training_paths(None, None, None)
            with self.assertRaises(FileNotFoundError):
                validate_training_paths(train, dev, checkpoint)
            train.write_text("invented")
            dev.write_text("invented")
            validate_training_paths(train, dev, checkpoint)
            self.assertFalse(checkpoint.exists())
            Path(str(checkpoint) + ".json").write_text("{}")
            with self.assertRaises(FileExistsError):
                validate_training_paths(train, dev, checkpoint)
            with self.assertRaises(FileNotFoundError):
                find_snapshot(root / "absent")
            self.assertEqual(find_snapshot(root), root.resolve())
