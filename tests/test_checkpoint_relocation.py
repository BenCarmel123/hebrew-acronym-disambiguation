"""Local checkpoint inspection and relocation on untrained, invented fixtures only."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from hebrew_acronyms.models.dictabert_cross_encoder import model as models
from hebrew_acronyms.models.dictabert_cross_encoder.training import TrainingConfig
from tests.fixtures.tiny import tiny_base_model

ROOT = Path(__file__).resolve().parents[1]


def guard(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.__new__"}:
        raise RuntimeError("Network forbidden in checkpoint fixture checks")
    if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        for root in (ROOT, ROOT.parent / "hebrew-acronym-disambiguation"):
            if any(path.is_relative_to(root / name) for name in ("data", "results", "weights", "sources")):
                raise RuntimeError("Research access forbidden in checkpoint fixture checks")


class CheckpointRelocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.addaudithook(guard)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="checkpoint-relocation-fixture-")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.snapshot = self.folder / "original-snapshot"
        self.snapshot.mkdir()
        for name in ("config.json", "tokenizer.json", "model.safetensors"):
            (self.snapshot / name).write_text("invented fixture: " + name)
        self.loader = patch.object(models, "build_base_model", side_effect=tiny_base_model)
        self.loader.start()
        self.addCleanup(self.loader.stop)
        self.config = TrainingConfig()
        tok, model, _, _ = models.build_cross_encoder(model_id=str(self.snapshot), revision="fixture-v1")
        self.checkpoint = self.folder / "untrained-fixture.pt"
        models.save_checkpoint(model, tok, self.checkpoint, self.config, {"fixture": True}, 0, 0.5)
        self.original_manifest = models.metadata_path(self.checkpoint).read_bytes()
        self.original_weights_hash = models.file_digest(self.checkpoint)
        self.copy = self.folder / "moved-snapshot"
        shutil.copytree(self.snapshot, self.copy)

    def test_inspection_never_loads_model_or_weights_and_reports_original_metadata(self):
        with patch.object(models, "build_cross_encoder") as build, patch.object(models.torch, "load") as load:
            report = models.inspect_checkpoint(self.checkpoint, snapshot_path=self.copy)
        build.assert_not_called()
        load.assert_not_called()
        self.assertTrue(report["relocated"])
        self.assertEqual(report["load_model_id"], str(self.copy.resolve()))
        self.assertEqual(report["metadata"]["training_config"]["seed"], self.config.seed)
        self.assertEqual(report["metadata"]["initialization"]["source"]["model_id"], str(self.snapshot))
        self.assertEqual(report["libraries"], models.library_versions())
        self.assertEqual(report["tokenizer_reconstruction"], "not yet checked")

    def test_missing_checkpoint_configuration_is_actionable(self):
        for checkpoint in (None, "", "   "):
            for call in (models.inspect_checkpoint, models.load_finetuned):
                with self.subTest(checkpoint=checkpoint, call=call.__name__):
                    with self.assertRaisesRegex(ValueError, "adjacent original JSON"):
                        call(checkpoint)

    def test_identical_relocated_snapshot_loads_without_original_and_manifest_stays_unchanged(self):
        self.snapshot.rename(self.folder / "unavailable-original")
        with self.assertRaises(FileNotFoundError):
            models.load_finetuned(self.checkpoint)
        _, loaded, _, _ = models.load_finetuned(self.checkpoint, snapshot_path=self.copy,
                                               expected_inputs={"fixture": True})
        self.assertEqual(loaded.initialization["source"]["model_id"], str(self.copy.resolve()))
        self.assertEqual(loaded.checkpoint_metadata["initialization"]["source"]["model_id"], str(self.snapshot))
        self.assertTrue(loaded.checkpoint_load_identity["relocated"])
        self.assertEqual(models.metadata_path(self.checkpoint).read_bytes(), self.original_manifest)
        self.assertEqual(models.file_digest(self.checkpoint), self.original_weights_hash)
        self.assertFalse(loaded.training)

    def test_different_model_or_tokenizer_contents_rejected_before_reconstruction(self):
        for filename in ("config.json", "tokenizer.json", "model.safetensors"):
            path = self.copy / filename
            original = path.read_bytes()
            path.write_bytes(original + b" changed")
            with self.subTest(filename=filename), patch.object(models, "build_cross_encoder") as build:
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    models.load_finetuned(self.checkpoint, snapshot_path=self.copy)
                build.assert_not_called()
            path.write_bytes(original)

    def test_missing_or_extra_recorded_file_rejected(self):
        extra = self.copy / "added.json"
        extra.write_text("new content")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            models.inspect_checkpoint(self.checkpoint, snapshot_path=self.copy)
        extra.unlink()
        (self.copy / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            models.inspect_checkpoint(self.checkpoint, snapshot_path=self.copy)

    def test_tokenizer_and_revision_reconstruction_checks_are_not_bypassed(self):
        with patch.object(models, "tokenizer_identity", return_value={"different": True}):
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                models.load_finetuned(self.checkpoint, snapshot_path=self.copy)
        original = models.source_identity
        def changed_revision(*args):
            return dict(original(*args), revision="different-revision")
        with patch.object(models, "source_identity", side_effect=changed_revision):
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                models.load_finetuned(self.checkpoint, snapshot_path=self.copy)

    def test_library_and_weight_checks_are_not_bypassed(self):
        libraries = dict(models.library_versions(), transformers="wrong-version")
        with patch.object(models, "library_versions", return_value=libraries), patch.object(models, "build_cross_encoder") as build:
            with self.assertRaisesRegex(ValueError, "library identity mismatch"):
                models.load_finetuned(self.checkpoint, snapshot_path=self.copy)
            build.assert_not_called()
        with self.checkpoint.open("ab") as stream:
            stream.write(b"changed fixture weights")
        with self.assertRaisesRegex(ValueError, "weights do not match"):
            models.load_finetuned(self.checkpoint, snapshot_path=self.copy)

    def test_hashless_source_cannot_be_relocated_and_original_overrides_stay_strict(self):
        tok, model, _, _ = models.build_cross_encoder(model_id="invented-hub-id", revision="fixture-v1")
        remote = self.folder / "remote-fixture.pt"
        models.save_checkpoint(model, tok, remote, self.config, {}, 0, 0.5)
        with self.assertRaisesRegex(ValueError, "requires original local file hashes"):
            models.load_finetuned(remote, snapshot_path=self.copy)
        for override in ({"model_id": "wrong"}, {"revision": "wrong"}, {"pooling": "marker"},
                         {"expected_inputs": {}}, {"config": TrainingConfig(seed=43)}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                models.load_finetuned(self.checkpoint, snapshot_path=self.copy, **override)
        with self.assertRaisesRegex(ValueError, "metadata is required"):
            models.inspect_checkpoint(self.folder / "missing.pt")


if __name__ == "__main__":
    unittest.main()
