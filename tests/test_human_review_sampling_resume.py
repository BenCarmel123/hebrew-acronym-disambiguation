"""Queue expansion must retain annotations without weakening source matching."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from hebrew_acronyms.human_review_data import source_identity, sampling_identity
from hebrew_acronyms.human_review_server import ReviewStore, SCHEMA


def data(ids):
    provenance = {'repository': 'fixture://only', 'files': [
        {'path': 'items.csv', 'sha256': 'a' * 64, 'bytes': 100},
        {'path': 'responses.csv', 'sha256': 'b' * 64, 'bytes': 200}]}
    identity = source_identity(provenance)
    queues = {'calibration': ids, 'evaluation': [], 'diagnosis': []}
    return {'dataset_id': identity, 'source_identity': identity,
            'sampling_plan_id': sampling_identity(queues, 'fixture', len(ids)),
            'provenance': provenance, 'queues': queues,
            'items': [{'id': i, 'answers': [{'id': i + ':a', 'system_id': 'a', 'auto_score': False},
                                           {'id': i + ':b', 'system_id': 'b', 'auto_score': True}]} for i in ['one', 'two']]}


class SamplingResumeTests(unittest.TestCase):
    def test_expansion_restart_json_csv_preserve_snapshot_and_decisions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'annotations.json'
            first = ReviewStore(data(['one']), path)
            def update(action, **kwargs):
                return first.update(dict(revision=first.snapshot()['revision'], item_id='one', action=action, **kwargs))
            annotation = {'schema_version': SCHEMA, 'annotator': 'QA_NOT_HUMAN',
                          'interpretation': 'invented independent attempt', 'interpretation_kind': 'interpretation',
                          'item_problems': ['undecidable'], 'answers': {
                              'one:a': {'system_id': 'a', 'quality': 'undecidable'},
                              'one:b': {'system_id': 'b', 'quality': 'partial'}}}
            update('draft', annotation=annotation)
            update('expose', stage='candidates')
            annotation['updated_interpretation'] = 'invented revised attempt'
            update('complete', annotation=annotation)
            saved = first.snapshot()
            restarted = ReviewStore(data(['one', 'two']), path)
            self.assertEqual(saved['records'], restarted.snapshot()['records'])
            self.assertNotEqual(saved['sampling_plan_id'], restarted.snapshot()['sampling_plan_id'])
            self.assertEqual(restarted.snapshot()['summary']['complete_items'], 1)
            imported = ReviewStore(data(['one', 'two']), Path(folder) / 'imported.json')
            imported.import_bundle(saved, 0)
            self.assertEqual(saved['records'], imported.snapshot()['records'])
            csv_imported = imported.csv_bundle(first.export_csv())
            self.assertEqual(saved['records'], csv_imported['records'])
            self.assertTrue(list(Path(folder).glob('annotations.json.before-migration.*.bak')))
            initial = saved['records']['one']['initial_interpretation']
            self.assertEqual(initial['text'], 'invented independent attempt')
            self.assertEqual(saved['records']['one']['draft']['updated_interpretation'], 'invented revised attempt')

    def test_changed_source_rejected_even_with_copied_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ReviewStore(data(['one']), Path(folder) / 'annotations.json')
            incoming = store.snapshot()
            incoming['provenance']['files'][0]['sha256'] = 'c' * 64
            with self.assertRaises(ValueError):
                store.import_bundle(incoming, 0)
            self.assertEqual(store.snapshot()['revision'], 0)
            self.assertFalse(store.path.exists())


if __name__ == '__main__':
    unittest.main()
