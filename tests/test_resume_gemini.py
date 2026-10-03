"""Request lifecycle fixtures; no network, credentials or research inputs."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from hebrew_acronyms import experimental_study as study
from hebrew_acronyms import resume_gemini_study as resume


class ResumeTests(unittest.TestCase):
    def test_retry_after_and_backoff(self):
        self.assertEqual(resume.retry_delay('90', 1), 90)
        self.assertEqual(resume.retry_delay('invalid', 2), 10)

    def test_persist_retry_skip_completed_and_ambiguous(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = {'root': folder, 'output_root': folder, 'gemini_model': 'gemini-3.8-flash',
                        'request_timeout': 120, 'gemini_generation_config': {'thinkingConfig': {'thinkingLevel': 'low'}}}
            a = study.new_artifact(study.fixture_rows(), 'fixture', settings, {})
            request_settings = {'generationConfig': settings['gemini_generation_config'], 'timeout_seconds': 120, 'max_attempts': 1}
            a['llm_runtimes'] = {'gemini': {'requested_model': settings['gemini_model'], 'model_version': None,
                                           'request_settings': request_settings}}
            study.save_artifact(a, Path(folder)/'fixture', create=True)
            good = {'response': 'A', 'status': 'response_received', 'requested_model': settings['gemini_model'],
                    'model_version': 'fixture-v1', 'finish_reason': 'STOP', 'request_settings': request_settings}
            # Populate exact fixture prompt fields through existing collection logic, then reset status.
            for mode in ('generate', 'select'):
                study.collect_responses(a, mode, lambda _: deepcopy(good), system='gemini')
            for record in a['records']:
                if record['system'] == 'gemini':
                    record.update(status='not_run', raw_response=None)
            records = [r for r in a['records'] if r['system'] == 'gemini']
            records[0].update(status='interrupted', error='unknown completion')
            study.save_study(a)
            transient = deepcopy(good)
            transient.update(response='', status='service_error', http_status=429, retryable=True, retry_after='17')
            replies = [transient, good, good, good]
            calls, delays = [], []
            def request(prompt, **kwargs):
                saved = study.load_artifact(Path(folder)/'fixture/study.json', expected_run_id='fixture')
                self.assertTrue(any(r['status'] == 'interrupted' and r.get('prompt') == prompt for r in saved['records']))
                calls.append(prompt)
                return deepcopy(replies[len(calls)-1])
            with patch('dotenv.load_dotenv'):
                resume.run(a, request=request, sleep=delays.append)
                resume.run(a, request=lambda *a, **k: self.fail('completed request resubmitted'), sleep=delays.append)
            self.assertEqual(len(calls), 4)
            self.assertIn(17, delays)
            self.assertEqual(records[0]['status'], 'interrupted')
            self.assertEqual(sum(r['status'] == 'response_received' for r in records), 3)


    def test_persistent_service_failure_stops_after_three_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = {'root': folder, 'output_root': folder, 'gemini_model': 'gemini-3.8-flash',
                        'request_timeout': 120, 'gemini_generation_config': {}}
            a = study.new_artifact(study.fixture_rows(), 'outage', settings, {})
            bound = {'generationConfig': {}, 'timeout_seconds': 120, 'max_attempts': 1}
            a['llm_runtimes'] = {'gemini': {'requested_model': settings['gemini_model'],
                                           'model_version': None, 'request_settings': bound}}
            response = {'response': '', 'status': 'service_error', 'requested_model': settings['gemini_model'],
                        'model_version': None, 'request_settings': bound, 'http_status': 503, 'retryable': True}
            for mode in ('generate', 'select'):
                study.collect_responses(a, mode, lambda _: deepcopy(response), system='gemini')
            for r in a['records']:
                r['status'] = 'not_run'
            study.save_artifact(a, Path(folder)/'outage', create=True)
            with patch('dotenv.load_dotenv'), patch.object(resume, 'gemini_response'):
                from unittest.mock import Mock
                request = Mock(return_value=response)
                resume.run(a, request=request, sleep=lambda _: None, stop_after_service_failures=1)
            self.assertEqual(request.call_count, 3)
            self.assertEqual(sum(r['status'] == 'service_error' for r in a['records']), 1)
            self.assertEqual(a['execution_events'][-1]['event'], 'service_circuit_break')


if __name__ == '__main__':
    unittest.main()
