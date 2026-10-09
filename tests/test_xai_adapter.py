"""Offline xAI schema, privacy, accounting and recovery contracts."""
from copy import deepcopy
import json
import os
import unittest
from unittest.mock import Mock, patch

from hebrew_acronyms.models.xai import eval as xai
from hebrew_acronyms.test_evaluation import _usage


def response(payload, status=200):
    return Mock(status_code=status, headers={}, json=Mock(return_value=payload))


class XAIAdapterTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'XAI_API_KEY': 'fixture-secret'})
        env.start()
        self.addCleanup(env.stop)
        self.payload = {'id': 'fixture-response', 'model': 'grok-4.7', 'status': 'completed',
            'output': [{'type': 'reasoning', 'encrypted_content': 'private-thought'},
                       {'type': 'message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': 'A'}]}],
            'usage': {'input_tokens': 100, 'output_tokens': 12, 'total_tokens': 112,
                      'output_tokens_details': {'reasoning_tokens': 10},
                      'input_tokens_details': {'cached_tokens': 80}, 'cost_in_usd_ticks': 1120000}}

    def call(self, payload=None, status=200):
        with patch.object(xai.requests, 'post', return_value=response(
                self.payload if payload is None else payload, status)) as post:
            result = xai.xai_response('unchanged prompt', model=xai.MODEL)
        post.assert_called_once()
        return result, post.call_args.kwargs

    def test_explicit_low_no_tools_single_final_and_no_reasoning_saved(self):
        result, kwargs = self.call()
        self.assertEqual(kwargs['json'], {'model': 'grok-4.7',
            'input': [{'role': 'user', 'content': 'unchanged prompt'}],
            'reasoning': {'effort': 'low'}, 'max_output_tokens': 1024,
            'tools': [], 'tool_choice': 'none', 'stream': False, 'store': False})
        self.assertFalse(kwargs['allow_redirects'])
        self.assertEqual(result['status'], 'response_received')
        self.assertEqual(result['response'], 'A')
        self.assertEqual(result['thought_parts_omitted'], 1)
        self.assertNotIn('private-thought', json.dumps(result))
        self.assertNotIn('fixture-secret', json.dumps(result))
        self.assertEqual(result['usage_metadata'], self.payload['usage'])

    def test_partial_wrong_model_tools_and_refusal_cannot_pass(self):
        for kind in ('partial', 'wrong-model', 'tool', 'refusal', 'multiple', 'malformed'):
            data = deepcopy(self.payload)
            if kind == 'partial':
                data.update(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
            elif kind == 'wrong-model':
                data['model'] = 'another-model'
            elif kind == 'tool':
                data['output'].append({'type': 'function_call', 'arguments': 'private'})
            elif kind == 'refusal':
                data['output'][1]['content'] = [{'type': 'refusal', 'refusal': 'private'}]
            elif kind == 'multiple':
                data['output'].append(deepcopy(data['output'][1]))
            else:
                data['output'][1]['content'] = None
            with self.subTest(kind=kind):
                result, _ = self.call(data)
                self.assertNotEqual(result['status'], 'response_received')
                self.assertNotIn('private', json.dumps(result))
                self.assertFalse(result['retryable'])
                if kind == 'partial':
                    self.assertEqual(result['response'], 'A')
                    self.assertEqual(result['finish_reason'], 'max_output_tokens')

    def test_reasoning_charged_once_and_unknown_usage_reserved(self):
        result, _ = self.call()
        self.assertEqual(_usage(result, 'xai'), {'input_tokens': 100, 'output_tokens': 2,
                                              'thinking_tokens': 10, 'cached_input_tokens': 80})
        for change in ({'total_tokens': 122}, {'output_tokens': -1},
                       {'output_tokens_details': {'reasoning_tokens': True}},
                       {'input_tokens_details': 'invalid'}):
            altered = deepcopy(result)
            altered['usage_metadata'].update(change)
            self.assertIsNone(_usage(altered, 'xai'))

    def test_billing_block_and_credentials_are_sanitized(self):
        result, _ = self.call({'error': {'message': 'fixture-secret private account'}}, 402)
        self.assertTrue(result['provider_blocked'])
        self.assertFalse(result['retryable'])
        self.assertNotIn('private', json.dumps(result))
        with patch.dict(os.environ, {}, clear=True), patch.object(xai.requests, 'post') as post:
            self.assertEqual(xai.xai_response('prompt', model=xai.MODEL)['attempts'], 0)
            post.assert_not_called()

    def test_lookup_no_generation_and_settings_fail_closed(self):
        with patch.object(xai.requests, 'get', return_value=response({'id': xai.MODEL})) as get, patch.object(xai.requests, 'post') as post:
            self.assertEqual(xai.inspect_xai_model(model=xai.MODEL)['status'], 'available')
            get.assert_called_once()
            post.assert_not_called()
        for kwargs in ({'model': 'other'}, {'effort': 'high'}, {'max_output_tokens': 0}):
            with patch.object(xai.requests, 'post') as post, self.assertRaises(ValueError):
                xai.xai_response('prompt', **{'model': xai.MODEL, **kwargs})
            post.assert_not_called()
