"""Resume Gemini against the exact prompts in an identified saved Qwen dev run."""
from copy import deepcopy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from hebrew_acronyms import experimental_study as study
from hebrew_acronyms.models.gemini.eval import gemini_response


def retry_delay(value, attempt):
    """Respect delta-seconds and HTTP-date Retry-After, with bounded backoff otherwise."""
    if value is not None:
        try:
            seconds = float(value)
            if math.isfinite(seconds):
                return max(0.0, seconds)
        except (ValueError, TypeError):
            try:
                return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return min(60, 5 * 2 ** (attempt - 1))


def prepare(source_path, source_run_id, root, output_root, run_id):
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    source = study.load_artifact(source_path, expected_run_id=source_run_id)
    if not source.get('cohort', {}).get('full_dev') or len(source['items']) != 62:
        raise ValueError('An identified complete 62-item dev source is required')
    settings = deepcopy(source['settings'])
    settings.update(root=str(root), output_root=str(output_root), run_id=run_id,
                    enable_qwen=False, enable_gemini=True, enable_encoder=False,
                    source_prompt_run_id=source_run_id,
                    retry_policy={'max_attempts_per_item': 3, 'pause_between_requests_seconds': 3,
                                  'transient_only': True, 'honor_retry_after': True})
    artifact = study.new_artifact(source['items'], run_id, settings, study.source_provenance(root))
    artifact['provenance']['source_sha256']['resume_gemini_study.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    artifact['cohort'] = deepcopy(source['cohort'])
    artifact['input_source'] = {'path': str(Path(source_path).resolve()), 'run_id': source_run_id,
                              'artifact_sha256': hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
                              'note': 'Exact saved input snapshot; prompts copied from identified Qwen records.'}
    by_key = {(r['item_id'], r['task']): r for r in source['records'] if r['system'] == 'qwen'}
    for record in artifact['records']:
        if record['system'] != 'gemini':
            continue
        old = by_key[(record['item_id'], record['task'])]
        for key in ('sentence', 'target_raw', 'prompt', 'prompt_sha256', 'prompt_sentence', 'shown_order'):
            record[key] = deepcopy(old[key])
        record['model'] = settings['gemini_model']
        record['model_revision'] = None
    study.save_artifact(artifact, Path(output_root) / run_id, create=True)
    study.connect_gemini(artifact)
    return artifact


def run(artifact, *, request=gemini_response, sleep=time.sleep, tasks=("select", "generate"),
        max_items_per_task=None, stop_after_service_failures=5):
    """Persist before/after each attempt. Ambiguous interrupted requests are never resent."""
    from dotenv import load_dotenv
    settings = artifact['settings']
    load_dotenv(Path(settings['root']) / '.env', override=False, interpolate=False)
    study.validate_artifact(artifact, expected_run_id=artifact['run_id'])
    records = [r for task in tasks for r in artifact['records']
               if r['system'] == 'gemini' and r['task'] == task]
    attempted = {task: 0 for task in tasks}
    consecutive_failures = 0
    for record in records:
        if record['status'] != 'not_run':
            continue
        if max_items_per_task is not None and attempted[record['task']] >= max_items_per_task:
            continue
        attempted[record['task']] += 1
        for attempt in range(1, 4):
            record.update(status='interrupted', error='Request started; completion not yet saved')
            study.save_study(artifact)
            response = request(record['prompt'], model=settings['gemini_model'],
                               timeout=settings['request_timeout'],
                               generation_config=settings['gemini_generation_config'])
            study._record_response(artifact, record, 'gemini', response)
            record.setdefault('attempt_history', []).append(deepcopy(response))
            study.save_study(artifact)
            if response.get('status') != 'service_error' or not response.get('retryable') or attempt == 3:
                break
            delay = retry_delay(response.get('retry_after'), attempt)
            print(json.dumps({'retry': record['item_id'], 'attempt': attempt, 'delay_seconds': delay}), flush=True)
            sleep(delay)
        counts = {task: sum(r['status'] != 'not_run' for r in records if r['task'] == task)
                  for task in ('select', 'generate')}
        print(json.dumps({'progress': counts, 'last_status': record['status']}, ensure_ascii=False), flush=True)
        html, _ = study.display_tables(artifact, max_items=62)
        (Path(settings['output_root']) / artifact['run_id'] / 'saved_results.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>Gemini saved dev results</title>' + html)
        consecutive_failures = consecutive_failures + 1 if record['status'] == 'service_error' else 0
        if consecutive_failures >= stop_after_service_failures:
            artifact.setdefault('execution_events', []).append({
                'event': 'service_circuit_break', 'consecutive_failed_items': consecutive_failures,
                'at_utc': datetime.now(timezone.utc).isoformat(),
                'note': 'Remaining unattempted records retained; resume only after service recovery.'})
            study.save_study(artifact)
            break
        sleep(3)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-run-id', required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    path = args.output_root / args.run_id / 'study.json'
    artifact = (study.load_artifact(path, expected_run_id=args.run_id) if path.exists() else
                prepare(args.source, args.source_run_id, args.root, args.output_root, args.run_id))
    if not study.llm_runtime(artifact, 'gemini'):
        study.connect_gemini(artifact)
    run(artifact)


if __name__ == '__main__':
    main()
