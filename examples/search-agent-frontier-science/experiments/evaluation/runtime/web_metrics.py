"""Content-free backend events and terminal-output audit counters."""
import json, os, re, time
from pathlib import Path
from collections import Counter

def emit(event, **fields):
    directory = os.environ.get('WEB_METRICS_DIR')
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f'backend-{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(time=time.time(), event=event, **fields)) + '\n')

def trace_metrics(trace):
    counts = Counter()
    messages = [n.get('message', {}) for n in trace.get('nodes', [])]
    results = {m.get('tool_call_id'): m.get('content') for m in messages if m.get('role') == 'tool'}
    for message in messages:
        for call in message.get('tool_calls') or []:
            if call.get('name') != 'webterminal':
                continue
            counts['terminal_calls'] += 1
            args = call.get('arguments', {})
            try:
                if isinstance(args, str): args = json.loads(args)
                command = args.get('command', '')
            except (ValueError, AttributeError): command = ''
            redirect = bool(re.search(r'https?://(?:[^/ ]+\.)?google\.com/(?:goto|url)\b', command))
            counts['redirect_open_calls'] += int(redirect)
            raw = results.get(call.get('id'))
            if raw is None:
                counts['missing_terminal_response'] += 1
                continue
            try:
                result = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(result, dict) or 'stdout' not in result: raise ValueError()
            except (ValueError, TypeError):
                counts['unparsed_terminal_response'] += 1
                continue
            empty = not str(result.get('stdout') or '').strip()
            error = bool(result.get('stderr')) or result.get('success') is False or result.get('exit_code', 0) != 0
            counts['terminal_errors'] += int(error)
            counts['terminal_truncated'] += int(bool(result.get('truncated')))
            if empty:
                counts['empty_stdout'] += 1
                counts['redirect_empty_stdout'] += int(redirect)
                if error: counts['empty_stdout_with_error'] += 1
                elif re.search(r'\|\s*(?:rg|grep)\b', command):
                    # Raw page body is unavailable here: do not claim a fetch failure.
                    counts['empty_filter_output_cause_unknown'] += 1
                else: counts['empty_unfiltered_output'] += 1
    return counts

def save_summary(path, counts):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    payload = dict(counts)
    payload['empty_stdout_rate'] = counts['empty_stdout'] / counts['terminal_calls'] if counts['terminal_calls'] else None
    payload['redirect_empty_stdout_rate'] = counts['redirect_empty_stdout'] / counts['redirect_open_calls'] if counts['redirect_open_calls'] else None
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)
