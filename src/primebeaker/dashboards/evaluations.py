"""Read saved benchmark summaries for optional dashboard result panels."""
from __future__ import annotations

from .store import read_json


def evaluation_results(config):
    panels = []
    for spec in config.data.get('evaluation_panels', []):
        series = []
        for source in spec.get('series', []):
            rows = []
            for point in source.get('points', []):
                row = {'step': point['step'], 'state': 'awaiting checkpoint', 'score': None}
                try:
                    summary = read_json(config.resolve(point['summary_file']))
                    summary = summary.get('policies', [summary])[0]
                    graded, expected = summary.get('graded'), summary.get('expected', 1000)
                    row.update(graded=graded, expected=expected, failed=summary.get('failed', 0), refused=summary.get('refused', 0))
                    value = summary.get('mean_weighted_score')
                    if summary.get('complete') and graded == expected and isinstance(value, (int, float)) and not isinstance(value, bool):
                        row.update(score=100 * value, state='completed')
                    else:
                        row['state'] = 'grading'
                except (OSError, ValueError, KeyError, IndexError, TypeError):
                    pass
                if row['state'] != 'completed':
                    for key in ['status_file', 'lifecycle_file']:
                        if point.get(key):
                            try:
                                status = read_json(config.resolve(point[key]))
                                row.update(state=status.get('state', row['state']))
                                if 'graded' in status: row['graded'] = status['graded']
                            except (OSError, ValueError):
                                pass
                    if point.get('submission_file') and config.resolve(point['submission_file']).exists() and row['state'] == 'awaiting checkpoint':
                        row['state'] = 'submitted'
                rows.append(row)
            series.append({k: source.get(k) for k in ['id', 'label', 'color', 'line_style']} | {'points': rows})
        panels.append({k: spec.get(k) for k in ['id', 'title', 'caveat']} | {'series': series})
    return {'panels': panels}
