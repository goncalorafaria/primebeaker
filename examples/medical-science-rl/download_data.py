#!/usr/bin/env python3
"""Download the published experiment files, pinned to the recorded release."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPOSITORY = 'graf/rubrichub-medical-science-rl-slice'
REVISION = '667c5d18eb9182c34919d427e71acec3b99c764e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--revision', default=REVISION)
    args = parser.parse_args()
    root = Path(snapshot_download(
        REPOSITORY, repo_type='dataset', revision=args.revision,
        local_dir=args.directory,
        allow_patterns=['original/**', 'EXPORT_METADATA.json', 'README.md', 'LICENSES.md']))
    manifest = json.loads((root / 'EXPORT_METADATA.json').read_text())
    for name, metadata in manifest['files'].items():
        path = root / name
        checksum = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                checksum.update(chunk)
        if path.stat().st_size != metadata['bytes'] or checksum.hexdigest() != metadata['sha256']:
            raise RuntimeError('Downloaded source artifact differs from the experiment slice: ' + name)
    print(json.dumps({'repository': REPOSITORY, 'revision': args.revision,
                      'train_data': str((root / 'original/rubric_rl').resolve()),
                      'eval_data': str((root / 'original/validation.jsonl').resolve()),
                      'training_rows': manifest['training_rows'],
                      'validation_rows': manifest['validation_rows']}, indent=2))


if __name__ == '__main__':
    main()
