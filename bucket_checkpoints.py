import json
import re
from pathlib import Path

from huggingface_hub import HfApi

MARKER = '_COMPLETE.json'


def upload_checkpoint(local_dir, bucket_uri, keep_last_k, api=None):
    
    if not isinstance(keep_last_k, int) or keep_last_k < 1:
        raise ValueError('bucket checkpoint keep_last_k must be a positive integer')

    prefix = 'hf://buckets/'
    if not bucket_uri.startswith(prefix):
        raise ValueError('checkpoint_bucket must start with hf://buckets/')

    parts = bucket_uri[len(prefix):].strip('/').split('/')
    if len(parts) < 3 or any(p in ('', '.', '..') for p in parts):
        raise ValueError('checkpoint_bucket must include a dedicated folder inside a bucket')

    bucket_id = '/'.join(parts[:2])
    run_prefix = '/'.join(parts[2:]) + '/'
    local_dir = Path(local_dir)
    tag = local_dir.name
    if not re.fullmatch(r'step_\d+', tag):
        raise ValueError(f'invalid checkpoint tag: {tag}')

    api = api or HfApi()
    api.sync_bucket(str(local_dir), bucket_uri.rstrip('/') + '/' + tag)
    api.batch_bucket_files(
        bucket_id,
        add=[(json.dumps({'tag': tag}).encode(), run_prefix + tag + '/' + MARKER)],
    )

    entries = list(api.list_bucket_tree(bucket_id, prefix=run_prefix, recursive=True))
    completed = set()

    for entry in entries:
        path = entry.path
        if not path.startswith(run_prefix):
            continue

        relative = path[len(run_prefix):].split('/')
        if len(relative) == 2 and relative[1] == MARKER and re.fullmatch(r'step_\d+', relative[0]):
            completed.add(relative[0])

    obsolete = set(sorted(completed, key=lambda s: int(s[5:]))[:-keep_last_k])
    deletes = []

    for entry in entries:
        if not entry.path.startswith(run_prefix) or getattr(entry, 'type', 'file') != 'file':
            continue

        relative = entry.path[len(run_prefix):].split('/')
        if len(relative) >= 2 and relative[0] in obsolete:
            deletes.append(entry.path)

    if deletes:
        api.batch_bucket_files(bucket_id, delete=deletes)
