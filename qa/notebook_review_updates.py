"""Apply the reviewed self-study corrections after a notebook's original builder.

The JSON is a declarative, auditable overlay, not an execution record. Exact
source matching fails closed if a generator changes. Historical result files
are never read, regenerated or modified. Changed code cells lose stale output.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
PATCH_FILE = Path(__file__).with_name('notebook_review_patches.json')


def review_notebook_file(path):
    path = Path(path).resolve()
    key = path.relative_to(ROOT).as_posix()
    patches = json.loads(PATCH_FILE.read_text(encoding='utf-8')).get(key, [])
    if not patches:
        return
    nb = json.loads(path.read_text(encoding='utf-8'))
    def source(cell):
        return ''.join(cell['source']).strip()
    for patch in patches:
        replacement = patch['replacement'].strip()
        if patch['kind'] == 'replace':
            matches = [c for c in nb['cells'] if source(c) == patch['match'].strip()]
            if not matches and any(source(c) == replacement for c in nb['cells']):
                continue
            if len(matches) != 1:
                raise ValueError(f'{key}: expected one source match, found {len(matches)}: {patch["match"][:80]}')
            cell = matches[0]
            cell['source'] = replacement.splitlines(keepends=True)
            cell.setdefault('metadata', {})['review_revision'] = 'self-study audit correction; not an execution'
            if cell['cell_type'] == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
        else:
            ident = 'review-' + patch['id']
            if any(c.get('metadata', {}).get('review_id') == ident for c in nb['cells']):
                continue
            before = patch.get('before')
            if before:
                positions = [i for i,c in enumerate(nb['cells']) if before in source(c)]
                if len(positions) != 1:
                    raise ValueError(f'{key}: ambiguous insertion anchor {before!r}')
                position = positions[0]
            else:
                position = 1
            cell = {'cell_type': patch['cell_type'], 'id': hashlib.sha256(ident.encode()).hexdigest()[:8],
                    'metadata': {'review_id': ident}, 'source': replacement.splitlines(keepends=True)}
            if cell['cell_type'] == 'code':
                cell.update(outputs=[], execution_count=None)
            nb['cells'].insert(position, cell)
    for index, cell in enumerate(nb['cells']):
        if not cell.get('id'):
            cell['id'] = hashlib.sha256(f'{key}:{index}:{source(cell)}'.encode()).hexdigest()[:12]
    nb.setdefault('metadata', {})['review_note'] = 'Source reviewed; changed/new code has no retained output. Original results remain historical author evidence.'
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    for relative in json.loads(PATCH_FILE.read_text(encoding='utf-8')):
        review_notebook_file(ROOT / relative)
