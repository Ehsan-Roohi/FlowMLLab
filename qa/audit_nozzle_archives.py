"""Locate exact published nozzle snapshots in ZIPs; never execute archived code."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def audit(path, expected):
    with Path(path).open('rb') as stream:
        archive_hash = sha(stream)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate entry names need manual disambiguation')
        matches, newline_matches, mismatches = [], [], []
        for name in names:
            base = PurePosixPath(name).name
            if base in expected:
                payload = archive.read(name)
                digest = hashlib.sha256(payload).hexdigest()
                canonical = hashlib.sha256(payload.replace(b'\r\n', b'\n')).hexdigest()
                row = {'entry': name, 'sha256': digest, 'lf_normalized_sha256': canonical}
                if digest == expected[base]:
                    matches.append(row)
                elif canonical == expected[base]:
                    newline_matches.append(row)
                else:
                    mismatches.append(row)
        candidates = [name for name in names if re.search(
            r'\.(?:f|for|f77|f90|f95|f03|f08|inp)$|(?:^|/)DS2V(?:D)?\.(?:dat|txt)$', name, re.I)]
        return {'archive': Path(path).name, 'archive_sha256': archive_hash,
                'entry_count': len(names), 'exact_published_snapshot_copies': matches,
                'published_snapshot_copies_after_CRLF_to_LF_only': newline_matches,
                'same_basename_different_bytes': mismatches,
                'unique_exact_snapshot_names': sorted({PurePosixPath(v['entry']).name for v in matches}),
                'unique_LF_equivalent_snapshot_names': sorted({PurePosixPath(v['entry']).name for v in newline_matches}),
                'candidate_solver_or_input_entries': candidates,
                'scope': 'Named solver/input search only. No nested archive extraction, unknown-extension moment decoding or solver reconstruction; absence is not proof that raw data do not exist elsewhere.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', action='append', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or any(a.output.resolve().is_relative_to(ROOT / f)
                                for f in ('data', 'results', 'notebooks', 'lectures')):
        p.error('Use a new scratch output file')
    source = json.loads((ROOT / 'results/mahdavi_deeponet/provenance.json').read_text())
    report = {'source_commit': source['source_commit'], 'archives': []}
    for path in a.archive:
        row = audit(path, source['source_files_sha256'])
        report['archives'].append(row)
        print(f"{path.name}: {len(row['unique_exact_snapshot_names'])} exact published snapshots; "
              f"{len(row['unique_LF_equivalent_snapshot_names'])} LF-equivalent snapshots; "
              f"{len(row['candidate_solver_or_input_entries'])} named source/input candidates", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
