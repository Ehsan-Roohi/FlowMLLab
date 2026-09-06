"""Read the author ZIP without executing its scripts or modifying course data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KB_SI = 1.380649e-23
SOURCE_COLUMNS = ('MA', 'TOV', 'P')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_ordered_point(archive, name):
    with archive.open(name) as raw, io.TextIOWrapper(raw, encoding='utf-8') as stream:
        title, variables, zone = (stream.readline() for _ in range(3))
        names = re.findall(r'"([^"]+)"', variables)
        if not variables.startswith('VARIABLES') or 'DATAPACKING=POINT' not in zone:
            raise ValueError(f'Unsupported header: {name}')
        dims = [re.search(rf'\b{axis}\s*=\s*(\d+)', zone) for axis in ('I', 'J')]
        if not all(dims) or len(names) != len(set(names)):
            raise ValueError(f'Ambiguous grid or variables: {name}')
        values = np.loadtxt(stream, ndmin=2)
        shape = [int(m.group(1)) for m in dims]
        if values.shape != (shape[0] * shape[1], len(names)):
            raise ValueError(f'Data/header shape mismatch: {name}')
    return names, values, shape


def compare_import(names, values, rows, targets):
    columns = [names.index(name) for name in SOURCE_COLUMNS]
    actual = values[rows][:, columns].astype(np.float32)
    if not np.array_equal(actual, targets):
        raise ValueError('Course targets differ from the archived MA/TOV/P columns')
    return len(rows)


def audit(archive_path):
    folder = ROOT / 'data/hypersonic_cylinder'
    manifest = json.loads((folder / 'manifest.json').read_text())
    archive_hash = digest(archive_path)
    if archive_hash != manifest['source_archive_sha256']:
        raise ValueError('Archive does not match the recorded source')
    npz = folder / manifest['artifact']
    if digest(npz) != manifest['artifact_sha256']:
        raise ValueError('Retained derivative hash mismatch')
    records = []
    with np.load(npz, allow_pickle=False) as data, zipfile.ZipFile(archive_path) as archive:
        entries = archive.namelist()
        if len(entries) != len(set(entries)):
            raise ValueError('Duplicate archive entry names')
        converters = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in
                      ('Allconvert.py', 'ConvertTriangulateGridToStructuredGrid.py',
                       'Convert_TriangulateGridToStructuredGrid.py')}
        for case in manifest['cases']:
            name = case['source_entry']
            names, values, shape = read_ordered_point(archive, name)
            if shape != case['source_grid']:
                raise ValueError('Source grid differs from the provenance manifest')
            mask = data['case_id'] == case['case_id']
            rows = data['source_row'][mask]
            matched = compare_import(names, values, rows, data['targets'][mask])
            if matched != case['retained_points']:
                raise ValueError('Retained row count differs from the manifest')
            selected = values[rows]
            nd, ttr, tov, pressure = [selected[:, names.index(n)] for n in ('ND', 'TTR', 'TOV', 'P')]
            if np.any(nd <= 0) or np.any(ttr <= 0) or np.any(pressure <= 0):
                raise ValueError('Nonpositive density, temperature or pressure in audit rows')
            # Consistency evidence only: interpolation does not preserve products.
            eos_error = np.abs(pressure - nd * KB_SI * ttr) / pressure
            x = selected[:, names.index('X')]
            upstream = x == x.min()
            records.append({
                'mach_inf': case['mach_inf'], 'entry': name, 'grid': shape,
                'matched_retained_rows': matched,
                'tov_range': [float(tov.min()), float(tov.max())],
                'p_range': [float(pressure.min()), float(pressure.max())],
                'upstream_edge_tov_median': float(np.median(tov[upstream])),
                'upstream_edge_p_median': float(np.median(pressure[upstream])),
                'p_vs_ND_kB_TTR_relative_error_quantiles':
                    np.quantile(eos_error, [.5, .99, 1]).tolist(),
            })
            print(f"M={case['mach_inf']:g}: {matched} targets match source MA/TOV/P", flush=True)
        result = {
            'source_archive': Path(archive_path).name, 'source_sha256': archive_hash,
            'derivative_sha256': manifest['artifact_sha256'],
            'source_columns_in_target_order': list(SOURCE_COLUMNS),
            'matched_retained_rows': sum(r['matched_retained_rows'] for r in records),
            'freestream_division_in_course_import': False,
            'converter_hashes_for_manual_review': converters,
            'fortran_entries': [n for n in entries if re.search(r'\.(f|for|f90)$', n, re.I)],
            'named_DS2V_input_entries': [n for n in entries if re.search(r'(^|/)DS2V(?:D)?\.(dat|inp)$', n, re.I)],
            'interpretation': 'Source MA/TOV/P, not verified freestream ratios. EOS check is SI-consistency evidence, not recovery of solver inputs or freestream reference constants.',
            'cases': records,
        }
    if digest(npz) != manifest['artifact_sha256']:
        raise RuntimeError('Retained derivative changed during audit')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or any(args.output.resolve().is_relative_to(ROOT / f)
                                   for f in ('data', 'results', 'notebooks', 'lectures')):
        parser.error('Use a new scratch output file, outside retained evidence')
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
