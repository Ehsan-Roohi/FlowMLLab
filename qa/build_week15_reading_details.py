"""Lossless detail crops of retained figures; never alter the original evidence.

This is a teaching display transformation, not a recomputation of CFD. Bounds
are pixels in the 2800 x 4725 originals. Each crop includes all seven flow rows,
with the repeated geometry strip removed. The original remains linked and shown.
"""
from pathlib import Path
import hashlib,json
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'lectures/source/week15_reading_details'

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for path in sorted((ROOT/'results/week15_postaudit').glob('*_g*_Re*.png')):
        if not path.name.startswith(('core_','extended_')):continue
        with Image.open(path) as im:
            if im.size!=(2800,4725):raise ValueError(f'Unexpected geometry: {path}')
            # Three panels: first 3 rows, last 4 rows, and the first CFD wake.
            # No resizing, color edits, or regenerated streamlines.
            for label,bounds in [('rows_1_3',(420,1605,2800,2910)),('rows_4_7',(420,2910,2800,4665)),('cfd_wake',(440,1605,2180,2050))]:
                output=OUT/f'{path.stem}_{label}.png';im.crop(bounds).save(output)
                records.append({'source':path.relative_to(ROOT).as_posix(),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'output':output.relative_to(ROOT).as_posix(),'bounds_pixels':list(bounds),'transformation':'lossless pixel crop; original scientific colors and aspect ratio preserved'})
    (OUT/'provenance.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':build()
