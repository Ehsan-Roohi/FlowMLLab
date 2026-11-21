"""Append verified startup teaching pages without losing existing lecture content."""
from pathlib import Path
import hashlib
import io
import json
import os
import sys
import fitz
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE_PAGES = 74


def build():
    from qa.build_cavity_startup_learning import build_appendix
    target = ROOT/'lectures/week01_2_pressure_velocity.pdf'
    manifest_path = ROOT/'qa/cavity_lecture_manifest.json'
    receipt_path = ROOT/'qa/cavity_startup_lecture.json'
    original = target.read_bytes()
    old_manifest = manifest_path.read_bytes()
    old_receipt = receipt_path.read_bytes() if receipt_path.exists() else None
    manifest = json.loads(old_manifest)
    if 'week01_2_pressure_velocity' not in manifest:
        raise ValueError('Missing original lecture provenance')
    base = PdfReader(io.BytesIO(original)); meta = dict(base.metadata or {})
    base_count = int(meta.get('/FlowMLLabStartupBasePages', BASE_PAGES))
    previous_appendix = int(meta.get('/FlowMLLabStartupAppendixPages',0))
    if base_count != BASE_PAGES or len(base.pages) != base_count+previous_appendix:
        raise ValueError('Unexpected lecture pages: inspect before replacing or appending any material')
    appendix_path = build_appendix(); appendix = PdfReader(appendix_path)
    writer = PdfWriter()
    writer.append(base, pages=(0,base_count), import_outline=True)
    writer.append(appendix)
    writer.add_outline_item('Physical-time cavity startup: movies, numerical sensitivity and exercises',base_count)
    metadata = {str(k):str(v) for k,v in meta.items()}
    original_sha = metadata.get('/FlowMLLabStartupOriginalPDFSHA256',hashlib.sha256(original).hexdigest())
    appendix_sha = hashlib.sha256(appendix_path.read_bytes()).hexdigest()
    metadata.update({'/FlowMLLabStartupBasePages':str(base_count),
                     '/FlowMLLabStartupAppendixPages':str(len(appendix.pages)),
                     '/FlowMLLabStartupOriginalPDFSHA256':original_sha,
                     '/FlowMLLabStartupAppendixSHA256':appendix_sha})
    writer.add_metadata(metadata)
    buffer=io.BytesIO(); writer.write(buffer); rebuilt=buffer.getvalue()
    check=PdfReader(io.BytesIO(rebuilt))
    for old,new in zip(base.pages[:base_count],check.pages[:base_count]):
        if old.get_contents().get_data()!=new.get_contents().get_data() or old.mediabox!=new.mediabox:
            raise ValueError('Original page stream or size changed')
    page_hashes=[]
    with fitz.open(stream=original,filetype='pdf') as before, fitz.open(stream=rebuilt,filetype='pdf') as after:
        for i in range(base_count):
            a=before[i].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
            b=after[i].get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False)
            if (a.width,a.height,a.samples)!=(b.width,b.height,b.samples):
                raise ValueError(f'Original rendered page {i+1} changed')
            page_hashes.append(hashlib.sha256(a.samples).hexdigest())
    receipt={'base_pages_preserved':base_count,'appendix_pages':len(appendix.pages),
             'total_pages':len(check.pages),'sha256':hashlib.sha256(rebuilt).hexdigest(),
             'appendix_sha256':appendix_sha,'original_pdf_sha256':original_sha,
             'input_pdf_sha256':hashlib.sha256(original).hexdigest(),
             'original_page_streams_and_sizes_preserved':True,
             'original_pages_rendered_identically':base_count,'render_dpi':108,
             'preserved_render_sha256':page_hashes}
    manifest['week01_2_pressure_velocity'].update(total_pages=len(check.pages),sha256=receipt['sha256'],startup_appendix_pages=len(appendix.pages))
    updates={target:rebuilt,manifest_path:(json.dumps(manifest,indent=2)+'\n').encode(),receipt_path:(json.dumps(receipt,indent=2)+'\n').encode()}
    previous={target:original,manifest_path:old_manifest,receipt_path:old_receipt}
    try:
        for path,payload in updates.items():
            temp=path.with_name(path.name+'.startup-tmp'); temp.write_bytes(payload); os.replace(temp,path)
    except Exception:
        for path,payload in previous.items():
            if payload is None:
                if path.exists(): path.unlink()
            else: path.write_bytes(payload)
        raise
    return receipt


if __name__=='__main__':
    result=build()
    print(json.dumps({k:v for k,v in result.items() if k!='preserved_render_sha256'},indent=2))
