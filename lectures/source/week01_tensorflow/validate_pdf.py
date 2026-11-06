from pathlib import Path
import json,re,hashlib
import pymupdf as fitz
import numpy as np
from PIL import Image
r=Path(__file__).resolve().parents[3]
s=r/"lectures/source/week01_tensorflow";p=r/"lectures/week01_02_tensorflow_keras.pdf"
d=fitz.open(p);fonts=set();colors=set();links=[];headings=[]
for i,page in enumerate(d):
 for b in page.get_text("dict")["blocks"]:
  for l in b.get("lines",[]):
   for span in l["spans"]:
    if span["text"].strip():fonts.add(span["font"]);colors.add(span["color"])
 for link in page.get_links():
  if link["kind"]==fitz.LINK_GOTO:assert 0<=link["page"]<len(d)
  if link["kind"]==fitz.LINK_URI:assert link["uri"].startswith("https://")
  links.append(dict(source_page=i+1,kind=link["kind"],uri=link.get("uri"),target_page=link.get("page")))
 headings.extend(int(n) for n in re.findall(r"(?m)^(\d{1,2})\. [A-Z]",page.get_text()))
assert all("TimesNewRoman" in f for f in fonts),fonts
assert colors=={0},colors
assert set(range(1,21))<=set(headings),headings
assert sum(v["kind"]==fitz.LINK_GOTO for v in links)==5,links
assert sum(v["kind"]==fitz.LINK_URI for v in links)>=16,links
images=[]
for fp in (s/"figures").glob("*.png"):
 with Image.open(fp) as im:
  a=np.asarray(im.convert("RGB"));assert np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,1],a[:,:,2]),fp
  images.append(dict(name=fp.name,monochrome=True,width=im.width,height=im.height))
report=dict(pages=len(d),fonts=sorted(fonts),text_colors=sorted(colors),clickable_links=len(links),links=links,images=images,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),all_20_teaching_sections_present=True)
(s/"format_validation.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k not in ("images","links")},indent=2))
