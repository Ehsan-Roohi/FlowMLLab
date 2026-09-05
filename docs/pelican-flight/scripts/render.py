"""Render unchanged animation SVGs at 960x540, 30 fps (CairoSVG, not screen capture).
Run node scripts/sample.cjs first. Requires Python lxml, cairosvg and ffmpeg.
"""
from pathlib import Path
import json,re,copy,subprocess
from lxml import etree
import cairosvg
ROOT=Path(__file__).resolve().parents[1]
for name in ['first','medium','light']:
 text=(ROOT/'originals'/f'{name}.html').read_text()
 svg=re.search(r'<svg[\s\S]*?</svg>',text).group()
 if 'xmlns=' not in svg.split('>')[0]:svg=svg.replace('<svg ','<svg xmlns="http://www.w3.org/2000/svg" ',1)
 root=etree.fromstring(svg.encode())
 style=etree.SubElement(root,'{http://www.w3.org/2000/svg}style')
 css=re.search(r'<style>([\s\S]*?)</style>',text).group(1)
 style.text='\n'.join(re.findall(r'\.(?:ink|line)\s*\{[^}]*\}',css))
 frames=json.load(open('/tmp/pelican-'+name+'.json'))
 dest=ROOT/'videos'/f'{name}.mp4'
 proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','image2pipe','-framerate','30','-i','-','-an','-c:v','libx264','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],stdin=subprocess.PIPE)
 for i,attrs in enumerate(frames):
  for el in root.iter():
   for key,value in attrs.get(el.get('id'),{}).items():el.set(key,value)
  png=cairosvg.svg2png(bytestring=etree.tostring(root),output_width=960,output_height=540)
  if i==0:(ROOT/'videos'/f'{name}.png').write_bytes(png)
  proc.stdin.write(png)
 proc.stdin.close()
 if proc.wait():raise RuntimeError(name)
 print(name,dest.stat().st_size,flush=True)
