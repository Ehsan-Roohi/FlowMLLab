"""Portable math delimiters for Jupyter, Colab and GitHub Markdown cells.
Fenced/inline code is preserved. Only prose math delimiters are normalized.
"""
from pathlib import Path
import json,re


def normalize_markdown(source):
    protected=[]
    def protect(match):
        protected.append(match[0]);return f'\x00CODE{len(protected)-1}\x00'
    # Preserve fenced examples and inline literals (including regexes).
    text=re.sub(r'(?m)^[ \t]*(`{3,}|~{3,})[^\n]*\n[\s\S]*?^[ \t]*\1[ \t]*$|(`+)([^`\n]+)\2',protect,source)
    text=re.sub(r'(?<!\\)\\\[([\s\S]*?)(?<!\\)\\\]',lambda m:'\n\n$$\n'+m[1].strip()+'\n$$\n\n',text)
    text=re.sub(r'(?<!\\)\\\(([\s\S]*?)(?<!\\)\\\)',lambda m:'$'+re.sub(r'\s*\n\s*',' ',m[1].strip())+'$',text)
    # Display delimiters must occupy their own lines in notebook Markdown.
    text=re.sub(r'(?<!\\)\$\$([\s\S]*?)(?<!\\)\$\$',lambda m:'\n\n$$\n'+m[1].strip()+'\n$$\n\n',text)
    text=re.sub(r'\n{3,}','\n\n',text)
    for i,value in enumerate(protected):text=text.replace(f'\x00CODE{i}\x00',value)
    return text.strip() if text!=source else source


def normalize_notebook(nb):
    changed=0
    for cell in nb['cells']:
        if cell['cell_type']!='markdown':continue
        original=cell['source'];source=original if isinstance(original,str) else ''.join(original)
        corrected=normalize_markdown(source)
        if corrected!=source:
            cell['source']=corrected if isinstance(original,str) else corrected.splitlines(keepends=True)
            changed+=1
    return changed


def normalize_file(path):
    path=Path(path);nb=json.loads(path.read_text(encoding='utf-8'));count=normalize_notebook(nb)
    if count:path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8',newline='\n')
    return count

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    for path in sorted((root/'notebooks').glob('week*/*.ipynb')):
        count=normalize_file(path)
        if count:print(path.relative_to(root).as_posix(),count)
