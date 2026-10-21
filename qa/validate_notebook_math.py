"""Audit portable notebook math without re-running or changing scientific code."""
from pathlib import Path
import copy,json,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from qa.notebook_math import normalize_markdown,normalize_notebook
import nbformat

def tokens(text):
    for delimiter in [r'\[',r'\]',r'\(',r'\)','$']:text=text.replace(delimiter,'')
    return re.sub(r'\s+','',text)

def main():
    for original in [r'\[D=\begin{bmatrix}1\\-1\end{bmatrix}\]',r'Inline \(p\prime=p_{new}-p_{old}\).',r'$$x+y$$']:
        fixed=normalize_markdown(original)
        assert normalize_markdown(fixed)==fixed
        assert tokens(original)==tokens(fixed)
    literal='```python\npattern = r"\\[x\\]"\n```\n\n`\\(literal\\)`'
    assert normalize_markdown(literal)==literal
    checked=changed=changed_cells=0
    for path in sorted((ROOT/'notebooks').glob('week*/*.ipynb')):
        relative=path.relative_to(ROOT).as_posix()
        current=json.loads(path.read_text(encoding='utf-8'));nbformat.validate(nbformat.from_dict(current))
        old=json.loads(subprocess.check_output(['git','show','HEAD:'+relative],cwd=ROOT).decode('utf-8'))
        assert len(old['cells'])==len(current['cells']),relative
        local=0
        for a,b in zip(old['cells'],current['cells']):
            assert a['cell_type']==b['cell_type']
            if a['cell_type']=='code':assert a==b,(relative,'Code or outputs changed')
            else:
                left=''.join(a['source']);right=''.join(b['source'])
                assert tokens(left)==tokens(right),(relative,'Math content or prose changed')
                assert normalize_markdown(right)==right,(relative,'Not normalized')
                if left!=right:local+=1
        clone=copy.deepcopy(current);assert normalize_notebook(clone)==0
        changed+=bool(local);changed_cells+=local;checked+=1
    result={'notebooks_checked':checked,'notebooks_reformatted':changed,'markdown_cells_reformatted':changed_cells,'code_cells_and_outputs_preserved':True,'math_and_prose_tokens_preserved':True,'normalization_idempotent':True,'scope':'Only Markdown delimiter/layout repair; no solver or model changes and no new scientific runs.'}
    target=ROOT/'results/notebook_math_validation.json';target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
