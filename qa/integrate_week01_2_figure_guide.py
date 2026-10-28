"""Keep lecture and notebook figure identifiers and captions consistent."""
from pathlib import Path
import json
import re
import sys
import nbformat

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from qa.refresh_week01_2_figures import CAPTIONS,caption_tex
SOURCE=ROOT/'lectures/source'
ASSETS=SOURCE/'week01_2_assets'
URL='https://raw.githubusercontent.com/Ehsan-Roohi/FlowMLLab/main/lectures/source/week01_2_assets/'

def expand(path):
    source=path.read_text(encoding='utf-8')
    return re.sub(r'\\input\{([^}]+)\}',lambda m:expand(path.parent/m.group(1)),source)

def main():
    names=re.findall(r'\\lessonfigure\{([^}]+)\}',expand(SOURCE/'week01_2_pressure_velocity.tex'))
    assert len(names)==20 and len(set(names))==20,names
    index={name:{'number':str(i),'caption':caption_tex(name),'image':name+'.png','lecture':True}
           for i,name in enumerate(names,1)}
    extra=json.loads((ASSETS/'supplementary_captions.json').read_text(encoding='utf-8'))
    for i,(name,caption) in enumerate(extra.items(),1):
        index[name]={'number':f'S{i}','caption':caption,'image':name+'.png','lecture':False}
    (ASSETS/'figure_index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8',newline='\n')
    def picture(name):
        entry=index[name]
        return f'![{name.replace("_"," ")}]({URL}{entry["image"]})\n\n**Figure {entry["number"]}.** {entry["caption"]}'
    def md(id,source):return nbformat.v4.new_markdown_cell(source,id=id)
    path=ROOT/'notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
    nb=nbformat.read(path,4)
    if any(c.id.startswith('coupling-page-') for c in nb.cells):
        from qa.sync_week01_2_teaching import sync_notebook
        nb=sync_notebook(nb)
        path.write_text(nbformat.writes(nb),encoding='utf-8',newline='\n')
        print(json.dumps({'lecture_figures':len(names),'supplementary_figures':len(extra),'synchronized_prerequisites':True}))
        return
    nb.cells=[c for c in nb.cells if not c.id.startswith('visual-caption-')]
    byid={c.id:c for c in nb.cells}
    byid['visual-hat-geometry'].source=r'''### Stored velocities and interpolated face values

'''+picture('hat_stored_locations')+r'''

Use local coordinates measured in grid spacings $h$, with the origin at pressure center $P$.
The pressure centers are $P=(0,0)$ and $E=(h,0)$. The stored horizontal velocity $u_e$
is at $(h/2,0)$, on their shared **pressure-cell face**. Its own **momentum control volume**
is the blue square: $0\leq x\leq h$, $-h/2\leq y\leq h/2$.

Blue circles store horizontal velocity components. In particular, $u_{\rm N}$ is
a horizontal component one spacing north of $u_e$; it is not a vertical velocity.
Green triangles store vertical components at pressure-cell faces. Open squares mark pressure.

'''+picture('hat_interpolated_locations')+r'''

The red crosses are evaluation points at the **momentum-volume faces**. Here a hat
means spatial interpolation. For example, the north cross at $(h/2,h/2)$ is halfway
between $u_e$ and $u_{\rm N}$, so

$$
\widehat u_{\rm N}=\frac{u_e+u_{\rm N}}2,\qquad
\widehat u_{\rm E}=\frac{u_e+u_{\rm E}}2.
$$

The west and east crosses coincide with pressure centers, while north and south
crosses lie at pressure-cell corners. The geometric coincidence does not change
which physical component is being evaluated.

'''+picture('north_face_transport')+r'''

At the north face, fluid crosses vertically with the normal advecting speed
$c_{\rm N}=(v_{n,P}+v_{n,E})/2$. It carries the horizontal component
$\widehat u_{\rm N}=(u_e+u_{\rm N})/2$.
For $h=0.25$, $c_{\rm N}=0.02$ and $\widehat u_{\rm N}=0.06$,

$$
Q_{\rm N}=h c_{\rm N}=0.005,\qquad
F^u_{\rm N}=Q_{\rm N}\widehat u_{\rm N}=0.00030.
$$

$Q_{\rm N}$ is volume flux per unit depth; $F^u_{\rm N}$ is horizontal-momentum
flux per unit density and unit depth. All example values are nondimensional.
The runnable calculation below also checks the other three faces, including
the outward signs: east and north positive, west and south negative.
The net momentum transport is $0.0015$ and its volume-normalized value is $0.024$.

In this face-flux example, a star marks a predictor and a prime marks a correction.
The later PIMPLE equations use $q_{\rm lag}^{(\ell)}$ and $p_{\rm lag}^{(\ell)}$
for outer-iteration inputs; their hats are not overloaded with a second meaning.
'''
    # Caption existing illustrations in place, preserving their teaching text.
    image_pattern=r'!\[[^\]]*\]\('+re.escape(URL)+r'([^/)]+)\.png\)'
    for c in nb.cells:
        if c.cell_type!='markdown' or c.id=='visual-hat-geometry':continue
        def replacement(m):
            name=m.group(1)
            if name not in index:raise ValueError(name)
            return picture(name)
        # Remove captions from a previous invocation before adding canonical ones.
        c.source=re.sub(r'\n\n\*\*Figure (?:\d+|S\d+)\.\*\*[^\n]*','',c.source)
        c.source=re.sub(image_pattern,replacement,c.source)

    insertions={
        'pressure-lesson-3':[md('visual-caption-momentum','### The horizontal-momentum volume\n\n'+picture('momentum_volume'))],
        '51fefcd5':[md('visual-caption-baseline','### Retained Re=100 fields and histories\n\n'+'\n\n'.join(picture(k) for k in ['profiles_re100','field_speed_re100','field_omega_re100','field_p_re100','convergence_re100']))],
        '1b614a07':[md('visual-caption-higher','### Retained higher-Reynolds-number comparisons\n\n'+'\n\n'.join(picture(k) for k in ['profiles_re400','profiles_re1000']+[k for k in extra if not k.startswith('review_')]))],
        'ee106a2c':[md('visual-caption-refinement','### Selected spatial refinement\n\n'+picture('refinement_re400')+'\n\n'+picture('refinement_re1000'))],
        'student-review-refinement-evidence':[md('visual-caption-independent','### Independent follow-up calculations\n\n'+'\n\n'.join(picture(k) for k in extra if k.startswith('review_')))]}
    # Retain all numerical loading, benchmark checks, tables and opt-in solves.
    # Only the old image-display/plotting blocks are replaced by captioned images.
    c=byid['51fefcd5'];c.source=c.source.split('for name in ["centerlines"',1)[0].rstrip()
    c=byid['1b614a07'];c.source=c.source.split('# Fresh diagnostic plot',1)[0].rstrip()
    c=byid['ee106a2c']
    if '# Keep inline images compact;' in c.source:
        head,tail=c.source.split('# Keep inline images compact;',1)
        tail=tail.split('# Fresh independent FD run:',1)[1]
        c.source=head.rstrip()+'\n\n# Fresh independent FD run:'+tail
    c=byid['student-review-refinement-evidence'];c.source=c.source.split('for name in ("mesh_and_methods"',1)[0].rstrip()
    # Optional fresh experiments retain their solvers and controls; captions
    # identify their figures separately from the archived lecture evidence.
    c=byid['327cdaea']
    if 'from IPython.display import Markdown' not in c.source:
        c.source='from IPython.display import Markdown\n'+c.source
    if '**Experiment Figure E100-' not in c.source:
        c.source=c.source.replace('for name,fig in figures(active,cells=CELLS).items():\n        display(fig); plt.close(fig)',
            'for figure_i, (name,fig) in enumerate(figures(active,cells=CELLS).items(), 1):\n'
            '        if name == "runtime":\n            plt.close(fig)\n            continue\n'
            '        display(fig)\n'
            '        display(Markdown(f"**Experiment Figure E100-{figure_i}.** Fresh Re=100 {name.replace(\'_\', \' \')}; h=1/{CELLS}. Recorded reference controls are identified in the table."))\n'
            '        plt.close(fig)')
    c=byid['1a08e6d5']
    if 'from IPython.display import Markdown' not in c.source:
        c.source='from IPython.display import Markdown\n'+c.source
    if '**Experiment Figure EH-' not in c.source:
        c.source=c.source.replace('for figure in comparison_figures(higher_runs).values():\n            display(figure); plt.close(figure)',
            'for figure_i, (name, figure) in enumerate(comparison_figures(higher_runs).items(), 1):\n'
            '            display(figure)\n'
            '            display(Markdown(f"**Experiment Figure EH-{figure_i}.** Fresh Re={HIGHER_RE} {name}; h=1/{HIGHER_CELLS}, convection={HIGHER_SCHEME}. Native stopping norms differ between FD and FV."))\n'
            '            plt.close(figure)')
    c=byid['ee106a2c']
    if 'from IPython.display import Markdown' not in c.source:
        c.source='from IPython.display import Markdown\n'+c.source
    if '**Experiment Figure EFD-1.' not in c.source:
        c.source=c.source.replace('display(fig);plt.close(fig)',
            'display(fig)\n    display(Markdown(f"**Experiment Figure EFD-1.** Fresh streamfunction-vorticity velocity profile and speed with streamlines, Re={STREAM_RE}, h=1/{STREAM_CELLS}. Black profile markers are Ghia reference samples."))\n    plt.close(fig)')
    # Changed display cells must not retain images from their earlier source.
    for id in ['51fefcd5','1b614a07','ee106a2c','student-review-refinement-evidence']:
        byid[id].outputs=[];byid[id].execution_count=None
    revised=[]
    for c in nb.cells:
        revised.append(c);revised.extend(insertions.get(c.id,[]))
    revised.append(md('visual-caption-resources',r'''## Guided reading and video lectures

1. **Momentum to pressure correction:** Greenshields and Weller, [CFD Direct, section 5.10](https://doc.cfd.direct/notes/cfd-general-principles/pressure-velocity-coupling). Follow the momentum diagonal, flux correction and substitution into continuity beside this notebook's scalar derivation.
2. **SIMPLE:** [CFD Direct, section 5.12](https://doc.cfd.direct/notes/cfd-general-principles/steady-state-solution) and Fluid Mechanics 101, [The SIMPLE Algorithm (YouTube)](https://www.youtube.com/watch?v=OOILoJ1zuiw). Identify the predictor, pressure correction, under-relaxation and next nonlinear iteration.
3. **PISO:** [CFD Direct, section 5.19](https://doc.cfd.direct/notes/cfd-general-principles/transient-solution) and Fluid Mechanics 101, [The PISO Algorithm (YouTube)](https://www.youtube.com/watch?v=ahdW5TKacok). Track which correction terms change while old-time data and momentum coefficients stay fixed. The video's non-orthogonal corrections address a separate mesh issue, absent on this uniform Cartesian grid.
4. **PIMPLE:** [CFD Direct, section 5.21](https://doc.cfd.direct/notes/cfd-general-principles/the-pimple-algorithm). Compare inner corrections with outer coefficient updates. Distinguish a time-accurate transient from a pseudo-transient steady calculation.
5. **Practical controls:** [OpenFOAM v13 User Guide, Solution and algorithm control](https://doc.cfd.direct/openfoam/user-guide-v13/fvsolution). Map `nCorrectors`, `nOuterCorrectors` and the pressure reference to the diagrams. Software defaults and collocated interpolation details differ from this Python MAC solver.

The [Fluid Mechanics 101 catalogue](https://www.fluidmechanics101.com/pages/lectures.html) lists the free YouTube lectures separately from paid slide collections. Paid slides are not required here. Figures 1-20 share identifiers with the lecture PDF; S-numbered figures are notebook supplements. The numerical examples above use the equations and conventions derived in this notebook.
'''))
    nb.cells=revised
    nbformat.validate(nb)
    path.write_text(nbformat.writes(nb),encoding='utf-8',newline='\n')
    print(json.dumps({'lecture_figures':len(names),'supplementary_figures':len(extra),'notebook_cells':len(nb.cells)}))

if __name__=='__main__':main()
