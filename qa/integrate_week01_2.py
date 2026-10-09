"""Add only navigation entries for the independent module; never edit Week 1."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
NOTE='notebooks/week01_2/W1_2_Cavity_Pressure_Velocity.ipynb'
COLAB='https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/'+NOTE


def insert_after(relative,marker,line):
    path=ROOT/relative
    text=path.read_text(encoding='utf-8')
    if line in text:return
    lines=text.splitlines()
    index=next(i for i,s in enumerate(lines) if marker in s)
    lines.insert(index+1,line)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    insert_after('README.md','| [1.1](weeks/',
        f'| [1.2](weeks/week01_2/README.md) | SIMPLE, PISO and PIMPLE pressure-velocity coupling; Re=100 cavity | [Week 1.2 lab]({NOTE}) | [Lecture 1.2](lectures/week01_2_pressure_velocity.pdf) |')
    insert_after('COURSE_MAP.md','| [1.1]',
        f'| [1.2]({NOTE}) | Pressure-velocity coupling and conservative face fluxes | Shared Python FV SIMPLE/PISO/PIMPLE; independent streamfunction-vorticity comparison at Re=100 | Ghia centerlines, matched-grid steady limits, momentum/continuity defects, pressure gauge, CPU cost and honest transient limits |')
    insert_after('notebooks/README.md','| Week 1.1 |',
        f'| Week 1.2 | Cavity pressure-velocity coupling: SIMPLE, PISO and PIMPLE (Re=100) | [Open in Colab]({COLAB}) |')
    insert_after('lectures/README.md','| 1.1 |',
        f'| 1.2 | [Pressure-velocity coupling](week01_2_pressure_velocity.pdf) (18) | Shared MAC FV operators, three algorithm flowcharts, comparative literature, Re=100 steady benchmark and one-step coupling study, higher-Re teaching plan | `{NOTE}` | [Algorithm/source guide](../notebooks/week01_2/README.md), [teaching supplement](../notebooks/week01_2/COMPARISON_AND_TEACHING.md) |')
    insert_after('START_HERE.md','| AI-assisted research-software ready |',
        f'| Pressure-velocity coupling ready | You have completed the original cavity lab and want SIMPLE/PISO/PIMPLE at Re=100 | `{NOTE}` | [Open in Colab]({COLAB}) |')
    directory=ROOT/'weeks/week01_2';directory.mkdir(parents=True,exist_ok=True)
    (directory/'README.md').write_text(f'''# Week 1.2 - Pressure-velocity coupling

An independent Re=100 extension after the original Week 1. The original
streamfunction-vorticity notebook remains unchanged.

![Executed Re=100 velocity and streamlines](../../results/week01_2_pressure_velocity/figures/speed.png)

[Open in Colab]({COLAB}) | [Lecture](../../lectures/week01_2_pressure_velocity.pdf) | [Measured report](../../results/week01_2_pressure_velocity/README.md) | [Detailed guide](../../notebooks/week01_2/README.md)

## Class sequence

1. Inspect the four-formulation centerline and field comparisons.
2. Derive the pressure operator from MAC face continuity.
3. Trace SIMPLE, PISO and PIMPLE on the same momentum equations.
4. Check conservation, pressure gauge and algorithm limits.
5. Recompute one method and explain its measured cost and benchmark error.
6. Use the three flowcharts to distinguish frozen inner corrections from
   outer coefficient updates. Inspect the measured one-step coupling study.
7. Discuss the comparative papers and design a matched-error transient test
   before expanding the Reynolds-number qualification.

[Expanded comparison, flowcharts and teaching plan](../../notebooks/week01_2/COMPARISON_AND_TEACHING.md)

## Student output

Velocity and pressure fields, Ghia centerlines, momentum and continuity
residuals, grid/refinement evidence and one scientifically bounded method choice.
Distinguish steady iteration from physical time. Only Re=100 steady behavior
is qualified; the next experiment is temporal refinement for a changing lid.

[Source](../../flowmllab/pressure_velocity.py) | [Tests](../../tests/test_pressure_velocity.py) | [Retained configurations and field hashes](../../results/week01_2_pressure_velocity/manifest.json)

[Return to course map](../../COURSE_MAP.md)
''',encoding='utf-8')
    path=ROOT/'weeks/README.md'
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(subprocess.check_output(['git','show','HEAD:weeks/README.md'],cwd=ROOT))
    text=path.read_text(encoding='utf-8')
    if 'week01_2/' not in text:
        lines=text.splitlines();i=next(i for i,s in enumerate(lines) if 'week01_1/' in s)
        original=lines[i]
        # Match the existing index table's column count, leaving all old rows intact.
        columns=len(original.split('|'))-2
        row=['[Week 1.2](week01_2/README.md)','Pressure-velocity coupling: SIMPLE, PISO and PIMPLE',
             '[Re=100 cavity lab](../'+NOTE+')','[Lecture](../lectures/week01_2_pressure_velocity.pdf)']
        if columns>4:row+=['Matched FV fields, Ghia profiles and conservation checks']*(columns-4)
        lines.insert(i+1,'| '+' | '.join(row[:columns])+' |')
        path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
