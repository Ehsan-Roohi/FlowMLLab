#!/usr/bin/env python3
"""CPU-only opt-in port of upstream emit/surf velocity averaging to emit/face.

Applies only to SPARTA 95b9abaa8bd548991cc3c3f1c58b34722f7ade74.
No change to the default algorithm, density, temperature, or particle weight.
"""
import hashlib
from pathlib import Path
import sys

def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source anchor missing or ambiguous')
    return text.replace(old, new)

def patch(root):
    root=Path(root)
    hashes={'fix_emit_face.cpp':'1e07e0b29b1eb5d708b7545854f84f42901b6fe9467093c1f953035a42f65251',
            'fix_emit_face.h':'7f3fbd42e0239c74f8695a890e094aa16af38686a573a6ed4310643933a968b3'}
    texts={}
    for name,digest in hashes.items():
        data=(root/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:
            raise ValueError('Unexpected source: '+name)
        texts[name]=data.decode()
    h=texts['fix_emit_face.h']
    h=replace(h,'    double *ntargetsp;', '    double vcom_window[3];    // CPU PTBOTH velocity average, reset on task rebuild\n    double *ntargetsp;')
    h=replace(h,'  int faces[6];','  int subsonic_window;\n  int faces[6];')
    c=texts['fix_emit_face.cpp']
    c=replace(c,'  subsonic_warning = 0;','  subsonic_warning = 0;\n  subsonic_window = 0;')
    c=replace(c,'  // task list and subsonic data structs', '''  if (subsonic_window > 0 &&
      (subsonic_style != PTBOTH || strcmp(style,"emit/face") != 0))
    error->all(FLERR,"Face window requires CPU emit/face with pressure and temperature");

  // task list and subsonic data structs''')
    anchor='    tasks[ntask].vstream[2] = particle->mixture[imix]->vstream[2];'
    c=replace(c,anchor,anchor+'''
    for (int j = 0; j < 3; j++)
      tasks[ntask].vcom_window[j] = tasks[ntask].vstream[j];''')
    anchor='    } else vstream[0] = vstream[1] = vstream[2] = 0.0;'
    c=replace(c,anchor,anchor+'''

    // Same exponential velocity averaging as upstream emit/surf.
    // Leave window=0 on the original arithmetic path exactly.
    if (subsonic_window > 0) {
      const double a = 1.0 / (subsonic_window + 1.0);
      for (int j = 0; j < 3; j++) {
        tasks[i].vcom_window[j] = a*vstream[j] +
          (1.0-a)*tasks[i].vcom_window[j];
        vstream[j] = tasks[i].vcom_window[j];
      }
    }''')
    anchor='''    return 3;
  }

  if (strcmp(arg[0],"twopass") == 0)'''
    c=replace(c,anchor,'''    if (narg > 3 && strcmp(arg[3],"window") == 0) {
      if (narg < 5) error->all(FLERR,"Missing face window value");
      subsonic_window = input->inumeric(FLERR,arg[4]);
      if (subsonic_window < 0) error->all(FLERR,"Negative face window");
      return 5;
    }
    return 3;
  }

  if (strcmp(arg[0],"twopass") == 0)''')
    (root/'fix_emit_face.h').write_text(h)
    (root/'fix_emit_face.cpp').write_text(c)
    print('CPU_FACE_WINDOW_PATCH_APPLIED')

if __name__=='__main__': patch(sys.argv[1])
