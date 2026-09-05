"""Plot verified, asynchronous Jager cylinder fields from the Unity export ZIP.

Only the first (cell-centre) Tecplot zone is used. No simulation, smoothing,
symmetry reflection, extrapolation or claimed pointwise model-error map.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle

CENTER = np.array([0.1524, 0.0])
RADIUS = 0.1524
N_INF = 4.247e20

def cell_zone(raw):
    stream = io.BytesIO(raw)
    columns = re.findall(r'"([^"]+)"', stream.readline().decode())
    zone = stream.readline().decode()
    match = re.search(r'\bI\s*=\s*(\d+)', zone)
    if not match: raise ValueError('Missing first-zone point count')
    count = int(match[1])
    values = np.loadtxt(stream, max_rows=count)
    if values.shape != (count, len(columns)) or not np.isfinite(values).all():
        raise ValueError('Invalid cell-centre zone')
    return columns, values

def exterior_triangulation(xy):
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    vertices = xy[tri.triangles]-CENTER
    mask = np.linalg.norm(vertices.mean(axis=1), axis=1) <= RADIUS
    # Reject every triangle with an edge crossing the solid disk.
    for i in range(3):
        a, b = vertices[:, i], vertices[:, (i+1)%3]
        delta = b-a
        fraction = np.clip(-np.sum(a*delta, axis=1)/np.sum(delta*delta, axis=1), 0, 1)
        closest = a+fraction[:, None]*delta
        mask |= np.linalg.norm(closest, axis=1) <= RADIUS
    tri.set_mask(mask)
    return tri

def main(archive, output, overview_only=False):
    output.mkdir(parents=True, exist_ok=True)
    records = {}
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('CONTOUR_EXPORT_MANIFEST.json'))
        for item in manifest['files']:
            name = item['path'].replace('\\', '/')
            if hashlib.sha256(z.read(name)).hexdigest() != item['sha256']:
                raise ValueError('Manifest hash mismatch: '+name)
        for name in z.namelist():
            if not name.endswith('/DS2FF.DAT'): continue
            label = 'DeepONet' if '/neural/' in name else 'Exact'
            if label in records: raise ValueError('Ambiguous field selection')
            columns, values = cell_zone(z.read(name))
            meta = dict(line.split('=', 1) for line in z.read(name.rsplit('/',1)[0]+'/METADATA.env').decode().splitlines() if '=' in line)
            if int(meta['MODE']) != {'Exact':1, 'DeepONet':2}[label]:
                raise ValueError('Mode does not match run label')
            xy = values[:, :2]
            if np.any(np.linalg.norm(xy-CENTER, axis=1)<RADIUS): raise ValueError('Cell centre inside solid')
            if len(np.unique(xy,axis=0))!=len(xy): raise ValueError('Duplicate cell centres')
            records[label] = dict(columns=columns, values=values, meta=meta, source=name)
    if set(records) != {'Exact','DeepONet'}: raise ValueError('Both fields required')
    if not np.array_equal(records['Exact']['values'][:,:2],records['DeepONet']['values'][:,:2]):
        raise ValueError('Different meshes: independent interpolation required')
    tri = exterior_triangulation(records['Exact']['values'][:,:2])
    xx, yy = np.meshgrid(np.linspace(-.2,.65,851), np.linspace(0,.4,401))
    solid = (xx-CENTER[0])**2+yy**2 <= RADIUS**2
    specs = [
        ('temperature','TTR',1.,'Translational temperature (K)','inferno',np.linspace(0,7500,51),[0,1500,3000,4500,6000,7500],False),
        ('mach','MA',1.,'Mach number','viridis',np.linspace(0,11,45),[0,2,4,6,8,10],False),
        ('pressure','P',1.,'Pressure (Pa)','magma',np.linspace(0,190,39),[0,40,80,120,160,190],False),
        ('density','ND',N_INF,'Number density / upstream density','cividis',np.geomspace(.08,32,49),[.1,.3,1,3,10,30],True),
    ]
    plt.rcParams.update({'font.size':10,'axes.titlesize':11,'svg.fonttype':'none','savefig.facecolor':'white'})
    overview, overview_axes = plt.subplots(4,2,figsize=(12,14),layout='constrained')
    diagnostics = {'archive':archive.name,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'manifest_hashes_verified':True,'asynchronous_snapshots':True,'grid_shape':list(xx.shape),
        'rendering':'Linear interpolation within a solid-masked triangulation; no smoothing or extrapolation',
        'masked_triangles':int(tri.mask.sum()),'runs':{},'fields':{}}
    for row,(stem,column,scale,title,cmap,levels,ticks,log) in enumerate(specs):
        pair,axes = plt.subplots(1,2,figsize=(12,4),layout='constrained')
        norm = LogNorm(levels[0],levels[-1]) if log else Normalize(levels[0],levels[-1])
        diagnostics['fields'][stem] = {'levels':levels.tolist(),'source_column':column,'scale_divisor':scale,'logarithmic_color':log}
        for col,label in enumerate(['Exact','DeepONet']):
            record=records[label]
            native=record['values'][:,record['columns'].index(column)]/scale
            if np.any(native<=0): raise ValueError('Nonpositive physical quantity')
            grid=mtri.LinearTriInterpolator(tri,native)(xx,yy)
            grid=np.ma.masked_where(solid,grid)
            diagnostics['runs'][label]={'source':record['source'],'metadata':record['meta'],'cell_count':len(native)}
            diagnostics['fields'][stem][label]={'native_min':float(native.min()),'native_max':float(native.max()),
                'finite_plotted_points':int(grid.count()),'out_of_color_range_native':int(((native<levels[0])|(native>levels[-1])).sum())}
            if not np.all(np.ma.getmaskarray(grid)[solid]): raise AssertionError('Solid not fully masked')
            for ax in [axes[col],overview_axes[row,col]]:
                filled=ax.contourf(xx,yy,grid,levels=levels,norm=norm,cmap=cmap,extend='neither',antialiased=False)
                ax.add_patch(Circle(CENTER,RADIUS,facecolor='#e9edf1',edgecolor='#303c49',linewidth=1.2,zorder=5))
                ax.set(xlim=(-.2,.65),ylim=(0,.4),xlabel='x (m)',ylabel='y (m)',aspect='equal')
                ax.set_title(f"{label} | NOUT {record['meta']['NOUT']} | tU/D = {float(record['meta']['TUD']):.3f}")
                ax.tick_params(direction='out')
        pair.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes,shrink=.85,pad=.02,ticks=ticks,label=title)
        overview.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=overview_axes[row,:],shrink=.85,pad=.015,ticks=ticks,label=title)
        pair.suptitle('Jäger Ar–Ar cylinder: '+title,fontsize=14)
        pair.supxlabel('Different output times and sampling windows; qualitative comparison only',fontsize=9)
        if not overview_only:
            pair.savefig(output/f'{stem}_exact_deeponet.png',dpi=320)
            pair.savefig(output/f'{stem}_exact_deeponet.svg')
        plt.close(pair)
        print('Saved',stem,flush=True)
    overview.suptitle('Jäger Ar–Ar cylinder | Exact-derived vs DeepONet-derived collision tables\nShared color scales; different output times; upper-half domain only',fontsize=14)
    overview.savefig(output/'contours_overview.png',dpi=260)
    plt.close(overview)
    (output/'contour_manifest.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
    print('Output:',output.resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,default=Path('tmp/abinitio_final/contours'))
    p.add_argument('--overview-only',action='store_true',help='Regenerate overview and diagnostics without individual figures')
    args=p.parse_args()
    main(args.archive,args.output,args.overview_only)
