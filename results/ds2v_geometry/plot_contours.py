import hashlib, io, json, os, re, sys, tarfile
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
def open_field(archive, case, label, snapshot):
    stem = f"{case}/samples/{snapshot}"
    field_path = f"{stem}/DS2FF.DAT"
    data = archive.extractfile(field_path).read()
    with io.BytesIO(data) as f:
        columns = re.findall(rb'"([^"]+)"', f.readline())
        header = f.readline()
        n = int(re.search(rb'\bI=\s*(\d+)', header)[1])
        found = int(re.search(rb'NOUT,MOUT,NM,.*?=\s*(\d+)', header)[1])
        if found != int(snapshot.split("-")[1]):
            raise ValueError("Snapshot number disagrees with Tecplot header")
        values = np.loadtxt(f, max_rows=n, ndmin=2)
    if values.shape != (n,len(columns)):
        raise ValueError(f"Unexpected native field shape {values.shape}")
    field = dict(zip((name.decode() for name in columns), values.T))
    keys = ("X", "Y", "MA", "ND", "TTR", "P")
    good = np.all([np.isfinite(field[k]) for k in keys], axis=0)
    good &= (field["ND"] > 0) & (field["Y"] >= 0)
    if good.sum() < 10000:
        raise ValueError("Insufficient valid DS2V sampling cells")
    xy = np.column_stack((field["X"][good],field["Y"][good]))/D
    quantities = {k:field[k][good] for k,*_ in FIELDS}
    # Density ratio for pure argon equals the number-density ratio.
    quantities["ND"] = quantities["ND"]/N_INF
    if any(np.any(v<0) for v in quantities.values()):
        raise ValueError("Unexpected negative native thermodynamic field")
    geom = json.load(archive.extractfile(f"{case}/geometry.json"))
    up = np.asarray(geom["vertices"])/D
    down = up[-2:0:-1]*[1,-1]
    polygon = np.vstack((up, down, up[:1]))
    window = json.load(archive.extractfile(f"{stem}/window.json"))
    return dict(label=label,xy=xy,fields=quantities,polygon=polygon,window=window,
                field_path=field_path,field_sha256=hashlib.sha256(data).hexdigest(),
                original_points=n,valid_points=int(good.sum()))


def interpolate(record, X, Y):
    xy = record["xy"]
    fields = record["fields"]
    # Reflect the native half-domain sampling cells across the solver symmetry plane.
    xy_full = np.vstack((xy,xy[xy[:,1]>0]*[1,-1]))
    index = np.r_[np.arange(len(xy)),np.flatnonzero(xy[:,1]>0)]
    points, ids = np.unique(xy_full,axis=0,return_index=True)
    tri = Delaunay(points, qhull_options='Qbb Qc Q12 QJ')
    print(record['label'], 'triangulated', flush=True)
    vertices = points[tri.simplices]
    middle = (vertices+np.roll(vertices,1,axis=1))/2
    polygon = PolygonPath(record["polygon"])
    # Prevent interpolation through the solid or across abnormally large gaps.
    bad = polygon.contains_points(vertices.mean(axis=1)).copy()
    bad |= polygon.contains_points(middle.reshape(-1,2)).reshape(-1,3).any(axis=1)
    bad |= np.max(np.linalg.norm(vertices-np.roll(vertices,1,axis=1),axis=2),axis=1)>.085
    sample = np.c_[X.ravel(),Y.ravel()]
    solid = polygon.contains_points(sample)
    simplex = tri.find_simplex(sample)
    keep = (simplex >= 0) & ~solid
    keep[keep] &= ~bad[simplex[keep]]
    keep[keep] &= np.isfinite(tri.transform[simplex[keep]]).all(axis=(1,2))
    transforms = tri.transform[simplex[keep]]
    weights = np.einsum("ijk,ik->ij", transforms[:,:2], sample[keep]-transforms[:,2])
    weights = np.column_stack((weights,1-weights.sum(axis=1)))
    corners = index[ids[tri.simplices[simplex[keep]]]]
    gridded = {}
    for k,arr in fields.items():
        v = np.ma.masked_all(sample.shape[0])
        v[keep] = np.einsum("ij,ij->i",weights,arr[corners])
        if v.count() < .80*np.count_nonzero(~solid):
            raise ValueError(f"Invalid plot coverage for {record['label']} {k}")
        gridded[k] = v.reshape(X.shape)
    print(record['label'], 'interpolated', flush=True)
    return gridded



root=Path(sys.argv[1])
out=root/"DS2V_Contours_Current"
out.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"]=str(out/".matplotlib")
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
from matplotlib.patches import Polygon
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import PowerNorm
from PIL import Image

D,U_INF,N_INF=.3048,2634.1,4.247e20
FIELDS=[
 ("MA","Mach number","mach",.5),
 ("ND",r"Density ratio $\rho/\rho_\infty$","density",5.),
 ("TTR","Translational temperature (K)","temperature",1000.),
 ("P","Pressure (Pa)","pressure",25.)]
sources={
 "long":root/"DS2V_Circle_Ellipse_Long_20261103",
 "holdout":root/"DS2V_DeepONet_Holdout_20261104",
 "orientation":root/"DS2V_Orientation_Pairs_20261105"}
hashes={}
for label,folder in sources.items():
 archive_path=folder/"results.tar.gz"
 actual=hashlib.sha256(archive_path.read_bytes()).hexdigest()
 assert actual==(folder/"SHA256SUMS").read_text().split()[0],archive_path
 hashes[label]=actual
print("All three archives SHA256 verified",flush=True)

x=np.linspace(-.64,2.10,460)
y=np.linspace(-1.20,1.20,410)
X,Y=np.meshgrid(x,y)
cache=out/".fields_cache.npz"
audit_path=out/"provenance.json"
ratios=(.600,.675,.750,.825,.875,.925)
if cache.exists() and audit_path.exists():
 audit=json.loads(audit_path.read_text())
 assert audit["source_archives_sha256"]==hashes
 records=audit["cases"]
 fields=[]
 with np.load(cache) as z:
  for i in range(len(records)):
   fields.append({k:np.ma.masked_invalid(z[f"{i}_{k}"]) for k,*_ in FIELDS})
 print("Loaded verified current rendering cache",flush=True)
else:
 records=[]
 fields=[]
 # Use the previously validated shock extraction, unchanged for all geometries.
 with tarfile.open(sources["holdout"]/"results.tar.gz") as tar:
  ns={"__name__":"archived_shock_extractor","__file__":"archived compare.py"}
  exec(compile(tar.extractfile("compare.py").read(),"archived compare.py","exec"),ns)
  get_shock=ns["shock"]
 specs=[
  ("long","fine_circle",1.0,"circle"),
  ("long","fine_ellipse",.75,"vertical"),
  ("holdout","ellipse_0825",.825,"vertical"),
  ("holdout","ellipse_0875",.875,"vertical"),
  ("holdout","ellipse_0925",.925,"vertical"),
  ("orientation","vert_0600",.6,"vertical"),
  ("orientation","vert_0675",.675,"vertical"),
  ("orientation","horiz_0600",.6,"horizontal"),
  ("orientation","horiz_0675",.675,"horizontal"),
  ("orientation","horiz_0750_retry1",.75,"horizontal"),
  ("orientation","horiz_0825",.825,"horizontal"),
  ("orientation","horiz_0875",.875,"horizontal"),
  ("orientation","horiz_0925",.925,"horizontal")]
 for source in sources:
  with tarfile.open(sources[source]/"results.tar.gz") as tar:
   names=tar.getnames()
   for src,case,ratio,orientation in specs:
    if src!=source:
     continue
    windows={}
    for name in names:
     if name.startswith(case+"/samples/out-") and name.endswith("/window.json"):
      w=json.load(tar.extractfile(name))
      assert w["target"] not in windows
      windows[w["target"]]=(name.split("/")[-2],w)
    assert set(windows)=={12,14,16,18,20},(case,windows)
    for p,q in ((16,18),(18,20)):
     old,new=windows[p][1],windows[q][1]
     assert abs(new["start_s"]-old["scheduled_reset_s"])<5e-10,(case,q)
     assert 1.8<(new["end_s"]-new["start_s"])*U_INF/D<2.2,(case,q)
    snapshot,w=windows[20]
    r=open_field(tar,case,case,snapshot)
    geometry=json.load(tar.extractfile(case+"/geometry.json"))
    a,b=geometry["a_m"],geometry["b_m"]
    assert np.isclose(min(a,b)/max(a,b),ratio)
    assert orientation=="circle" or (a<b if orientation=="vertical" else a>b)
    raw={"X":r["xy"][:,0]*D,"Y":r["xy"][:,1]*D,"ND":r["fields"]["ND"]*N_INF}
    shock={str(level):1000*get_shock(raw,.006,.004,.85/320,level)[0] for level in (2.2,2.5,2.8)}
    # A separate late window measures temporal drift at the same threshold.
    previous=windows[18][0]
    shock_previous=None
    if f"{case}/samples/{previous}/DS2FF.DAT" in names:
     native=tar.extractfile(f"{case}/samples/{previous}/DS2FF.DAT").read()
     with io.BytesIO(native) as stream:
      columns=[c.decode() for c in re.findall(rb'"([^"]+)"',stream.readline())]
      header=stream.readline()
      count=int(re.search(rb"\bI=\s*(\d+)",header)[1])
      cols=[columns.index(k) for k in ("X","Y","ND")]
      values=np.loadtxt(stream,max_rows=count,usecols=cols,ndmin=2)
     keep=(values[:,2]>0)&(values[:,1]>=0)&np.isfinite(values).all(axis=1)
     prevraw=dict(zip(("X","Y","ND"),values[keep].T))
     shock_previous=1000*get_shock(prevraw,.006,.004,.85/320,2.5)[0]
    assert all(np.isfinite(v) and v>0 for v in shock.values()) and (shock_previous is None or np.isfinite(shock_previous))
    r.update(source=source,case=case,ratio=ratio,orientation=orientation,a_m=a,b_m=b,
             time_range=[w[k]*U_INF/D for k in ("start_s","end_s")],
             previous_window=windows[18][1],shock_mm=shock,
             shock_previous_mm=shock_previous,shock_drift_mm=None if shock_previous is None else shock["2.5"]-shock_previous)
    fields.append(interpolate(r,X,Y))
    records.append({k:r[k] for k in ("source","case","ratio","orientation","a_m","b_m",
       "field_path","field_sha256","valid_points","time_range","window","previous_window",
       "shock_mm","shock_previous_mm","shock_drift_mm")}|{"polygon":r["polygon"].tolist()})
    print(case,"shock_mm",shock["2.5"],"drift_mm",r["shock_drift_mm"],flush=True)
 assert len(records)==13
 windows=np.asarray([r["time_range"] for r in records])
 assert np.max(np.ptp(windows,axis=0))<.12
 assert np.max(windows[:,0])-np.min(windows[:,0])<.12
 audit={"source_archives_sha256":hashes,"cases":records,
   "display_grid":[len(x),len(y)],"extent":[-.64,2.10,-1.20,1.20],
   "symmetry":"Lower half mirrored from actual native DS2V half-domain.",
   "density":"Pure-argon density normalized by N_inf=4.247e20/m^3.",
   "reference_length_m":D,
   "circle":"One reference solution repeated in both rightmost panels.",
   "geometry_note":"The same semiaxes are exchanged in each orientation pair. Horizontal q=.75 uses a .75deg phase offset of interior boundary vertices on the same ellipse.",
   "sampling":"Historical vertical cases and fresh orientation cases use matched late native blocks near tU_inf/D=18..20. Earlier adaptive-reset blocks excluded."}
 np.savez_compressed(cache,**{f"{i}_{k}":f[k].filled(np.nan).astype(np.float32) for i,f in enumerate(fields) for k,*_ in FIELDS})
 audit_path.write_text(json.dumps(audit,indent=2)+"\n")

lookup={(r["orientation"],round(r["ratio"],3)):i for i,r in enumerate(records)}
circle=lookup[("circle",1.)]
plt.rcParams.update({"font.family":"DejaVu Serif","mathtext.fontset":"dejavuserif",
 "font.size":11,"axes.linewidth":1.1,"pdf.fonttype":42})
scales={}
isovalues={"MA":[1,2,4,6,8],"ND":[1.5,2.5,4,6,10,15,20],
           "TTR":[500,1000,2000,4000,6000],"P":[5,10,25,50,100,150]}
with PdfPages(out/"all_fields_rgb.pdf") as pdf:
 for key,title,stem,step in FIELDS:
  vmax=max(float(f[key].max()) for f in fields)
  top=float(step*np.ceil(vmax/step))
  norm=PowerNorm(gamma=.5,vmin=0,vmax=top) if key=="ND" else None
  levels=top*np.linspace(0,1,49)**2 if key=="ND" else np.linspace(0,top,49)
  ticks=[0,1,2.5,5,10,top] if key=="ND" else np.linspace(0,top,6)
  if key=="ND":
   title=title+" (square-root color spacing)"
  scales[key]=dict(limits=[0,top],actual_max=vmax,cmap="jet",normalization="square-root (gamma=.5)" if key=="ND" else "linear",intervals=48)
  fig=plt.figure(figsize=(23.6,8.9),facecolor="white")
  gs=fig.add_gridspec(2,8,width_ratios=[1]*7+[.065],left=.045,right=.969,bottom=.18,top=.835,wspace=.22,hspace=.53)
  fig.suptitle(title,fontsize=24,y=.985)
  fig.text(.5,.925,r"DS2V  |  Argon, $M_\infty=10$, $T_\infty=200$ K, $T_w=500$ K  |  Matched late averages near $tU_\infty/D=18$–$20$",
           ha="center",fontsize=12)
  for row,orientation in enumerate(("vertical","horizontal")):
   for col,ratio in enumerate((*ratios,1.)):
    index=circle if ratio==1 else lookup[(orientation,round(ratio,3))]
    r,f=records[index],fields[index]
    ax=fig.add_subplot(gs[row,col])
    img=ax.contourf(X,Y,f[key],levels=levels,cmap="jet",norm=norm,antialiased=False)
    img.set_rasterized(True)
    iso=[v for v in isovalues[key] if 0<v<top]
    contours=ax.contour(X,Y,f[key],levels=iso,colors="black",linewidths=.32,alpha=.40)
    contours.set_rasterized(True)
    ax.add_patch(Polygon(r["polygon"],closed=True,facecolor="white",edgecolor="black",lw=1.15))
    delta=r["shock_mm"]["2.5"]
    ax.scatter(-delta/1000/D,0,marker="o",s=18,facecolor="white",edgecolor="black",linewidth=.7,zorder=9)
    ax.set(xlim=(x[0],x[-1]),ylim=(y[0],y[-1]),aspect="equal",
           xticks=[-.5,0,.5,1,1.5,2],yticks=[-1,-.5,0,.5,1],xlabel="$x/D$")
    ax.tick_params(direction="in",top=True,right=True,labelleft=col==0,pad=3,labelsize=10)
    if col==0:ax.set_ylabel("$y/D$")
    if row==0:
     ax.set_title("Circle" if ratio==1 else f"$q={ratio:.3f}$",fontsize=15,pad=11)
    ax.text(.5,-.30,rf"$\Delta={delta:.1f}$ mm",transform=ax.transAxes,ha="center",va="top",fontsize=11)
  cax=fig.add_subplot(gs[:,7])
  cb=fig.colorbar(img,cax=cax,ticks=ticks)
  cb.ax.tick_params(direction="in",pad=4,labelsize=11)
  fig.text(.008,.71,"Vertical ellipses",rotation=90,ha="center",va="center",fontsize=15)
  fig.text(.008,.325,"Horizontal ellipses",rotation=90,ha="center",va="center",fontsize=15)
  fig.text(.5,.070,r"$q=$ minor/major semiaxis.  $D=0.3048$ m; common nose at $x=0$.  Each paired geometry has the same two semiaxes.",
           ha="center",fontsize=11)
  fig.text(.5,.032,r"White marker: axis shock from $\rho/\rho_\infty=2.5$.  Lower half mirrored.  Rightmost panels repeat the same circular reference.",
           ha="center",fontsize=10.5)
  path=out/f"{stem}_rgb.png"
  fig.savefig(path,dpi=230)
  pdf.savefig(fig,dpi=230)
  plt.close(fig)
  with Image.open(path) as im:
   rgb=im.convert("RGB");dpi=im.info["dpi"]
  rgb.save(path,dpi=dpi)
  print("Rendered",path.name,scales[key],flush=True)
audit["color_scales"]=scales
audit_path.write_text(json.dumps(audit,indent=2)+"\n")
with (out/"shock_comparison.csv").open("w",newline="") as stream:
 writer=csv.writer(stream)
 writer.writerow(["orientation","minor_major","a_over_b","shock_mm_rho2p5","threshold2p2_mm","threshold2p8_mm","previous_window_shock_mm","drift_mm"])
 for r in sorted(records,key=lambda a:(a["orientation"],a["ratio"])):
  writer.writerow([r["orientation"],r["ratio"],r["a_m"]/r["b_m"],r["shock_mm"]["2.5"],r["shock_mm"]["2.2"],r["shock_mm"]["2.8"],r["shock_previous_mm"],r["shock_drift_mm"]])
for r in sorted(records,key=lambda a:(a["orientation"],a["ratio"])):
 print(r["orientation"],r["ratio"],"Delta",round(r["shock_mm"]["2.5"],3),"late drift",None if r["shock_drift_mm"] is None else round(r["shock_drift_mm"],3),flush=True)
