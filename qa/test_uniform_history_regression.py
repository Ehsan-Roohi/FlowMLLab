import sys
sys.path.insert(0,sys.argv[1])
import numpy as np
from flowmllab.cylinder_lbm import simulate_cylinder
for n in (201,200,3):
 r=simulate_cylinder(100,nx=80,ny=32,diameter=4,center=(20,15.5),inflow_velocity=.05,steps=n,history_stride=5,statistics_start=min(20,n-1),startup_ramp_steps=min(20,n),collision_model="trt",cylinder_boundary="bouzidi")
 assert np.all(np.diff(r["time"])==5)
 assert np.isfinite(r["u"]).all()
 print("PASS steps=",n,"samples=",len(r["time"]))
