"""Explicit scalar-field metrics inspired by PDEBench, independently implemented.

Integrals are integrals of the supplied scalar, not automatically mass/energy.
An ROI edge is not a physical wall. Zero-norm relative errors are undefined.
"""
import numpy as np


def relative_l2(estimate,reference):
    a,b=np.asarray(estimate,float),np.asarray(reference,float)
    if a.shape!=b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Need finite matching nonempty arrays')
    norm=float(np.linalg.norm(b))
    return None if norm==0 else float(np.linalg.norm(a-b)/norm)


def field_metrics(estimate,reference,weights=None,edge_mask=None):
    a,b=np.asarray(estimate,float),np.asarray(reference,float)
    relative_l2(a,b)
    if a.ndim!=3:
        raise ValueError('Expected (time,y,x) scalar fields')
    w=np.ones(a.shape[1:]) if weights is None else np.asarray(weights,float)
    if w.shape!=a.shape[1:] or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:
        raise ValueError('Invalid cell-area weights')
    e=a-b; weighted=e*np.sqrt(w); truth=b*np.sqrt(w)
    norms=np.linalg.norm(truth.reshape(len(a),-1),axis=1)
    frame_error=np.linalg.norm(weighted.reshape(len(a),-1),axis=1)
    out={'relative_l2':relative_l2(a*np.sqrt(w),truth),
         'rmse':float(np.sqrt(np.sum(e*e*w)/(len(a)*w.sum()))),
         'maximum_absolute_error':float(np.max(abs(e[:,w>0]))),
         'worst_frame_relative_l2':None if np.any(norms==0) else float(np.max(frame_error/norms)),
         'zero_reference_frames':int(np.count_nonzero(norms==0)),
         'mean_absolute_scalar_integral_error':float(np.mean(abs(np.sum(e*w,axis=(-2,-1))))) }
    if edge_mask is not None:
        edge=np.asarray(edge_mask)
        if edge.dtype!=bool or edge.shape!=w.shape or not np.any(edge & (w>0)):
            raise ValueError('Invalid edge mask')
        out['edge_relative_l2']=relative_l2((a*np.sqrt(w))[:,edge],truth[:,edge])
    return out


def temporal_spectrum(estimate,reference,dt,band=None):
    a,b=np.asarray(estimate,float),np.asarray(reference,float)
    relative_l2(a,b)
    if a.ndim!=1 or len(a)<8 or not np.isfinite(dt) or dt<=0:
        raise ValueError('Need at least eight samples and positive dt')
    f=np.fft.rfftfreq(len(a),dt);window=np.hanning(len(a))
    fa=np.fft.rfft((a-a.mean())*window);fb=np.fft.rfft((b-b.mean())*window)
    use=f>0
    if band is not None:
        low,high=band
        if not 0<=low<high<=.5/dt:
            raise ValueError('Band exceeds Nyquist or is unordered')
        use &= (f>=low)&(f<=high)
    if not np.any(use):
        raise ValueError('No resolved frequency bins in requested band')
    pa,pb=abs(fa[use])**2,abs(fb[use])**2;bins=f[use]
    has_signal=float(pb.max())>0
    peak=int(np.argmax(pb))
    return {'frequency_resolution':float(1/(len(a)*dt)),
      'reference_peak':float(bins[peak]) if has_signal else None,
      'predicted_peak':float(bins[np.argmax(pa)]) if pa.max()>0 else None,
      'power_spectrum_relative_l2':relative_l2(pa,pb),
      'phase_error_at_reference_peak':float(np.angle(fa[use][peak]*np.conj(fb[use][peak]))) if has_signal and pa[peak]>0 else None}
