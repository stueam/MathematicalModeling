"""Independent angle-gap diagnostics, never used to certify or stop a policy."""
import math

import numpy as np
import shapely

from q4.core import DOMAIN


def sample_domain(spacing):
    values=np.arange(-1800.,1800.+spacing/2,spacing)
    x,y=np.meshgrid(values,values,indexing='ij')
    xy=np.column_stack((x.ravel(),y.ravel()))
    xy=xy[np.linalg.norm(xy,axis=1)<=1800.]
    theta=np.arange(720)*math.tau/720
    boundary=np.column_stack((1800*np.cos(theta),1800*np.sin(theta)))
    return np.vstack((xy,boundary,shapely.get_coordinates(DOMAIN)))


def angle_gaps(samples,negative_points):
    """Largest uncovered angular gap among points within the true R minimum.

    A <=180-degree gap suffices at a sampled location because the emission
    semicircle is closed. Samples remain a diagnostic, not a continuum proof.
    """
    if not negative_points:
        return np.full(len(samples),360.)
    points=np.asarray(negative_points,dtype=float)
    delta=points[None]-np.asarray(samples)[:,None]
    lengths=np.linalg.norm(delta,axis=2)
    angle=np.where(lengths<=1000.,np.arctan2(delta[:,:,1],delta[:,:,0])%math.tau,np.inf)
    angle.sort(axis=1)
    count=np.isfinite(angle).sum(axis=1)
    result=np.full(len(samples),360.)
    for i,n in enumerate(count):
        if np.any(lengths[i]<=1e-10):
            result[i]=0.
        elif n:
            a=angle[i,:n]
            result[i]=float(np.max(np.diff(np.r_[a,a[0]+math.tau]))*180/math.pi)
    return result


def audit_absence(belief,spacings):
    groups={}
    for c,channel in belief.channels.items():
        if channel.status=='absent_certified':
            points=tuple(sorted(set(channel.negatives)))
            groups.setdefault(points,[]).append(c)
    result=[]
    for points,channels in groups.items():
        grids=[]
        for spacing in spacings:
            samples=sample_domain(spacing)
            gaps=angle_gaps(samples,points)
            bad=np.flatnonzero(gaps>180.+1e-7)
            grids.append({'spacing_m':spacing,'samples':len(samples),'uncovered_samples':len(bad),
                          'maximum_gap_deg':float(np.max(gaps)),
                          'counterexamples':[{'position':samples[i].tolist(),'gap_deg':float(gaps[i])} for i in bad[:8]]})
        result.append({'channels':channels,'negative_points':len(points),'grids':grids})
    return {'kind':'independent_sample_diagnostic_not_completion_proof','groups':result,
            'all_sample_checks_pass':all(g['uncovered_samples']==0 for r in result for g in r['grids']),
            'count_16_completion_without_absence':len(belief.known)==16 and not groups}
