"""Multi-start open-tour improvement for the existing route scheduler."""
import numpy as np
from joint_strategy import open_tour as original

def improved_tour(start,points):
    if len(points)<4:return original(start,points)
    p=np.vstack([start,points]);d=np.linalg.norm(p[:,None]-p[None,:],axis=2);n=len(points)
    def score(r):return sum(d[a,b] for a,b in zip([0]+r[:-1],r))
    seeds=[None]+sorted(range(1,n+1),key=lambda i:d[0,i])[:min(n,7)]
    best=None;bestcost=float('inf')
    for first in seeds:
        if first is None:r=[i+1 for i in original(start,points)]
        else:
            r=[first];remaining=set(range(1,n+1))-{first}
            while remaining:
                j=min(remaining,key=lambda j:(d[r[-1],j],j));r.append(j);remaining.remove(j)
        for _ in range(100):
            change=None;gain=1e-6
            for i in range(n):
                prev=0 if i==0 else r[i-1]
                for j in range(i+1,n):
                    delta=d[prev,r[i]]-d[prev,r[j]]
                    if j+1<n:delta+=d[r[j],r[j+1]]-d[r[i],r[j+1]]
                    if delta>gain:gain=delta;change=(i,j)
            if change is None:break
            i,j=change;r[i:j+1]=r[i:j+1][::-1]
        cost=score(r)
        if cost<bestcost:bestcost=cost;best=r
    return [i-1 for i in best]
