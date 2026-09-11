"""两个不同目标函数；R 在方案一解析边际化，在方案二始终固定。"""
import math
import numpy as np
import shapely
from shapely.geometry import Polygon
from geometry import (TAU, disk, wedge, first_region, cells_for, quadrature,
                      diameter, radius, bearing_contributions, classify)


class Base:
    def __init__(self, config, rmax):
        self.c = config
        self.s1 = np.array([config['x1'], config['y1']])
        self.rmax = rmax
        self.eps = math.radians(config['epsilon_deg'])
        self.width = math.radians(config['angle_bin_width_deg'])
        self.nbins = round(TAU/self.width)
        self.p1 = first_region(config, rmax)
        if self.p1.area <= 0:
            raise ValueError('第一次正常示向度与位置/半径不相容：P1 为空或面积为零。')
        self.cells = cells_for(config, self.p1, rmax)
        self.base_cells, self.xy, self.areas = quadrature(config, self.cells)

    def regions(self, s):
        near_disk = disk(s, 5, min(self.c['arc_tolerance_m'], .001))
        receive_disk = disk(s, self.rmax, self.c['arc_tolerance_m'])
        return (self.p1.intersection(near_disk),
                self.p1.intersection(receive_disk).difference(near_disk),
                self.p1.difference(receive_disk))

    def bearing_region(self, direction, s, theta, eps=None):
        reach = np.linalg.norm(s) + self.c['L'] + 1
        return direction.intersection(wedge(s, theta, self.eps if eps is None else eps, reach))

    def pack(self, near, none, bins, losses, rn, rz, regions=None, diagnostics=None, angle_values=None):
        pd = float(bins.sum())
        error = abs(near+none+pd-1)
        if error > 1e-9:
            raise ArithmeticError(f'概率不守恒：误差 {error}')
        return dict(J=float(np.dot(bins,losses)+near*rn+none*rz),
                    p_near=float(near), p_direction=pd, p_no_signal=float(none),
                    bins=bins, losses=losses, near_loss=rn, no_signal_loss=rz,
                    representative_angle_deg=np.degrees((np.arange(self.nbins)+.5)*self.width)
                    if angle_values is None else np.degrees(angle_values)%360,
                    normalization_error=error, regions=regions or {}, diagnostics=diagnostics or {})

    def repeated(self, loss):
        bins = np.zeros(self.nbins)
        bins[int((math.radians(self.c['theta1_deg']) % TAU)/self.width) % self.nbins] = 1
        return self.pack(0,0,bins,np.full(self.nbins,loss),0,0,{'unchanged':self.p1},
                         angle_values=np.full(self.nbins,math.radians(self.c['theta1_deg'])))


class FixedRadius(Base):
    metric = 'expected_full_region_diameter_m'

    def __init__(self, config):
        super().__init__(config, config['R0'])
        self.weights = self.areas / self.areas.sum()

    def evaluate(self, s, details=False):
        s = np.asarray(s)
        if np.linalg.norm(s-self.s1) < 1e-9:
            return self.repeated(diameter(self.p1))
        _, xy, areas = quadrature(self.c, self.cells, s, self.rmax)
        w = areas/areas.sum()
        delta = xy-s
        d = np.linalg.norm(delta,axis=1)
        phi = np.arctan2(delta[:,1],delta[:,0])
        near, direct, none = classify(d,self.rmax)
        di = np.flatnonzero(direct)
        ids,ks,alpha = bearing_contributions(phi[direct],self.eps,self.width)
        contrib = w[di[ids]]*alpha
        bins = np.bincount(ks,weights=contrib,minlength=self.nbins)
        pn,pz = float(w[near].sum()),float(w[none].sum())
        gn,gd,gz = self.regions(s)
        # near/no_signal 概率直接用完整裁剪面积，避免 5m 小圆被粗网格漏采。
        masses = np.array([gn.area,gd.area,gz.area])
        pn,pd,pz = masses/masses.sum()
        if bins.sum() > 0:
            bins *= pd/bins.sum()
        elif pd > 0:
            # 极薄的 direction 区域：重新对该区域进行面积求积，不能丢失分支。
            from geometry import split_region
            parts = split_region(gd)
            pts = np.array([(q.representative_point().x,q.representative_point().y) for q in parts])
            ph = np.arctan2(pts[:,1]-s[1],pts[:,0]-s[0])
            ids,ks,alpha = bearing_contributions(ph,self.eps,self.width)
            ww = np.array([q.area for q in parts])/self.p1.area
            bins = np.bincount(ks,weights=ww[ids]*alpha,minlength=self.nbins)
            bins *= pd/bins.sum()
        losses = np.zeros(self.nbins)
        angles = (np.arange(self.nbins)+.5)*self.width
        fallback = 0
        regions = {'near':gn,'no_signal':gz}
        for k in np.flatnonzero(bins > 0):
            theta = (k+.5)*self.width
            p = self.bearing_region(gd,s,theta)
            if p.area <= 0:
                # 正概率档的中心可能不在可达支持内。以档支持交集内部点构造
                # 一个档内且可达的测角；不会把正概率分支设为零损失。
                support = self.bearing_region(gd,s,theta,self.eps+self.width/2)
                if support.area <= 0:
                    raise ArithmeticError('概率求积与几何支持不一致；请减小网格和圆弧误差。')
                q = support.representative_point()
                ph = math.atan2(q.y-s[1],q.x-s[0])
                ph = theta + (ph-theta+math.pi) % TAU-math.pi
                lo,hi = max(k*self.width,ph-self.eps),min((k+1)*self.width,ph+self.eps)
                angles[k] = (lo+hi)/2
                p = self.bearing_region(gd,s,angles[k])
                if p.area <= 0:
                    raise ArithmeticError('有效档内求积点仍为空。')
                fallback += 1
            losses[k] = diameter(p)
            if details and k in np.argsort(bins)[-3:]:
                regions[f'bearing_{math.degrees(theta):.3f}_deg'] = p
        return self.pack(pn,pz,bins,losses,diameter(gn),diameter(gz),regions,
                         {'quadrature_points':len(xy),'effective_angle_fallbacks':fallback,
                          'direction_quadrature_mass':float(w[direct].sum()),
                          'direction_geometry_mass':pd,'truncated_probability':0.},angles)


class Bayesian(Base):
    metric = 'expected_credible_region_cover_radius_m'

    def __init__(self, config):
        super().__init__(config, config['R_max'])
        self.d1 = np.linalg.norm(self.xy-self.s1,axis=1)
        self.surv1 = self.survival(self.d1)
        raw = self.areas*self.surv1
        if raw.sum() <= 0:
            raise ValueError('第一次观测的后验概率为零。')
        self.weights = raw/raw.sum()

    def survival(self, distance):
        return np.clip((self.c['R_max']-np.maximum(self.c['R_min'],distance))/
                       (self.c['R_max']-self.c['R_min']),0,1)

    def credible(self, weights, support=None):
        """单元密度排序到至少 credible_mass；覆盖完整所选裁剪单元。"""
        idx = np.flatnonzero(weights > 0)
        if not len(idx):
            return 0.,Polygon(),0.
        geoms = np.array(self.base_cells,dtype=object)[idx]
        if support is not None:
            geoms = shapely.intersection(geoms,support)
        areas = shapely.area(geoms)
        if np.any(areas <= 0):
            raise ArithmeticError('贝叶斯正概率单元没有几何支持，请加密圆弧/位置网格。')
        mass = weights[idx]
        order = np.argsort(-mass/areas,kind='stable')
        accumulated = np.cumsum(mass[order])
        count = min(len(order),np.searchsorted(accumulated,self.c['credible_mass']*mass.sum())+1)
        chosen = order[:count]
        region = shapely.union_all(geoms[chosen])
        return radius(region),region,float(mass[chosen].sum()/mass.sum())

    def evaluate(self, s, details=False):
        s = np.asarray(s)
        if np.linalg.norm(s-self.s1) < 1e-9:
            r,p,mass = self.credible(self.weights)
            result = self.repeated(r)
            result['regions'] = {'unchanged_credible':p}
            return result
        delta = self.xy-s
        d2 = np.linalg.norm(delta,axis=1)
        phi = np.arctan2(delta[:,1],delta[:,0])
        # 对同一个固定未知 R 解析积分：P(R>=d2 | R>=d1)。
        reception = np.divide(self.survival(np.maximum(self.d1,d2)),self.surv1,
                              out=np.zeros_like(d2),where=self.surv1>0)
        near = d2 <= 5
        wn = self.weights*near
        wz = self.weights*(~near)*(1-reception)
        wd = self.weights*(~near)*reception
        ids,ks,alpha = bearing_contributions(phi,self.eps,self.width)
        contrib = wd[ids]*alpha
        bins = np.bincount(ks,weights=contrib,minlength=self.nbins)
        losses = np.zeros(self.nbins)
        gn,gd,_ = self.regions(s)
        # no_signal 的存在性要求 d2 > max(R_min,d1)。
        # d2>d1 是以 S1/S2 垂直平分线为界的半平面，逐单元裁剪。
        none_support = self.p1.difference(disk(s,self.c['R_min'],self.c['arc_tolerance_m']))
        v = s-self.s1
        midpoint = (s+self.s1)/2
        u = v/np.linalg.norm(v)
        t = np.array([-u[1],u[0]])
        reach = 4*(np.linalg.norm(s)+np.linalg.norm(self.s1)+self.c['L']+self.rmax)
        # 离 S2 更远的一侧：dot(G-midpoint, S2-S1)<0。
        half = Polygon([midpoint+t*reach,midpoint-t*reach,
                        midpoint-t*reach-u*reach,midpoint+t*reach-u*reach])
        none_support = none_support.intersection(half)
        rn,pn,_ = self.credible(wn,gn)
        rz,pz,_ = self.credible(wz,none_support)
        regions = {'near_credible':pn,'no_signal_credible':pz}
        for k in np.flatnonzero(bins > 0):
            m = ks == k
            weights = np.bincount(ids[m],weights=contrib[m],minlength=len(self.xy))
            theta = (k+.5)*self.width
            # 方案一观测就是角度档，必须取整档支持的并集。
            support = self.bearing_region(gd,s,theta,self.eps+self.width/2)
            losses[k],p,_ = self.credible(weights,support)
            if details and k in np.argsort(bins)[-3:]:
                regions[f'bin_{math.degrees(theta):.3f}_deg_credible'] = p
        return self.pack(wn.sum(),wz.sum(),bins,losses,rn,rz,regions,
                         {'posterior_cells':len(self.xy),'truncated_probability':0.,
                          'R_integration':'analytic_uniform_conditional_same_R'})
