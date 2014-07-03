#!/usr/bin/env python

import sys
import numpy
import scipy.optimize
import pyscf.lib.logger as log


IMP_AND_BATH  = 1
IMP_BLK       = 2
IMP_BATH_DIAG = 3
NO_BATH_BLK   = 4
DIAG_BLK      = 5
IMP_DIAG      = 6
NO_IMP_BLK    = 7
TRACE_IMP     = 8

NO_CONSTRAINT = 0

LINE_SEARCH_PLAIN = 1
LINE_SEARCH_SHARP = 2
LINE_SEARCH_WOLFE = 3
LINE_SEARCH = LINE_SEARCH_WOLFE

######################################
class ImpBathDM(object):
    '''Fit potential for density matrix on impurity and bath'''
    def __init__(self, nemb, nimp):
        self._nemb = nemb
        self._nimp = nimp
        self._nd = nemb
        self._x = None

    def tensor_v2dm(self, e, c, nocc, v_V):
        self._x = v_V.forImpBathDM(e, c, nocc)
        return self._x

    def grad_hessian_approx(self, e, c, nocc, dm_ref_alpha, v2dm):
        n = self._nd
        x0 = v2dm.reshape(n,n,-1)
        x1 = x0.transpose(1,0,2)
        x0 = x0.reshape(n*n,-1)
        x1 = x1.reshape(n*n,-1)
        h = numpy.dot(x0.T, x1) * 2

        ddm = self.diff_den_mat(c, nocc, dm_ref_alpha)
        g = numpy.dot(ddm.flatten(), x0)
        return h, g

    def grad_hessian(self, *args):
        return self.grad_hessian_approx(*args)

    def diff_den_mat(self, c, nocc, dm_ref_alpha):
        nd = self._nd
        dm0 = numpy.dot(c[:nd,:nocc], c[:nd,:nocc].T)
        return dm0 - dm_ref_alpha[:nd,:nd]

    def diff_dm_diag(self, c, nocc, dm_ref_alpha):
        return self.diff_den_mat(c, nocc, dm_ref_alpha).diagonal()

    def v2dmforImpDiagLinearConstr(self):
        idx = numpy.arange(self._nimp)
        return self._x[idx,idx].reshape(self._nimp,-1)
    def v2dmforImpTraceLinearConstr(self):
        return self._x[:self._nimp].trace().reshape(1,-1)

class ImpDM(ImpBathDM):
    def __init__(self, nemb, nimp):
        ImpBathDM.__init__(self, nemb, nimp)
        self._nd = nimp
    def tensor_v2dm(self, e, c, nocc, v_V):
        self._x = v_V.forImpDM(e, c, nocc)
        return self._x

class ImpDiagDM(ImpBathDM):
    def __init__(self, nemb, nimp):
        ImpBathDM.__init__(self, nemb, nimp)
        self._nd = nimp
    def tensor_v2dm(self, e, c, nocc, v_V):
        self._x = v_V.forImpDiagDM(e, c, nocc)
        return self._x
    def grad_hessian_approx(self, e, c, nocc, dm_ref_alpha, v2dm):
        n = self._nd
        x0 = v2dm.reshape(n,-1)
        h = numpy.dot(x0.T, x0) * 2
        ddm = self.diff_den_mat(c, nocc, dm_ref_alpha)
        g = numpy.dot(ddm, x0)
        return h, g
    def diff_den_mat(self, c, nocc, dm_ref_alpha):
        nd = self._nd
        dm0 = numpy.dot(c[:nd,:nocc], c[:nd,:nocc].T).diagonal()
        return dm0 - dm_ref_alpha.diagonal()[:nd]
    def diff_dm_diag(self, c, nocc, dm_ref_alpha):
        return self.diff_den_mat(c, nocc, dm_ref_alpha)
    def v2dmforImpDiagLinearConstr(self):
        return self._x[:self._nimp]
    def v2dmforImpTraceLinearConstr(self):
        return sum(self._x[:self._nimp]).reshape(1,-1)

class NoBathDM(ImpBathDM):
    def tensor_v2dm(self, e, c, nocc, v_V):
        self._x = v_V.forNoBathDM(e, c, nocc)
        return self._x
    def diff_den_mat(self, c, nocc, dm_ref_alpha):
        nd = self._nd
        ddm = numpy.dot(c[:,:nocc], c[:,:nocc].T) - dm_ref_alpha
        ddm[self._nimp:,self._nimp:] = 0
        return ddm
    def diff_dm_diag(self, c, nocc, dm_ref_alpha):
        ddm = self.diff_den_mat(c, nocc, dm_ref_alpha)
        return ddm[:self._nimp].diagonal()

######################################
def mat_v_to_mat_dm1(e, c, nocc, nd, nv):
    '''in AO representation, DM1 = X * V'''
    nmo = e.shape[0]
    nvir = nmo - nocc

    eia = 1 / (e[:nocc].reshape(nocc,1) - e[nocc:])
    tmpcc = numpy.empty((nmo,nd,nv))
    for i in range(nmo):
        ci = c[:,i]
        for t in range(nd):
            for u in range(nv):
                tmpcc[i,t,u] = ci[t] * ci[u]
    v = tmpcc.reshape(nmo,nd*nv)
    _x = reduce(numpy.dot, (v[nocc:].T, eia.T, v[:nocc]))
    _x = _x.reshape(nd,nv,nd,nv)
    x0 = _x.transpose(0,2,1,3)
    x1 = x0.transpose(1,0,3,2)
    return x0 + x1

def mat_v_to_diag_dm1(e, c, nocc, nd, nv):
    x = mat_v_to_mat_dm1(e, c, nocc, nd, nv)
    return numpy.array([x[i,i] for i in range(nd)])

def diag_v_to_diag_dm1(e, c, nocc, nd, nv):
    nmo = e.shape[0]
    nvir = nmo - nocc
    eia = 1 / (e[:nocc].reshape(nocc,1) - e[nocc:])
    tmpcc = numpy.empty((nmo,nd,nv))
    for i in range(nmo):
        ci = c[:,i]
        for t in range(nd):
            for u in range(nv):
                tmpcc[i,t,u] = ci[t] * ci[u]
    v = tmpcc.reshape(nmo,nd*nv)
    vi = numpy.dot(eia, v[nocc:])
    _x = [numpy.dot(v[:nocc,i], vi[:,i]) for i in range(nd*nv)]
    return numpy.array(_x).reshape(nd,nv) * 2

def symm_trans_mat_for_hermit(n):
    # transformation matrix to remove the antisymmetric mode
    # usym is the symmetrized vector corresponding to symmetric component.
    usym = numpy.zeros((n*n, n*(n+1)/2))
    for i in range(n):
        for j in range(i):
            usym[i*n+j,i*(i+1)/2+j] = numpy.sqrt(.5)
            usym[j*n+i,i*(i+1)/2+j] = numpy.sqrt(.5)
            # if considering the weights of the off-diagonal terms
            # usym[i*n+j,i*(i+1)/2+j] = numpy.sqrt(2)
            # usym[j*n+i,i*(i+1)/2+j] = numpy.sqrt(2)
        usym[i*n+i,i*(i+1)/2+i] = 1
    return usym

class ImpBathV(object):
    def __init__(self, nemb, nimp):
        self._nemb = nemb
        self._nimp = nimp
        self._nv = nemb
        self._usymm = symm_trans_mat_for_hermit(self._nv)

    def compress(self, vfit):
        v1 = numpy.empty(self._nv*(self._nv+1)/2)
        k = 0
        for i in range(self._nv):
            for j in range(i+1):
                v1[k] = vfit[i,j]
                k += 1
        return v1
    def decompress(self, vfit):
        v1 = numpy.zeros((self._nemb, self._nemb))
        k = 0
        for i in range(self._nv):
            for j in range(i+1):
                v1[i,j] = v1[j,i] = vfit[k]
                k += 1
        return v1

    def remove_asymm_mode(self, x):
        nn = self._usymm.shape[0]
        return numpy.dot(x.reshape(-1,nn), self._usymm)

    def forImpBathDM(self, e, c, nocc):
        x = mat_v_to_mat_dm1(e, c, nocc, self._nemb, self._nv)
        return self.remove_asymm_mode(x)
    def forImpDM(self, e, c, nocc):
        x = mat_v_to_mat_dm1(e, c, nocc, self._nimp, self._nv)
        return self.remove_asymm_mode(x)
    def forImpDiagDM(self, e, c, nocc):
        x = mat_v_to_diag_dm1(e, c, nocc, self._nimp, self._nv)
        return self.remove_asymm_mode(x)
    def forNoBathDM(self, e, c, nocc):
        x = mat_v_to_mat_dm1(e, c, nocc, self._nemb, self._nv)
        x[self._nimp:,self._nimp:] = 0
        return self.remove_asymm_mode(x)


class ImpV(ImpBathV):
    def __init__(self, nemb, nimp):
        self._nemb = nemb
        self._nimp = nimp
        self._nv = nimp
        self._usymm = symm_trans_mat_for_hermit(self._nv)

class ImpDiagV(ImpBathV):
    def __init__(self, nemb, nimp):
        self._nemb = nemb
        self._nimp = nimp
        self._nv = nimp
        self._usymm = None
    def forImpBathDM(self, e, c, nocc):
        return None
    def forImpDM(self, e, c, nocc):
        return None
    def forImpDiagDM(self, e, c, nocc):
        return diag_v_to_diag_dm1(e, c, nocc, self._nimp, self._nv)
    def forNoBathDM(self, e, c, nocc):
        return None
    def compress(self, vfit):
        return vfit.diagonal()[:self._nv]
    def decompress(self, vfit):
        v1 = numpy.zeros((self._nemb, self._nemb))
        for i in range(self._nv):
            v1[i,i] = vfit[i]
        return v1

class NoBathV(ImpBathV):
    def __init__(self, nemb, nimp):
        self._nemb = nemb
        self._nimp = nimp
        self._nv = nemb
        kick_bath = []
        self._idxi = []
        self._idxj = []
        k = 0
        for i in range(nemb):
            for j in range(i+1):
                if i < nimp or j < nimp:
                    kick_bath.append(k)
                    self._idxi.append(i)
                    self._idxj.append(j)
                k += 1
        self._usymm = symm_trans_mat_for_hermit(self._nv)[:,kick_bath]

    def compress(self, vfit):
        return vfit[self._idxi, self._idxj]
    def decompress(self, vfit):
        v1 = numpy.zeros((self._nemb, self._nemb))
        v1[self._idxi, self._idxj] = vfit
        v1[self._idxj, self._idxi] = vfit
        return v1

######################################
class DmFitObj(object):
    def __init__(self, fock0, nocc, nimp, dm_ref_alpha, v_V, dm_V):
        self._fock0 = fock0
        self._nocc = nocc
        self._nimp = nimp
        self._dm_ref_alpha = dm_ref_alpha
        self._v_V = v_V
        self._dm_V = dm_V
        self.h = None
        self.g = None
        self.update(self.init_guess())

    # for leastsq
    def diff_dm(self, vfit):
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        ddm = self._dm_V.diff_den_mat(c, self._nocc, self._dm_ref_alpha)
        return ddm.flatten()
    def jac_ddm(self, vfit):
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        return self._dm_V.tensor_v2dm(e, c, self._nocc, self._v_V)

    # for Newton-CG
    def norm_ddm(self, vfit):
        return numpy.linalg.norm(self.diff_dm(vfit))
    def hess(self, vfit):
        return self.h
    def grad(self, vfit):
        return self.g
    def update(self, vfit):
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        v2dm = self._dm_V.tensor_v2dm(e, c, self._nocc, self._v_V)
        self.h, self.g = self._dm_V.grad_hessian(e, c, self._nocc, \
                                                 self._dm_ref_alpha, v2dm)
    def init_guess(self):
        return self._v_V.compress(numpy.zeros_like(self._fock0))

class DmFitImpDiagLinearConstr(DmFitObj):
# augment the hessian with Lagrange multiplier
# min f = f0 + gx + x^T h x + ..., with linear constraints ax = b
# /h  a^T\ /x\ = /-g\
# \a  0  / \v/   \ b/
    def update(self, vfit):
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        v2dm = self._dm_V.tensor_v2dm(e, c, self._nocc, self._v_V)
        h, g = self._dm_V.grad_hessian(e, c, self._nocc, \
                                       self._dm_ref_alpha, v2dm)
        a = self._dm_V.v2dmforImpDiagLinearConstr()
        ddm = self._dm_V.diff_den_mat(c, self._nocc, self._dm_ref_alpha)
        c0 = c[:self._nimp,:self._nocc]
        dm0 = numpy.dot(c0, c0.T)
        ddm = dm0.diagonal() - self._dm_ref_alpha.diagonal()[:self._nimp]
        nv = h.shape[0]
        self.h = numpy.empty((nv+self._nimp,nv+self._nimp))
        self.h[:nv,:nv] = h
        self.h[:nv,nv:] = a.T
        self.h[nv:,:nv] = a
        self.h[nv:,nv:] = 0
        self.g = numpy.hstack((g, ddm))
    def init_guess(self):
        return numpy.hstack((self._v_V.compress(numpy.zeros_like(self._fock0)), \
                             numpy.zeros(self._nimp)))

class DmFitImpTraceLinearConstr(DmFitObj):
# augment the hessian with Lagrange multiplier
# min f = f0 + gx + x^T h x + ..., with linear constraints ax = b
# /h  a^T\ /x\ = /-g\
# \a  0  / \v/   \ b/
    def update(self, vfit):
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        v2dm = self._dm_V.tensor_v2dm(e, c, self._nocc, self._v_V)
        h, g = self._dm_V.grad_hessian(e, c, self._nocc, \
                                       self._dm_ref_alpha, v2dm)
        a = self._dm_V.v2dmforImpTraceLinearConstr()
        c0 = c[:self._nimp,:self._nocc]
        dm0 = numpy.dot(c0, c0.T)[:self._nimp].trace()
        ddm = dm0 - self._dm_ref_alpha[:self._nimp].trace()
        nv = h.shape[0]
        self.h = numpy.empty((nv+1,nv+1))
        self.h[:nv,:nv] = h
        self.h[:nv,-1] = a
        self.h[-1,:nv] = a
        self.h[-1,-1] = 0
        self.g = numpy.hstack((g, ddm))
    def init_guess(self):
        return numpy.hstack((self._v_V.compress(numpy.zeros_like(self._fock0)), \
                             numpy.zeros(1)))

class DmFitImpDiagWeightConstr(DmFitObj):
    def __init__(self, fock0, nocc, nimp, dm_ref_alpha, v_V, dm_V):
        assert(not isinstance(dm_V, ImpDiagDM))
        DmFitObj.__init__(self, fock0, nocc, nimp, dm_ref_alpha, v_V, dm_V)

    def diff_dm(self, vfit):
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        ddm = self._dm_V.diff_den_mat(c, self._nocc, self._dm_ref_alpha)
        for i in range(self._nimp):
            ddm[i,i] *= self._weight
        return ddm.flatten()
    def jac_ddm(self, vfit):
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        x = self._jac_ddm_common(e, c)
        m = x.shape[-1]
        return x.reshape(-1, m)
    def _jac_ddm_common(self, e, c):
        ddm = self._dm_V.diff_den_mat(c, self._nocc, self._dm_ref_alpha)
        norm_ddm = numpy.linalg.norm(ddm)
        ddmdiag = numpy.linalg.norm(ddm[:self._nimp].diagonal())
        self._weight = min(1e6, 1+norm_ddm/ddmdiag)
        x = self._dm_V.tensor_v2dm(e, c, self._nocc, self._v_V)
        n = self._dm_V._nd
        m = x.shape[-1]
        x = x.reshape(n,n,m)
        for i in range(self._nimp):
            x[i,i] *= self._weight
            ddm[i,i] *= self._weight
        return x

    def update(self, vfit):
# the weighed h,g can also calculated by
# _dm_V.grad_hessian + weight**2*ImpDiagDM.grad_hessian
# use the following code for efficiency
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        x0 = self._jac_ddm_common(e, c)
        self.h, self.g = self._dm_V.grad_hessian(e, c, self._nocc, \
                                                 self._dm_ref_alpha, x0)

class DmFitImpTraceWeightConstr(DmFitObj):
    def __init__(self, *args):
        DmFitObj.__init__(self, *args)

    def diff_dm(self, vfit):
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        ddm = self._dm_V.diff_den_mat(c, self._nocc, self._dm_ref_alpha)
        if isinstance(self._dm_V, ImpDiagDM):
            ddmdiag = ddm[:self._nimp].sum()
        else:
            ddmdiag = ddm[:self._nimp].trace()
        return numpy.hstack((ddm.flatten(), self._weight*ddmdiag))
    def jac_ddm(self, vfit):
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        x = self._jac_ddm_common(e, c)
        m = x.shape[-1]
        return x.reshape(-1, m)
    def _jac_ddm_common(self, e, c):
        ddm = self._dm_V.diff_den_mat(c, self._nocc, self._dm_ref_alpha)
        norm_ddm = numpy.linalg.norm(ddm)
        self._weight = min(1e6, 1+norm_ddm/abs(ddmdiag))
        n = self._dm_V._nd
        x = self._dm_V.tensor_v2dm(e, c, self._nocc, self._v_V)
        if isinstance(self._dm_V, ImpDiagDM):
            ddmdiag = ddm[:self._nimp].sum()
            y = sum(x[:self._nimp])
        else:
            ddmdiag = ddm[:self._nimp].trace()
            y = x.reshape(n,n,-1)[:self._nimp].trace()
        return numpy.vstack((x.reshape(-1,y.size), self._weight*y))

    def update(self, vfit):
        e, c = numpy.linalg.eigh(self._fock0+self._v_V.decompress(vfit))
        x = self._jac_ddm_common(e, c)
        y = x[-1].reshape(1,-1)
        h1 = numpy.dot(y.T, y) * 2
        g1 = ddmdiag * x[-1]
        h, g = self._dm_V.grad_hessian(e, c, self._nocc, self._dm_ref_alpha, \
                                       x[:-1])
        self.h = h + self._weight**2*h1
        self.g = g + self._weight**2*g1


def select_v(fit_domain, nemb, nimp):
    if fit_domain == IMP_AND_BATH:
        return ImpBathV(nemb, nimp)
    elif fit_domain == IMP_BLK:
        return ImpV(nemb, nimp)
    elif fit_domain == IMP_BATH_DIAG:
        return None
    elif fit_domain == NO_BATH_BLK:
        return NoBathV(nemb, nimp)
    elif fit_domain == DIAG_BLK:
        return None
    elif fit_domain == IMP_DIAG:
        return ImpDiagV(nemb, nimp)
    elif fit_domain == NO_IMP_BLK:
        return None
    elif fit_domain == TRACE_IMP:
        return None

def select_dm(fit_domain, nemb, nimp):
    if fit_domain == IMP_AND_BATH:
        return ImpBathDM(nemb, nimp)
    elif fit_domain == IMP_BLK:
        return ImpDM(nemb, nimp)
    elif fit_domain == IMP_BATH_DIAG:
        return None
    elif fit_domain == NO_BATH_BLK:
        return NoBathDM(nemb, nimp)
    elif fit_domain == DIAG_BLK:
        return None
    elif fit_domain == IMP_DIAG:
        return ImpDiagDM(nemb, nimp)
    elif fit_domain == NO_IMP_BLK:
        return None
    elif fit_domain == TRACE_IMP:
        return None


####################
# simple version of scfopt.step_by_eigh_min
def step_by_eigh_min(h, g, threshold=1e-6):
    ''' h * x = g '''
    n = h.shape[0]
    h = numpy.array(h)
    w, u = numpy.linalg.eigh(h)
    g = numpy.array(g)
    idx = []
    step = []
    for i,wi in enumerate(w):
        if wi > threshold:
            b1 = numpy.dot(u[:,i], g) / wi
            idx.append(i)
            step.append(b1)
        elif wi < -1e-10: # avoid local maxima
            idx.append(i)
            b1 = .1 * numpy.dot(u[:,i], g) / (1e-9-wi)
            step.append(b1)
    if idx:
        step = numpy.array(step)
        idx = numpy.array(idx)
        x = numpy.dot(u[:,idx], step)
    else:
        x = numpy.zeros_like(g)
    return x

def line_search(dev, fn, dx, lim=1, floating=1e-5, val0=None, title=''):
    if val0 is None:
        val0 = fn(numpy.zeros_like(dx))
    def line_search_iter(val_old, step, lim):
        alpha = 0
        for a in numpy.arange(step, lim, step):
            val_new = fn(dx*a)
            log.debug(dev, 'line_search %s, factor = %.9g, val_old = %.9g, val_new = %.9g', \
                      title, a, val_old, val_new)
            if val_old*(1+floating) < val_new:
                break
            else:
                alpha = a
                val_old = val_new
        if alpha > 0 or step < 1e-3:
            log.debug(dev, 'line_search %s, factor = %.9g', title, alpha)
            return alpha * dx, val_old
        else:
            return line_search_iter(val_old, step*.5, step)
    return line_search_iter(val0, lim*.25, lim*1.6)

def line_search_sharp(dev, fn, dx, lim=1, floating=1e-5, val0=None, title=''):
    if val0 is None:
        val0 = fn(numpy.zeros_like(dx))
    def line_search_iter(val_old, alpha, step, lim):
        if step < 1e-3 or alpha > lim:
            log.debug(dev, 'sharp line_search %s, factor = %.9g', title, alpha)
            return alpha*dx, val_old
        else:
            val_new = fn(dx*(alpha+step))
            log.debug(dev, 'sharp line_search %s, factor = %.9g, val_old = %.9g, val_new = %.9g', \
                      title, alpha+step, val_old, val_new)
            if val_old*(1+floating) < val_new:
                return line_search_iter(val_old, alpha, step*.4, alpha+step)
            else:
                return line_search_iter(val_new, alpha+step, step, lim)
    return line_search_iter(val0, 0, lim*.5, lim*2.1)

def line_search_wolfe(dev, fn, dx, c1=1e-4, val0=None, grad0=None, \
                      minstep=2e-3, title=''):
    if val0 is None:
        val0 = fn(numpy.zeros_like(dx))
    if grad0 is None:
        slope = -1e-8
    else:
        slope = -1e-8 + numpy.dot(grad0.flatten(), dx.flatten()) * c1
    def line_search_iter(val_old, alpha, step, lim):
        val_new = fn(dx*(alpha+step))
        log.debug(dev, 'wolfe line_search %s, factor = %.9g, val_old = %.9g, val_new = %.9g', \
                  title, alpha+step, val_old, val_new)
        if alpha > 1e2:
            return alpha, val_old
        elif val_old+slope*(alpha+step) < val_new:
            # outside the trust region of Wolfe conditions
            if step < minstep or alpha > 1.99:
                return alpha, val_old
            else:
                return line_search_iter(val_old, alpha, step*.4, alpha+step*.79)
        elif alpha+step > lim:
            return line_search_iter(val_new, alpha+step, step*3, alpha+step*5.99)
        else:
            return line_search_iter(val_new, alpha+step, step, lim)
    alpha, xnew = line_search_iter(val0, 0, .5, 1.49)
    log.debug(dev, 'wolfe line_search %s, factor = %.9g', title, alpha)
    return alpha*dx, xnew

# refer to scfopt.find_emb_potential
def newton_gauss(dev, fun, x0, grad, hess, callback, thrd=1e-12, maxiter=10):
    x = numpy.copy(x0)
    thrd_grad = thrd*1e2
    for it in range(maxiter):
        val = fun(x)
        if val < thrd:
            break
        g = grad(x)
        if numpy.linalg.norm(g) < thrd_grad:
            break
        h = hess(x)
        dx = step_by_eigh_min(h, -g, min(1e-6,val*.2))

        if LINE_SEARCH == LINE_SEARCH_PLAIN:
            dx,val1 = line_search(dev, lambda d:fun(x+d), \
                                  dx, title='newton-gauss')
        elif LINE_SEARCH == LINE_SEARCH_SHARP:
            dx,val1 = line_search_sharp(dev, lambda d:fun(x+d), \
                                        dx, floating=1e-3, \
                                        title='newton-gauss')
        elif LINE_SEARCH == LINE_SEARCH_WOLFE:
            dx,val1 = line_search_wolfe(dev, lambda d:fun(x+d), \
                                        dx, title='newton-gauss')
        x += dx
        callback(x)
    return x


####################
def fit_solver(dev, fock0, nocc, nimp, dm_ref_alpha, \
               v_domain, dm_domain, constr):
    nemb = fock0.shape[0]
    v_V  = select_v(v_domain, nemb, nimp)
    dm_V = select_dm(dm_domain, nemb, nimp)
    if constr == NO_CONSTRAINT:
        fitp = DmFitObj(fock0, nocc, nimp, dm_ref_alpha, v_V, dm_V)
        if 1:
            #x = scipy.optimize.minimize(fitp.norm_ddm, fitp.init_guess(), \
            #                            method='Newton-CG', \
            #                            jac=fitp.grad, hess=fitp.hess, \
            #                            tol=1e-8, callback=fitp.update, \
            #                            options={'maxiter':6,'disp':False}).x
            x = scipy.optimize.leastsq(fitp.diff_dm, fitp.init_guess(), \
                                       Dfun=fitp.jac_ddm, ftol=1e-8)[0]
        else:
            x = newton_gauss(dev, fitp.norm_ddm, fitp.init_guess(), \
                             fitp.grad, fitp.hess, fitp.update, 1e-8, 6)
    elif constr == TRACE_IMP:
        #fitp = DmFitImpTraceLinearConstr(fock0, nocc, nimp, dm_ref_alpha, \
        #                                 v_V, dm_V)
        if 0:
            # slow, sometimes not so accurate, but works in almost all occasions
            fitp = DmFitImpTraceWeightConstr(fock0, nocc, nimp, dm_ref_alpha, \
                                             v_V, dm_V)
            x = scipy.optimize.leastsq(fitp.diff_dm, fitp.init_guess(), \
                                       Dfun=fitp.jac_ddm, ftol=1e-8)[0]
            #x = newton_gauss(dev, fitp.norm_ddm, fitp.init_guess(), \
            #                 fitp.grad, fitp.hess, fitp.update, 1e-8, 6)
        else:
            fitp = DmFitObj(fock0, nocc, nimp, dm_ref_alpha, v_V, dm_V)
            def ddm_diag(vfit):
                e, c = numpy.linalg.eigh(fock0+v_V.decompress(vfit))
                return dm_V.diff_dm_diag(c, nocc, dm_ref_alpha)[:nimp].sum()
            def grad(vfit):
                fitp.update(vfit)
                return fitp.grad(vfit)
            #cons = {'type': 'eq', 'fun': ddm_diag}
            dmdiag_V = ImpDiagDM(fock0.shape[0], nimp)
            def jac(vfit):
                e, c = numpy.linalg.eigh(fock0+v_V.decompress(vfit))
                v2dm = dmdiag_V.tensor_v2dm(e, c, nocc, v_V)
                return sum(v2dm.reshape(nimp,-1))
            cons = {'type': 'eq', 'fun': ddm_diag, 'jac': jac}
            x = scipy.optimize.minimize(fitp.norm_ddm, fitp.init_guess(), \
                                        method='SLSQP', jac=grad, tol=1e-8, \
                                        constraints=cons, \
                                        options={'maxiter':6,'disp':0}).x
    else:
# Linear Constraint methods cannot ensure the diagonal DM completely match
# diagonal dm_ref. Maybe diagonal DM is not really linear constraint
# weighted constraint works better
        #fitp = DmFitImpDiagLinearConstr(fock0, nocc, nimp, dm_ref_alpha, \
        #                                v_V, dm_V)
        if 0:
            # slow, sometimes not so accurate, but works in almost all occasions
            fitp = DmFitImpDiagWeightConstr(fock0, nocc, nimp, dm_ref_alpha, \
                                            v_V, dm_V)
            x = scipy.optimize.leastsq(fitp.diff_dm, fitp.init_guess(), \
                                       Dfun=fitp.jac_ddm, ftol=1e-8)[0]
            #x = newton_gauss(dev, fitp.norm_ddm, fitp.init_guess(), \
            #                 fitp.grad, fitp.hess, fitp.update, 1e-8, 6)
        else:
            fitp = DmFitObj(fock0, nocc, nimp, dm_ref_alpha, v_V, dm_V)
            def ddm_diag(vfit):
                e, c = numpy.linalg.eigh(fock0+v_V.decompress(vfit))
                return dm_V.diff_dm_diag(c, nocc, dm_ref_alpha)[:nimp]
            def grad(vfit):
                fitp.update(vfit)
                return fitp.grad(vfit)
            #cons = {'type': 'eq', 'fun': ddm_diag}
            dmdiag_V = ImpDiagDM(fock0.shape[0], nimp)
            def jac(vfit):
                e, c = numpy.linalg.eigh(fock0+v_V.decompress(vfit))
                v2dm = dmdiag_V.tensor_v2dm(e, c, nocc, v_V)
                return v2dm.reshape(nimp,-1)
            cons = {'type': 'eq', 'fun': ddm_diag, 'jac': jac}
            x = scipy.optimize.minimize(fitp.norm_ddm, fitp.init_guess(), \
                                        method='SLSQP', jac=grad, tol=1e-8, \
                                        constraints=cons, \
                                        options={'maxiter':6,'disp':0}).x
    vfit = v_V.decompress(x)

    e, c = numpy.linalg.eigh(fock0+vfit)
    ddm = dm_V.diff_den_mat(c, nocc, dm_ref_alpha)
    if ddm.ndim == 2:
        ddiag = ddm[:nimp].diagonal()
    else:
        ddiag = ddm[:nimp]
    log.debug(dev, 'ddm diag = %s', ddiag)
    log.debug(dev, 'norm(ddm) = %.8g, norm(dv) = %.8g, trace_imp(ddm) = %.8g', \
              numpy.linalg.norm(ddm), numpy.linalg.norm(vfit), \
              ddiag.sum())
    return vfit


def numfitor(dev, get_dm, walkers, dm_ref, \
             v_inc_base, title=''):
    def mspan(dv):
        v_inc = numpy.zeros_like(v_inc_base)
        for k,(i,j) in enumerate(walkers):
            v_inc[i,j] = v_inc[j,i] = dv[k]
        return v_inc
    def ddm_method(dv):
        return dm_ref - get_dm(v_inc_base+mspan(dv))
    x0 = numpy.zeros(len(walkers))
    x = scipy.optimize.leastsq(ddm_method, x0, ftol=1e-8)[0]
    ddm = ddm_method(x)
    log.debug(dev, 'ddm %s = %s', title, ddm)
    log.debug(dev, 'norm(ddm) = %.8g, norm(dv) = %.8,g', \
              numpy.linalg.norm(ddm), numpy.linalg.norm(x))
    return mspan(x)

###############################
def fit_solver_quiet(fock0, nocc, nimp, dm_ref_alpha, \
                     v_domain, dm_domain, constr):
    dev = lambda: None
    dev.fout = sys.stdout
    dev.verbose = 0
    return fit_solver(dev, fock0, nocc, nimp, dm_ref_alpha, \
                      v_domain, dm_domain, constr)
