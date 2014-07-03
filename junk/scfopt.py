#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import numpy
import scipy.sparse.linalg as sparse
import pyscf.lib.logger as log
import hfdm
from hfdm import line_search, line_search_sharp, line_search_wolfe

# fitting can be started with big HESSIAN_THRESHOLD and small TUNE_FAC, after
# the gross fitting converged, then lower the HESSIAN_THRESHOLD to include
# more and more "mode" (eigenfunction of hessian)
MAX_ITER = 20
CONV_THRESHOLD = 1e-5
HESSIAN_THRESHOLD = 1e-6
TUNE_FAC = .2
STEP_SIZE_CONTR = .4#TUNE_FAC
MAX_STEP_SIZE = 100

LINE_SEARCH_PLAIN = hfdm.LINE_SEARCH_PLAIN # 1
LINE_SEARCH_SHARP = hfdm.LINE_SEARCH_SHARP # 2
LINE_SEARCH_WOLFE = hfdm.LINE_SEARCH_WOLFE # 3
LINE_SEARCH       = LINE_SEARCH_WOLFE

FIT_DM_IMP_AND_BATH = 1
FIT_DM_IMP_ONLY = 2
FIT_DM_IMP_DIAG = 3
FIT_DM_NO_BATH_BLOCK = 4
FIT_DM_IMP_ONLY_DIAG_CONSTRAINT = 5
FIT_DM_IMP_ONLY_NELE_CONSTRAINT = 6
FIT_DM_IMP_BATH_DIAG_CONSTRAINT = 12
FIT_DM_IMP_BATH_NELE_CONSTRAINT = 7
FIT_DM_NO_IMP_BLOCK = 8
FIT_DM_NO_IMP_NELE_CONSTRAINT = 9
FIT_DM_NO_IMP_DIAG_CONSTRAINT = 10
FIT_DM_LIEB_FNL = 11

FIT_DM_METHOD = 3

def solve_lineq_by_SVD(a, b, threshold=1e-6):
    ''' a * x = b '''
    t, w, vH = numpy.linalg.svd(numpy.array(a))
    lst = w > threshold
    tb = numpy.dot(t[:,lst].T.conj(), numpy.array(b).flatten())
    wtb = tb.T * (1/w[lst])
    return numpy.dot(vH[lst].T.conj(), wtb.T), w.shape[0] - lst.shape[0]

def solve_lineq_by_eigh(a, b, threshold=1e-6):
    w, v = numpy.linalg.eigh(numpy.array(a))
    lst = w > threshold
    vb = numpy.dot(v[:,lst].T.conj(), numpy.array(b).flatten())
    wvb = vb.T * (1/w[lst])
    return numpy.dot(v[:,lst], wvb.T), w.shape[0] - lst.shape[0]

def step_by_SVD(h, g, threshold=1e-6):
    '''determine step by using SVD to solve h * x = g'''
    t, w, vH = numpy.linalg.svd(h)
    t = numpy.array(t)
    vH = numpy.array(vH)
    if threshold > 1e-7:
        threshold = 1e-7
    elif threshold < 1e-14:
        threshold = 1e-14
    idx = []
    step = []
    n_singular = w.__len__()
    for i,wi in enumerate(w):
        # to avoid big step size in optimization, this threshold should be larger
        if wi > threshold:
            ##idx.append(i)
            ##n_singular -= 1
            ##b1 = numpy.dot(t[:,i], g) / wi
            ##if b1 > MAX_STEP_SIZE:
            ##    step.append(MAX_STEP_SIZE)
            ##elif b1 < -MAX_STEP_SIZE:
            ##    step.append(-MAX_STEP_SIZE)
            ##else:
            ##    step.append(b1)
            idx.append(i)
            n_singular -= 1
            b1 = numpy.dot(t[:,i], g) / wi
            step.append(b1)
    # normalize the step_size
    step = numpy.array(step)
    t = max(abs(step))
    if t > MAX_STEP_SIZE:
        step *= MAX_STEP_SIZE / t
    if idx:
        idx = numpy.array(idx)
        x = numpy.dot(vH[idx,:].T.conj(), step)
    else:
        x = numpy.zeros_like(g)
    return x, n_singular

def step_by_eigh_min(h, g, threshold=1e-6):
    ''' h * x = g '''
    if threshold > HESSIAN_THRESHOLD:
        threshold = HESSIAN_THRESHOLD
    elif threshold < 1e-14:
        threshold = 1e-14
    n = h.shape[0]
    h = numpy.array(h)
    w, u = numpy.linalg.eigh(h)
    g = numpy.array(g)
    idx = []
    step = []
    n_singular = w.__len__()
    # w = w - w[0] + threshold # trust region method h -> h-lambda
    for i,wi in enumerate(w):
        # very small eigenvalue will cause unreasonable step size
        # dynamically adjust this threshold to avoid big step
        if wi > threshold:
            ##b1 = STEP_SIZE_CONTR * numpy.dot(u[:,i], g) / (threshold*0 + wi)
            ###if abs(b1) < 3e-1:
            ###    step.append(b1)
            ###    idx.append(i)
            ###    n_singular -= 1
            ### Levenberg algorithm
            ##idx.append(i)
            ##n_singular -= 1
            ##if b1 > MAX_STEP_SIZE:
            ##    step.append(MAX_STEP_SIZE)
            ##elif b1 < -MAX_STEP_SIZE:
            ##    step.append(-MAX_STEP_SIZE)
            ##else:
            ##    step.append(b1)
            #b1 = STEP_SIZE_CONTR * numpy.dot(u[:,i], g) / wi
            b1 = numpy.dot(u[:,i], g) / wi
            idx.append(i)
            n_singular -= 1
            step.append(b1)
        elif wi < -1e-10: # avoid local maxima
            ##idx.append(i)
            ### *.2 to avoid big step
            ##b1 = STEP_SIZE_CONTR * numpy.dot(u[:,i], g) / (1e-9-wi)
            ##if b1 > MAX_STEP_SIZE:
            ##    step.append(MAX_STEP_SIZE)
            ##elif b1 < -MAX_STEP_SIZE:
            ##    step.append(-MAX_STEP_SIZE)
            ##else:
            ##    step.append(b1)
            idx.append(i)
            b1 = .1 * numpy.dot(u[:,i], g) / (1e-9-wi)
            step.append(b1)
        #else:
        # eigenvector assoc. w~0 should be excluded. They correspond to
        # neither symmetric nor anti-symmetric solution
        #    idx.append(i)
        #    wnew.append(1)
    # normalize step size
    if idx:
        step = numpy.array(step)
        t = max(abs(step))
        if t > MAX_STEP_SIZE:
            step *= MAX_STEP_SIZE / t
        idx = numpy.array(idx)
        x = numpy.dot(u[:,idx], step)
    else:
        x = numpy.zeros_like(g)
    return x, n_singular

def aug_hessian_min(h, g, threshold=1e-6):
# IJQC, 54, 329
# JCP, 73, 382
    if threshold > HESSIAN_THRESHOLD:
        threshold = HESSIAN_THRESHOLD
    elif threshold < 1e-14:
        threshold = 1e-14
    n = h.shape[0]+1
    ah = numpy.empty((n,n))
    ah[0,0] = 0
    ah[1:,0] = ah[0,1:] = g
    ah[1:,1:] = h
    w, u = numpy.linalg.eigh(ah)
    shift = w[0]
#TODO should exclude the modes with w~0
    h = h - numpy.eye(n-1) * w[0]
    return step_by_eigh_min(h, g, threshold)

def step_by_eigh_max(h, g, threshold=1e-6):
    ''' h * x = g '''
    if threshold > HESSIAN_THRESHOLD:
        threshold = HESSIAN_THRESHOLD
    elif threshold < 1e-14:
        threshold = 1e-14
    n = h.shape[0]
    h = numpy.array(h) - numpy.eye(n) * threshold * 0
    w, u = numpy.linalg.eigh(h)
    g = numpy.array(g)
    idx = []
    step = []
    n_singular = w.__len__()
    for i,wi in enumerate(w):
        if wi < -threshold:
            idx.append(i)
            n_singular -= 1
            b1 = 5e-1 * numpy.dot(u[:,i], g) / (threshold + wi)
            if b1 > MAX_STEP_SIZE:
                step.append(MAX_STEP_SIZE)
            elif b1 < -MAX_STEP_SIZE:
                step.append(-MAX_STEP_SIZE)
            else:
                step.append(b1)
        elif wi > 1e-10: # avoid local minima
            idx.append(i)
            b1 = 2e-1 * numpy.dot(u[:,i], g) / (-1e-9-wi)
            if b1 > MAX_STEP_SIZE:
                step.append(MAX_STEP_SIZE)
            elif b1 < -MAX_STEP_SIZE:
                step.append(-MAX_STEP_SIZE)
            else:
                step.append(b1)
    if idx:
        idx = numpy.array(idx)
        x = numpy.dot(u[:,idx], numpy.array(step))
    else:
        x = numpy.zeros_like(g)
    return x, n_singular

def symm_trans_mat_for_hermit(n):
    # transformation matrix to remove the antisymmetric component
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

def solve_lineq_for_indep_var(h, g, threshold=1e-12):
    '''use symmetry x = x^T for equation h x = g'''
    threshold = float(threshold)
    nimp = int(numpy.sqrt(h.shape[0]))
    usym = symm_trans_mat_for_hermit(nimp)
    h1 = reduce(numpy.dot, (usym.T.conj(), h, usym))
    g1 = numpy.dot(usym.T, numpy.array(g).flatten())

    # when c[core,vir] ~ 0, h[core,core,core,core] ~ 0,
    # to avoid singularity problem, use SVD solver instead of numpy.linalg.solve
    #x, nsg = step_by_SVD(h1, g1, threshold)
    if FIT_DM_METHOD == FIT_DM_LIEB_FNL:
        x, nsg = step_by_eigh_max(h1, g1, threshold)
    else:
        x, nsg = step_by_eigh_min(h1, g1, threshold)
        #x, nsg = aug_hessian_min(h1, g1, threshold)
    vfit = numpy.dot(usym, x).reshape(nimp,nimp)
    return vfit, nsg

def trans_mat_v_imp_to_dm1(e, c, nocc):
    '''in AO representation, DM1 = X * V'''
    nmo = e.shape[0]
    nvir = nmo - nocc
    nao = c.shape[0]

    eia = 1 / (e[:nocc].reshape(-1,1) - e[nocc:])
    tmpcc = numpy.empty((nmo,nao,nao))
    for i in range(nmo):
        for t in range(nao):
            for u in range(nao):
                tmpcc[i,t,u] = c[t,i] * c[u,i]
    v = tmpcc.reshape(nmo,nao*nao)
    _x = reduce(numpy.dot, (v[nocc:].T, eia.T, v[:nocc]))
    _x = numpy.array(_x).reshape(nao,nao,nao,nao)
    x0 = _x.transpose(0,3,1,2)
    x1 = x0.transpose(1,0,3,2)
    return x0 + x1

##################################################
class ImpPot4ImpBathDM:
    '''Fit potential for density matrix on impurity and bath'''
    def __init__(self):
        self._x = None

    def set_tensor_x(self, e, c, nocc, nimp):
        self._x = trans_mat_v_imp_to_dm1(e, c, nocc)
    def get_tensor_x(self, e, c, nocc, nimp):
        if self._x is None:
            self.set_tensor_x(e, c, nocc, nimp)
        return self._x

#ABORT    def hessian_2e_fac(self, c, eri_full, nocc, nimp):
#ABORT        nmo = c.shape[1]
#ABORT        nvir = nmo - nocc
#ABORT        j_ai = numpy.empty((nimp,nimp,nvir,nocc))
#ABORT        k_ai = numpy.empty((nimp,nimp,nvir,nocc))
#ABORT        k_ia = numpy.empty((nimp,nimp,nvir,nocc))
#ABORT        c_v = numpy.mat(c[:,nocc:])
#ABORT        c_o = numpy.mat(c[:,:nocc])
#ABORT        for r in range(nimp):
#ABORT            for s in range(nimp):
#ABORT                tmp = c_v.H * eri_full[r,s] * c_o
#ABORT                j_ai[r,s] = tmp
#ABORT                tmp = c_v.T * eri_full[r,:,:,s] * c_o
#ABORT                k_ai[r,s] = tmp
#ABORT                k_ia[s,r] = tmp
#ABORT        t = numpy.eye(nimp*nimp) \
#ABORT          - numpy.dot((2 * j_ai - k_ai).reshape(nimp*nimp,nvir*nocc), \
#ABORT                      self._ersai.reshape(nimp*nimp,nvir*nocc).T) \
#ABORT          + numpy.dot((2 * j_ai - k_ia).reshape(nimp*nimp,nvir*nocc), \
#ABORT                      self._ersia.reshape(nimp*nimp,nvir*nocc).T)
#ABORT        return t

    def hessian_approx(self, e, c, nocc, nimp):
        # Gauss-Newton algorithm
        _x = self.get_tensor_x(e, c, nocc, nimp)
        nao = c.shape[0]
        x0 = _x[:nao,:nao,:nimp,:nimp]
        x1 = x0.transpose(1,0,2,3)
        x0 = x0.reshape(nao*nao,nimp*nimp)
        x1 = x1.reshape(nao*nao,nimp*nimp)
        h = numpy.dot(x0.T, x1) * 2
        return h

#FIXME: bug maybe exist in analytical hessian
    def hessian(self, e, c, nocc, nimp, ddm):
        '''analytical hessian'''
        nao = c.shape[0]
        nmo = e.shape[0]
        nvir = nmo - nocc

        eia = 1 / (e[:nocc].reshape(-1,1) - e[nocc:])
        tmpcc = numpy.empty((nao,nao,nvir))
        for r in range(nao):
            for u in range(nao):
                for a in range(nvir):
                    tmpcc[r,u,a] = c[r,a+nocc] * c[u,a+nocc]
        tmpce = numpy.empty((nvir,nocc,nao))
        for a in range(nvir):
            for i in range(nocc):
                for u in range(nao):
                    tmpce[a,i,u] = c[u,i] * eia[i,a]
        _y = numpy.dot(tmpcc.reshape(nao*nao,nvir), \
                       tmpce.reshape(nvir,nocc*nao))
        _y = _y.reshape(nao,nao,nocc,nao).transpose(2,0,1,3)
        tmpcc = numpy.empty((nocc,nimp,nao))
        for i in range(nocc):
            for r in range(nimp):
                for u in range(nao):
                    tmpcc[i,r,u] = c[r,i] * c[u,i]
        tmpce = numpy.empty((nvir,nao,nocc))
        for a in range(nvir):
            for u in range(nao):
                for i in range(nocc):
                    tmpce[a,u,i] = c[u,a+nocc] * eia[i,a]
        _z = numpy.dot(tmpce.reshape(nvir*nao,nocc), \
                       tmpcc.reshape(nocc,nimp*nao))
        _z = _z.reshape(nvir,nao,nimp,nao).transpose(0,2,1,3)
        tmp1 = numpy.dot(_y[:,:nimp].reshape(nocc*nimp,nao*nao), \
                         numpy.array(ddm).flatten())
        tmp1 = numpy.dot(tmp1.reshape(nocc, nimp).T, \
                         _y[:,:nimp,:nimp,:nimp].reshape(nocc,nimp*nimp*nimp))
        tmp2 = numpy.dot(_z.reshape(nvir*nimp,nao*nao), \
                         numpy.array(ddm).flatten())
        tmp2 = numpy.dot(tmp2.reshape(nvir, nimp).T, \
                         _z[:,:,:nimp,:nimp].reshape(nvir,nimp*nimp*nimp))
        tmp2.reshape(nimp,nimp,nimp,nimp).transpose(1,0,2,3)
        h1 = (tmp1 - tmp2).reshape(nimp*nimp,nimp*nimp)
        h1 = h1.reshape(nimp,nimp,nimp,nimp)
        h2 = h1.transpose(1,0,3,2)
        _y1 = _y[:,:nao,:nimp,:nimp].transpose(1,0,2,3)
        _y2 = _y1.transpose(0,1,3,2)
        tmp3 = numpy.dot(ddm, _y1.reshape(nao, nocc*nimp*nimp))
        h3 = tmp3.reshape(nao*nocc,nimp*nimp).T * _y2.reshape(nao*nocc,nimp*nimp)
        h1 = ((h1 + h2).reshape(nimp*nimp,nimp*nimp) + h3) * 2
        h0 = self.hessian_approx(e, c, nocc, nimp)
        h = h0 - h1 - h1.T
        return h

    def gradient(self, e, c, nocc, nimp, ddm):
        nao = c.shape[0]
        _x = self.get_tensor_x(e, c, nocc, nimp)
        x0 = _x[:nao,:nao,:nimp,:nimp].reshape(nao*nao,nimp*nimp)
        g = numpy.dot(numpy.array(ddm).flatten(), x0)
        return g

    def diff_den_mat(self, dm_ref, c, nocc, nimp):
        dm0 = numpy.dot(c[:,:nocc], c[:,:nocc].T.conj())
        return .5*dm_ref - dm0 # *.5 for closed shell

    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp,
                            deepgrad=False):
        e, c = numpy.linalg.eigh(fock0)
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)
        self.set_tensor_x(e, c, nocc, nimp)

        g = self.gradient(e, c, nocc, nimp, ddm)
        if deepgrad:
            x = g
            nsg = -1
        else:
            if False and norm_ddm > 1e-1:
                h = self.hessian(e, c, nocc, nimp, ddm)
            else:
                #log.debug(dev, 'approximate hessian')
                h = self.hessian_approx(e, c, nocc, nimp)
            x, nsg = solve_lineq_for_indep_var(h, g, norm_ddm*.1)
#ABORT        if False and eri_full is not None:
#ABORT            # first order Fock should include the 2e contribution.
#ABORT            # don't know why it doesn't help convergence.
#ABORT            t = self.hessian_2e_fac(c, eri_full, nocc, nimp)
#ABORT            x = numpy.dot(t, x)
        # tune step size by this factor to accelerate convergence
        if LINE_SEARCH:
            dv = numpy.zeros_like(fock0)
            dv[:nimp,:nimp] = x.reshape(nimp,nimp)
            def norm_ddm_method(dv):
                e, c = numpy.linalg.eigh(fock0+dv)
                ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
                return numpy.linalg.norm(ddm)
            x,val = line_search_sharp(dev, norm_ddm_method, dv, \
                                      floating=1e-3, \
                                      title='for fitting imp+bath')
            x = x[:nimp,:nimp]
        else:
            x = x.reshape(nimp,nimp)
        norm_x = numpy.linalg.norm(x)
        log.debug(dev, 'homo = %.9g, lumo = %.9g', \
                  e[nocc-1], e[nocc])
        log.debug(dev, 'norm(ddm) = %.6g, norm(dv) = %.6g, singular = %d', \
                  rm_ddm, norm_x, nsg)
        return x

#ABORT    def hessian_dbg(self, e, c, nocc, nimp):
#ABORT        ndm = c.shape[0]
#ABORT        nvir = e.shape[0] - nocc
#ABORT        eai = 1 / numpy.array(numpy.mat(e[nocc:]).T - numpy.mat(e[:nocc])).flatten()
#ABORT        def vfit_to_den_mat(v):
#ABORT            vai = numpy.mat(c[:nimp,nocc:]).H * v.reshape(nimp,nimp) \
#ABORT                    * c[:nimp,:nocc]
#ABORT            dmai = (-numpy.array(vai).flatten() * eai).reshape(nvir,nocc)
#ABORT            dmia = dmai.T.conj()
#ABORT            dm = numpy.mat(c[:,nocc:]) * dmai * numpy.mat(c[:,:nocc]).H \
#ABORT                    + numpy.mat(c[:,:nocc]) * dmia * numpy.mat(c[:,nocc:]).H
#ABORT            return numpy.array(dm).flatten()
#ABORT        def vfit_to_den_mat_T(dm):
#ABORT            dmai = numpy.mat(c[:,nocc:]).T * dm.reshape(ndm,ndm) \
#ABORT                    * c[:,:nocc].conj()
#ABORT            vai = (-numpy.array(dmai).flatten() * eai).reshape(nvir,nocc)
#ABORT            via = vai.T.conj()
#ABORT            v = numpy.mat(c[:nimp,nocc:]).conj() * vai * c[:nimp,:nocc].T \
#ABORT                    + numpy.mat(c[:nimp,:nocc]).conj() * via * c[:nimp,nocc:].T
#ABORT            return numpy.array(v).flatten()
#ABORT        return sparse.LinearOperator((ndm*ndm,nimp*nimp), vfit_to_den_mat, \
#ABORT                                     rmatvec=vfit_to_den_mat_T)
#ABORT
#ABORT    def generate_pot_dbg(self, dev, dm_ref, fock0, nocc, nimp):
#ABORT        '''based on the given series of {v_k}, generate v_{k+1}'''
#ABORT        e, c = numpy.linalg.eigh(fock0)
#ABORT        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
#ABORT        norm_ddm = numpy.linalg.norm(ddm)
#ABORT        self.set_tensor_x(e, c, nocc, nimp)
#ABORT
#ABORT        if True or norm_ddm < 1e-4:
#ABORT            h = self.hessian_approx(e, c, nocc, nimp)
#ABORT            g = self.gradient(e, c, nocc, nimp, ddm)
#ABORT            x, nsg = solve_lineq_for_indep_var(h, g, norm_ddm*.1)
#ABORT        else:
#ABORT            h = self.hessian_dbg(e, c, nocc, nimp)
#ABORT
#ABORT            if ddm.shape[0] == 1:
#ABORT                x = h.rmatvec(numpy.array(ddm).flatten())
#ABORT            else:
#ABORT                x = sparse.lsqr(h, numpy.array(ddm).flatten(), \
#ABORT                                atol=1e-5, show=False)[0]
#ABORT
#ABORT        x = x.reshape(nimp,nimp) * STEP_SIZE_CONTR
#ABORT        norm_x = numpy.linalg.norm(x)
#ABORT        log.debug(dev, 'norm(ddm) = %.6g, norm(v) = %.6g', \
#ABORT                  norm_ddm, norm_x)
#ABORT        return x

    def generate_pot(self, dev, dm_ref, fock0, nocc, nimp):
        if 1:
            return self.generate_pot_approx(dev, dm_ref, fock0, nocc, nimp)
        else:
            return self.generate_pot_dbg(dev, dm_ref, fock0, nocc, nimp)


##################################################
# impurity
class ImpPot4ImpDM(ImpPot4ImpBathDM):
    '''Fit potential for density matrix on impurity site'''
    def __init__(self):
        ImpPot4ImpBathDM.__init__(self)

    def diff_den_mat(self, dm_ref, c, nocc, nimp):
        dm0 = numpy.dot(c[:,:nocc], c[:,:nocc].T.conj())
        return (.5*dm_ref - dm0)[:nimp,:nimp] # *.5 for closed shell

    def gradient(self, e, c, nocc, nimp, ddm):
        return ImpPot4ImpBathDM.gradient(self, e, c[:nimp,:], nocc, nimp, ddm)

    def hessian_approx(self, e, c, nocc, nimp):
        return ImpPot4ImpBathDM.hessian_approx(self, e, c[:nimp,:], nocc, nimp)

    def hessian(self, e, c, nocc, nimp, ddm):
        return ImpPot4ImpBathDM.hessian(self, e, c[:nimp,:], nocc, nimp, ddm)

#ABORT    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp):
#ABORT        '''based on the given series of {v_k}, generate v_{k+1}'''
#ABORT        e, c = numpy.linalg.eigh(fock0)
#ABORT        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
#ABORT        norm_ddm = numpy.linalg.norm(ddm)
#ABORT        self.set_tensor_x(e, c, nocc, nimp)
#ABORT
#ABORT        if False and norm_ddm > 1e-2:
#ABORT            #h = self.hessian(e, c, nocc, nimp, ddm)
#ABORT            log.debug(dev, 'linearing search with gradient descent')
#ABORT            _x = self.get_tensor_x(e, c, nocc, nimp)
#ABORT            _x = _x[:nimp,:nimp,:nimp,:nimp].reshape(nimp*nimp,nimp*nimp)
#ABORT            x, nsg = step_by_SVD(_x, numpy.array(ddm).flatten(), \
#ABORT                                  norm_ddm*.1)
#ABORT        else:
#ABORT            #log.debug(dev, 'approximate hessian')
#ABORT            h = self.hessian_approx(e, c, nocc, nimp)
#ABORT            g = self.gradient(e, c, nocc, nimp, ddm)
#ABORT
#ABORT            x, nsg = solve_lineq_for_indep_var(h, g, norm_ddm*.1)
#ABORT        x = x.reshape(nimp,nimp) * STEP_SIZE_CONTR
#ABORT        norm_x = numpy.linalg.norm(x)
#ABORT        log.debug(dev, 'norm(ddm) = %.6g, norm(dv) = %.6g, singular = %d', \
#ABORT                  norm_ddm, norm_x, nsg)
#ABORT        return x
#ABORT
#ABORT    def hessian_dbg(self, e, c, nocc, nimp):
#ABORT        return ImpPot4ImpBathDM.hessian_dbg(self, e, c[:nimp,:], nocc, nimp)


##################################################
# fitting with constraints: diagonal term on impurity sites or trace of
# impurity sites.
def solve_lagrangian(h, g, x_diag, ddm, threshold=1e-12):
    threshold = float(threshold)
    if threshold > 1e-10:
        threshold = 1e-10
    elif threshold < 1e-14:
        threshold = 1e-14

    nimp = int(numpy.sqrt(h.shape[0]))
    usym = symm_trans_mat_for_hermit(nimp)

    x_sym = numpy.dot(usym.T, x_diag.T)

    h1 = reduce(numpy.dot, (usym.T.conj(), h, usym))
    e, v = numpy.linalg.eigh(h1)
    vx = numpy.dot(v[:,abs(e)>threshold].T, x_sym)
    tmp = vx.T * (1/e[abs(e)>threshold])
    ha = numpy.dot(tmp, vx)

    g1 = numpy.dot(usym.T, numpy.array(g).flatten())
    vg = numpy.dot(v[:,abs(e)>threshold].T, g1)

    if ha.shape[0] == 1:
        dne = ddm.trace()
        #dynm_thr = 1e-4#min(dne*1e-2, 1e-4)
        dynm_thr = 1e-12#max(numpy.linalg.norm(ddm), 1e-7)
        if dne > dynm_thr:
            ga = numpy.array(ddm).trace() - dynm_thr - numpy.dot(tmp, vg)
        elif dne < -dynm_thr:
            ga = numpy.array(ddm).trace() + dynm_thr - numpy.dot(tmp, vg)
        else:
            ga = 0
        if abs(ha) > 1e-12:
            alpha = float(ga/ha)
        else:
            alpha = 0
    else:
        ga = numpy.array(ddm).diagonal() - numpy.dot(tmp, vg)
        try:
            alpha = numpy.linalg.solve(ha, ga)
        except:
            alpha, nsg = solve_lineq_by_SVD(ha, ga, 1e-12)
            #log.debug(dev, '    lagrangian singularity = %d', nsg)
            if nsg > 0:
                print 'lagrangian singularity = %d' % nsg
    return alpha

def solve_lagrangian1(h, g, x_diag, ddm, threshold=1e-12):
    threshold = float(threshold)
    if threshold > 1e-10:
        threshold = 1e-10
    elif threshold < 1e-14:
        threshold = 1e-14

    nimp = int(numpy.sqrt(h.shape[0]))
    usym = symm_trans_mat_for_hermit(nimp)

    x_sym = numpy.dot(usym.T, x_diag.T)

    h1 = reduce(numpy.dot, (usym.T.conj(), h, usym))
    g1 = numpy.dot(usym.T, numpy.array(g).flatten())

    dne = ddm.trace()
    #dynm_thr = 1e-4#min(dne*1e-2, 1e-4)
    dynm_thr = 1e-12#max(numpy.linalg.norm(ddm), 1e-7)
    if dne > dynm_thr:
        ga = numpy.array(ddm).trace() - dynm_thr - numpy.dot(tmp, vg)
    elif dne < -dynm_thr:
        ga = numpy.array(ddm).trace() + dynm_thr - numpy.dot(tmp, vg)
    else:
        ga = 0
    if abs(ha) > 1e-12:
        alpha = float(ga/ha)
    else:
        alpha = 0
    return alpha

class ImpPot4ImpDM_DiagConstr(ImpPot4ImpDM):
    def __init__(self):
        ImpPot4ImpDM.__init__(self)

    def step_with_lagrangian(self, dev, h, g, ddm, threshold=1e-12):
        # FIXME: increase alpha to make sure the constraints domain the
        # optimization
        # first few steps optimize the DM difference, then use big alpha to
        # make sure diagonal diff meets 0
        nimp = int(numpy.sqrt(h.shape[0]))
        _x = self.get_tensor_x(None, None, None, None)
        x_diag = _x.diagonal()[:nimp,:nimp,:nimp].reshape(nimp,nimp*nimp)
        alpha = solve_lagrangian(h, g, x_diag, ddm, threshold)
        g += numpy.dot(alpha, x_diag.reshape(nimp,nimp*nimp))
        vfit, nsg = solve_lineq_for_indep_var(h, g, threshold)
        return vfit, nsg

    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp):
        e, c = numpy.linalg.eigh(fock0)
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)
        self.set_tensor_x(e, c, nocc, nimp)

        h = self.hessian_approx(e, c, nocc, nimp)
        g = self.gradient(e, c, nocc, nimp, ddm)
        x, nsg = self.step_with_lagrangian(dev, h, g, ddm, ddm.trace())
        x = x.reshape(nimp,nimp)
        norm_x = numpy.linalg.norm(x)
        log.debug(dev, 'homo = %.9g, lumo = %.9g', \
                  e[nocc-1], e[nocc])
        log.debug(dev, 'constraint on diag terms. norm(ddm) = %.6g, ' \
                  'norm_diag = %.6g, norm(dv) = %.6g, singular = %d', \
                  norm_ddm, numpy.linalg.norm(ddm.diagonal()[:nimp]), norm_x, nsg)
        return x

class ImpPot4ImpDM_NeleConstr(ImpPot4ImpDM_DiagConstr):
    def step_with_lagrangian(self, dev, h, g, ddm, threshold=1e-12):
        #return ImpPot4ImpBathDM_NeleConstr(self, dev, h, g, ddm, threshold)
        nimp = int(numpy.sqrt(h.shape[0]))
        _x = self.get_tensor_x(None, None, None, None)
        x_diag = reduce(lambda x,y: x+y, _x.diagonal()[:nimp,:nimp,:nimp])
        x_diag = x_diag.flatten()
        alpha = solve_lagrangian(h, g, x_diag, ddm, threshold)
        g = g + x_diag.flatten() * alpha
        vfit, nsg = solve_lineq_for_indep_var(h, g, threshold)
        return vfit, nsg

    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp):
        e, c = numpy.linalg.eigh(numpy.array(fock0))
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)
        self.set_tensor_x(e, c, nocc, nimp)

        h = self.hessian_approx(e, c, nocc, nimp)
        g = self.gradient(e, c, nocc, nimp, ddm)
        x, nsg = self.step_with_lagrangian(dev, h, g, ddm, ddm.trace())
        x = x.reshape(nimp,nimp)
        norm_x = numpy.linalg.norm(x)
        log.debug(dev, 'homo = %.9g, lumo = %.9g', \
                  e[nocc-1], e[nocc])
        log.debug(dev, 'constraints on nelec_imp. norm(ddm) = %.6g, ' \
                  'norm_diag = %.6g, norm(dv) = %.6g, singular = %d', \
                  norm_ddm, abs(ddm.trace()), norm_x, nsg)
        return x

class ImpPot4ImpBathDM_DiagConstr(ImpPot4ImpBathDM):
    def __init__(self):
        ImpPot4ImpBathDM.__init__(self)

    def step_with_lagrangian(self, dev, h, g, ddm, threshold=1e-12):
        import fitter
        nimp = int(numpy.sqrt(h.shape[0]))
        _x = self.get_tensor_x(None, None, None, None)
        x_diag = _x.diagonal()[:nimp,:nimp,:nimp].reshape(nimp,nimp*nimp)
        usym = symm_trans_mat_for_hermit(nimp)
        h1 = reduce(numpy.dot, (usym.T.conj(), h, usym))
        g1 = numpy.dot(usym.T, g)
        x_diag_sym = numpy.dot(x_diag, usym)

        vfit, nsg = fitter.lin_constrain(h1, g1, x_diag_sym, \
                                         ddm.diagonal()[:nimp], 1e-12)
        vfit = numpy.dot(usym,vfit)
        return vfit, nsg

    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp):
        e, c = numpy.linalg.eigh(fock0)
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)
        self.set_tensor_x(e, c, nocc, nimp)

        h = self.hessian_approx(e, c, nocc, nimp)
        g = self.gradient(e, c, nocc, nimp, ddm)
        x, nsg = self.step_with_lagrangian(dev, h, g, ddm, ddm.trace())
        vfit = numpy.zeros_like(fock0)
        vfit[:nimp,:nimp] = x.reshape(nimp,nimp)
        norm_x = numpy.linalg.norm(vfit)
        #log.debug(dev, 'homo = %.9g, lumo = %.9g', \
        #          e[nocc-1], e[nocc])
        e, c = numpy.linalg.eigh(fock0+vfit)
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)
        log.debug(dev, 'constraint on diag terms. norm(ddm) = %.6g, ' \
                  'norm_diag = %.6g, norm(dv) = %.6g, singular = %d', \
                  norm_ddm, numpy.linalg.norm(ddm.diagonal()[:nimp]), norm_x, nsg)
        return vfit

class ImpPot4ImpBathDM_NeleConstr(ImpPot4ImpBathDM_DiagConstr):
    def step_with_lagrangian(self, dev, h, g, ddm, threshold=1e-12):
        nimp = int(numpy.sqrt(h.shape[0]))
        _x = self.get_tensor_x(None, None, None, None)
        x_diag = reduce(lambda x,y: x+y, _x.diagonal()[:nimp,:nimp,:nimp])
        x_diag = x_diag.flatten()
        alpha = solve_lagrangian(h, g, x_diag, ddm, threshold)
        g = g + x_diag.flatten() * alpha
        vfit, nsg = solve_lineq_for_indep_var(h, g, threshold)
        return vfit, nsg

    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp):
        e, c = numpy.linalg.eigh(fock0)
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)
        self.set_tensor_x(e, c, nocc, nimp)

        h = self.hessian_approx(e, c, nocc, nimp)
        g = self.gradient(e, c, nocc, nimp, ddm)
        x, nsg = self.step_with_lagrangian(dev, h, g, ddm, ddm.trace())
        x = x.reshape(nimp,nimp)
        norm_x = numpy.linalg.norm(x)
        log.debug(dev, 'homo = %.9g, lumo = %.9g', \
                  e[nocc-1], e[nocc])
        log.debug(dev, 'constraints on nelec_imp. norm(ddm) = %.6g, ' \
                  'norm_diag = %.6g, norm(dv) = %.6g, singular = %d', \
                  norm_ddm, abs(ddm.trace()), norm_x, nsg)
        return x

##################################################
# Diagonal term of fitting potential -> diagonal term of density matrix
class ImpPot4ImpDiag:
    '''Fit diagonal term of one-electron potential for given Fock operator to
    minimize the distance of the diagonal terms of density matrix on impurity
    between HF and fragment FCI'''
    def __init__(self):
        pass

    def get_tensor_x(self, e, c, nocc, nimp):
        nmo = e.shape[0]
        nvir = nmo - nocc

        eai = 1 / (e[nocc:].reshape(-1,1) - e[:nocc])
        ersai = numpy.empty((nvir,nocc))
        _x = numpy.empty((nimp,nimp))
        c_imp = c[:nimp]
        for t in range(nimp):
            #ersai = -numpy.array(numpy.mat(c_imp[r,nocc:]).T * c_[r,:nocc]) \
            #        * numpy.array(eai)
            for a in range(nvir):
                for i in range(nocc):
                    ersai[a,i] =-c_imp[t,nocc+a] * c_imp[t,i] * eai[a,i]
            #tmp = reduce(numpy.dot, (c_imp[:,nocc:], ersai,\
            #                         c_imp[:,:nocc].T.conj()))
            #_x[t] = tmp.diagonal()
            tmp = numpy.dot(c_imp[:,nocc:], ersai)
            for u in range(nimp):
                _x[t,u] = numpy.dot(tmp[u],c_imp[u,:nocc])
        return _x

    def hessian(self, e, c, nocc, nimp):
        _x = self.get_tensor_x(e, c, nocc, nimp)
        wx = _x * 2
        h = numpy.dot(wx, wx.T)
        return h * 2

    def gradient(self, e, c, nocc, nimp, ddm):
        _x = self.get_tensor_x(e, c, nocc, nimp)
        g = numpy.dot(ddm, _x.T) * 2
        return g

    def diff_den_mat(self, dm_ref, c, nocc, nimp):
        dm0 = numpy.dot(c[:,:nocc], c[:,:nocc].T.conj())
        return numpy.diagonal(.5*dm_ref - dm0)[:nimp]

    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp):
        e, c = numpy.linalg.eigh(fock0)
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)

        h = self.hessian(e, c, nocc, nimp)
        g = self.gradient(e, c, nocc, nimp, ddm)

        # In very rare situation, the occupied or virtual orbitals ~ 0, the
        # hessian is singular. random potential can possibly fix this problem.
        #nvir = e.shape[0] - nocc
        #tt = numpy.empty((nocc,nvir,nimp))
        #for i in range(nocc):
        #    for a in range(nvir):
        #        for x in range(nimp):
        #            tt[i,a,x] = c[x,a+nocc] * c[x,i]
        #print numpy.linalg.svd(tt.reshape(nocc*nvir,nimp))[1]

        #x, nsg = step_by_SVD(h, numpy.array(g).flatten(), norm_ddm*.1)
        x, nsg = step_by_eigh_min(h, numpy.array(g).flatten(), norm_ddm*.1)
        if LINE_SEARCH:
            dv = numpy.zeros_like(fock0)
            dv[:nimp,:nimp] = numpy.diag(x) * 4
            def norm_ddm_method(dv):
                e, c = numpy.linalg.eigh(fock0+dv)
                ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
                return numpy.linalg.norm(ddm)
            x,val = line_search_sharp(dev, norm_ddm_method, dv, \
                                      floating=1e-3, \
                                      title='for fitting imp-diag')
            x = x.diagonal()[:nimp]

        norm_x = numpy.linalg.norm(x)
        log.debug(dev, 'homo = %.9g, lumo = %.9g', \
                  e[nocc-1], e[nocc])
        log.debug(dev, 'norm(ddm) = %.6g, norm(dv) = %.6g, singular = %d', \
                  rm_ddm, norm_x, nsg)
        return numpy.diag(x)

#ABORT    def generate_pot_finitie(self, dev, dm_ref, fock0, nocc, nimp):
#ABORT        '''based on the given series of {v_k}, generate v_{k+1}'''
#ABORT        e, c = numpy.linalg.eigh(fock0)
#ABORT        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
#ABORT        norm_ddm = numpy.linalg.norm(ddm)
#ABORT
#ABORT        h = numpy.empty((nimp*(nimp+1)/2, nimp))
#ABORT        k = 0
#ABORT        for i in range(nimp):
#ABORT            for j in range(i+1):
#ABORT                fock1 = fock0.copy()
#ABORT                fock1[i,j] += 1e-2
#ABORT                if i != j:
#ABORT                    fock1[j,i] += 1e-2
#ABORT                e, c = numpy.linalg.eigh(fock1)
#ABORT                h[k] = self.diff_den_mat(dm_ref, c, nocc, nimp) * 1e2
#ABORT                k += 1
#ABORT
#ABORT        #v, nsg = step_by_SVD(h.T, ddm, norm_ddm*.1)
#ABORT        v, nsg = step_by_eigh_min(h.T, ddm, norm_ddm*.1)
#ABORT        x = numpy.empty((nimp,nimp))
#ABORT        k = 0
#ABORT        for i in range(nimp):
#ABORT            for j in range(i+1):
#ABORT                x[i,j] = x[j,i] = v[k]
#ABORT                k += 1
#ABORT
#ABORT        norm_x = numpy.linalg.norm(x)
#ABORT        log.debug(dev, 'norm(ddm) = %.6g, norm(dv) = %.6g', \
#ABORT                  norm_ddm, norm_x))
#ABORT        #fock1 = fock0.copy()
#ABORT        #fock1[:nimp,:nimp] += x
#ABORT        #e, c = numpy.linalg.eigh(fock1)
#ABORT        #ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
#ABORT        #norm_ddm = numpy.linalg.norm(ddm)
#ABORT        #log.debug(dev, 'norm(ddm) = %.6g,', norm_ddm)
#ABORT        return x

    def generate_pot(self, dev, dm_ref, fock0, nocc, nimp):
        if 1:
            return self.generate_pot_approx(dev, dm_ref, fock0, nocc, nimp)
        else:
            return self.generate_pot_finitie(dev, dm_ref, fock0, nocc, nimp)



##################################################
# density matrix of impurity and off-diagonal terms of impurity-bath
class ImpPotNoBathBlock(ImpPot4ImpBathDM):
    def diff_den_mat(self, dm_ref, c, nocc, nimp):
        dm0 = numpy.dot(c[:,:nocc], c[:,:nocc].T.conj())
        ddm = .5*dm_ref - dm0 # *.5 for closed shell
        ddm[nimp:,nimp:] = 0
        return ddm

    def get_tensor_x(self, e, c, nocc, nimp):
        nao = c.shape[0]
        _x = ImpPot4ImpBathDM.get_tensor_x(self, e, c, nocc, nao)
        _x[nimp:,nimp:] = 0
        _x[:,:,nimp:,nimp:] = 0
        return _x

    def hessian_approx(self, e, c, nocc, nimp):
        # Gauss-Newton algorithm
        _x = self.get_tensor_x(e, c, nocc, nimp)
        nao = c.shape[0]
        x0 = _x[:nao,:nao,:nao,:nao]
        x1 = x0.transpose(1,0,2,3)
        x0 = x0.reshape(nao*nao,nao*nao)
        x1 = x1.reshape(nao*nao,nao*nao)
        h = numpy.dot(x0.T, x1) * 2
        return h

    def gradient(self, e, c, nocc, nimp, ddm):
        nao = c.shape[0]
        _x = self.get_tensor_x(e, c, nocc, nimp)
        g = numpy.dot(_x.reshape(nao*nao,nao*nao), \
                      ddm.flatten().T).reshape(nao,nao)
        g[nimp:,nimp:] = 0
        return numpy.array(g + g.T.conj()).flatten()
        #x0 = _x[:nao,:nao,:nimp,:nimp].reshape(nao*nao,nimp*nimp)
        #g = numpy.dot(numpy.array(ddm).flatten(), x0)
        #g[nimp:,nimp:] = 0
        #return g

    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp):
        e, c = numpy.linalg.eigh(fock0)
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)

        h = self.hessian_approx(e, c, nocc, nimp)
        g = self.gradient(e, c, nocc, nimp, ddm)

        #x, nsg = step_by_SVD(h, numpy.array(g).flatten(), 1e-10)
        x, nsg = step_by_eigh_min(h, numpy.array(g).flatten(), norm_ddm*.1)
        nao = c.shape[0]
        x = x.reshape(nao,nao)
        norm_x = numpy.linalg.norm(x)
        log.debug(dev, 'homo = %.9g, lumo = %.9g', \
                  e[nocc-1], e[nocc])
        log.debug(dev, 'norm(ddm) = %.6g, norm(dv) = %.6g, singular = %d', \
                  rm_ddm, norm_x, nsg)
        return x

#######################################
# fit potential on impurity sites for the density matrix except the elements
# on impurity sites
class ImpPot4NoImpDm(ImpPot4ImpBathDM):
    def diff_den_mat(self, dm_ref, c, nocc, nimp):
        dm0 = numpy.dot(c[:,:nocc], c[:,:nocc].T.conj())
        ddm = .5*dm_ref - dm0 # *.5 for closed shell
        nao = c.shape[0]
        ddm_lst = []
        for i in range(nimp, nao):
            for j in range(i+1):
                ddm_lst.append(ddm[i,j])
        return numpy.array(ddm_lst)

    def set_tensor_x(self, e, c, nocc, nimp):
        _x = trans_mat_v_imp_to_dm1(e, c, nocc)
        nao = c.shape[0]
        x_lst = []
        for i in range(nimp, nao):
            for j in range(i+1):
                x_lst.append(_x[i,j,:nimp,:nimp])
        self._x = numpy.array(x_lst)

    def hessian_approx(self, e, c, nocc, nimp):
        _x = self.get_tensor_x(e, c, nocc, nimp)
        nd = _x.shape[0]
        x0 = _x.reshape(nd, nimp*nimp)
        h = numpy.dot(x0.T, x0)
        return h + h.T

    def gradient(self, e, c, nocc, nimp, ddm):
        _x = self.get_tensor_x(e, c, nocc, nimp)
        nd = _x.shape[0]
        x0 = _x.reshape(nd, nimp*nimp)
        g = numpy.dot(ddm, x0)
        return g

    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp):
        e, c = numpy.linalg.eigh(fock0)
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)
        self.set_tensor_x(e, c, nocc, nimp)

        h = self.hessian_approx(e, c, nocc, nimp)
        g = self.gradient(e, c, nocc, nimp, ddm)

        x, nsg = solve_lineq_for_indep_var(h, g, norm_ddm*.1)
        x = x.reshape(nimp,nimp)
        norm_x = numpy.linalg.norm(x)
        log.debug(dev, 'homo = %.9g, lumo = %.9g', \
                  e[nocc-1], e[nocc])
        log.debug(dev, 'norm(ddm) = %.6g, norm(dv) = %.6g, singular = %d', \
                  rm_ddm, norm_x, nsg)
        return x

class ImpPot4NoImpDm_NeleConstr(ImpPot4NoImpDm):
    def step_with_lagrangian(self, dev, h, g, ddm, diag_idx, threshold=1e-12):
        _x = self.get_tensor_x(None, None, None, None)
        nimp = int(numpy.sqrt(h.shape[0]))
        x_diag = reduce(lambda x,y: x+y, _x[diag_idx])
        x_diag = x_diag.flatten()
        alpha = solve_lagrangian(h, g, x_diag, numpy.diag(ddm[diag_idx]), \
                                 threshold)
        g = g + x_diag.flatten() * alpha
        vfit, nsg = solve_lineq_for_indep_var(h, g, threshold)
        return vfit, nsg

    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp):
        e, c = numpy.linalg.eigh(fock0)
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)
        self.set_tensor_x(e, c, nocc, nimp)

        nao = c.shape[0]
        diag_idx = []
        n = 0
        for i in range(nimp, nao):
            for j in range(i+1):
                if i == j:
                    diag_idx.append(n)
                n += 1
        diag_idx = numpy.array(diag_idx)

        h = self.hessian_approx(e, c, nocc, nimp)
        g = self.gradient(e, c, nocc, nimp, ddm)
        x, nsg = self.step_with_lagrangian(dev, h, g, ddm, diag_idx, \
                                           ddm[diag_idx].sum())
        x = x.reshape(nimp,nimp)
        norm_x = numpy.linalg.norm(x)
        log.debug(dev, 'homo = %.9g, lumo = %.9g', \
                  e[nocc-1], e[nocc])
        log.debug(dev, 'constraints on nelec_imp. norm(ddm) = %.6g, ' \
                  'norm_diag = %.6g, norm(dv) = %.6g, singular = %d', \
                  norm_ddm, abs(ddm[diag_idx].sum()), norm_x, nsg)
        return x

class ImpPot4NoImpDm_DiagConstr(ImpPot4NoImpDm):
    def step_with_lagrangian(self, dev, h, g, ddm, e,c,nocc,nimp,threshold=1e-12):
        _x = trans_mat_v_imp_to_dm1(e, c, nocc)
        x_diag = _x.diagonal()[:nimp,:nimp,:nimp].reshape(nimp,nimp*nimp)
        alpha = solve_lagrangian(h, g, x_diag, ddm, threshold)
        g += numpy.dot(alpha.flatten(), x_diag)
        vfit, nsg = solve_lineq_for_indep_var(h, g, threshold)
        return vfit, nsg

    def generate_pot_approx(self, dev, dm_ref, fock0, nocc, nimp):
        e, c = numpy.linalg.eigh(fock0)
        ddm = self.diff_den_mat(dm_ref, c, nocc, nimp)
        norm_ddm = numpy.linalg.norm(ddm)
        self.set_tensor_x(e, c, nocc, nimp)

        dm0 = numpy.dot(c[:nimp,:nocc], c[:nimp,:nocc].T.conj())
        ddm_imp = numpy.array(.5*dm_ref[:nimp,:nimp] - dm0)

        h = self.hessian_approx(e, c, nocc, nimp)
        g = self.gradient(e, c, nocc, nimp, ddm)
        x, nsg = self.step_with_lagrangian(dev, h, g, ddm_imp, \
                                           e,c,nocc,nimp,ddm_imp.trace())
        x = x.reshape(nimp,nimp)
        norm_x = numpy.linalg.norm(x)
        log.debug(dev, 'homo = %.9g, lumo = %.9g', \
                  e[nocc-1], e[nocc])
        log.debug(dev, 'constraints on nelec_imp. norm(ddm) = %.6g, ' \
                  'norm_diag = %.6g, norm(dv) = %.6g, singular = %d', \
                  norm_ddm, numpy.linalg.norm(ddm_imp.diagonal()), norm_x, nsg)
        return x


def find_emb_potential_nobath(dev, dm_ref, fock0, nocc, nimp):
    '''find pseudo potential v which minimize |dm_ref - dm(fock0+v)|^2'''
    fitpot = ImpPotNoBathBlock()

    e, c = numpy.linalg.eigh(fock0)
    ddm = fitpot.diff_den_mat(dm_ref, c, nocc, nimp)
    norm_ddm0 = numpy.linalg.norm(ddm)
    log.debug(dev, 'before fitting, norm(dm_ref-dm) = %.6g', norm_ddm0)

    iter_count = 0
    norm_dv = 1
    vfit = numpy.zeros_like(fock0)
    while iter_count < MAX_ITER and norm_dv > CONV_THRESHOLD:
        dv = fitpot.generate_pot(dev, dm_ref, fock0+vfit, nocc, nimp)
        vfit += dv
        norm_dv = numpy.linalg.norm(dv)
        iter_count += 1
    log.debug(dev, 'fitting iteration = %d', iter_count)

    e, c = numpy.linalg.eigh(fock0+vfit)
    ddm = fitpot.diff_den_mat(dm_ref, c, nocc, nimp)
    norm_ddm = numpy.linalg.norm(ddm)
    log.debug(dev, 'after fitting, norm(dm_ref-dm) = %.6g', norm_ddm)

    if  norm_ddm0 < norm_ddm:
        #log.stdout(dev, 'optimized fitting v was not found')
        log.warn(dev, 'optimized fitting v was not found')
    return vfit

def find_emb_potential(dev, dm_ref, fock0, nocc, nimp):
    '''find pseudo potential v which minimize |dm_ref - dm(fock0+v)|^2'''
    if FIT_DM_METHOD == FIT_DM_IMP_AND_BATH:
        fitpot = ImpPot4ImpBathDM()
    elif FIT_DM_METHOD == FIT_DM_IMP_ONLY:
        fitpot = ImpPot4ImpDM()
    elif FIT_DM_METHOD == FIT_DM_IMP_ONLY_DIAG_CONSTRAINT:
        fitpot = ImpPot4ImpDM_DiagConstr()
    elif FIT_DM_METHOD == FIT_DM_IMP_ONLY_NELE_CONSTRAINT:
        fitpot = ImpPot4ImpDM_NeleConstr()
    elif FIT_DM_METHOD == FIT_DM_IMP_BATH_NELE_CONSTRAINT:
        fitpot = ImpPot4ImpBathDM_NeleConstr()
    elif FIT_DM_METHOD == FIT_DM_IMP_DIAG:
        fitpot = ImpPot4ImpDiag()
    elif FIT_DM_METHOD == FIT_DM_NO_IMP_BLOCK:
        fitpot = ImpPot4NoImpDm()
    elif FIT_DM_METHOD == FIT_DM_NO_IMP_NELE_CONSTRAINT:
        fitpot = ImpPot4NoImpDm_NeleConstr()
    elif FIT_DM_METHOD == FIT_DM_NO_IMP_DIAG_CONSTRAINT:
        fitpot = ImpPot4NoImpDm_DiagConstr()
    elif FIT_DM_METHOD == FIT_DM_NO_BATH_BLOCK:
        return find_emb_potential_nobath(dev, dm_ref, fock0, nocc, nimp)
    elif FIT_DM_METHOD == FIT_DM_LIEB_FNL:
        return find_pseudo_v_for_lieb_fnl(dev, dm_ref, fock0, nocc, nimp)
    elif FIT_DM_METHOD == FIT_DM_IMP_BATH_DIAG_CONSTRAINT:
        fitpot = ImpPot4ImpBathDM_DiagConstr()
    else:
        raise KeyError('unkonw FIT_DM_METHOD')

    e, c = numpy.linalg.eigh(fock0)
    ddm = fitpot.diff_den_mat(dm_ref, c, nocc, nimp)
    norm_ddm0 = numpy.linalg.norm(ddm)
    log.debug(dev, 'before fitting, norm(dm_ref-dm) = %.6g', norm_ddm0)

    iter_count = 0
    norm_dv = 1
    vfit = numpy.zeros_like(fock0)
    #import scf.diis
    #adiis = scf.diis.DIIS(dev)
    while iter_count < MAX_ITER and norm_dv > CONV_THRESHOLD:
        dv = fitpot.generate_pot(dev, dm_ref, fock0+vfit, nocc, nimp)
        #dv,norm_ddm = fitpot.generate_pot(dev, dm_ref, fock0+vfit, nocc, nimp)
        #if norm_ddm - norm_ddm0 > 1e-3:
        #    print 'warn', iter_count, norm_ddm0, norm_ddm
        #    dv = 0
        #norm_ddm0 = norm_ddm
        vfit[:nimp,:nimp] += dv[:nimp,:nimp] * STEP_SIZE_CONTR
        #vfit = adiis.update(vfit)
        norm_dv = numpy.linalg.norm(dv)
        iter_count += 1
    log.debug(dev, 'fitting iteration = %d', iter_count)

    e, c = numpy.linalg.eigh(fock0+vfit)
    ddm = fitpot.diff_den_mat(dm_ref, c, nocc, nimp)
    norm_ddm = numpy.linalg.norm(ddm)
# should use true hessian instead
#ABORT    if norm_ddm > CONV_THRESHOLD:
#ABORT        import scf.diis
#ABORT        adiis = scf.diis.DIIS(dev)
#ABORT        vfit = adiis.update(vfit)
#ABORT        iter_count = 0
#ABORT        norm_ddm = norm_ddm0
#ABORT        while iter_count < MAX_ITER and norm_ddm > CONV_THRESHOLD:
#ABORT            #dv = fitpot.generate_pot(dev, dm_ref, fock0+vfit, nocc, nimp)
#ABORT            e, c = numpy.linalg.eigh(fock0+vfit)
#ABORT            ddm = fitpot.diff_den_mat(dm_ref, c, nocc, nimp)
#ABORT            norm_ddm = numpy.linalg.norm(ddm)
#ABORT            dv = -fitpot.gradient(e, c, nocc, nimp, ddm)
#ABORT            vfit[:nimp,:nimp] += numpy.diag(dv)
#ABORT            #vfit[:nimp,:nimp] += dv.reshape(nimp,nimp)
#ABORT            #vfit = adiis.update(vfit)
#ABORT            norm_dv = numpy.linalg.norm(dv)
#ABORT            iter_count += 1
#ABORT            log.debug(dev, 'DIIS fitting iter = %d, norm(dm_ref-dm) = %.6g ' \
#ABORT                      'norm(dv) = %.6g', iter_count, norm_ddm, norm_dv)
    log.debug(dev, 'after fitting, norm(dm_ref-dm) = %.6g', norm_ddm)

#    if FIT_DM_METHOD == FIT_DM_IMP_DIAG \
#       and (norm_ddm1 > 1e-4 or numpy.linalg.norm(vfit) > 1):
#        # In very rare situation, the impurity components in occupied and
#        # virtual orbitals are ~ 0. It leads to the hessian being very
#        # singular. Using random potential as initial guess to mix the
#        # occupied and virtual orbitals can improve the hessian.
#        # Don't know how to find proper condition to know when this happens.
#
#        #vfit = numpy.zeros_like(fock0)
#
#        vfit[:nimp,:nimp] = numpy.diag(numpy.random.random(nimp)) * 1e-3
#
##        vfit[:nimp,:nimp] = (.5 * dm_ref[:nimp,:nimp] \
##                - c[:nimp,:nocc] * numpy.mat(c[:nimp,:nocc]).T) \
##                * min(norm_ddm1,norm_ddm0)
#
##        e, c = numpy.linalg.eigh(fock0)
##        for i in range(min(nimp,nocc)):
##            c[i,i] += 1e-1
##        for i in range(min(nimp,fock0.shape[0]-nocc)):
##            c[i,i+nocc] += 1e-1
##        vfit[:nimp,:nimp] = numpy.array(c[:nimp,:]) * e \
##                * numpy.mat(c[:nimp,:]).H \
##                - fock0[:nimp,:nimp]
#
#        log.debug(dev, '    norm of initial guess = %.6g', \
#                  numpy.linalg.norm(vfit))
#        iter_count = 0
#        norm_dv = 1
#        while iter_count < MAX_ITER and norm_dv > CONV_THRESHOLD:
#            dv = fitpot.generate_pot(dev, dm_ref, fock0+vfit, nocc, nimp)
#            vfit[:nimp,:nimp] += dv
#            norm_dv = numpy.linalg.norm(dv)
#            iter_count += 1
#        log.debug(dev, '    with initial guess, fitting iteration = %d', iter_count)
#        log.debug(dev, '    with initial guess, after fitting, norm(dm_ref-dm) = %.6g', norm_ddm1)
#
#        e, c = numpy.linalg.eigh(fock0+vfit)
#        ddm = fitpot.diff_den_mat(dm_ref, c, nocc, nimp)
#        norm_ddm1 = numpy.linalg.norm(ddm)

    if  norm_ddm0 < norm_ddm:
        #log.stdout(dev, 'optimized fitting v was not found')
        log.warn(dev, 'optimized fitting v was not found')

    return vfit

def find_emb_potential_damp(dev, dm_ref, fock0, nocc, nimp):
    return find_emb_potential(dev, dm_ref, fock0, nocc, nimp) * TUNE_FAC
    #fitpot = FitImpPotential()
    #if not True:
    #    return fitpot.generate_pot_scf_dbg(dev, dm_ref, fock0, nocc, nimp)
    #else:
    #    return fitpot.generate_pot_scf(dev, dm_ref, fock0, nocc, nimp)


#######################################
# FIXME
def find_pseudo_v_for_lieb_fnl(dev, dm_ref, fock0, nocc, nimp):
    def generate_pot(dev, dm_ref, fock0, nocc, nimp):
        nao = fock0.shape[0]
        e, c = numpy.linalg.eigh(fock0)
        dm0 = numpy.dot(c[:nimp,:nocc], c[:nimp,:nocc].T.conj())
        ddm = numpy.array(.5*dm_ref[:nimp,:nimp] - dm0)
        norm_ddm = numpy.linalg.norm(ddm)

        _x = trans_mat_v_imp_to_dm1(e, c, nocc)
        x0 = _x[:,:,:nimp,:nimp].reshape(nao*nao,nimp*nimp)

        g = numpy.dot(fock0.flatten(), x0) - ddm.flatten()

        nvir = e.shape[0] - nocc
        eia = 1 / (e[:nocc].reshape(-1,1) - e[nocc:])
        tmpcc = numpy.empty((nao,nao,nvir))
        for r in range(nao):
            for u in range(nao):
                for a in range(nvir):
                    tmpcc[r,u,a] = c[r,a+nocc] * c[u,a+nocc]
        tmpce = numpy.empty((nvir,nocc,nao))
        for a in range(nvir):
            for i in range(nocc):
                for u in range(nao):
                    tmpce[a,i,u] = c[u,i] * eia[i,a]
        _y = numpy.dot(tmpcc.reshape(nao*nao,nvir), \
                       tmpce.reshape(nvir,nocc*nao))
        _y = _y.reshape(nao,nao,nocc,nao).transpose(2,0,1,3)
        tmpcc = numpy.empty((nocc,nimp,nao))
        for i in range(nocc):
            for r in range(nimp):
                for u in range(nao):
                    tmpcc[i,r,u] = c[r,i] * c[u,i]
        tmpce = numpy.empty((nvir,nao,nocc))
        for a in range(nvir):
            for u in range(nao):
                for i in range(nocc):
                    tmpce[a,u,i] = c[u,a+nocc] * eia[i,a]
        _z = numpy.dot(tmpce.reshape(nvir*nao,nocc), \
                       tmpcc.reshape(nocc,nimp*nao))
        _z = _z.reshape(nvir,nao,nimp,nao).transpose(0,2,1,3)

        tmp1 = numpy.dot(_y[:,:nimp].reshape(nocc*nimp,nao*nao), \
                         numpy.array(fock0).flatten())
        tmp1 = numpy.dot(tmp1.reshape(nocc, nimp).T, \
                         _y[:,:nimp,:nimp,:nimp].reshape(nocc,nimp*nimp*nimp))
        tmp2 = numpy.dot(_z.reshape(nvir*nimp,nao*nao), \
                         numpy.array(fock0).flatten())
        tmp2 = numpy.dot(tmp2.reshape(nvir, nimp).T, \
                         _z[:,:,:nimp,:nimp].reshape(nvir,nimp*nimp*nimp))
        tmp2.reshape(nimp,nimp,nimp,nimp).transpose(1,0,2,3)
        h1 = (tmp1 - tmp2).reshape(nimp*nimp,nimp*nimp)
        h1 = h1.reshape(nimp,nimp,nimp,nimp)
        h2 = h1.transpose(1,0,3,2)
        _y1 = _y[:nocc,:nao,:nimp,:nimp].transpose(1,0,2,3)
        _y2 = _y1.transpose(0,1,3,2)
        tmp3 = numpy.dot(fock0, _y1.reshape(nao, nocc*nimp*nimp))
        h3 = tmp3.reshape(nao*nocc,nimp*nimp).T * _y2.reshape(nao*nocc,nimp*nimp)
        h1 = ((h1 + h2).reshape(nimp*nimp,nimp*nimp) + h3) * 2
        h = h1 + _x[:nimp,:nimp,:nimp,:nimp].reshape(nimp*nimp,nimp*nimp)

        vfit, nsg = solve_lineq_for_indep_var(h, g)
        vfit = vfit.reshape(nimp,nimp)
        log.debug(dev, 'homo = %.9g, lumo = %.9g', \
                  e[nocc-1], e[nocc])
        log.debug(dev, 'norm(ddm), norm(dv) = %.6g, singular = %d', \
                  norm_ddm, norm_dv, nsg)
        return vfit

    iter_count = 0
    norm_dv = 1
    vfit = numpy.zeros_like(fock0)
    while iter_count < MAX_ITER and norm_dv > CONV_THRESHOLD:
        dv = generate_pot(dev, dm_ref, fock0+vfit, nocc, nimp)
        vfit[:nimp,:nimp] += dv
        norm_dv = numpy.linalg.norm(dv)
        iter_count += 1
    log.debug(dev, 'fitting iteration = %d', iter_count)
    return vfit



if __name__ == '__main__':
    import sys
    #nao = 6
    #nocc = 4
    #a = (numpy.sin(numpy.arange(nao*nao))+numpy.arange(0,nao*nao*.1,.1)).reshape(nao,nao)
    #t, w, v = numpy.linalg.svd(a)
    #f = numpy.mat(v * numpy.arange(nao)) * numpy.mat(v).H
    #l = numpy.zeros_like(w)
    #l[:nocc] = 1
    #l[nocc+1] = .05
    #l[nocc] = .1
    #l[nocc-1] -= .1
    #l[nocc-2] -= .05
    #dref = (v * l) * numpy.mat(v).H
    #mol = type('Temp', (), {'verbose': 5, 'fout': sys.stdout}) # temporary object
    #print find_emb_potential(mol, dref, f, nocc, 2)

    dref = numpy.zeros((4,4))
    dref[:2,:2] = 1
    dref[2:,2:] = 1
    f = numpy.ones((4,4)) * .1 - numpy.eye(4)#- numpy.diag(1/(1+numpy.arange(4.)))
    #f = numpy.random.random((4,4))
    #f = f + f.T
    e, c = numpy.linalg.eigh(f)
    print numpy.dot(c[:,:3], c[:,:3].T)
    mol = type('Temp', (), {'verbose': 5, 'fout': sys.stdout}) # temporary object
    print find_emb_potential(mol, dref, f, 3, 3)
