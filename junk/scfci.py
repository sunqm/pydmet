#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

# 1. run hf with fit_pot_global
# 2. update embs {imp, bath} with new hf
# 3. fit new pot globally/locally

'''
DMET-SCF: dual-level SCF (fragment-FCI + entire-sys-SCF)
'''

import os, sys
import tempfile
import commands
import copy
import pickle

import numpy
import scipy.optimize
import scipy.linalg.flapack as lapack

from pyscf import gto
from pyscf import scf
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
#import dmet
#import ci.fci
import scfopt
import molpro_fcidump as molpro
from pyscf import ao2mo

MAX_ITER = 40
DMET_WITH_NAO = False
DMET_WITH_META_LOWDIN = False
RAND_INIT = False

FITTING_WITHOUT_SCF = 1
FITTING_WITH_SCF = 2
FITTING_1SHOT = 3
FITTING_FCI_POT = 4
FITTING_LIEB_POT = 5
FITTING_APPROX = 1
WITH_HOPPING = False

NO_FIXED_DM = 1
FIXED_CI_DM = 2
FIXED_MF_DM = 3
NO_FIXED_DM_BACKWARDS = 4
GLOBAL_FIT = NO_FIXED_DM

IMP_AND_BATH = 1
IMP_BLK = 2
IMP_BATH_DIAG = 3
NO_BATH_BLK = 4
DIAG_BLK = 5
IMP_DIAG = 6
VFIT_DOMAIN = IMP_BLK

FCI_WITH_ENV_POT = 1
FOLLOW_STATE = False
FCI_SOLVER = True
DAMPING = True

DMFET_THRED = 1e-9


def dot(a, b):
    return numpy.dot(numpy.array(a).flatten(), numpy.array(b).flatten())

def flatten(lsts):
    return [i for lst in lsts for i in lst]


def full_eri_ao2mo(eri_ao, mo_coeff):
    nao, nmo = mo_coeff.shape
    eri0 = numpy.empty((nao,nao,nao,nao))
    for i in range(nao):
        for j in range(i+1):
            for k in range(nao):
                for l in range(k+1):
                    ij = i*(i+1)/2+j
                    kl = k*(k+1)/2+l
                    eri0[i,j,k,l] = eri0[i,j,l,k] = eri0[j,i,k,l] \
                            = eri0[j,i,l,k] = eri_ao[ij,kl]
    eri_ao = eri0
    eri1 = numpy.dot(eri_ao.reshape(nao*nao*nao,nao), mo_coeff)
    eri1 = numpy.dot(mo_coeff.T, eri1.reshape(nao,nao*nao*nmo))
    eri1 = eri1.reshape(nmo*nao,nao*nmo).transpose()
    eri1 = numpy.dot(eri1.reshape(nao*nmo*nmo,nao), mo_coeff)
    eri1 = numpy.dot(mo_coeff.T, eri1.reshape(nao,nmo*nmo*nmo))
    return ao2mo.gen_int2e_from_full_eri(eri1.reshape(nmo,nmo,nmo,nmo))

# eri has 4-fold symmetry ijkl=ijlk=jikl=jilk
def partial_eri_ao2mo(eri_ao, mo_coeff):
    try:
        if mo_coeff.flags.c_contiguous:
            mo_coeff = mo_coeff.copy('F')
        return ao2mo.partial_eri_ao2mo_o3(eri_ao, mo_coeff)
    except:
        nao, nmo = mo_coeff.shape
        tmp = numpy.empty((nao,nao))
        eri1 = numpy.empty((nao,nao,nmo,nmo))
        nao_pair = nao * (nao+1) / 2
        eri_ao1 = numpy.empty((nao_pair,nao_pair))
        for i in range(nao_pair):
            for j in range(i+1):
                eri_ao1[i,j] = eri_ao1[j,i] = eri_ao[i*(i+1)/2+j]
        for i in range(nao):
            for j in range(i+1):
                ij = i*(i+1)/2+j
                v = eri_ao1[ij]
                for k in range(nao):
                    for l in range(k+1):
                        kl = k*(k+1)/2+l
                        tmp[k,l] = tmp[l,k] = v[kl]
                t1 = reduce(numpy.dot, (mo_coeff.T, tmp, mo_coeff))
                eri1[i,j] = eri1[j,i] = t1
        eri1 = numpy.dot(mo_coeff.T, eri1.reshape(nao,-1))
        eri1 = eri1.reshape(nmo*nao,nmo*nmo).transpose()
        eri1 = numpy.dot(eri1.reshape(nmo*nmo*nmo,nao), mo_coeff)
        return ao2mo.gen_int2e_from_full_eri(eri1.reshape(nmo,nmo,nmo,nmo))

def dump_fci_input(mol, emb, filefci):
    impmol = emb.mol

    fcinp = open(filefci, 'w')
    molpro.head(emb.impbas_coeff.shape[1], emb.nelectron, fcinp)

    if emb._eri is not None:
        molpro.write_eri_in_molpro_format(emb._eri, fcinp)
    elif emb.entire_scf._eri is not None:
        eri = partial_eri_ao2mo(emb.entire_scf._eri, emb.impbas_coeff)
        molpro.write_eri_in_molpro_format(eri, fcinp)
    else:
        molpro.eri(impmol, emb.impbas_coeff, fcinp)

    if emb._pure_hcore is not None:
        h1e = emb._pure_hcore + emb._vhf_env
    else:
        h1e = emb.get_hcore(emb.mol)
    molpro.write_hcore_in_molpro_format(h1e, fcinp)

    fcinp.write(' %.16g  0  0  0  0\n' % 0)
    ucinp.close()

def ext_fci_run(filefci, n_imp_site):
    file_rdm1 = filefci + '.rdm1'
    cmd = '%s --basis CoreH --subspace-dimension 200 --save-rdm1 %s'\
            ' --work-memory 1000 --ptrace %s %s' \
            % (dmet.fci.FCI_EXE, file_rdm1, n_imp_site, filefci)
    rec = commands.getoutput(cmd)
    rdm1 = dmet.fci.read_dm(file_rdm1)
    os.remove(file_rdm1)
    return rdm1, rec

#def frag_fci(mol, emb, vfit=0):
#    if vfit is not 0:
#        vhf_env_bak = emb._vhf_env.copy()
#        nv = vfit.shape[0]
#        emb._vhf_env[:nv,:nv] += vfit
#    nimp = emb.dim_of_impurity()
#    filefci = tempfile.mktemp('.fcinp')
#    dump_fci_input(mol, emb, filefci)
#    rdm1, rec = ext_fci_run(filefci, nimp)
#    os.remove(filefci)
#    if vfit is not 0:
#        emb._vhf_env = vhf_env_bak
#    return rdm1, rec
def frag_fci(mol, emb, vfit=0):
    if emb._pure_hcore is not None:
        h1e = emb._pure_hcore + emb._vhf_env
    else:
        h1e = emb.get_hcore(emb.mol)
    if vfit is not 0:
        nv = vfit.shape[0]
        h1e[:nv,:nv] += vfit

    if emb._eri is not None:
        int2e = emb._eri
    elif emb.entire_scf._eri is not None:
        int2e = partial_eri_ao2mo(emb.entire_scf._eri, emb.impbas_coeff)
    else:
        int2e = ao2mo.gen_int2e_ao2mo(mol, emb.impbas_coeff)

    nemb = emb.num_of_impbas()
    rdm1 = numpy.empty((nemb,nemb))
    nimp = emb.dim_of_impurity()
    nelec = emb.nelectron
    rec = ci.fci._run(mol, nelec, h1e, int2e, 0, rdm1, ptrace=nimp)
    res = {'rdm1': rdm1, \
           'etot': ci.fci.find_fci_key(rec, 'STATE 1 ENERGY'), \
           'e1frag': numpy.dot(h1e[:nimp].reshape(-1),rdm1.reshape(-1)), \
           'e2frag': ci.fci.find_fci_key(rec, 'STATE 1 pTraceSys'),
           'rec': rec}
    return res


def get_emb_fci_e1_e2(emb, cires, with_env_pot=FCI_WITH_ENV_POT):
    rdm1 = cires['rdm1']
    nimp = emb.dim_of_impurity()
    log.debug(emb, 'FCI energy of (frag + bath) %.12g', cires['etot'])

    if emb._pure_hcore is not None:
        h1e = emb._pure_hcore
    else:
        h1e = emb.mat_ao2impbas(emb.entire_scf.get_hcore(emb.mol))
    nelec_frag = numpy.trace(rdm1[:nimp,:nimp])
    e1_frag = dot(rdm1[:nimp,:nimp], h1e[:nimp,:nimp])
    e1_bath = dot(rdm1[:nimp,nimp:], h1e[:nimp,nimp:])
    if with_env_pot:
        if isinstance(emb.env_fit_pot,int):
            e1_vfit = 0
        else:
            e1_vfit = dot(rdm1[:nimp], emb.env_fit_pot[:nimp])
    else:
        e1_vfit = 0
    e1 = e1_frag + e1_bath + e1_vfit
    log.debug(emb, 'e1 = fragment + bath + fitenv')
    log.debug(emb, '   = %.12g = %.12g + %.12g + %.12g', \
              e1, e1_frag, e1_bath, e1_vfit)
    e2env_hf = dot(rdm1[:nimp], emb._vhf_env[:nimp]) * .5
    e2 = res['e2frag']
    log.debug(emb, 'fragment e1 = %.12g, e2env_hf = %.12g, FCI pTraceSys = %.12g', \
              e1, e2env_hf, e2)
    log.debug(emb, 'fragment e2env_hf = %.12g, FCI pTraceSys = %.12g, nelec = %.12g', \
              e2env_hf, e2, nelec_frag)
    e_frag = e1 + e2env_hf + e2
    return e_frag, nelec_frag


def map_frag_to_bas_idx(mol, frag_group):
    lbl = mol.labels_of_spheric_GTO()
    atm_basidx = [[] for i in range(mol.natm)]
    for ib, s in enumerate(lbl):
        ia = s[0]
        atm_basidx[ia].append(ib)
    def _remove_bas_if_not_on_frag(atm_lst):
        bas_on_a = []
        for ia in atm_lst:
            bas_on_a.extend(atm_basidx[ia])
        return numpy.array(bas_on_a)
    bas_idx_group = []
    for m, frags in enumerate(frag_group):
        if isinstance(frags[0], int):
            bas_idx_group.append(_remove_bas_if_not_on_frag(frags))
        else:
            tmp_group = []
            for atm_lst in frags:
                tmp_group.append(_remove_bas_if_not_on_frag(atm_lst))
            bas_idx_group.append(tmp_group)
    return bas_idx_group

def loop_all_frags(mol, frag_group):
    frag_bas_idx = map_frag_to_bas_idx(mol, frag_group)
    for emb_id, frags in enumerate(frag_group):
        if isinstance(frags[0], int):
            yield emb_id, frags, frag_bas_idx[emb_id]
        else:
            for k, atm_lst in enumerate(frags):
                yield emb_id, atm_lst, frag_bas_idx[emb_id][k]

def loop_uniq_frags(mol, frag_group):
    frag_bas_idx = map_frag_to_bas_idx(mol, frag_group)
    for emb_id, atm_lst in enumerate(frag_group):
        if isinstance(atm_lst[0], int):
            yield emb_id, atm_lst, frag_bas_idx[emb_id]
        else:
            yield emb_id, atm_lst[0], frag_bas_idx[emb_id][0]

def mat_orthao2ao(mol, mat, orth_coeff):
    '''matrix represented on orthogonal basis to the representation on
    non-orth AOs'''
    c_inv = numpy.dot(orth_coeff.T, mol.intor_symmetric('cint1e_ovlp_sph'))
    mat_on_ao = reduce(numpy.dot, (c_inv.T.conj(), mat, c_inv))
    return mat_on_ao

# v_inc_base should be Hermitian
def numfit_imp_pot_old(mol, emb, get_dm, dm_ref, \
                   v_inc_base, finite_inc_fac, dgrad=False):
    nd = v_inc_base.shape[0]
    v_inc = v_inc_base.copy()
    dm_base = get_dm(emb, v_inc_base)
    resps = []
    for i in range(nd):
        for j in range(i):
            v_inc[i,j] += finite_inc_fac
            v_inc[j,i] += finite_inc_fac
            dm = get_dm(emb, v_inc)
            v_inc[i,j] -= finite_inc_fac
            v_inc[j,i] -= finite_inc_fac
            resps.append(dm-dm_base)
        v_inc[i,i] += finite_inc_fac
        dm = get_dm(emb, v_inc)
        v_inc[i,i] -= finite_inc_fac
        resps.append(dm-dm_base)

    resps = numpy.array(resps) * (1/finite_inc_fac)
    ddm = dm_ref - dm_base
    x, grad, sgl = gen_approx_step(mol, resps, ddm, dgrad)

    def norm_ddm_method(dv):
        v_inc = numpy.zeros_like(v_inc_base)
        dvi = iter(dv)
        for i in range(nd):
            for j in range(i+1):
                v_inc[i,j] = v_inc[j,i] = v_inc_base[i,j]+dvi.next()
        dm = get_dm(emb, v_inc)
        ddm = dm_ref - dm
        return numpy.linalg.norm(ddm)
    dv,norm_ddm = scfopt.line_search_wolfe(mol, norm_ddm_method, x, grad0=grad, \
                                           title='for numerical on-imp fitting')
    v_inc = numpy.zeros_like(v_inc_base)
    dvi = iter(dv)
    for i in range(nd):
        for j in range(i+1):
            v_inc[i,j] = v_inc[j,i] = dvi.next()

    log.debug(mol, 'after num_fitting, norm(dm_ref-dm[v]) = %.9g,' \
              ' norm(dv) = %.9g, singular = %d', \
              norm_ddm, numpy.linalg.norm(v_inc), sgl)
    return v_inc, norm_ddm

# numfit_global_pot+gen_global_walker => numfit_global_pot_old
def numfit_global_pot_old(mol, embsys, get_dm, dm_ref, \
                      v_inc_base, finite_inc_fac, dgrad=False):
    v_inc = v_inc_base.copy()
    dm_base = get_dm(v_inc_base)
    resps = []
    for m, atm_lst, bas_idx in embsys.all_frags:
        for i0, i in enumerate(bas_idx):
            for j0 in range(i0):
                j = bas_idx[j0]
                v_inc[i,j] += finite_inc_fac
                v_inc[j,i] += finite_inc_fac
                dm = get_dm(v_inc)
                v_inc[i,j] -= finite_inc_fac
                v_inc[j,i] -= finite_inc_fac
                resps.append(dm-dm_base)
            v_inc[i,i] += finite_inc_fac
            dm = get_dm(v_inc)
            v_inc[i,i] -= finite_inc_fac
            resps.append(dm-dm_base)

    resps = numpy.array(resps) * (1/finite_inc_fac)
    ddm = dm_ref - dm_base
    x, grad, sgl = gen_approx_step(mol, resps, ddm, dgrad)

    v_inc = numpy.zeros_like(v_inc_base)
    x_iter = iter(x)
    for m, atm_lst, bas_idx in embsys.all_frags:
        for i0, i in enumerate(bas_idx):
            for j0 in range(i0+1):
                j = bas_idx[j0]
                v_inc[i,j] = v_inc[j,i] = x_iter.next()

    def norm_ddm_method(dv):
        dm = get_dm(v_inc_base+dv)
        ddm = dm_ref - dm
        return numpy.linalg.norm(ddm)
    v_inc,norm_ddm = scfopt.line_search_sharp(mol, norm_ddm_method, v_inc, \
                                              floating=1e-3, \
                                              title='for numerical global fitting')

    log.debug(mol, 'after num_fitting, norm(dm_ref-dm[v]) = %.9g,' \
              ' norm(dv) = %.9g, singular = %d', \
              norm_ddm, numpy.linalg.norm(v_inc), sgl)
    return v_inc, norm_ddm



def numfitor(dev, get_dm, walkers, dm_ref, \
             v_inc_base, finite_inc_fac, dgrad=False, title=''):
    v_inc = v_inc_base.copy()
    for i in range(v_inc.shape[0]):
        v_inc[i,i] += i*1e-11
    dm_base = get_dm(v_inc)
    resps = []
    for i,j in walkers:
        fac = finite_inc_fac
        v_inc[i,j] += fac
        v_inc[j,i] = v_inc[i,j]
        dm_inc = get_dm(v_inc)-dm_base
        v_inc[i,j] -= fac
        v_inc[j,i] = v_inc[i,j]
        while numpy.linalg.norm(dm_inc) > 5e-3 and fac > 1e-7:
            fac = fac * .2
            v_inc[i,j] += fac
            v_inc[j,i] = v_inc[i,j]
            dm_inc = get_dm(v_inc)-dm_base
            v_inc[i,j] -= fac
            v_inc[j,i] = v_inc[i,j]
        resps.append(dm_inc*(1./fac))
    resps = numpy.array(resps)
    ddm = dm_ref - dm_base
    log.debug(dev, 'before num_fitting %s, norm_ddm = %.9g', \
              title, numpy.linalg.norm(ddm))
    x, grad, sgl = gen_approx_step(dev, resps, ddm, dgrad, DMFET_THRED)

    v_inc = numpy.zeros_like(v_inc_base)
    x_iter = iter(x)
    for i,j in walkers:
        v_inc[i,j] = v_inc[j,i] = x_iter.next()

    def norm_ddm_method(dv):
        dm = get_dm(v_inc_base+dv)
        return numpy.linalg.norm(dm_ref - dm)
    v_inc,norm_ddm = scfopt.line_search_wolfe(dev, norm_ddm_method, v_inc, \
                                              val0=numpy.linalg.norm(dm_ref-dm_base), \
                                              minstep=1e-2,title=title)
    #v_inc,norm_ddm = scfopt.line_search_sharp(dev, norm_ddm_method, v_inc, \
    #                                          floating=1e-3, title=title)

    log.debug(dev, 'after num_fitting %s, norm(dm_ref-dm[v]) = %.9g,' \
              ' norm(dv) = %.9g, singular = %d', \
              title, norm_ddm, numpy.linalg.norm(v_inc), sgl)
#CHECK    norm_ddm = norm_ddm_method(v_inc)
    return v_inc, norm_ddm
#ABORT
#ABORTdef numfit_imp_pot(mol, emb, fn, walkers, val_ref, \
#ABORT                   v_inc_base, finite_inc_fac, dgrad=False):
#ABORT    val_base = fn(emb, v_inc_base)
#ABORT    v_inc = v_inc_base.copy()
#ABORT    resps = []
#ABORT    for i,j in walkers:
#ABORT        v_inc[i,j] += finite_inc_fac
#ABORT        v_inc[j,i] = v_inc[i,j]
#ABORT        resps.append(fn(emb, v_inc)-val_base)
#ABORT        v_inc[i,j] -= finite_inc_fac
#ABORT        v_inc[j,i] = v_inc[i,j]
#ABORT
#ABORT    resps = numpy.array(resps) * (1/finite_inc_fac)
#ABORT    ddm = val_ref - val_base
#ABORT    log.debug(mol, 'before local num_fitting norm_ddm = %.9g', \
#ABORT              numpy.linalg.norm(ddm))
#ABORT    x, grad, sgl = gen_approx_step(mol, resps, ddm, dgrad)
#ABORT
#ABORT    v_inc = numpy.zeros_like(v_inc_base)
#ABORT    x_iter = iter(x)
#ABORT    for i,j in walkers:
#ABORT        v_inc[i,j] = v_inc[j,i] = x_iter.next()
#ABORT    def norm_ddm_method(dv):
#ABORT        ddm = val_ref - fn(emb, v_inc_base+dv)
#ABORT        return numpy.linalg.norm(ddm)
#ABORT    dv,norm_ddm = scfopt.line_search_wolfe(mol, norm_ddm_method, v_inc, \
#ABORT                                      val0=numpy.linalg.norm(val_ref-val_base))
#ABORT    log.debug(mol, 'after local num_fitting, norm_ddm = %.9g, norm(dv) = %.9g, singular = %d', \
#RT              % (norm_ddm, numpy.linalg.norm(dv), sgl)
#ABORT    v_add = v_inc_base + dv
#ABORT    return v_add, norm_ddm

def gen_global_uniq_walker(mol, embsys):
    walker_group = [[] for i in embsys.frag_group]
    for m, atm_lst, bas_idx in embsys.all_frags:
        w = []
        for i0, i in enumerate(bas_idx):
            for j0 in range(i0+1):
                w.append((i,bas_idx[j0]))
        if isinstance(embsys.frag_group[m][0], int):
            walker_group[m] = w
        else:
            walker_group[m].append(w)
    for m,atm_lst in enumerate(embsys.frag_group):
        if not isinstance(atm_lst[0], int):
            walker_group[m] = map(zip, *walker_group[m])
    return flatten(walker_group)

def gen_global_walker(mol, embsys):
    walker_group = []
    for m, atm_lst, bas_idx in embsys.all_frags:
        for i0, i in enumerate(bas_idx):
            for j0 in range(i0+1):
                walker_group.append((i,bas_idx[j0]))
    return walker_group

# thrd can affect the accuracy of fitting.
# When it is very small, fitting should therotically be accurate, but
# practically it suffers from numerical instability. Besides small threshold
# may cause big step size which is not wanted in fitting.  When it becomes
# larger, fitting often cannot converge to 1e-4.  This has be numerically
# proven in the simple system of H2.
def gen_approx_step(dev, resps, ddm, dgrad, thrd=1e-6):
    norm_ddm = numpy.linalg.norm(ddm)
    log.debug(dev, 'before num_fitting, norm(dm_ref-dm[v]) = %.9g', \
              norm_ddm)

    g = numpy.dot(resps, ddm)
    if dgrad:
        log.debug(dev, 'numerical fitting with deepest gradient')
        x = g * 10
        sgl = -1
    else:
        log.debug(dev, 'numerical fitting with newton method')
        h = numpy.dot(resps, resps.T)
        scfopt.MAX_STEP_SIZE = 100
        x, sgl = scfopt.step_by_eigh_min(h, g, thrd)
    grad = -g
    return x, grad, sgl

def run_hf_with_ext_pot(mol, entire_scf, vext_on_ao, follow_state=False):
    def _dup_entire_scf(mol, entire_scf):
        #eff_scf = entire_scf.__class__(mol)
        eff_scf = copy.copy(entire_scf)
        eff_scf.verbose = 0#entire_scf.verbose
        eff_scf.scf_threshold = 1e-10#entire_scf.scf_threshold
        eff_scf.diis_space = 6#entire_scf.diis_space
        eff_scf.scf_conv = False
        return eff_scf
    eff_scf = _dup_entire_scf(mol, entire_scf)

    # FIXME: ground state strongly depends on initial guess.
    # when previous SCF does not converge, the initial guess will be incorrect
    # and leads to incorrect MF ground state.
    # In this case, use old initial guess instead.
    if entire_scf.scf_conv:
        dm = entire_scf.calc_den_mat(entire_scf.mo_coeff, entire_scf.mo_occ)
        def _init_guess_method(mol):
            return entire_scf.hf_energy, dm
    else:
        log.warn(eff_scf, "use old initial guess")
        hf_e_old, dm_old = entire_scf.init_guess_method(mol)
        def _init_guess_method(mol):
            return hf_e_old, dm_old
    eff_scf.init_guess_method = _init_guess_method

    def _get_hcore(mol):
        h = scf.hf.SCF.get_hcore(mol)
        if isinstance(eff_scf, scf.hf_symm.RHF):
            return symm.symmetrize_matrix(h+vext_on_ao, mol.symm_orb)
        elif isinstance(eff_scf, scf.hf_symm.UHF):
            return (symm.symmetrize_matrix(h+vext_on_ao[0], mol.symm_orb), \
                    symm.symmetrize_matrix(h+vext_on_ao[1], mol.symm_orb))
        else:
            return h + vext_on_ao
    eff_scf.get_hcore = _get_hcore

    if follow_state:
        eff_scf.mo_coeff = entire_scf.mo_coeff
        eff_scf.mo_occ = numpy.zeros_like(entire_scf.mo_energy)
        eff_scf.mo_occ[:mol.nelectron/2] = 2
        def _occ_follow_state(mol, mo_energy, mo_coeff):
            s = entire_scf.get_ovlp(mol)
            prj = reduce(numpy.dot, (mo_coeff.T, s, eff_scf.mo_coeff))
            mo_occ = numpy.zeros_like(mo_energy)
            for i,occ in enumerate(eff_scf.mo_occ):
                if occ > 0:
                    imax = abs(prj[i]).argmax()
                    prj[:,imax] = 0
                    mo_occ[imax] = 2
                    if imax == i and i < mol.nelectron/2:
                        log.info(mol, 'occupied MO %d energy=%.15g occ=2.0', \
                                 i+1, mo_energy[i])
                    else:
                        log.info(mol, ' ** occupied MO %d energy=%.15g occ=2.0', \
                                 imax+1, mo_energy[i])
            for i,occ in enumerate(mo_occ):
                if occ == 0:
                    if i < mol.nelectron/2:
                        log.info(mol, ' ** LUMO=%d energy= %.15g occ=0.0', \
                                 i+1, mo_energy[i])
                    else:
                        log.info(mol, 'LUMO=%d energy= %.15g occ=0.0', \
                                 i+1, mo_energy[i])
                    break
            eff_scf.mo_coeff = mo_coeff
            eff_scf.mo_occ[:] = mo_occ
            return mo_occ
        eff_scf.set_mo_occ = _occ_follow_state

    log.debug(mol, 'SCF for entire molecule with fitting potential')
    eff_scf.scf_conv, eff_scf.hf_energy, eff_scf.mo_energy, \
            eff_scf.mo_occ, eff_scf.mo_coeff \
            = eff_scf.scf_cycle(mol, eff_scf.scf_threshold, dump_chk=False)

    # must release the modified get_hcore to get pure hcore for other function
    del(eff_scf.get_hcore)
    #del(eff_scf.init_guess_method)
    return eff_scf

def log_homo_lumo(mol, mo_occ, mo_e):
    e_homo = mo_e[mo_occ>0][-1]
    e_lumo = mo_e[mo_occ==0][0]
    log.debug(mol, '                      homo = %.9g, lumo = %.9g', \
              e_homo, e_lumo)

class EmbSys(object):
    def __init__(self, mol, entire_scf, frag_group, init_v=None):
        self.frag_group = frag_group
        self.all_frags = [i for i in loop_all_frags(mol, frag_group)]
        self.uniq_frags = [i for i in loop_uniq_frags(mol, frag_group)]
        try:
            with open(init_v, 'r') as f:
                self.v_global, v_add_on_ao = pickle.load(f)
            self.entire_scf = run_hf_with_ext_pot(mol, entire_scf, v_add_on_ao)
        except:
            nao = mol.num_NR_function()
            self.v_global = numpy.zeros((nao,nao))
            #if RAND_INIT:
            #    for m, atm_lst, bas_idx in self.all_frags:
            #        nimp = bas_idx.__len__()
            #        v = numpy.random.randn(nimp*nimp).reshape(nimp,nimp)
            #        v = (v + v.T) * .1
            #        for i, j in enumerate(bas_idx):
            #            self.v_global[j,bas_idx] = v[i]
            #self.entire_scf = run_hf_with_ext_pot(mol, entire_scf, self.v_global)
            self.entire_scf = entire_scf

        self.embs = self.setup_embs(mol, self.entire_scf, frag_group)
        self.orth_coeff = self.embs[0].orth_coeff

    #def __copy__(self):
    #    new = self.__class__()

    def setup_embs(self, mol, entire_scf, frag_group):
        embs = []
        for m, atm_lst, bas_idx in loop_uniq_frags(mol, frag_group):
            emb = dmet.hf.RHF(entire_scf)
            emb.occ_env_cutoff = 1e-14
            emb.set_embsys(atm_lst)
            if DMET_WITH_META_LOWDIN:
                emb.set_ao_with_atm_scf()
                emb.init_with_meta_lowdin_ao()
            elif DMET_WITH_NAO:
                emb.init_with_nao()
            emb._pure_hcore = None
            embs.append(emb)

        orth_coeff = embs[0].get_orth_ao(mol)
        for m, emb in enumerate(embs):
            emb.orth_coeff = orth_coeff
        self.setup_embs_without_pot_on_site(mol, embs, self.v_global)
        return embs

    def setup_embs_without_pot_on_site(self, mol, embs, v_global):
        eff_scf = self.entire_scf
        s = eff_scf.get_ovlp(mol)
        sc = numpy.dot(s, eff_scf.mo_coeff)
        fock0 = numpy.dot(sc*eff_scf.mo_energy, sc.T.conj())
        for emb in embs:
            emb.init_dmet_scf(mol)
            f = reduce(numpy.dot, (emb.impbas_coeff.T, fock0, emb.impbas_coeff))
            emb.mo_energy, emb.mo_coeff_on_imp = numpy.linalg.eigh(f)
            emb.mo_coeff = numpy.dot(emb.impbas_coeff, emb.mo_coeff_on_imp)
            emb.hf_energy = 0
            if eff_scf._eri is not None:
                emb._eri = partial_eri_ao2mo(eff_scf._eri, emb.impbas_coeff)

        if FCI_WITH_ENV_POT == 3:
            entire_scf_dm = eff_scf.calc_den_mat(eff_scf.mo_coeff, \
                                                 eff_scf.mo_occ)
            v_scaled = numpy.zeros_like(v_global)
            for emb in embs:
                nele = emb.nelectron
                nimp = emb.dim_of_impurity()
                dm = numpy.dot(emb.mo_coeff_on_imp[:,:nele/2],
                               emb.mo_coeff_on_imp[:,:nele/2].T) * 2
                tmp = v_global[emb.bas_on_frag][:,emb.bas_on_frag] \
                        * (dm[:nimp,:nimp].trace() / nele)
                for i,j in enumerate(emb.bas_on_frag):
                    v_scaled[j,emb.bas_on_frag] = tmp[i]
        elif FCI_WITH_ENV_POT == 4:
            orth_coeff = embs[0].orth_coeff
            dm = eff_scf.calc_den_mat(eff_scf.mo_coeff, eff_scf.mo_occ)
            vhf = reduce(numpy.dot, (orth_coeff.T, \
                                     eff_scf.get_eff_potential(mol, dm), \
                                     orth_coeff))
        elif FCI_WITH_ENV_POT == 5:
            c_inv = numpy.dot(embs[0].orth_coeff.T.conj(), s)
            mo_orth = numpy.dot(c_inv, eff_scf.mo_coeff)
            entire_scf_dm = eff_scf.calc_den_mat(mo_orth, eff_scf.mo_occ)

        for emb in embs:
            # setup environment potential, exclude the potential of the present fragment
            nimp = emb.dim_of_impurity()
            if FCI_WITH_ENV_POT == 1:
                emb.env_fit_pot = emb.mat_orthao2impbas(v_global)
                emb.env_fit_pot[:nimp,:nimp] = 0
            if FCI_WITH_ENV_POT == 2:
            # scale the potential with the num. electrons in environment
                emb.env_fit_pot = emb.mat_orthao2impbas(v_global) \
                        * float(emb.env_orb.shape[1]*2)/mol.nelectron
                emb.env_fit_pot[:nimp,:nimp] = 0
            elif FCI_WITH_ENV_POT == 3:
            # scale the potential with the num. electrons in each fragment
                emb.env_fit_pot = emb.mat_orthao2impbas(v_scaled)
                emb.env_fit_pot[:nimp,:nimp] = 0
            elif FCI_WITH_ENV_POT == 4:
            # scale the potential with the density matrix elements
                env_orb = numpy.dot(orth_coeff, emb.env_orb)
                dm_env = numpy.dot(env_orb, env_orb.T.conj()) * 2
                vhf_env_ao = eff_scf.get_eff_potential(mol, dm_env)
                vhf_env = reduce(numpy.dot, (orth_coeff.T, vhf_env_ao, orth_coeff))
                a = numpy.zeros_like(vhf_env)
                for i in range(vhf_env.shape[0]):
                    for j in range(vhf_env.shape[1]):
                        if abs(vhf[i,j] > 1e-10):
                            a[i,j] = vhf_env[i,j] / vhf[i,j]
                emb.env_fit_pot = emb.mat_orthao2impbas(v_global * a)
            elif FCI_WITH_ENV_POT == 5:
                # first transform DM to embedding basis, then scale the
                # potential with the density matrix elements.
                # it actually set emb.env_fit_pot[:nimp,:nimp] = 0 because
                # env_orb has zero-coefficients on impurity sites
                dm_env = numpy.dot(emb.env_orb,emb.env_orb.T.conj()) * 2
                a = numpy.zeros_like(dm_env)
                for i in range(dm_env.shape[0]):
                    for j in range(dm_env.shape[1]):
                        if abs(entire_scf_dm[i,j]) > 1e-10:
                            a[i,j] = dm_env[i,j] / entire_scf_dm[i,j]
                emb.env_fit_pot = emb.mat_orthao2impbas(v_global * a)
            else:
                emb.env_fit_pot = 0
            #vfit_imp = v_global[emb.bas_on_frag][:,emb.bas_on_frag]
            #emb.env_fit_pot[:nimp,:nimp] -= vfit_imp
        return embs

    def update_embsys_vglobal(self, mol, v_add):
        v_add_ao = mat_orthao2ao(mol, v_add, self.orth_coeff)
        eff_scf = run_hf_with_ext_pot(mol, self.entire_scf, v_add_ao)

        self.v_global = v_add
        self.entire_scf = eff_scf

        for emb in self.embs:
            emb.entire_scf = eff_scf
        self.setup_embs_without_pot_on_site(mol, self.embs, v_add)
        return eff_scf

    def frag_fci_solver(self, mol, emb):
        return frag_fci(mol, emb, emb.env_fit_pot)

    def assemble_frag_fci_energy(self, mol):
        val_tot = 0
        nelec = 0
        for m, emb in enumerate(self.embs):
            cires = self.frag_fci_solver(mol, emb)
            val_frag, nelec_frag = get_emb_fci_e1_e2(emb, cires)

            if isinstance(self.frag_group[m][0], int):
                val_tot += val_frag
                nelec += nelec_frag
            else: # degenerated fragments
                val_tot += val_frag * self.frag_group[m].__len__()
                nelec += nelec_frag * self.frag_group[m].__len__()
        log.info(mol, 'DMET-FCI-in-HF of entire system energy = %.12g', val_tot)
        return val_tot, nelec


    def gen_all_frag_ddm_fci_mf(self, mol):
        ''' DM difference between FCI and model-MF'''
        ddm_group = []
        for m, emb in enumerate(self.embs):
            cires = self.frag_fci_solver(mol, emb)
            dm_ref = cires['rdm1']
            nimp = emb.dim_of_impurity()
            s = emb.entire_scf.get_ovlp(mol)
            sc = reduce(numpy.dot, (emb.impbas_coeff.T.conj(), s, \
                                    emb.entire_scf.mo_coeff))
            fock0 = reduce(numpy.dot, (sc, emb.entire_scf.mo_energy, sc.T.conj()))
            nocc = emb.nelectron/2
            e, c = numpy.linalg.eigh(fock0)
            ddm = dm_ref - numpy.dot(c[:,:nocc], c[:,:nocc].T) * 2
            ddm_group.append(ddm)
        return ddm_group


    def assemble_to_blockmat(self, mol, v_group):
        '''assemble matrix on impuity sites to the diagonal block'''
        nao = self.orth_coeff.shape[1]
        v_add = numpy.zeros((nao,nao))
        for m, atm_lst, bas_idx in self.all_frags:
            nimp = self.embs[m].bas_on_frag.__len__()
            vfrag = numpy.array(v_group[m][:nimp,:nimp])
            for i, j in enumerate(bas_idx):
                v_add[j,bas_idx] = vfrag[i,:]
        return v_add


    def assemble_to_fullmat(self, mol, dm_group):
        '''assemble matrix of the embsys to the full matrix'''
        nao = self.orth_coeff.shape[1]
        dm_big = numpy.zeros((nao,nao))
        for m, atm_lst, bas_idx in self.all_frags:
            emb = self.embs[m]
            nimp = emb.dim_of_impurity()
            dm_ab = numpy.dot(dm_group[m][:nimp,nimp:], emb.bath_orb.T)
            dm_ab[:,emb.bas_on_frag] = dm_group[m][:nimp,:nimp]
            dm_big[emb.bas_on_frag] = dm_ab
        return dm_big

# pseudo potential and hopping matrix
    def assemble_frag_v_hopping(self, mol, v_group):
        '''represent the pseudo potential of the whole system in original
        non-orth AO basis'''
        nao = self.orth_coeff.shape[1]
        v_add = numpy.zeros((nao,nao))
        for m, atm_lst, bas_idx in self.all_frags:
            emb = self.embs[m]
            nimp = emb.dim_of_impurity()
            vfrag = numpy.array(v_group[m][:nimp,:nimp])
            hop = numpy.dot(vfrag, emb.bath_orb.T)
            for i, j in enumerate(bas_idx):
                v_add[j] = hop[i]
                v_add[j,emb.bas_on_frag] = vfrag[i,:nimp]
        return (v_add + v_add.T) * .5

    def get_all_frag_fci_dm(self, mol):
        dm_group = []
        for m, emb in enumerate(self.embs):
            cires = self.frag_fci_solver(mol, emb)
            dm_group.append(cires['rdm1'])
        return dm_group



def fit_without_local_scf_iter(mol, emb):
    cires = frag_fci(mol, emb, emb.env_fit_pot)
    dm_ref = cires['rdm1']
    nimp = emb.dim_of_impurity()
    # this fock matrix includes the pseudo potential of present fragment
    s = emb.entire_scf.get_ovlp(mol)
    sc = reduce(numpy.dot, (emb.impbas_coeff.T.conj(), s, \
                            emb.entire_scf.mo_coeff))
    fock0 = numpy.dot(sc*emb.entire_scf.mo_energy, sc.T.conj())
    nocc = emb.nelectron/2

    # The damped potential does not minimize |dm_ref - dm(fock0+v)|^2,
    # but it helps convergence
    if DAMPING:
        dv = scfopt.find_emb_potential_damp(mol, dm_ref, fock0, \
                                            nocc, nimp)
    else:
        dv = scfopt.find_emb_potential(mol, dm_ref, fock0, nocc, nimp)
    #vfit_imp = embsys.v_global[emb.bas_on_frag][:,emb.bas_on_frag]
    #return vfit_imp + dv[:nimp,:nimp]
    if isinstance(emb.env_fit_pot, int):
        return dv[:nimp,:nimp]
    else:
        return emb.env_fit_pot[:nimp,:nimp] + dv[:nimp,:nimp]

def fit_pot_without_local_scf(mol, embsys):
    v_group = []
    for m, emb in enumerate(embsys.embs):
        log.debug(mol, 'update local vfit for fragment %s', \
                  str(emb.imp_atoms))
        v_group.append(fit_without_local_scf_iter(mol, emb))
    return v_group

def fit_pot_1shot(mol, embsys, frag_id=0):
    v_group = []
    for emb in embsys.embs:
        nimp = emb.dim_of_impurity()
        v_group.append(numpy.zeros((nimp,nimp)))
    v_group[frag_id] = embsys.fit_without_local_scf_iter(mol, embsys.embs[frag_id])
    return v_group

def fit_pot_with_local_scf(embsys, mol):
    def fit_scfly(mol, emb):
        nimp = emb.dim_of_impurity()
        cires = frag_fci(mol, emb, emb.env_fit_pot)
        dm_ref = cires['rdm1']

        # optimize initial guess to accelerate fragment-SCF convergence
        def _init_guess_method(mol):
            return 0, dm_ref
        _vhf_env_bak = emb._vhf_env.copy()
        emb._vhf_env += emb.env_fit_pot

        emb.init_guess_method = _init_guess_method
        emb.setup_eri_on_impbas(mol)
        nocc = emb.nelectron / 2

        # use existed fitting potential as initial guess
        dv = embsys.v_global[emb.bas_on_frag][:,emb.bas_on_frag]
        icyc = 0
        dv_norm = 1
        if scfopt.FIT_DM_METHOD == scfopt.FIT_DM_IMP_AND_BATH:
            fitpot = scfopt.ImpPot4ImpBathDM()
        elif scfopt.FIT_DM_METHOD == scfopt.FIT_DM_IMP_ONLY:
            fitpot = scfopt.ImpPot4ImpDM()
        elif scfopt.FIT_DM_METHOD == scfopt.FIT_DM_IMP_DIAG:
            fitpot = scfopt.ImpPot4ImpDiag()
        elif scfopt.FIT_DM_METHOD == scfopt.FIT_DM_IMP_ONLY_DIAG_CONSTRAINT:
            fitpot = scfopt.ImpPot4ImpDM_DiagConstr()
        elif scfopt.FIT_DM_METHOD == scfopt.FIT_DM_IMP_ONLY_NELE_CONSTRAINT:
            fitpot = scfopt.ImpPot4ImpDM_NeleConstr()
        while icyc < scfopt.MAX_ITER and dv_norm > scfopt.CONV_THRESHOLD:
            emb._vhf_env[:nimp,:nimp] += dv
            scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff_on_imp \
                    = emb.scf_cycle(mol, dump_chk=False)

            dm = emb.calc_den_mat(mo_coeff_on_imp, mo_occ)
            if scfopt.FIT_DM_METHOD == scfopt.FIT_DM_IMP_AND_BATH:
                norm_ddm1 = numpy.linalg.norm(dm_ref - dm)
            elif scfopt.FIT_DM_METHOD == scfopt.FIT_DM_IMP_ONLY:
                norm_ddm1 = numpy.linalg.norm((dm_ref - dm)[:nimp,:nimp])
            elif scfopt.FIT_DM_METHOD == scfopt.FIT_DM_IMP_DIAG:
                norm_ddm1 = numpy.linalg.norm(numpy.diagonal(dm_ref - dm)[:nimp])
            elif scfopt.FIT_DM_METHOD == scfopt.FIT_DM_IMP_ONLY_DIAG_CONSTRAINT:
                norm_ddm1 = numpy.linalg.norm((dm_ref - dm)[:nimp,:nimp])
            elif scfopt.FIT_DM_METHOD == scfopt.FIT_DM_IMP_ONLY_NELE_CONSTRAINT:
                norm_ddm1 = numpy.linalg.norm((dm_ref - dm)[:nimp,:nimp])
            else:
                norm_ddm1 = numpy.linalg.norm((dm_ref - dm)[:nimp])
            if icyc == 0:
                norm_ddm0 = norm_ddm1
                log.debug(mol, 'before fitting, norm(dm_ref-dm) = %.12g', \
                          norm_ddm1)

            fock = numpy.dot(mo_coeff_on_imp*mo_energy, mo_coeff_on_imp)
            dv = fitpot.generate_pot(mol, dm_ref, fock, nocc, nimp)
            dv_norm = numpy.linalg.norm(dv)
            log.info(mol, '  fragment-iter = %d norm(dm-dm_ref) = %.12g' \
                     'norm(v_{k+1}-v_k) = %.12g', icyc, norm_ddm1, dv_norm)
            icyc += 1
        log.debug(mol, 'after fitting, norm(dm_ref-dm) = %.12g', norm_ddm1)

        v_acc = emb._vhf_env[:nimp,:nimp] - \
                (_vhf_env_bak[:nimp,:nimp] + emb.env_fit_pot[:nimp,:nimp])
        emb._vhf_env = _vhf_env_bak
        emb.release_eri()
        return v_acc

    v_group = []
    for m, emb in enumerate(embsys.embs):
        log.debug(mol, 'update vfit SCFly for fragment %s', \
                  str(emb.imp_atoms))
        v_group.append(fit_scfly(mol, emb))
    return v_group


# fitting potential for FCI
def fit_vfci_fixed_mf_dm(embsys, mol):
    def fit_vfci_fixed_mf_dm_iter(mol, emb):
        s = embsys.entire_scf.get_ovlp(mol)
        mo = reduce(numpy.dot, (emb.impbas_coeff.T.conj(), \
                                s, embsys.entire_scf.mo_coeff))
        occ = emb.entire_scf.mo_occ
        emb_mf_dm = numpy.dot(mo[:,occ>0], mo[:,occ>0].T.conj()) * 2

        cires = frag_fci(mol, emb, emb.env_fit_pot)
        dm_base = cires['rdm1']
        vhf_env_bak = emb._vhf_env.copy()
        nimp = emb.dim_of_impurity()
        if FCI_WITH_ENV_POT:
            emb._vhf_env += emb.env_fit_pot
        else:
            emb._vhf_env[:nimp,:nimp] += emb.env_fit_pot[:nimp,:nimp]

        def get_fci_dm_imp(v_inc):
            cires = frag_fci(mol, emb, v_inc[:nimp,:nimp])
            return cires['rdm1'][:nimp,:nimp].flatten()

        def num_fitting(finite_inc_fac, dm_base, v_inc_base, dm_ref, \
                        max_cycle, dgrad=False):
            nimp = emb.dim_of_impurity()
            walker = [(i,j) for i in range(nimp) for j in range(i+1)]
            v_inc, norm_ddm = numfitor(emb, get_fci_dm_imp, walker, \
                                       dm_ref[:nimp,:nimp].flatten(), \
                                       v_inc_base, finite_inc_fac, dgrad)
            if max_cycle < 1 or norm_ddm < 1e-5:
                return v_inc_base + v_inc
            else:
                return num_fitting(finite_inc_fac, dm_base, v_inc_base+v_inc, \
                                   dm_ref, max_cycle-1, dgrad)

        def approx_analytic_fitting(dm_base, v_inc_base, dm_ref, max_cycle=10):
            ddm = numpy.array(dm_ref[:nimp,:nimp] - dm_base[:nimp,:nimp])
            fitopt = scfopt.ImpPot4ImpDM()
            csc = reduce(numpy.dot, (emb.impbas_coeff.T.conj(), s, \
                                     emb.entire_scf.mo_coeff))
            fock0 = reduce(numpy.dot, (csc, emb.entire_scf.mo_energy, \
                                       csc.T.conj()))
            nocc = emb.nelectron/2

            istep = 0
            ddm_norm = 1
            v_add = v_inc_base
            while istep < max_cycle and ddm_norm > 1e-5:
                e, c = numpy.linalg.eigh(fock0)
                fitopt.set_tensor_x(e, c, nocc, nimp)
                h_approx = fitopt.hessian_approx(e, c, nocc, nimp)
                g_approx = fitopt.gradient(e, c, nocc, nimp, ddm)
                x, nsg = scfopt.step_by_eigh_min(h_approx, \
                                                 numpy.array(g_approx).flatten())
                v_add[:nimp,:nimp] += x.reshape(nimp,nimp)
                fci_dm = get_fci_rdm1(emb, v_add)
                ddm = numpy.array(dm_ref[:nimp,:nimp] - fci_dm[:nimp,:nimp])
                ddm_norm = numpy.linalg.norm(ddm)
                istep += 1
                log.debug(mol, 'iter = %d, ddm_norm = %.9g, dv_norm = %.9g', \
                          istep, ddm_norm, numpy.linalg.norm(x))
            return v_add


        ddm = numpy.array(emb_mf_dm[:nimp,:nimp] - dm_base[:nimp,:nimp])
        log.debug(mol, 'norm(fci_dm-mf_dm) = %.9g', numpy.linalg.norm(ddm))
        v_add = numpy.zeros((nimp,nimp))
        if not True:
            v_add = approx_analytic_fitting(dm_base, v_add, emb_mf_dm)
        else:
            #v_add = approx_analytic_fitting(dm_base, v_add, emb_mf_dm, 2)
            fci_dm = get_fci_rdm1(emb, v_add)
            v_add = num_fitting(1e-4, fci_dm, v_add, emb_mf_dm, 7)

        fci_dm = get_fci_rdm1(emb, v_add)
        ddm = emb_mf_dm[:nimp,:nimp] - fci_dm[:nimp,:nimp]
        log.debug(mol, 'after fitting norm(fci_dm-mf_dm) = %.9g', numpy.linalg.norm(ddm))

        emb._vhf_env = vhf_env_bak
        # The potential removes the correlation in FCI to match mean-field
        # density matrix.  So the correlation potential should be -V
        return -v_add

    v_group = []
    for m, emb in enumerate(embsys.embs):
        log.debug(mol, 'Fitting FCI-DM to match MF-DM for fragment %s', \
                  str(emb.imp_atoms))
        v_group.append(fit_vfci_fixed_mf_dm_iter(mol, emb))
    return v_group


def fit_lieb_pot(embsys, mol):
    def fit_lieb_pot_iter(mol, emb):
        cires = frag_fci(mol, emb, emb.env_fit_pot)
        dm_ref = cires['rdm1']
        nimp = emb.dim_of_impurity()
        # this fock matrix includes the pseudo potential of target fragment
        s = embsys.entire_scf.get_ovlp(mol)
        sc = reduce(numpy.dot, (emb.impbas_coeff.T.conj(), s, \
                                emb.entire_scf.mo_coeff))
        fock0 = reduce(numpy.dot, (sc, emb.entire_scf.mo_energy, sc.T.conj()))
        nocc = emb.nelectron/2
        if True:
            dv = scfopt.find_emb_potential_damp(mol, dm_ref, fock0, \
                                                nocc, nimp)
        else:
            dv = scfopt.find_emb_potential(mol, dm_ref, fock0, nocc, nimp)
        vfit_imp = embsys.v_global[emb.bas_on_frag][:,emb.bas_on_frag]
        return vfit_imp + dv[:nimp,:nimp]

    v_group = []
    for m, emb in enumerate(embsys.embs):
        log.debug(mol, 'Fitting Lieb-pot for fragment %s', \
                  str(emb.imp_atoms))
        v_group.append(fit_lieb_pot_iter(mol, emb))
    return v_group


##################################################
def dump_frag_prop_mat(mol, frag_group, frag_mat_group):
    '''dump fragment potential or density matrix'''
    for m, atm_lst in enumerate(frag_group):
        try:
            mol.fout.write('fragment %s\n' % str(atm_lst))
            fmt = '    %10.5f' * frag_mat_group[m].shape[1] + '\n'
            for c in numpy.array(frag_mat_group[m]):
                mol.fout.write(fmt % tuple(c))
        except:
            # frag_mat_group is 0-d array
            pass


def dmet_scf_cycle(mol, embsys):
    _diis = scf.diis.DIIS(mol)
    #_diis.diis_space = 6
    v_add = embsys.v_global
    e_tot = 0
    icyc = 0

    embsys.update_embsys_vglobal(mol, v_add)
    for icyc in range(MAX_ITER):
        v_add_old = v_add
        e_tot_old = e_tot

        # ABORT
        if DMET_WITH_NAO:
            orth_coeff = embsys.embs[0].get_orth_ao(mol)
            for emb in embsys.embs:
                emb.orth_coeff = orth_coeff
            embsys.orth_coeff = orth_coeff

        log.debug(mol, '  HF energy = %.12g', embsys.entire_scf.hf_energy)
        if FITTING_APPROX == FITTING_WITH_SCF:
            v_group = fit_pot_with_local_scf(embsys, mol)
        elif FITTING_APPROX == FITTING_WITHOUT_SCF:
            v_group = fit_pot_without_local_scf(mol, embsys)
        elif FITTING_APPROX == FITTING_1SHOT:
            v_group = fit_pot_1shot(mol, embsys, 0)
        elif FITTING_APPROX == FITTING_LIEB_POT:
            v_group = fit_lieb_pot(embsys, mol)
        elif FITTING_APPROX == FITTING_FCI_POT:
            v_group = fit_vfci_fixed_mf_dm(mol)

        if WITH_HOPPING:
            v_add = embsys.assemble_to_fullmat(mol, v_group)
        else:
            v_add = embsys.assemble_to_blockmat(mol, v_group)
        #v_add = _diis.update(v_add)
        embsys.update_embsys_vglobal(mol, v_add)
        e_tot, nelec = embsys.assemble_frag_fci_energy(mol)

        if mol.verbose >= param.VERBOSE_DEBUG:
            log.debug(mol, 'fitting fragment potential')
            dump_frag_prop_mat(mol, embsys.frag_group, v_group)

        dv_norm = numpy.linalg.norm(v_add_old - v_add)
        log.info(mol, 'macro iter = %d, e_tot = %.12g, nelec = %g, dv_norm = %g', \
                 icyc, e_tot, nelec, dv_norm)
        de = abs(1-e_tot_old/e_tot)
        log.info(mol, '                 delta_e = %g, (~ %g%%)', \
                 e_tot-e_tot_old, de * 100)

        if dv_norm < scfopt.CONV_THRESHOLD and de < scfopt.CONV_THRESHOLD*.1:
            break
        #import sys
        #if icyc > 1: sys.exit()
    return e_tot, v_group

def scdmet(mol, entire_scf, frag_group, init_v=None, sav_v=None):
    log.info(mol, '==== start DMET SCF ====')
    embsys = EmbSys(mol, entire_scf, frag_group, init_v)
    e_tot, v_group = dmet_scf_cycle(mol, embsys)
    if WITH_HOPPING:
        v_add = embsys.assemble_to_fullmat(mol, v_group)
    else:
        v_add = embsys.assemble_to_blockmat(mol, v_group)

    log.info(mol, '====================')
    if mol.verbose >= param.VERBOSE_DEBUG:
        log.debug(mol, 'V_fitting in AO representation')
        v_add_ao = mat_orthao2ao(mol, v_add, embsys.orth_coeff)
        fmt = '    %10.5f' * v_add_ao.shape[1] + '\n'
        for c in numpy.array(v_add_ao):
            mol.fout.write(fmt % tuple(c))

    e_tot, nelec = embsys.assemble_frag_fci_energy(mol)
    log.log(mol, 'macro iter = X, e_tot = %.11g, +nuc = %.11g, nelec = %.8g', \
            e_tot, e_tot+mol.nuclear_repulsion(), nelec)
    if isinstance(sav_v, str):
        v_add_ao = mat_orthao2ao(mol, v_add, embsys.orth_coeff)
        with open(sav_v, 'w') as f:
            pickle.dump((v_add,v_add_ao), f)
    return e_tot

def dmet_1shot(mol, entire_scf, frag_group, init_v=None, sav_v=None):
    log.info(mol, '==== start DMET 1 shot ====')
    global RAND_INIT
    RAND_INIT = False
    embsys = EmbSys(mol, entire_scf, frag_group)
    e_tot, nelec = embsys.assemble_frag_fci_energy(mol)
    log.log(mol, 'e_tot = %.11g, +nuc = %.11g, nelec = %.8g', \
            e_tot, e_tot+mol.nuclear_repulsion(), nelec)
    return e_tot

def scdmet_1shot(mol, entire_scf, frag_group, init_v=None, sav_v=None):
    global RAND_INIT
    RAND_INIT = False
    global FITTING_APPROX
    FITTING_APPROX = FITTING_1SHOT
    return scdmet(mol, entire_scf, frag_group, init_v, sav_v)


# fitting potential and hopping matrix
def scdmet_hopping(mol, entire_scf, frag_group, init_v=None, sav_v=None):
    scfopt.FIT_DM_METHOD = 4
    global WITH_HOPPING
    WITH_HOPPING = True
    return scdmet(mol, entire_scf, frag_group)


# fitting lieb functional
def scdmet_lieb(mol, entire_scf, frag_group, init_v=None, sav_v=None):
    #scfopt.FIT_DM_METHOD = 4
    global FITTING_APPROX
    FITTING_APPROX = FITTING_LIEB_POT
    return scdmet(mol, entire_scf, frag_group, init_v, sav_v)


# backwards fitting: MF DM fixed
def scdmet_fci_pot(mol, entire_scf, frag_group):
    global FITTING_APPROX
    FITTING_APPROX = FITTING_FCI_POT

    log.info(mol, '==== start DMET SCF ====')
    embsys = EmbSys(mol, entire_scf, frag_group)
    e_tot, v_group = dmet_scf_cycle(mol, embsys)
    v_add = embsys.assemble_to_blockmat(mol, v_group)
    embsys.update_embsys_vglobal(mol, v_add)

    log.info(mol, '====================')
    e_tot, nelec = embsys.assemble_frag_fci_energy(mol)
    log.log(mol, 'macro iter = X, e_tot = %.11g, +nuc = %.11g, nelec = %.8g', \
            e_tot, e_tot+mol.nuclear_repulsion(), nelec)
    return e_tot

# optmize potential of entire system with fixed local FCI density matrix. Then
# update local FCI density matrix.
def scdmet_vglobal(mol, entire_scf, frag_group, init_v=None, sav_v=None):
    embsys = EmbSys(mol, entire_scf, frag_group, init_v)
    v_add = embsys.v_global
    e_tot = 0
    istep = 0
    embsys.update_embsys_vglobal(mol, v_add)
    sc = numpy.dot(entire_scf.get_ovlp(mol), embsys.orth_coeff)
    nocc = mol.nelectron / 2
    mo = numpy.dot(sc.T.conj(), embsys.entire_scf.mo_coeff)
    dm_mf0 = numpy.dot(mo[:,:nocc], mo[:,:nocc].T) * 2
    for istep in range(MAX_ITER):
        v_add_old = v_add
        e_tot_old = e_tot
        if GLOBAL_FIT == NO_FIXED_DM:
            v_add = numfit_global_pot_no_fixed_dm(mol, embsys)
            embsys.update_embsys_vglobal(mol, v_add)
            v_add = numfit_global_pot_no_fixed_dm(mol, embsys, dgrad=True)
        elif GLOBAL_FIT == FIXED_CI_DM:
            v_add = numfit_global_pot_with_fixed_fci_dm(mol, embsys)
            embsys.update_embsys_vglobal(mol, v_add)
            v_add = numfit_global_pot_with_fixed_fci_dm(mol, embsys, dgrad=True)
            #v_add = fit_global_pot_with_fixed_fci_dm(mol, embsys)
        elif GLOBAL_FIT == FIXED_MF_DM:
            v_add = numfit_global_pot_with_fixed_mf_dm(mol, embsys, dm_ref=dm_mf0)
            embsys.update_embsys_vglobal(mol, v_add)
            v_add = numfit_global_pot_with_fixed_mf_dm(mol, embsys, dm_ref=dm_mf0, dgrad=True)
            #v_add = numfit_global_pot_with_fixed_mf_dm(mol, embsys)
            #embsys.update_embsys_vglobal(mol, v_add)
            #v_add = numfit_global_pot_with_fixed_mf_dm(mol, embsys, dgrad=True)
        elif GLOBAL_FIT == NO_FIXED_DM_BACKWARDS:
            v_add = numfit_global_pot_backwards(mol, embsys)
            embsys.update_embsys_vglobal(mol, v_add)
            v_add = numfit_global_pot_backwards(mol, embsys, dgrad=True)
        embsys.update_embsys_vglobal(mol, v_add)
        e_tot, nelec = embsys.assemble_frag_fci_energy(mol)

        if mol.verbose >= param.VERBOSE_DEBUG:
            log.debug(mol, 'fragment fitting potential')
            v_group = []
            for m, atm_lst, bas_idx in embsys.uniq_frags:
                v_group.append(v_add[bas_idx][:,bas_idx])
            dump_frag_prop_mat(mol, embsys.frag_group, v_group)

            log.debug(mol, 'fragment DM')
            sc = numpy.dot(entire_scf.get_ovlp(mol), embsys.orth_coeff)
            nocc = mol.nelectron / 2
            mo = numpy.dot(sc.T.conj(), embsys.entire_scf.mo_coeff)
            dm_mf = numpy.dot(mo[:,:nocc], mo[:,:nocc].T)
            dm_group = []
            for m, atm_lst, bas_idx in embsys.uniq_frags:
                dm_group.append(dm_mf[bas_idx][:,bas_idx])
            dump_frag_prop_mat(mol, embsys.frag_group, dm_group)

        dv_norm = numpy.linalg.norm(v_add_old - v_add)
        log.info(mol, 'macro iter = %d, e_tot = %.12g, nelec = %g, dv_norm = %g', \
                 istep, e_tot, nelec, dv_norm)
        de = abs(1-e_tot_old/e_tot)
        log.info(mol, '                 delta_e = %g, (~ %g%%)', \
                 e_tot-e_tot_old, de * 100)
        if dv_norm < scfopt.CONV_THRESHOLD and de < scfopt.CONV_THRESHOLD*.1:
            break

    if mol.verbose >= param.VERBOSE_DEBUG:
        log.debug(mol, '** mo_coeff of MF sys (on orthogonal AO) **')
        c = numpy.dot(numpy.linalg.inv(embsys.orth_coeff), \
                      embsys.entire_scf.mo_coeff)
        scf.hf.dump_orbital_coeff(mol, c)

    log.info(mol, 'macro iter = X, e_tot = %.9g, +nuc = %.9g, nelec = %.9g', \
             e_tot, e_tot+mol.nuclear_repulsion(), nelec)
    if isinstance(sav_v, str):
        v_add_ao = mat_orthao2ao(mol, v_add, embsys.orth_coeff)
        with open(sav_v, 'w') as f:
            pickle.dump((v_add,v_add_ao), f)
    return e_tot

def fit_global_pot_with_fixed_fci_dm(mol, embsys, dgrad=False, dm_ref=None):
    # optmize potential of entire system with fixed local FCI DM.
    sc = numpy.dot(entire_scf.get_ovlp(mol), embsys.orth_coeff)
    mo = numpy.dot(sc.T.conj(), embsys.entire_scf.mo_coeff)
    mo_e = embsys.entire_scf.mo_energy
    nocc = mol.nelectron / 2
    nmo = mo_e.shape[0]
    nvir = nmo - nocc

    eia = 1 / (e[:nocc].reshape(-1,1) - e[nocc:]).flatten()
    cc_ia = numpy.empty((nmo*nmo,nocc*nvir))
    cc_ai = numpy.empty((nmo*nmo,nocc*nvir))
    idx = []
    p0 = 0
    p1 = 0
    for m, atm_lst, bas_idx in embsys.all_frags:
        nimp = embsys.embs[m].dim_of_impurity()
        tmpcc_ai = numpy.empty((nimp,nimp,nocc,nvir))
        for t0,t in enumerate(bas_idx):
            for u0,u in enumerate(bas_idx):
                for i in range(nocc):
                    for a in range(nvir):
                        tmpcc_ai[t0,u0,i,a] = mo[t,nocc+a] * mo[u,i]
        tmpcc_ia = numpy.rollaxis(tmpcc_ai, 1, 0)
        cc_ai[p0:p0+nimp*nimp,:] = tmpcc_ai.reshape(nimp*nimp,nocc*nvir)
        cc_ia[p0:p0+nimp*nimp,:] = tmpcc_ia.reshape(nimp*nimp,nocc*nvir)

        for i in bas_idx:
            for j in bas_idx:
                idx.append(i*nmo+j)
        p0 += nimp * nimp
    dm_group = embsys.get_all_frag_fci_dm(mol)
    if dm_ref is None:
        fci_dm = embsys.assemble_to_blockmat(mol, dm_group).flatten()[idx]
    else:
        fci_dm = dm_ref.flatten()[idx]

    def symmetrize(x):
        xsym = numpy.empty_like(x)
        p0 = 0
        p1 = 0
        # remove anti-symmetric part
        for m, atm_lst, bas_idx in embsys.all_frags:
            nimp = bas_idx.__len__()
            u = scfopt.symm_trans_mat_for_hermit(nimp)
            xsym[:,p1:p1+nimp*(nimp+1)/2] = numpy.dot(x[:,p0:p0+nimp*nimp], u)
            p0 += nimp * nimp
            p1 += nimp * (nimp + 1) / 2
        return xsym[:,:p1]

    def desymmetrize(x):
        xtmp = numpy.empty(nmo*nmo)
        p0 = 0
        p1 = 0
        for m, atm_lst, bas_idx in embsys.all_frags:
            nimp = bas_idx.__len__()
            u = scfopt.symm_trans_mat_for_hermit(nimp)
            xtmp[p0:p0+nimp*nimp] = numpy.dot(u, x[p1:p1+nimp*(nimp+1)/2])
            p0 += nimp * nimp
            p1 += nimp * (nimp + 1) / 2
        return xtmp

    dm_mf = numpy.dot(mo[:,:nocc], mo[:,:nocc].T)
    ddm = fci_dm * .5 - dm_mf.flatten()[idx]
    resps = numpy.dot(cc_ai[:p0] * eia, cc_ai[:p0].T) \
            + numpy.dot(cc_ia[:p0] * eia, cc_ia[:p0].T)
    resps = resps.T

    if dgrad:
        x, grad, sgl = gen_approx_step(mol, resps, ddm, dgrad)
    else:
        resps = symmetrize(resps).T
        x, grad, sgl = gen_approx_step(mol, resps, ddm, dgrad)
        x = desymmetrize(x)

    dv = numpy.zeros(nmo*nmo)
    dv[idx] = x
    dv = dv.reshape(nmo,nmo)

    fock0 = numpy.dot(mo * mo_e, mo.T)
    def norm_ddm_method(dv):
        e, c = numpy.linalg.eigh(fock0+dv)
        dm_mf = numpy.dot(c[:,:nocc], c[:,:nocc].T)
        ddm = fci_dm * .5 - dm_mf.flatten()[idx]
        return numpy.linalg.norm(ddm)
    dv, norm_ddm = scfopt.line_search_sharp(mol, norm_ddm_method, dv, \
                                            floating=1e-3, \
                                            title='analytic global fitting')

    log.debug(mol, 'after fitting, norm(dm_ref-dm_mf) = %.9g,' \
              ' norm(dv) = %.9g, singular = %d', \
              norm_ddm, numpy.linalg.norm(dv), sgl)
    e, c = numpy.linalg.eigh(fock0+dv)
    log.debug(mol, '                  ??? homo = %.9g, lumo = %.9g', \
              e[nocc-1], e[nocc])
    e_tot, nelec = embsys.assemble_frag_fci_energy(mol)
    return e_tot, nelec, embsys.v_global + dv

def numfit_global_pot_with_fixed_fci_dm(mol, embsys, dgrad=False, dm_ref=None):
    # optmize potential of entire system with fixed FCI DM.
    sc = numpy.dot(entire_scf.get_ovlp(mol), embsys.orth_coeff)
    mo = numpy.dot(sc.T.conj(), embsys.entire_scf.mo_coeff)
    mo_e = embsys.entire_scf.mo_energy
    nocc = mol.nelectron / 2
    nmo = mo_e.shape[0]

    idx = []
    for m, atm_lst, bas_idx in embsys.all_frags:
        for i in bas_idx:
            for j in bas_idx:
                idx.append(i*nmo+j)
    dm_group = embsys.get_all_frag_fci_dm(mol)
    if dm_ref is None:
        dm_ref = embsys.assemble_to_blockmat(mol, dm_group).flatten()[idx]
    else:
        dm_ref = dm_ref.flatten()[idx]
    def get_mf_rdm1(v_add):
        embsys.update_embsys_vglobal(mol, v_add)
        mo = numpy.dot(sc.T.conj(), embsys.entire_scf.mo_coeff)
        dm = numpy.dot(mo[:,:nocc], mo[:,:nocc].T)
        return dm.flatten()[idx]
    v_inc, norm_ddm = numfitor(embsys, get_mf_rdm1, \
                               gen_global_uniq_walker(mol, embsys), \
                               dm_ref*.5, embsys.v_global, 1e-4, dgrad)
    log.debug(mol, '                  ??? homo = %.9g, lumo = %.9g', \
              embsys.entire_scf.mo_energy[nocc-1], \
              embsys.entire_scf.mo_energy[nocc])
    return embsys.v_global+v_inc

def numfit_global_pot_no_fixed_dm(mol, embsys, dgrad=False):
    # min|DM_FCI[V] - DM_MF[V]|^2. Neither FCI nor MF DM are fixed. Both are
    # the functional of fitting potential
    sc = numpy.dot(embsys.entire_scf.get_ovlp(mol), embsys.orth_coeff)
    nocc = mol.nelectron / 2
    nao = embsys.orth_coeff.shape[1]

    idx = []
    for m, atm_lst, bas_idx in embsys.all_frags:
        for i in bas_idx:
            for j in bas_idx:
                idx.append(i*nao+j)

    def ddm_fci_mf(v_global):
        embsys.update_embsys_vglobal(mol, v_global)
        mo = numpy.dot(sc.T.conj(), embsys.entire_scf.mo_coeff)
        dm_mf = numpy.dot(mo[:,:nocc], mo[:,:nocc].T) * 2

        dm_group = embsys.get_all_frag_fci_dm(mol)
        fci_dm = embsys.assemble_to_blockmat(mol, dm_group)
        return fci_dm.flatten()[idx] - dm_mf.flatten()[idx]

    v_inc, norm_ddm = numfitor(embsys, ddm_fci_mf, \
                               gen_global_uniq_walker(mol, embsys), \
                               0, embsys.v_global, 1e-4, dgrad)
    return embsys.v_global+v_inc

def numfit_global_pot_with_fixed_mf_dm(mol, embsys, dgrad=False, dm_ref=None):
    sc = numpy.dot(embsys.entire_scf.get_ovlp(mol), embsys.orth_coeff)
    nocc = mol.nelectron / 2
    nao = embsys.orth_coeff.shape[1]

    idx = []
    for m, atm_lst, bas_idx in embsys.all_frags:
        for i in bas_idx:
            for j in bas_idx:
                idx.append(i*nao+j)

    mo = numpy.dot(sc.T.conj(), embsys.entire_scf.mo_coeff)
    if dm_ref is None:
        dm_ref = numpy.dot(mo[:,:nocc], mo[:,:nocc].T).flatten()[idx]
    else:
        dm_ref = dm_ref.flatten()[idx]
    def get_fci_dm(v_global):
        embsys.update_embsys_vglobal(mol, v_global)
        for emb in embsys.embs:
            nimp = emb.dim_of_impurity()
            # "-" in v_loc/v_inc/v_global means to remove the correlation from CI calcluation
            v_loc = v_global[emb.bas_on_frag][:,emb.bas_on_frag]
            emb.env_fit_pot[:nimp,:nimp] = -v_loc
        dm_group = embsys.get_all_frag_fci_dm(mol)
        fci_dm = embsys.assemble_to_blockmat(mol, dm_group)
        return fci_dm.flatten()[idx]
    v_inc, norm_ddm = numfitor(embsys, get_fci_dm, \
                               gen_global_uniq_walker(mol, embsys), \
                               dm_ref, embsys.v_global, .5e-4, dgrad)
    return embsys.v_global+v_inc


# DM_MF(H-[V1..VN])
# DM_CI(H+[V1..VN]+{-V1..-VN}), {V1..VN} is transformed from [V1..VN] to bath
# min(DM_MF - DM_CI)
def numfit_global_pot_backwards(mol, embsys, dgrad=False):
    sc = numpy.dot(embsys.entire_scf.get_ovlp(mol), embsys.orth_coeff)
    nao = embsys.orth_coeff.shape[1]

    idx = []
    for m, atm_lst, bas_idx in embsys.all_frags:
        for i in bas_idx:
            for j in bas_idx:
                idx.append(i*nao+j)

    # "-" in v_loc/v_inc/v_global means the removed correlation from CI calcluation
    def get_fci_dm(v_global):
        eff_scf = embsys.update_embsys_vglobal(mol, v_global)
        mo = numpy.dot(sc.T.conj(), eff_scf.mo_coeff[:,eff_scf.mo_occ>0])
        dm_mf = numpy.dot(mo, mo.T)
        for emb in embsys.embs:
            nimp = emb.dim_of_impurity()
            v_loc = -v_global[emb.bas_on_frag][:,emb.bas_on_frag]
            emb.env_fit_pot[:nimp,:nimp] = v_loc
        dm_group = embsys.get_all_frag_fci_dm(mol)
        fci_dm = embsys.assemble_to_blockmat(mol, dm_group)
        return fci_dm.flatten()[idx] * .5 - dm_mf.flatten()[idx]
    v_inc, norm_ddm = numfitor(embsys, get_fci_dm, \
                               gen_global_uniq_walker(mol, embsys), \
                               0, embsys.v_global, .5e-4, dgrad)
    return embsys.v_global+v_inc


if __name__ == '__main__':
    import scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = "out_scfci"

    mol.atom.extend([
        ['C' , ( 0. , 0.  , 0.)],
        ['H' , ( 0.7, 0.7 , 0.7)],
        ['H' , ( 0.7,-0.7 ,-0.7)],
        ['H' , (-0.7, 0.7 ,-0.7)],
        ['H' , (-0.7,-0.7 , 0.7)] ])
    mol.basis = {'C': 'sto_6g',
                 'H': 'sto_6g',}
    mol.grids = {'C': (100,110),
                 'H': (100,110)}
    mol.build()

    rhf = scf.RHF(mol)
    #rhf.scf_threshold = 1e-10
    rhf.chkfile = 'ch4.chkfile'
    rhf.init_guess('chkfile')
    print "E=", rhf.scf(mol)

    frag_group = [(0,), ((1,), (2,), (3,), (4,),) ]
    #frag_group = [(0,), (1,2,3,4,)]
    #frag_group = [(0,1,), (2,3,4,)]
    #frag_group = [(0,1,2,), (3,4,)]
    #frag_group = [(0,1,2,3,), (4,)]
    #print scdmet(mol, rhf, frag_group) #-52.2588738758
    #print dmet_1shot(mol, rhf, frag_group) #-52.2674000845
    #print scdmet_1shot(mol, rhf, frag_group)
    #print scdmet_fci_pot(mol, rhf, frag_group)
    print scdmet_vglobal(mol, rhf, frag_group)

