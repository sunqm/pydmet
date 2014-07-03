#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

'''
DMET Hartree-Fock
'''

__author__ = 'Qiming Sun <osirpt.sun@gmail.com>'
__version__ = '$ 0.2 $'

import os
import tempfile
import numpy
import scipy.linalg.flapack as lapack
import h5py

import gto
from pyscf import scf
from pyscf import lib
import pyscf.lib.parameters as param
import pyscf.lib.logger as log
import pyscf.lib.pycint as pycint
from pyscf import ao2mo

def lowdin_orth_coeff(s):
    ''' new basis is |mu> c^{lowdin}_{mu i} '''
    e, v, info = lapack.dsyev(s)
    return numpy.dot(v/numpy.sqrt(e), v.T.conj())

def schmidt_orth_coeff(s):
    c = numpy.linalg.cholesky(s)
    return numpy.linalg.inv(c).T.conj()

def pre_orth_ao_atm_scf(mol):
    atm_scf = scf.atom_hf.get_atm_nrhf_result(mol)
    nbf = mol.num_NR_function()
    c = numpy.zeros((nbf, nbf))
    p0 = 0
    for ia in range(mol.natm):
        symb = mol.symbol_of_atm(ia)
        if atm_scf.has_key(symb):
            e_hf, mo_e, mo_occ, mo_c = atm_scf[symb]
        else:
            symb = mol.pure_symbol_of_atm(ia)
            e_hf, mo_e, mo_occ, mo_c = atm_scf[symb]
        p1 = p0 + mo_e.size
        c[p0:p1,p0:p1] = mo_c
        p0 = p1
    log.debug(mol, 'use SCF AO instead of input basis')
    return c

def pre_orth_ao_projected_ano(mol):
    raise('FIX ME')
    import copy
    pmol = mol.copy()

    nbf = mol.num_NR_function()
    c = numpy.eye(nbf)
    p0 = 0
    for ia in range(mol.natm):
        bras = []
        for ib in range(mol.nbas):
            if mol.atom_of_bas(ib) == ia:
                bras.append(ib)

        symb = mol.symbol_of_atm(ia)
        basis_ano = gto.basis.ano[symb]
        k0 = pmol.nbas
        pmol._bas.extend(pmol.make_bas_env_by_atm_id(ia, basis_ano))
        pmol.nbas = pmol._bas.__len__()
        kets = range(k0,pmol.nbas)

        s = pmol.intor_cross('cint1e_ovlp_sph', bras, bras)
        sinv = numpy.linalg.inv(s)
        cross = pmol.intor_cross('cint1e_ovlp_sph', bras, kets)
        c_ao = numpy.dot(sinv, cross)

        idx_ano = [[] for i in range(6)]
        k = 0
        for b in basis_ano:
            l = b[0]
            nctr = b[1].__len__() - 1
            idx_ano[l].extend(range(k, k+nctr*(l*2+1)))
            k += nctr * (l*2+1)
        idx_ao = [[] for i in range(6)]
        k = 0
        for b in mol.basis[symb]:
            l = b[0]
            nctr = b[1].__len__() - 1
            idx_ao[l].extend(range(k, k+nctr*(l*2+1)))
            k += nctr * (l*2+1)
        for l in range(6):
            len_ao = idx_ao[l].__len__()
            for i in idx_ao[l]:
                for j0, j in enumerate(idx_ano[l]):
                    if j0 < len_ao:
                        c[p0+i,p0+idx_ao[l][j0]] = c_ao[i,j]
        p0 += c_ao.shape[0]

    s = mol.intor_symmetric('cint1e_ovlp_sph')
    for i,a in enumerate(numpy.diagonal(reduce(numpy.dot,(c.T.conj(),s,c)))):
        c[:,i] *= 1 / numpy.sqrt(a)
    log.debug(mol, 'use projected ANO instead of input basis')
    return c

# scf_method is not required unless orth_method is 'nao'
def orthogonalize_ao(mol, scf_method, pre_orth_ao=None, orth_method='meta_lowdin'):
    s = mol.intor_symmetric('cint1e_ovlp_sph')

    if pre_orth_ao is None:
        nbf = mol.num_NR_function()
        pre_orth_ao = numpy.eye(nbf)

    if orth_method == 'lowdin':
        log.debug(mol, 'orthogonalize AOs with lowdin scheme')
        c_orth = numpy.dot(pre_orth_ao, \
                lowdin_orth_coeff(reduce(numpy.dot, (pre_orth_ao.T.conj(), s,
                                                     pre_orth_ao))))
    elif orth_method == 'nao':
        log.debug(mol, 'orthogonalize AOs with NAO')
        o = dmet.nao.NAO(mol, scf_method)
        c_orth = o.nao_coeff(mol)
    else: # meta_lowdin: divide ao into core, valence and Rydberg sets,
          # orthogonalizing within each set
        log.debug(mol, 'orthogonalize AOs with meta lowdin scheme')
        nbf = mol.num_NR_function()
        weight = numpy.ones(nbf)
        c_orth = dmet.nao._nao_sub(mol, weight, pre_orth_ao)
    # adjust phase
    sc = numpy.dot(s, c_orth)
    for i in range(c_orth.shape[1]):
        if sc[i,i] < 0:
            c_orth[:,i] *= -1
    return c_orth

def select_ao_on_fragment(mol, atm_lst, bas_idx=[]):
    log.info(mol, 'atm_lst of impurity sys: %s', \
             str(atm_lst))
    log.info(mol, 'extra bas_idx of impurity sys: %s', \
             str(bas_idx))
    lbl = mol.labels_of_spheric_GTO()
    bas_on_a = []
    for ia in atm_lst: # ensure the order of imp_site consiste with imp_atoms
        for ib, s in enumerate(lbl):
            if s[0] == ia:
                bas_on_a.append(ib)
    if bas_idx:
        return bas_on_a + [i for i in bas_idx if i not in bas_on_a]
    else:
        return bas_on_a

def _pick_bath_idx(w, num_bath, occ_env_cutoff):
    sorted_w = numpy.argsort(abs(w-.5)) # order the entanglement
                                        # so most important bath comes frist
    if num_bath == -1:
        bath_idx = [i for i in sorted_w \
                    if occ_env_cutoff<w[i]<1-occ_env_cutoff]
        env_idx = [i for i in sorted_w if w[i]<occ_env_cutoff]
        rest_idx = [i for i in sorted_w if w[i]>1-occ_env_cutoff]
    else:
# we prefer the bath which captures the electrons in the impurity virtual
# space, because more electrons will be in the virtual space when DZ,TZ basis
# is used. The orbitals with occ>0.99? never be considered as bath
        #bath_idx = sorted_w[:num_bath]
        bath_idx = [i for i in sorted_w if w[i]<1-occ_env_cutoff][:num_bath]
        thrd = min(w[bath_idx[-1]], 1-w[bath_idx[-1]]) - 1e-12
        env_idx = [i for i in sorted_w if w[i] < thrd]
        rest_idx = [i for i in sorted_w if w[i]>1-thrd]
    return bath_idx, env_idx, rest_idx

def decompose_den_mat(emb, dm_orth, bas_on_frag, num_bath=-1):
    nimp = bas_on_frag.__len__()
    frag_dm = dm_orth[bas_on_frag,:][:,bas_on_frag]
    log.debug(emb, 'entanglement weight (= sqrt(occs)),  occ')
    w, pre_nao = numpy.linalg.eigh(frag_dm)
    idx, not_idx, rest_idx = _pick_bath_idx(w, num_bath, emb.occ_env_cutoff)
    if emb.verbose >= param.VERBOSE_DEBUG:
        for i in idx:
            log.debug(emb, '%d th weight = %12.9f => bath (%s)', \
                      i, w[i], ('acceptor' if w[i]>0.5 else 'donor'))
        for i in not_idx:
            log.debug(emb, '%d th weight = %12.9f => env', i, w[i])
        for i in rest_idx:
            log.debug(emb, '%d th weight = %12.9f => rest/imp', i, w[i])
        log.debug(emb, 'potentially change in embsys charge = %12.9f', \
                  sum(w[not_idx])+sum(w[rest_idx]-1))
    pre_bath = numpy.dot(dm_orth[:,bas_on_frag], \
                         pre_nao[:,idx]) / numpy.sqrt(w[idx])
    if pre_bath.shape[1] > 0:
        mo_bath[bas_on_frag] = 0
        norm = 1/numpy.sqrt(1-w[idx]**2)
        bath_orb = mo_bath * norm
    else:
        bath_orb = pre_bath
    log.debug(emb, 'number of bath orbital = %d', bath_orb.shape[1])
    if emb.verbose >= param.VERBOSE_DEBUG:
        log.debug(emb, ' ** bath orbital coefficients (on orthogonal basis) **')
        scf.hf.dump_orbital_coeff(emb.mol, bath_orb)

    proj = -numpy.dot(bath_orb, bath_orb.T)
    for i in range(bath_orb.shape[0]):
        proj[i,i] += 1
    for i in bas_on_frag:
        proj[i,i] -= 1
    dm_env = reduce(numpy.dot, (proj, dm_orth, proj))
    w, env_orb = numpy.linalg.eigh(dm_env)
    env_orb = env_orb[:,w>.1] #remove orbitals with eigenvalue ~ 0
    return numpy.eye(nimp), bath_orb, env_orb

def decompose_orbital(emb, mo_orth, bas_on_frag, num_bath=-1):
    log.debug(emb, 'occupied mo shape = %d, %d', *mo_orth.shape)
    log.debug(emb, 'number of basis on fragment = %d', \
              bas_on_frag.__len__())

    log.debug(emb, '*** decompose orbitals to fragment sites, '\
              'bath, env orbitals ***')

    fmo = mo_orth[bas_on_frag]
    pre_nao, w1, pre_env_h = numpy.linalg.svd(fmo)
    mo1 = numpy.dot(mo_orth, pre_env_h.T.conj())
    w = numpy.zeros(mo_orth.shape[1])
    w[:w1.size] = w1   # when nimp < nmo, adding 0s by the end

    idx, not_idx, rest_idx = _pick_bath_idx(w1**2, num_bath, emb.occ_env_cutoff)
    env_idx = not_idx + range(w1.size, mo_orth.shape[1])
    mo_bath = mo1[:,idx]
    env_orb = mo1[:,env_idx]
    log.info(emb, 'number of proto bath orbital = %d', mo_bath.shape[1])
    log.info(emb, 'number of env orbitals = %d', env_orb.shape[1])
    log.debug(emb, 'entanglement weight (= sqrt(occs)),  occ')
    if emb.verbose >= param.VERBOSE_DEBUG:
        for i in idx:
            log.debug(emb, '%d th weight = %12.9f, %12.9f  => bath (%s)', \
                      i, w[i], w[i]**2, \
                      ('acceptor' if w[i]**2>0.5 else 'donor'))
        for i in env_idx:
            log.debug(emb, '%d th weight = %12.9f, %12.9f  => env', \
                      i, w[i], w[i]**2)
        for i in rest_idx:
            log.debug(emb, '%d th weight = %12.9f, %12.9f => rest/imp', \
                      i, w[i], w[i]**2)
        log.debug(emb, 'potentially change in embsys charge = %12.9f', \
                  sum(w[not_idx]**2)+sum(w[rest_idx]**2-1))
        #log.debug(emb, ' ** env orbital coefficients (on orthogonal basis)**')
        #scf.hf.dump_orbital_coeff(emb.mol, env_orb)

    if mo_bath.shape[1] > 0:
#ABORT        u, s, vh = numpy.linalg.svd(mo_bath[bas_on_frag,:])
#ABORT        log.debug(emb, 'number of frag sites = %d', \
#ABORT                  bas_on_frag.__len__())
#ABORT        log.debug(emb, 'lowdin orthogonalized fragment sites:')
#ABORT        for i, si in enumerate(s):
#ABORT            if si > emb.lindep_cutoff:
#ABORT                log.debug(emb, '%d th sqrt(eigen_value) = %12.9f  => frag', i, si)
#ABORT            else:
#ABORT                log.debug(emb, '%d th sqrt(eigen_value) = %12.9f', i, si)
        mo_bath[bas_on_frag] = 0
        norm = 1/numpy.sqrt(1-w[idx]**2)
        bath_orb = mo_bath * norm
    else:
        bath_orb = mo_bath
    #if emb.verbose >= param.VERBOSE_DEBUG:
    #    log.debug(emb, ' ** bath orbital coefficients (on orthogonal basis) **')
    #    scf.hf.dump_orbital_coeff(emb.mol, bath_orb)

    nimp = bas_on_frag.__len__()
    return numpy.eye(nimp), bath_orb, env_orb

def padding0(mat, dims):
    mat0 = numpy.zeros(dims)
    n, m = mat.shape
    mat0[:n,:m] = mat
    return mat0

class RHF(scf.hf.RHF):
    '''Non-relativistic restricted Hartree-Fock DMET'''
    def __init__(self, entire_scf, orth_ao=None):
        self.verbose = entire_scf.verbose
        self.fout = entire_scf.fout

        self.lindep_cutoff = 1e-12
# * non-zero occ_env_cutoff can remove core electrons from bath but will
#   introduce non-integer total electrons.  In such case, the sum of the
#   fragment electron density/energy does not reproduce the density/energy of
#   the whole system
# * large occ_env_cutoff may remove some electrons from the fragments to the
#   envrionment orbitals.  Comparing to the population analysis with the full
#   calculation, the pop of framgent due to impurity will show postive
#   charge/less electrons.
# * large occ_env_cutoff may have orthognality problem. When occ_env_cutoff ~ 0,
#   fragment/bath/env orbitals are orthogonal to each other.  If
#   occ_env_cutoff is not 0, the orthognality between fragment and env will be
#   destroyed.
# * when occ_env_cutoff is 0, HF-in-HF embedding does not need extra scf
#   because the space of occupied orbitals is not changed and the total HF
#   potential due to the embedding system is the same to the original one.
#   But when this value > 0, it can happen that few impurity orbitals are not
#   orthogoanl to the env_orb.  Projecting HF-potential of env_orb into
#   the impurity basis (imp+bath) may bring a little contribution from
#   HF-potential of env_orb.  The impurity basis may recount a few
#   contribution of env_orb.  This causes extra SCF in impurity and a slight
#   difference between the converged imp_scf solution and the original SCF
#   result.
        self.occ_env_cutoff = 1e-3 # an MO is considered as env_orb when imp_occ < 1e-3
        self.num_bath = -1
        self.chkfile = entire_scf.chkfile

        #if not entire_scf.scf_conv:
        #    log.info(self, "SCF again before DMET.")
        #    entire_scf.scf_cycle(mol, entire_scf.scf_threshold*1e2)
        self.entire_scf = entire_scf
        self.mol = entire_scf.mol

        self.imp_atoms = []
        self.imp_basidx = []

#ABORT        self.set_bath_orth_by_svd()
        if orth_ao is None:
            self.set_ao_with_input_basis()
            #self.set_ao_with_atm_scf()
            #self.set_ao_with_projected_ano()
            self.init_with_lowdin_ao()
            #self.init_with_meta_lowdin_ao()
            self.orth_coeff = None
        else:
            self.orth_coeff = orth_ao

        # when bath is truncated, impurity site and env-core-orb are
        # non-orthogonal to each other.  Project env-core-orb from impurity.
        self.orth_imp_to_env = False

        # imp_site, bath_orb, env_orb are represented on orthogonal basis
        self.imp_site = 1
        self.bath_orb = None
        self.env_orb = None
        self.impbas_coeff = None
        self.nelectron = None
        self._vhf_env = 0
        self.diis_start_cycle = 3
        self.diis_space = entire_scf.diis_space
        self.scf_threshold = entire_scf.scf_threshold
        self.max_scf_cycle = entire_scf.max_scf_cycle
        self.direct_scf = entire_scf.direct_scf
        self.direct_scf_threshold = entire_scf.direct_scf_threshold
        self.level_shift_factor = 0
        self.damp_factor = 0
        self.scf_conv = False
        self.direct_scf = True
        self.direct_scf_threshold = 1e-13
        self._eri = None

    def dump_options(self):
        log.info(self, '\n')
        log.info(self, '******** DMET options *************')
        log.info(self, 'method = %s', self.__doc__)
        log.info(self, 'linear dependent cutoff = %g', self.lindep_cutoff)
        log.info(self, 'bath/env cutoff = %g', self.occ_env_cutoff)
        log.info(self, 'num_bath = %g\n', self.num_bath)

    def decompose_den_mat(self, dm_orth):
        return decompose_den_mat(self, dm_orth*.5, self.bas_on_frag, self.num_bath)
    def decompose_orbital(self, mo_orth):
        return decompose_orbital(self, mo_orth, self.bas_on_frag, self.num_bath)

# use set_bath_orth_by_svd by default, see decompose_orbital, decompose_den_mat
#ABORT    def set_bath_orth_by_svd(self):
#ABORT        def bath_orth(mo_bath):
#ABORT            log.debug(self, 'lowdin orthogonalized bath orbitals:')
#ABORT            u, s, vh = numpy.linalg.svd(mo_bath)
#ABORT            n_bath = (s>self.lindep_cutoff).sum()
#ABORT            log.info(self, 'number of bath orbitals = %d', n_bath)
#ABORT            for i, si in enumerate(s):
#ABORT                if si > self.lindep_cutoff:
#ABORT                    log.debug(self, '%d th sqrt(eigen_value) = %12.9f  => bath', i, si)
#ABORT                else:
#ABORT                    log.debug(self, '%d th sqrt(eigen_value) = %12.9f', i, si)
#ABORT            bath = numpy.dot(mo_bath, vh[s>self.lindep_cutoff,:].T.conj()) \
#ABORT                    / s[s>self.lindep_cutoff]
#ABORT            return bath
#ABORT        self.orthogonalize_bath = bath_orth
#ABORT
#ABORT    def set_bath_orth_by_qr(self):
#ABORT        def bath_orth(mo_bath):
#ABORT            log.debug(self, 'Schmidt orthogonalized bath orbitals:')
#ABORT            v, r = numpy.linalg.qr(mo_bath[:,::-1])
#ABORT            s = r.diagonal()
#ABORT            n_bath = (s>self.lindep_cutoff).sum()
#ABORT            log.info(self, 'number of bath orbitals = %d', n_bath)
#ABORT            for i, si in enumerate(s):
#ABORT                if si > self.lindep_cutoff:
#ABORT                    log.debug(self, '%d th norm = %12.9f  => bath', i, si)
#ABORT                else:
#ABORT                    log.debug(self, '%d th norm = %12.9f', i, si)
#ABORT            bath = v[:,s>self.lindep_cutoff]
#ABORT            return bath
#ABORT        self.orthogonalize_bath = bath_orth

##################################################
# scf for impurity
    def init_dmet_scf(self, mol=None):
        if mol is None:
            mol = self.mol
        if self.orth_coeff is None:
            self.orth_coeff = self.get_orth_ao(mol)
        self.release_eri()
        self.bas_on_frag = select_ao_on_fragment(mol, self.imp_atoms, \
                                                 self.imp_basidx)
        c_inv = numpy.dot(self.orth_coeff.T, self.entire_scf.get_ovlp(mol))
        mo_orth = numpy.dot(c_inv, self.entire_scf.mo_coeff[:,self.entire_scf.mo_occ>1e-15])
        # self.imp_site, self.bath_orb, self.env_orb are based on orth-orbitals
        self.imp_site, self.bath_orb, self.env_orb = \
                self.decompose_orbital(mo_orth)
        self.impbas_coeff = self.cons_impurity_basis()
        dd = self.dets_ovlp(mol, self.impbas_coeff)
        log.info(self, 'overlap of determinants before SCF = %.15g', dd)
        self.nelectron = mol.nelectron - self.env_orb.shape[1] * 2
        log.info(self, 'number of electrons for impurity  = %d', \
                 self.nelectron)

        self._vhf_env = self.init_vhf_env(mol, self.env_orb)

    def init_vhf_env(self, mol, env_orb):
        log.debug(self, 'init Hartree-Fock environment')
        env_orb = numpy.dot(self.orth_coeff, env_orb)
        dm_env = numpy.dot(env_orb, env_orb.T.conj()) * 2
        #vj, vk = scf.hf.get_vj_vk(pycint.nr_vhf_o3, mol, dm_env)
        vhf_env_ao = scf.hf.RHF.get_eff_potential(self.entire_scf, self.mol, dm_env)
        hcore = scf.hf.RHF.get_hcore(mol)
        self.energy_by_env = lib.trace_ab(dm_env, hcore) \
                           + lib.trace_ab(dm_env, vhf_env_ao) * .5
        return self.mat_ao2impbas(vhf_env_ao)

    def init_guess_method(self, mol):
        log.debug(self, 'init guess based on entire MO coefficients')
        s = self.entire_scf.get_ovlp(mol)
        eff_scf = self.entire_scf
        entire_scf_dm = eff_scf.calc_den_mat(eff_scf.mo_coeff, eff_scf.mo_occ)
        env_orb = numpy.dot(self.orth_coeff, self.env_orb)
        dm_env = numpy.dot(env_orb, env_orb.T.conj()) * 2
        cs = numpy.dot(self.impbas_coeff.T.conj(), s)
        dm = reduce(numpy.dot, (cs, entire_scf_dm-dm_env, cs.T.conj()))
        hf_energy = 0
        return hf_energy, dm

    def dump_scf_option(self):
        log.info(self, '******** DMET SCF options *************')
        log.info(self, 'damping factor = %g', self.damp_factor)
        log.info(self, 'level shift factor = %g', self.level_shift_factor)
        log.info(self, 'DIIS start cycle = %d', self.diis_start_cycle)
        log.info(self, 'DIIS space = %d', self.diis_space)
        log.info(self, 'SCF threshold = %g', self.scf_threshold)
        log.info(self, 'max. SCF cycles = %d', self.max_scf_cycle)

    def mat_ao2impbas(self, mat):
        c = self.impbas_coeff
        mat_emb = reduce(numpy.dot, (c.T.conj(), mat, c))
        return mat_emb

    def mat_orthao2impbas(self, mat):
        a = numpy.dot(self.imp_site.T, mat[self.bas_on_frag])
        b = numpy.dot(self.bath_orb.T, mat)
        ab = numpy.vstack((a,b))
        a = numpy.dot(ab[:,self.bas_on_frag], self.imp_site)
        b = numpy.dot(ab, self.bath_orb)
        return numpy.hstack((a,b))

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        h1e = self.mat_ao2impbas(self.entire_scf.get_hcore(mol)) \
                + self._vhf_env
        return h1e

    def get_ovlp(self, mol=None):
        if mol is None:
            mol = self.mol
        s1e = self.mat_ao2impbas(self.entire_scf.get_ovlp(mol))
        return s1e

    def set_mo_occ(self, mo_energy, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = self.nelectron / 2
        mo_occ[:nocc] = 2
        if nocc < mo_occ.size:
            log.debug(self, 'HOMO = %.12g, LUMO = %.12g,', \
                      mo_energy[nocc-1], mo_energy[nocc])
        else:
            log.debug(self, 'HOMO = %.12g,', mo_energy[nocc-1])
        log.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    def calc_den_mat(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff_on_imp
        if mo_occ is None:
            mo_occ = self.mo_occ
        nbf = mo_coeff.shape[0]
        mo = mo_coeff[:,mo_occ>0]
        dm = numpy.dot(mo, mo.T.conj()) * 2
        log.debug(self, 'density.diag = %s', dm.diagonal())
        return dm

    def eri_on_impbas(self, mol):
        if self.entire_scf._eri is not None:
            eri = ao2mo.incore.full(self.entire_scf._eri, self.impbas_coeff)
        else:
            eri = ao2mo.direct.full_iofree(self.entire_scf._eri, \
                                           self.impbas_coeff)
        return eri

    def release_eri(self):
        self._eri = None

#    def dot_eri_dm(self, mol, dm):
#        if self._eri is None:
#            self._eri = self.eri_on_impbas(mol)
#        return scf._vhf.vhf_jk_o2(self._eri, dm)
#
#    def get_eff_potential(self, mol, dm, dm_last=0, vhf_last=0):
#        vj, vk = self.dot_eri_dm(mol, dm)
#        vhf = vj - vk * .5
#        return vhf

    def get_eff_potential(self, mol, dm, dm_last=0, vhf_last=0):
        dm = reduce(numpy.dot, (self.impbas_coeff, dm, \
                                self.impbas_coeff.T))
        vhf_ao = scf.hf.RHF.get_eff_potential(self.entire_scf, self.mol, dm)
        return self.mat_ao2impbas(vhf_ao)

    def frag_non_symm_projector(self, s1e):
        '''project operator of fragment. Its definition is not unique
        Non-symmtric <bra|, |ket> leads
            P = |mu> <mu'|     where <mu'| = S^{-1}<mu|;
            P^A_{ij} = \sum_{k\in A} S_{ik} S^{-1}_kj}
        Lowdin-symmetric orthogonalized <bra|, |ket> gives rise to
            P^A_{ij} = \sum_{k\in A} S^{-1/2}_{ik} S^{-1/2}_{kj}
        The non-symmetric projection is consistent with the Mulliken pop.
            chg^A = Tr(D P^A S) = \sum_{i\in A,j} D_{ij}S_{ji}
        Here the project adopts the non-symmetric form'''
        nimp = self.dim_of_impurity()
        s_inv = numpy.linalg.inv(s1e)
        return numpy.dot(s1e[:,:nimp], s_inv[:nimp,:])

    def calc_frag_elec_energy(self, mol, vhf, dm):
        ''' Calculate meanfiled fragment electronic energy:
            E = <\Psi|H|\Psi> = \sum_A <\Psi^A|H|\Psi> = \sum_A E^A
            E^A = <\Psi^A|H|\Psi>
                = \sum_\mu <\psi^A|h|\mu> D_{\mu\psi^A}
                + 1/2 \sum_\mu (J-K)_{\psi^A\mu} D_{\mu\psi^A}
* When the occ_env_cutoff is large, the so obtained fragment sites are not
complete basis sets for electron density. In such fragment basis sets, the
density matrix cannot represent all the electrons.  The number of electron
represented by the density matrix (= trace(D) = \sum elec_frag) would be less
then the total number of electron.  The sum of the fragment electronic energy
which does not include all electrons would be less than the total electronic
energy.
* If vhf is 0 and dm is post-HF density matrix, the "fragment energy"
will be the sum of one electron energy and the mean-filed energy of
environment two-electron part'''
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        proj = self.frag_non_symm_projector(s1e)
        dm_frag = numpy.dot(dm, proj)

        # ne = Tr(D S)
        # ne^A = Tr(D P^A S)
        nelec_frag = lib.trace_ab(dm_frag, s1e)
        log.info(self, 'number of electrons in fragment = %.15g', \
                 nelec_frag.real)

        e = lib.trace_ab(dm_frag, h1e-self._vhf_env) \
                + lib.trace_ab(dm_frag, vhf+self._vhf_env) * .5
        log.info(self, 'fragment electronic energy = %.15g', e.real)
        log.debug(self, ' ~ total energy (non-variational) = %.15g', \
                  lib.trace_ab(dm, h1e) + lib.trace_ab(dm, vhf)*.5 \
                  + self.energy_by_env)
        return e.real, nelec_frag.real

    def imp_scf(self):
        if self.orth_coeff is None:
            self.orth_coeff = self.get_orth_ao(self.mol)

        self.dump_options()
        self.init_dmet_scf(self.mol)
        self.dump_scf_option()

        self.scf_conv, self.hf_energy, self.mo_energy, self.mo_occ, \
                self.mo_coeff_on_imp \
                = self.scf_cycle(self.mol, self.scf_threshold, \
                                 dump_chk=False)

        log.info(self, 'impurity MO energy')
        for i in range(self.mo_energy.size):
            if self.mo_occ[i] > 0:
                log.info(self, 'impurity occupied MO %d energy = %.15g occ=%g', \
                         i+1, self.mo_energy[i], self.mo_occ[i])
            else:
                log.info(self, 'impurity virtual MO %d energy = %.15g occ=%g', \
                         i+1, self.mo_energy[i], self.mo_occ[i])

        #e_nuc = self.nuclear_repulsion(self.mol)
        #log.log(self, 'impurity sys nuclear repulsion = %.15g', e_nuc)
        if self.scf_conv:
            log.log(self, 'converged impurity sys electronic energy = %.15g', \
                    self.hf_energy)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'electronic energy = %.15g after %d cycles.', \
                    self.hf_energy, self.max_scf_cycle)

        # mo_coeff_on_imp based on embedding basis + bath
        # mo_coeff based on AOs
        self.mo_coeff = numpy.dot(self.impbas_coeff, self.mo_coeff_on_imp)
        s = self.entire_scf.get_ovlp(self.mol)
        mo0 = self.entire_scf.mo_coeff[:,self.entire_scf.mo_occ>0]
        mo1 = numpy.hstack((self.mo_coeff[:,self.mo_occ>0], \
                            numpy.dot(self.orth_coeff, self.env_orb)))
        norm = 1/numpy.sqrt(numpy.linalg.det( \
                reduce(numpy.dot, (mo1.T.conj(), s, mo1))))
        ovlp = numpy.linalg.det(reduce(numpy.dot, (mo0.T.conj(), s, mo1))) * norm
        # ovlp**2 because of the beta orbital contribution
        log.info(self, 'overlap of determinants after SCF = %.15g', (ovlp**2))

        dm = self.calc_den_mat(self.mo_coeff_on_imp, self.mo_occ)
        vhf = self.get_eff_potential(self.mol, dm)
        self.e_frag, self.n_elec_frag = \
                self.calc_frag_elec_energy(self.mol, vhf, dm)
        log.log(self, 'fragment electronic energy = %.15g', self.e_frag)
        log.log(self, 'fragment electron number = %.15g', self.n_elec_frag)
        #self.frag_mulliken_pop()
        return self.e_frag

    def nuclear_repulsion(self, mol):
        e = 0
        for j, ja in enumerate(self.imp_atoms):
            q2 = mol.charge_of_atm(ja)
            r2 = numpy.array(mol.coord_of_atm(ja))
            for i in range(j):
                ia = self.imp_atoms[i]
                q1 = mol.charge_of_atm(ia)
                r1 = numpy.array(mol.coord_of_atm(ia))
                r = numpy.linalg.norm(r1-r2)
                e += q1 * q2 / r
        for j in range(mol.natm):
            if j not in self.imp_atoms:
                q2 = mol.charge_of_atm(j)
                r2 = numpy.array(mol.coord_of_atm(j))
                for i in self.imp_atoms:
                    q1 = mol.charge_of_atm(i)
                    r1 = numpy.array(mol.coord_of_atm(i))
                    r = numpy.linalg.norm(r1-r2)
                    e += q1 * q2 / r
        return e

    def frag_mulliken_pop(self):
        '''Mulliken M_ij = D_ij S_ji, Mulliken chg_i = \sum_j M_ij'''
        mol = self.mol
        log.info(self, ' ** Mulliken pop (on impurity basis)  **')
        s1e = self.get_ovlp(self.mol)
        c_inv = numpy.dot(self.orth_coeff.T, s1e)
        c_frag = numpy.dot(c_inv, self.impbas_coeff)
        dm = self.calc_den_mat(self.mo_coeff_on_imp, self.mo_occ)
        dm_frag = numpy.dot(dm, self.frag_non_symm_projector(s1e))
        dm = reduce(numpy.dot, (c_frag, dm_frag, c_frag.T.conj()))
        pop = dm.diagonal()
        label = mol.labels_of_spheric_GTO()

        for i, s in enumerate(label):
            if s[0] in self.imp_atoms:
                log.info(self, 'pop of  %d%s %s%4s ' % s \
                         + ' %10.5f' % pop[i])

        log.info(self, ' ** Mulliken atomic charges  **')
        chg = numpy.zeros(mol.natm)
        for i, s in enumerate(label):
            if s[0] in self.imp_atoms:
                chg[s[0]] += pop[i]
        frag_charge = 0
        for ia in self.imp_atoms:
            symb = mol.symbol_of_atm(ia)
            nuc = mol.charge_of_atm(ia)
            log.info(self, 'charge of  %d%s =   %10.5f', \
                     ia, symb, nuc - chg[ia])
            frag_charge += nuc - chg[ia]
        log.info(self, 'charge of embsys = %10.5f', frag_charge)

        if self.num_bath != -1:
            self.diff_dm()

    def diff_dm(self):
        # diff between the SCF DM and DMET DM for fragment block
        mol = self.mol
        s = self.entire_scf.get_ovlp(self.mol)
        c_inv = numpy.dot(self.orth_coeff.T, s)
        eff_scf = self.entire_scf
        mo = numpy.dot(c_inv, eff_scf.mo_coeff)
        dm0 = eff_scf.calc_den_mat(mo, eff_scf.mo_occ)
        # in case impurity sites are not the AO orbitals
        mo = reduce(numpy.dot, (c_inv, self.impbas_coeff, self.mo_coeff_on_imp))
        dm1 = numpy.dot(mo*self.mo_occ, mo.T)
        dm1 += numpy.dot(self.env_orb, self.env_orb.T)*2
        norm = numpy.linalg.norm((dm0-dm1)[self.bas_on_frag][:,self.bas_on_frag])
        log.info(self, 'norm(diff of imp-DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm, norm/len(self.bas_on_frag))
        norm = numpy.linalg.norm(dm0-dm1)
        log.info(self, 'norm(diff of entire DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm, norm/dm0.shape[0])

        bas_off_frag = [i for i in range(s.shape[0]) \
                        if i not in self.bas_on_frag]
        p = reduce(numpy.dot, (c_inv, self.impbas_coeff, self.impbas_coeff.T, c_inv.T))
        norm0 = numpy.linalg.norm((numpy.dot(dm0,p)-dm0)[self.bas_on_frag][:,bas_off_frag])
        norm1 = numpy.linalg.norm((dm0-dm1)[self.bas_on_frag][:,bas_off_frag])
        log.info(self, 'before SCF norm(diff off-diagonal DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm0, norm0/numpy.sqrt(len(self.bas_on_frag)*len(bas_off_frag)))
        log.info(self, 'after SCF norm(diff off-diagonal DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm1, norm1/numpy.sqrt(len(self.bas_on_frag)*len(bas_off_frag)))
        norm1 = numpy.linalg.norm((dm0-dm1)[self.bas_on_frag])
        log.info(self, 'after SCF norm(diff frag-band DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm1, norm1/numpy.sqrt(len(self.bas_on_frag)*s.shape[0]))


# scf for impurity end
##################################################

    def get_orth_ao(self, mol):
        return orthogonalize_ao(mol, self.entire_scf, \
                                self.pre_orth_ao(mol), \
                                self.orth_ao_method)

    def init_with_lowdin_ao(self):
        self.orth_ao_method = 'lowdin'

    def init_with_nao(self):
        self.orth_ao_method = 'nao'

    def init_with_meta_lowdin_ao(self):
        self.orth_ao_method = 'meta_lowdin'

    def pre_orth_ao(self, mol):
        return numpy.eye(mol.num_NR_function())

    def set_ao_with_input_basis(self):
        try:
            del(self.pre_orth_ao)
        except:
            pass

    def set_ao_with_atm_scf(self):
        self.pre_orth_ao = pre_orth_ao_atm_scf

    def set_ao_with_projected_ano(self):
        self.pre_orth_ao = pre_orth_ao_projected_ano
##################################################

    def set_embsys(self, atm_lst):
        assert(max(atm_lst) < self.mol.natm)
        self.imp_atoms = atm_lst

    def set_bath(self, atm_lst):
        assert(max(atm_lst) < self.mol.natm)
        self.imp_atoms = filter(lambda n: n not in atm_lst, \
                                range(self.mol.natm))

    def append_embsys(self, atm_lst):
        assert(max(atm_lst) < self.mol.natm)
        self.imp_atoms = set(list(atm_lst) + list(self.imp_atoms))

    def append_bath(self, atm_lst):
        assert(max(atm_lst) < self.mol.natm)
        if self.imp_atoms == []:
            self.set_bath(atm_lst)
        else:
            self.imp_atoms = filter(lambda n: n not in atm_lst, \
                                    self.imp_atoms)

    def dim_of_impurity(self):
        return self.imp_site.shape[1]
    def num_of_impbas(self):
        return self.impbas_coeff.shape[1]

##################################################

    def cons_impurity_basis(self):
        a = numpy.dot(self.orth_coeff[:,self.bas_on_frag], self.imp_site)
        b = numpy.dot(self.orth_coeff, self.bath_orb)
        if self.orth_imp_to_env:
            return self.suborth_imp_to_env(numpy.hstack((a,b)))
        else:
            return numpy.hstack((a,b))

    def suborth_imp_to_env(self, impbas_coeff):
        c = numpy.hstack((numpy.dot(self.orth_coeff, self.env_orb), \
                          impbas_coeff))
        s = self.entire_scf.get_ovlp(self.mol)
        t = schmidt_orth_coeff(reduce(numpy.dot, (c.T.conj(), s, c)))
        off = self.env_orb.shape[1]
        impbas_coeff = numpy.dot(c, t)[:,off:]
        return impbas_coeff

## scheme 1: GHO is based on the AO of boundary atom and thereby
## non-orthogonal to the impurity sys
#    def gho_on_lowdin_aos1(self, hyb, gho_atm_lst):
#        idx = gho.gho_index(self.mol, gho_atm_lst[0])
#        nbf = self.mol.num_NR_function()
#        v = numpy.zeros((nbf, 4))
#        v[idx] = hyb
#        return numpy.mat(v)
## scheme 2: GHO is based on the orthogonalized AOs
#    def gho_on_lowdin_aos2(self, hyb, gho_atm_lst):
#        idx = gho.gho_index(self.mol, gho_atm_lst[0])
#        v = numpy.zeros((nbf, 4))
#        v[idx] = hyb
#
#        # fix the phase of s and p
#        s = self.entire_scf.get_ovlp(self.mol)
#        sh = (numpy.mat(s) * v)[idx,:]
#        if (hyb[0,0] * sh[0,0]) * (hyb[1:,0].T * sh[1:,0]) < 0:
#            hyb[0] = -hyb[0]
#            v = numpy.dot(self.orth_coeff[:,idx], hyb)
#        # based on AO basis
#        return v

    def set_gho_pseudo_bath(self, gho_atm_lst, inc_1s=False):
        assert(self.mol.pure_symbol_of_atm(gho_atm_lst[0]) == 'C')
        self.append_bath(gho_atm_lst)
        self.num_bath = 1
        if inc_1s:
            for i, s in enumerate(self.mol.labels_of_spheric_GTO()):
                if s[0] == gho_atm_lst[0] and s[2] == '1s':
                    self.imp_basidx = [i]
                    break

        def cons_impbas():
            import gho
            log.info(self, 'replace bath orbital with GHOs')
            g = gho.GHO()
            gho_orb = g.hybrid_coeff(self.mol, gho_atm_lst)
            gho_idx = gho.gho_index(self.mol, gho_atm_lst[0])
            ovlp = numpy.dot(self.bath_orb[gho_idx,:].T.conj(), gho_orb)
            for i,c in enumerate(ovlp):
                log.debug(self, '<bath_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
            p_hybs = numpy.dot(ovlp.T,ovlp).diagonal()
            log.debug(self, '<gho_i|bath><bath|gho_i> = %s', str(p_hybs))
            u, w, v = numpy.linalg.svd(ovlp)
            log.debug(self, 'SVD <gho|bath> = %s', str(w))

            if self.env_orb.shape[1] > 0:
                ovlp = numpy.dot(self.env_orb[gho_idx,:].T.conj(), gho_orb)
                for i,c in enumerate(ovlp):
                    log.debug(self, '<env_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
                p_hybs = numpy.dot(ovlp.T,ovlp).diagonal()
                log.debug(self, '<gho_i|env><env|gho_i> = %s', str(p_hybs))
                u, w, v = numpy.linalg.svd(ovlp)
                log.debug(self, 'SVD <gho|env> = %s', str(w))

            coord0 = self.mol.coord_of_atm(gho_atm_lst[0])
            dists = [numpy.linalg.norm(self.mol.coord_of_atm(i)-coord0) \
                     for i in self.imp_atoms]
            bondatm = self.imp_atoms[numpy.argmin(dists)]
            bath1 = self.bath_orb[gho_idx,0]/numpy.linalg.norm(self.bath_orb[gho_idx,0])
            log.debug(self, 'bath_1 hybrid = sp^%4.3f, angle to bond = %.6g', \
                      gho.sp_hybrid_level(bath1), \
                      gho.angle_to_bond(self.mol, gho_atm_lst[0], \
                                        bondatm, bath1))
            log.debug(self, 'GHO-active hybrid = sp^%4.3f, angle to bond = %.6g', \
                      gho.sp_hybrid_level(gho_orb[:,0]), \
                      gho.angle_to_bond(self.mol, gho_atm_lst[0], \
                                        bondatm, gho_orb[:,0]))
            cosovlp = numpy.dot(bath1[1:4],gho_orb[1:4,0]) \
                    / numpy.linalg.norm(bath1[1:4]) \
                    / numpy.linalg.norm(gho_orb[1:4,0])
            log.debug(self, 'angle between GHO and bath_1 = %.6g', \
                      numpy.arccos(cosovlp))

            a = numpy.dot(self.orth_coeff[:,self.bas_on_frag], self.imp_site)
            b = numpy.dot(self.orth_coeff[:,gho_idx], gho_orb[:,:1])
            impbas_coeff = numpy.hstack((a,b))
            if self.orth_imp_to_env:
                impbas_coeff = self.suborth_imp_to_env(impbas_coeff)
            return impbas_coeff
        self.cons_impurity_basis = cons_impbas


##################################################
    def bath_delta_nuc_vhf(self, mol):
        # nuclear attraction matrix not in fragment
        nbf = mol.num_NR_function()
        bnuc = numpy.zeros((nbf,nbf))
        for ia in range(mol.natm):
            if ia not in self.imp_atoms:
                mol.set_rinv_orig(mol.coord_of_atm(ia))
                chg = mol.charge_of_atm(ia)
                bnuc += -chg * self.entire_scf.get_ovlp(self.mol)

        bnuc = self.mat_ao2impbas(bnuc)
        print bnuc + self._vhf_env

#TODO:
    def set_link_atom_pseudo_bath(self):
        pass

    def dets_ovlp(self, mol, orbs):
        '''det(<i*|i>):  |i*> = P|i>,  P = |x>S^{-1}<x|'''
        mo0 = self.entire_scf.mo_coeff[:,self.entire_scf.mo_occ>0]
        s = self.entire_scf.get_ovlp(self.mol)
        orbs1 = numpy.hstack((orbs, numpy.dot(self.orth_coeff, self.env_orb)))
        tmp = reduce(numpy.dot, (orbs1.T.conj(), s, orbs1))
        proj = reduce(numpy.dot, (orbs1, numpy.linalg.inv(tmp), \
                                  orbs1.T.conj()))
        ovlp = reduce(numpy.dot, (mo0.T.conj(), s, proj, s, mo0))
        # <ovlp>**2 because of the beta orbital contribution
        return numpy.linalg.det(ovlp)**2


##################################################
class UHF(RHF, scf.hf.UHF):
    '''Non-relativistic unrestricted Hartree-Fock DMET'''
    def __init__(self, entire_scf, orth_ao=None):
        assert(isinstance(entire_scf, scf.hf.UHF) \
              or isinstance(entire_scf, scf.hf_symm.UHF))
        RHF.__init__(self, entire_scf, orth_ao)
        self.nelectron_alpha = None
        self.nelectron_beta = None

    def decompose_den_mat(self, dm_orth):
        imp_a, bath_a, env_a = decompose_den_mat(self, dm_orth[0], \
                                                 self.bas_on_frag, self.num_bath)
        imp_b, bath_b, env_b = decompose_den_mat(self, dm_orth[1], \
                                                 self.bas_on_frag, self.num_bath)
        #if bath_a.shape[1] == bath_b.shape[1]:
        #    cc = [bath_a,bath_b]
        #elif bath_a.shape[1] > bath_b.shape[1]:
        #    cc = [bath_a,padding0(bath_b,bath_a.shape)]
        #else:
        #    cc = [padding0(bath_a,bath_b.shape),bath_b]
        return (imp_a,imp_b), (bath_a,bath_b), (env_a,env_b)

    def decompose_orbital(self, mo_orth):
        imp_a, bath_a, env_a = decompose_orbital(self, mo_orth[0], \
                                                 self.bas_on_frag, self.num_bath)
        if mo_orth[1].size > 0:
            imp_b, bath_b, env_b = decompose_orbital(self, mo_orth[1], \
                                                     self.bas_on_frag, self.num_bath)
        else:
            nao = bath_a.shape[0]
            imp_b  = imp_a
            bath_b = numpy.zeros((nao,0))
            env_b  = numpy.zeros_like(env_a)
        #if bath_a.shape[1] == bath_b.shape[1]:
        #    cc = [bath_a,bath_b]
        #elif bath_a.shape[1] > bath_b.shape[1]:
        #    cc = [bath_a,padding0(bath_b,bath_a.shape)]
        #else:
        #    cc = [padding0(bath_a,bath_b.shape),bath_b]
        return (imp_a,imp_b), (bath_a,bath_b), (env_a,env_b)

    def cons_impurity_basis(self):
        a_a = numpy.dot(self.orth_coeff[:,self.bas_on_frag], self.imp_site[0])
        a_b = numpy.dot(self.orth_coeff[:,self.bas_on_frag], self.imp_site[1])
        b_a = numpy.dot(self.orth_coeff, self.bath_orb[0])
        b_b = numpy.dot(self.orth_coeff, self.bath_orb[1])
        impbas = (numpy.hstack((a_a,b_a)), \
                  numpy.hstack((a_b,b_b)))
        if self.orth_imp_to_env:
            impbas = self.suborth_imp_to_env(impbas)
        return impbas

    def suborth_imp_to_env(self, impbas_coeff):
        c_a = numpy.hstack((numpy.dot(self.orth_coeff, self.env_orb[0]), \
                            impbas_coeff[0]))
        c_b = numpy.hstack((numpy.dot(self.orth_coeff, self.env_orb[1]), \
                            impbas_coeff[1]))
        s = self.entire_scf.get_ovlp(self.mol)[0]
        t_a = schmidt_orth_coeff(reduce(numpy.dot, (c_a.T.conj(), s, c_a)))
        t_b = schmidt_orth_coeff(reduce(numpy.dot, (c_b.T.conj(), s, c_b)))
        impbas_coeff = (numpy.dot(c_a, t_a)[:,self.env_orb[0].shape[1]:], \
                        numpy.dot(c_b, t_b)[:,self.env_orb[1].shape[1]:])
        return impbas_coeff

    def init_dmet_scf(self, mol):
        if self.orth_coeff is None:
            self.orth_coeff = self.get_orth_ao(mol)
        self.bas_on_frag = select_ao_on_fragment(mol, self.imp_atoms, \
                                                 self.imp_basidx)
        c_inv = numpy.dot(self.orth_coeff.T, self.entire_scf.get_ovlp(self.mol)[0])
        mo_a, mo_b = self.entire_scf.mo_coeff
        occ_a, occ_b = self.entire_scf.mo_occ
        mo_orth_a = numpy.dot(c_inv, mo_a[:,self.entire_scf.mo_occ[0]>1e-15])
        mo_orth_b = numpy.dot(c_inv, mo_b[:,self.entire_scf.mo_occ[1]>1e-15])
        # self.imp_site, self.bath_orb, self.env_orb are based on orth-orbitals
        self.imp_site, self.bath_orb, self.env_orb = \
                self.decompose_orbital((mo_orth_a,mo_orth_b))
        ovlp = numpy.dot(self.bath_orb[0].T,self.bath_orb[1])[:4,:4]
        for i,c in enumerate(ovlp):
            log.debug(self, ('<bath_alpha_%d|bath_beta> = ' % i) \
                      + '%10.5f'*len(c), *c)
        self.impbas_coeff = self.cons_impurity_basis()
        dd = self.dets_ovlp(mol, self.impbas_coeff)
        log.info(self, 'overlap of determinants before SCF = %.15g', dd)
        self.nelectron_alpha = self.entire_scf.nelectron_alpha \
                - self.env_orb[0].shape[1]
        self.nelectron_beta = mol.nelectron \
                - self.entire_scf.nelectron_alpha \
                - self.env_orb[1].shape[1]
        log.info(self, 'alpha / beta electrons for impurity = %d / %d', \
                 self.nelectron_alpha, self.nelectron_beta)

        self._vhf_env = self.init_vhf_env(mol, self.env_orb)

    def init_vhf_env(self, mol, env_orb):
        log.debug(self, 'init Hartree-Fock environment')
        env_a = numpy.dot(self.orth_coeff, env_orb[0])
        env_b = numpy.dot(self.orth_coeff, env_orb[1])
        dm_env = numpy.array([numpy.dot(env_a, env_a.T.conj()), \
                              numpy.dot(env_b, env_b.T.conj())])
        vhf_env_ao = scf.hf.UHF.get_eff_potential(self.entire_scf, self.mol, dm_env)
        hcore = scf.hf.UHF.get_hcore(mol)
        self.energy_by_env = lib.trace_ab(dm_env[0], hcore[0]) \
                           + lib.trace_ab(dm_env[1], hcore[1]) \
                           + lib.trace_ab(dm_env[0], vhf_env_ao[0]) * .5 \
                           + lib.trace_ab(dm_env[1], vhf_env_ao[1]) * .5
        return self.mat_ao2impbas(vhf_env_ao)

    def mat_ao2impbas(self, mat):
        c_a = self.impbas_coeff[0]
        c_b = self.impbas_coeff[1]
        if mat.ndim == 2:
            mat_a = reduce(numpy.dot, (c_a.T.conj(), mat, c_a))
            mat_b = reduce(numpy.dot, (c_b.T.conj(), mat, c_b))
        else:
            mat_a = reduce(numpy.dot, (c_a.T.conj(), mat[0], c_a))
            mat_b = reduce(numpy.dot, (c_b.T.conj(), mat[1], c_b))
        return (mat_a,mat_b)

    def mat_orthao2impbas(self, mat):
        c = numpy.zeros_like(self.impbas_coeff)
        nimp = self.dim_of_impurity()
        c[0,self.bas_on_frag,:nimp] = self.imp_site[0]
        c[1,self.bas_on_frag,:nimp] = self.imp_site[1]
        nemb_a = nimp + self.bath_orb[0].shape[1]
        nemb_b = nimp + self.bath_orb[1].shape[1]
        c[0,:,nimp:nemb_a] = self.bath_orb[0]
        c[1,:,nimp:nemb_b] = self.bath_orb[1]
        if mat.ndim == 2:
            mat_a = reduce(numpy.dot, (c[0].T.conj(), mat, c[0]))
            mat_b = reduce(numpy.dot, (c[1].T.conj(), mat, c[1]))
        else:
            mat_a = reduce(numpy.dot, (c[0].T.conj(), mat[0], c[0]))
            mat_b = reduce(numpy.dot, (c[1].T.conj(), mat[1], c[1]))
        return (mat_a,mat_b)

    def dim_of_impurity(self):
        return self.imp_site[0].shape[1]
    def num_of_impbas(self):
        return self.impbas_coeff[0].shape[1]


# **** impurity SCF ****
    def check_dm_converge(self, dm, dm_last, scf_threshold):
        delta_dm = abs(dm[0]-dm_last[0]).sum() + abs(dm[1]-dm_last[1]).sum()
        dm_change = delta_dm/(abs(dm_last[0]).sum()+abs(dm_last[1]).sum())
        log.info(self, '          sum(delta_dm)=%g (~ %g%%)\n', \
                 delta_dm, dm_change*100)
        return dm_change < scf_threshold*1e2

    def init_guess_method(self, mol):
        log.debug(self, 'init guess based on entire MO coefficients')
        s = self.entire_scf.get_ovlp(self.mol)[0]
        eff_scf = self.entire_scf
        entire_scf_dm = eff_scf.calc_den_mat(eff_scf.mo_coeff, eff_scf.mo_occ)
        env_a = numpy.dot(self.orth_coeff, self.env_orb[0])
        env_b = numpy.dot(self.orth_coeff, self.env_orb[1])
        dm_a = numpy.dot(env_a, env_a.T.conj())
        dm_b = numpy.dot(env_b, env_b.T.conj())
        cs_a = numpy.dot(self.impbas_coeff[0].T.conj(), s)
        cs_b = numpy.dot(self.impbas_coeff[1].T.conj(), s)
        dm_a = reduce(numpy.dot, (cs_a, entire_scf_dm[0]-dm_a, cs_a.T.conj()))
        dm_b = reduce(numpy.dot, (cs_b, entire_scf_dm[1]-dm_b, cs_b.T.conj()))
        hf_energy = 0
        return hf_energy, numpy.array((dm_a,dm_b))

    def get_eff_potential(self, mol, dm, dm_last=0, vhf_last=0):
        dm_a = reduce(numpy.dot, (self.impbas_coeff[0], dm[0], \
                                  self.impbas_coeff[0].T))
        dm_b = reduce(numpy.dot, (self.impbas_coeff[1], dm[1], \
                                  self.impbas_coeff[1].T))
        dm_ao = numpy.array((dm_a, dm_b))
        vhf_ao = scf.hf.UHF.get_eff_potential(self.entire_scf, self.mol, dm_ao)
        return self.mat_ao2impbas(vhf_ao)

    def get_hcore(self, mol):
        h1e = self.mat_ao2impbas(scf.hf.RHF.get_hcore(mol))
        return (h1e[0]+self._vhf_env[0], h1e[1]+self._vhf_env[1])

    def make_fock(self, h1e, vhf):
        return (h1e[0]+vhf[0], h1e[1]+vhf[1])

    def eig(self, fock, s):
        c_a, e_a, info = lapack.dsygv(fock[0], s[0])
        c_b, e_b, info = lapack.dsygv(fock[1], s[1])
        return (e_a,e_b), (c_a,c_b), info

    def set_mo_occ(self, mo_energy, mo_coeff=None):
        mo_occ = [numpy.zeros_like(mo_energy[0]), \
                  numpy.zeros_like(mo_energy[1])]
        mo_occ[0][:self.nelectron_alpha] = 1
        mo_occ[1][:self.nelectron_beta]  = 1
        if self.nelectron_alpha < mo_energy[0].size:
            log.debug(self, 'alpha nocc = %d, HOMO = %.12g, LUMO = %.12g,', \
                      self.nelectron_alpha, \
                      mo_energy[0][self.nelectron_alpha-1], \
                      mo_energy[0][self.nelectron_alpha])
        else:
            log.debug(self, 'alpha nocc = %d, HOMO = %.12g, no LUMO,', \
                      self.nelectron_alpha, \
                      mo_energy[0][self.nelectron_alpha-1])
        log.debug(self, '  mo_energy = %s', mo_energy[0])
        log.debug(self, 'beta  nocc = %d, HOMO = %.12g, LUMO = %.12g,', \
                  self.nelectron_beta, \
                  mo_energy[0][self.nelectron_beta-1], \
                  mo_energy[0][self.nelectron_beta])
        log.debug(self, '  mo_energy = %s', mo_energy[1])
        return mo_occ

    def calc_den_mat(self, mo_coeff, mo_occ):
        mo_a = mo_coeff[0][:,mo_occ[0]>0]
        mo_b = mo_coeff[1][:,mo_occ[1]>0]
        dm_a = numpy.dot(mo_a, mo_a.T.conj())
        dm_b = numpy.dot(mo_b, mo_b.T.conj())
        log.debug(self, 'alpha density.diag = %s', dm_a.diagonal())
        log.debug(self, 'beta  density.diag = %s', dm_b.diagonal())
        return (dm_a,dm_b)

    def init_diis(self):
        udiis = scf.diis.SCF_DIIS(self)
        udiis.diis_space = self.diis_space
        #udiis.diis_start_cycle = self.diis_start_cycle
        def scf_diis(cycle, s, d, f):
            if cycle >= self.diis_start_cycle:
                sdf_a = reduce(numpy.dot, (s[0], d[0], f[0]))
                sdf_b = reduce(numpy.dot, (s[1], d[1], f[1]))
                erra = (sdf_a.T.conj()-sdf_a).flatten()
                errb = (sdf_b.T.conj()-sdf_b).flatten()
                errvec = numpy.hstack((erra, errb))
                udiis.err_vec_stack.append(errvec)
                log.debug(self, 'diis-norm(errvec) = %g', \
                          numpy.linalg.norm(errvec))
                if udiis.err_vec_stack.__len__() > udiis.diis_space:
                    udiis.err_vec_stack.pop(0)
                f1 = numpy.hstack((f[0].flatten(),f[1].flatten()))
                f1 = scf.diis.DIIS.update(udiis, f1)
                f = (f1[:f[0].size].reshape(f[0].shape), \
                     f1[f[0].size:].reshape(f[1].shape))
            if cycle < self.diis_start_cycle-1:
                f = (self.damping(s[0], d[0], f[0], self.damp_factor), \
                     self.damping(s[1], d[1], f[1], self.damp_factor))
                f = (self.level_shift(s[0],d[0],f[0],self.level_shift_factor), \
                     self.level_shift(s[1],d[1],f[1],self.level_shift_factor))
            else:
                fac = self.level_shift_factor \
                        * numpy.exp(self.diis_start_cycle-cycle-1)
                f = (self.level_shift(s[0], d[0], f[0], fac), \
                     self.level_shift(s[1], d[1], f[1], fac))
            return f
        return scf_diis

    def imp_scf(self):
        if self.orth_coeff is None:
            self.orth_coeff = self.get_orth_ao(self.mol)

        self.dump_options()
        self.init_dmet_scf(self.mol)
        self.dump_scf_option()

        self.scf_conv, self.hf_energy, self.mo_energy, self.mo_occ, \
                self.mo_coeff_on_imp \
                = self.scf_cycle(self.mol, self.scf_threshold, \
                                 dump_chk=False)

        def dump_mo_energy(mo_energy, mo_occ, title=''):
            log.info(self, 'impurity %s MO energy', title)
            for i in range(mo_energy.size):
                if mo_occ[i] > 0:
                    log.info(self, 'impurity %s occupied MO %d energy = %.15g', \
                             title, i+1, mo_energy[i])
                else:
                    log.info(self, 'impurity %s virtual MO %d energy = %.15g', \
                             title, i+1, mo_energy[i])
        dump_mo_energy(self.mo_energy[0], self.mo_occ[0], 'alpha')
        dump_mo_energy(self.mo_energy[1], self.mo_occ[1], 'beta')

        if self.scf_conv:
            log.log(self, 'converged impurity sys electronic energy = %.15g', \
                    self.hf_energy)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'electronic energy = %.15g after %d cycles.', \
                    self.hf_energy, self.max_scf_cycle)

        # mo_coeff_on_imp based on embedding basis + bath
        # mo_coeff based on AOs
        c_a = numpy.dot(self.impbas_coeff[0], self.mo_coeff_on_imp[0])
        c_b = numpy.dot(self.impbas_coeff[1], self.mo_coeff_on_imp[1])
        self.mo_coeff = (c_a,c_b)
        s = self.entire_scf.get_ovlp(self.mol)[0]
        mo0_a = self.entire_scf.mo_coeff[0][:,self.entire_scf.mo_occ[0]>0]
        mo0_b = self.entire_scf.mo_coeff[1][:,self.entire_scf.mo_occ[1]>0]
        mo1_a = numpy.hstack((c_a[:,self.mo_occ[0]>0], \
                              numpy.dot(self.orth_coeff, self.env_orb[0])))
        mo1_b = numpy.hstack((c_b[:,self.mo_occ[1]>0], \
                              numpy.dot(self.orth_coeff, self.env_orb[1])))
        norm = 1/numpy.sqrt( \
                numpy.linalg.det(reduce(numpy.dot,  (mo1_a.T.conj(),s,mo1_a)))\
                * numpy.linalg.det(reduce(numpy.dot,(mo1_b.T.conj(),s,mo1_b))))
        ovlp = numpy.linalg.det(reduce(numpy.dot,  (mo0_a.T.conj(),s,mo1_a))) \
               * numpy.linalg.det(reduce(numpy.dot,(mo0_b.T.conj(),s,mo1_b)))
        log.info(self, 'overlap of determinants after SCF = %.15g', abs(ovlp * norm))

        dm = self.calc_den_mat(self.mo_coeff_on_imp, self.mo_occ)
        vhf = self.get_eff_potential(self.mol, dm)
        self.e_frag, self.n_elec_frag = \
                self.calc_frag_elec_energy(self.mol, vhf, dm)
        log.log(self, 'fragment electronic energy = %.15g', self.e_frag)
        log.log(self, 'fragment electron number = %.15g', self.n_elec_frag)
        self.frag_mulliken_pop()
        return self.e_frag

    def calc_frag_elec_energy(self, mol, vhf, dm):
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(self.mol)
        proj_a = self.frag_non_symm_projector(s1e[0])
        proj_b = self.frag_non_symm_projector(s1e[1])
        dm_frag_a = numpy.dot(dm[0], proj_a)
        dm_frag_b = numpy.dot(dm[1], proj_b)

        # ne = Tr(D S)
        # ne^A = Tr(D P^A S)
        nelec_frag = lib.trace_ab(dm_frag_a, s1e[0]) \
                + lib.trace_ab(dm_frag_b, s1e[1])
        log.info(self, 'number of electrons in fragment = %.15g', \
                 nelec_frag.real)

        e = lib.trace_ab(dm_frag_a, h1e[0]-self._vhf_env[0]) \
                + lib.trace_ab(dm_frag_b, h1e[1]-self._vhf_env[1]) \
                + lib.trace_ab(dm_frag_a, vhf[0] + self._vhf_env[0]) * .5 \
                + lib.trace_ab(dm_frag_b, vhf[1] + self._vhf_env[1]) * .5
        log.info(self, 'fragment electronic energy = %.15g', e.real)
        log.debug(self, ' ~ total energy (non-variational) = %.15g', \
                  (lib.trace_ab(dm[0], h1e[0])+lib.trace_ab(dm[1], h1e[1]) \
                   + lib.trace_ab(dm[0], vhf[0])*.5 \
                   + lib.trace_ab(dm[1], vhf[1])*.5 \
                   + self.energy_by_env))
        return e.real, nelec_frag.real

    def frag_mulliken_pop(self):
        '''Mulliken M_ij = D_ij S_ji, Mulliken chg_i = \sum_j M_ij'''
        mol = self.mol
        log.info(self, ' ** Mulliken pop alpha/beta (on impurity basis)  **')
        c_inv = numpy.dot(self.orth_coeff.T, self.entire_scf.get_ovlp(mol)[0])
        c_frag_a = numpy.dot(c_inv, self.impbas_coeff[0])
        c_frag_b = numpy.dot(c_inv, self.impbas_coeff[1])
        dm = self.calc_den_mat(self.mo_coeff_on_imp, self.mo_occ)
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        dm_frag_a = numpy.dot(dm[0], self.frag_non_symm_projector(s1e[0]))
        dm_frag_b = numpy.dot(dm[1], self.frag_non_symm_projector(s1e[1]))
        dm_a = reduce(numpy.dot, (c_frag_a, dm_frag_a, c_frag_a.T.conj()))
        dm_b = reduce(numpy.dot, (c_frag_b, dm_frag_b, c_frag_b.T.conj()))
        pop_a = dm_a.diagonal()
        pop_b = dm_b.diagonal()
        label = mol.labels_of_spheric_GTO()

        for i, s in enumerate(label):
            if s[0] in self.imp_atoms:
                log.info(self, 'pop of  %d%s %s%4s ' % s \
                         + ' %10.5f   / %10.5f' % (pop_a[i], pop_b[i]))

        log.info(self, ' ** Mulliken atomic charges  **')
        chg = numpy.zeros(mol.natm)
        for i, s in enumerate(label):
            if s[0] in self.imp_atoms:
                chg[s[0]] += pop_a[i] + pop_b[i]
        frag_charge = 0
        for ia in self.imp_atoms:
            symb = mol.symbol_of_atm(ia)
            nuc = mol.charge_of_atm(ia)
            log.info(self, 'charge of  %d%s =   %10.5f', \
                     ia, symb, nuc - chg[ia])
            frag_charge += nuc - chg[ia]
        log.info(self, 'charge of embsys = %10.5f', frag_charge)

        # diff between the SCF DM and DMET DM for fragment block
        if self.num_bath != -1:
            self.diff_dm()

    def diff_dm(self):
        mol = self.mol
        s = self.entire_scf.get_ovlp(self.mol)[0]
        c_inv = numpy.dot(self.orth_coeff.T, s)
        eff_scf = self.entire_scf
        mo_a = numpy.dot(c_inv, eff_scf.mo_coeff[0])
        mo_b = numpy.dot(c_inv, eff_scf.mo_coeff[1])
        dm0 = eff_scf.calc_den_mat((mo_a,mo_b), eff_scf.mo_occ)
        # in case impurity sites are not the AO orbitals
        mo_a = reduce(numpy.dot, (c_inv, self.impbas_coeff[0], \
                                  self.mo_coeff_on_imp[0]))
        mo_b = reduce(numpy.dot, (c_inv, self.impbas_coeff[1], \
                                  self.mo_coeff_on_imp[1]))
        dm1 = (numpy.dot(mo_a*self.mo_occ[0],mo_a.T),
               numpy.dot(mo_b*self.mo_occ[1],mo_b.T))
        dm1 = (dm1[0] + numpy.dot(self.env_orb[0], self.env_orb[0].T), \
               dm1[1] + numpy.dot(self.env_orb[1], self.env_orb[1].T))
        norm_a = numpy.linalg.norm((dm1[0]-dm0[0])[self.bas_on_frag][:,self.bas_on_frag])
        norm_b = numpy.linalg.norm((dm1[1]-dm0[1])[self.bas_on_frag][:,self.bas_on_frag])
        norm = numpy.sqrt(norm_a**2+norm_b**2)
        log.info(self, 'norm(diff of imp-DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm, norm/numpy.sqrt(2)/len(self.bas_on_frag))
        norm = numpy.sqrt(numpy.linalg.norm(dm0[0]-dm1[0])**2 \
                          +numpy.linalg.norm(dm0[1]-dm1[1])**2)
        log.info(self, 'norm(diff of entire DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm, norm/numpy.sqrt(2)/dm0[0].shape[0])

        bas_off_frag = [i for i in range(s.shape[0]) \
                        if i not in self.bas_on_frag]
        p_a = reduce(numpy.dot, (c_inv, self.impbas_coeff[0],
                                 self.impbas_coeff[0].T, c_inv.T))
        p_b = reduce(numpy.dot, (c_inv, self.impbas_coeff[1],
                                 self.impbas_coeff[1].T, c_inv.T))
        norm0a = numpy.linalg.norm((numpy.dot(dm0[0],p_a)-dm0[0])[self.bas_on_frag][:,bas_off_frag])
        norm0b = numpy.linalg.norm((numpy.dot(dm0[1],p_b)-dm0[1])[self.bas_on_frag][:,bas_off_frag])
        norm0 = numpy.sqrt((norm0a**2+norm0b**2)/2)
        norm1a = numpy.linalg.norm((dm0[0]-dm1[0])[self.bas_on_frag][:,bas_off_frag])
        norm1b = numpy.linalg.norm((dm0[1]-dm1[1])[self.bas_on_frag][:,bas_off_frag])
        norm1 = numpy.sqrt((norm1a**2+norm1b**2)/2)
        log.info(self, 'before SCF norm(diff off-diagonal DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm0, norm0/numpy.sqrt(len(self.bas_on_frag)*len(bas_off_frag)))
        log.info(self, 'after SCF norm(diff off-diagonal DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm1, norm1/numpy.sqrt(len(self.bas_on_frag)*len(bas_off_frag)))
        norm1a = numpy.linalg.norm((dm0[0]-dm1[0])[self.bas_on_frag])
        norm1b = numpy.linalg.norm((dm0[1]-dm1[1])[self.bas_on_frag])
        norm1 = numpy.sqrt((norm1a**2+norm1b**2)/2)
        log.info(self, 'after SCF norm(diff frag-band DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm1, norm1/numpy.sqrt(len(self.bas_on_frag)*s.shape[0]))


# **** GHO ****
    def set_gho_pseudo_bath(self, gho_atm_lst, inc_1s=False):
        assert(self.mol.pure_symbol_of_atm(gho_atm_lst[0]) == 'C')
        self.append_bath(gho_atm_lst)
        self.num_bath = 1
        if inc_1s:
            for i, s in enumerate(self.mol.labels_of_spheric_GTO()):
                if s[0] == gho_atm_lst[0] and s[2] == '1s':
                    self.imp_basidx = [i]
                    break

        def cons_impbas():
            import gho
            log.info(self, 'replace bath orbital with GHOs')
            g = gho.GHO()
            gho_orb = g.hybrid_coeff(self.mol, gho_atm_lst)
            gho_idx = gho.gho_index(self.mol, gho_atm_lst[0])

            ovlp_a = numpy.dot(self.bath_orb[0][gho_idx,:].T.conj(), gho_orb)
            for i,c in enumerate(ovlp_a):
                log.debug(self, 'alpha <bath_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
            p_hybs = numpy.dot(ovlp_a.T,ovlp_a).diagonal()
            log.debug(self, 'alpha <gho_i|bath><bath|gho_i> = %s', str(p_hybs))
            u, w, v = numpy.linalg.svd(ovlp_a)
            log.debug(self, 'alpha SVD <gho|bath> = %s', str(w))

            ovlp_b = numpy.dot(self.bath_orb[1][gho_idx,:].T.conj(), gho_orb)
            for i,c in enumerate(ovlp_b):
                log.debug(self, 'beta <bath_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
            p_hybs= numpy.dot(ovlp_b.T,ovlp_b).diagonal()
            log.debug(self, 'beta <gho_i|bath><bath|gho_i> = %s', str(p_hybs))
            u, w, v = numpy.linalg.svd(ovlp_b)
            log.debug(self, 'beta SVD <gho|bath> = %s', str(w))

            if self.env_orb[1].shape[1] > 0:
                ovlp_a = numpy.dot(self.env_orb[0][gho_idx,:].T.conj(), gho_orb)
                for i,c in enumerate(ovlp_a):
                    log.debug(self, 'alpha <env_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
                p_hybs = numpy.dot(ovlp_a.T,ovlp_a).diagonal()
                log.debug(self, 'alpha <gho_i|env><env|gho_i> = %s', str(p_hybs))
                u, w, v = numpy.linalg.svd(ovlp_a)
                log.debug(self, 'alpha SVD <gho|env> = %s', str(w))
                ovlp_b = numpy.dot(self.env_orb[1][gho_idx,:].T.conj(), gho_orb)
                for i,c in enumerate(ovlp_b):
                    log.debug(self, 'beta <env_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
                p_hybs = numpy.dot(ovlp_b.T,ovlp_b).diagonal()
                log.debug(self, 'beta <gho_i|env><env|gho_i> = %s', str(p_hybs))
                u, w, v = numpy.linalg.svd(ovlp_b)
                log.debug(self, 'beta SVD <gho|env> = %s', str(w))

            coord0 = self.mol.coord_of_atm(gho_atm_lst[0])
            dists = [numpy.linalg.norm(self.mol.coord_of_atm(i)-coord0) \
                     for i in self.imp_atoms]
            bondatm = self.imp_atoms[numpy.argmin(dists)]
            bath1a = self.bath_orb[0][gho_idx,0]/numpy.linalg.norm(self.bath_orb[0][gho_idx,0])
            bath1b = self.bath_orb[1][gho_idx,0]/numpy.linalg.norm(self.bath_orb[1][gho_idx,0])
            log.debug(self, 'alpha bath_1 hybrid = sp^%4.3f, angle to bond = %.6g', \
                      gho.sp_hybrid_level(bath1a), \
                      gho.angle_to_bond(self.mol, gho_atm_lst[0], \
                                        bondatm, bath1a))
            log.debug(self, 'beta bath_1 hybrid = sp^%4.3f, angle to bond = %.6g', \
                      gho.sp_hybrid_level(bath1b), \
                      gho.angle_to_bond(self.mol, gho_atm_lst[0], \
                                        bondatm, bath1b))
            log.debug(self, 'GHO-active hybrid = sp^%4.3f, angle to bond = %.6g', \
                      gho.sp_hybrid_level(gho_orb[:,0]), \
                      gho.angle_to_bond(self.mol, gho_atm_lst[0], \
                                        bondatm, gho_orb[:,0]))
            cosovlpa = numpy.dot(bath1a[1:4],gho_orb[1:4,0]) \
                    / numpy.linalg.norm(bath1a[1:4]) \
                    / numpy.linalg.norm(gho_orb[1:4,0])
            cosovlpb = numpy.dot(bath1b[1:4],gho_orb[1:4,0]) \
                    / numpy.linalg.norm(bath1b[1:4]) \
                    / numpy.linalg.norm(gho_orb[1:4,0])
            log.debug(self, 'angle between GHO and bath_1 (alpha, beta) = %.6g, %.6g', \
                      numpy.arccos(cosovlpa), numpy.arccos(cosovlpb))

            a = numpy.dot(self.orth_coeff[:,self.bas_on_frag], self.imp_site[0])
            b = numpy.dot(self.orth_coeff[:,gho_idx], gho_orb[:,:1])
            c = numpy.hstack((a,b))
            impbas_coeff = numpy.array((c,c))
            if self.orth_imp_to_env:
                impbas_coeff = self.suborth_imp_to_env(impbas_coeff)
            return impbas_coeff
        self.cons_impurity_basis = cons_impbas

    def dets_ovlp(self, mol, orbs):
        '''det(<i*|i>):  |i*> = P|i>,  P = |x>S^{-1}<x|'''
        mo_a = self.entire_scf.mo_coeff[0][:,self.entire_scf.mo_occ[0]>0]
        mo_b = self.entire_scf.mo_coeff[1][:,self.entire_scf.mo_occ[1]>0]
        s = self.entire_scf.get_ovlp(self.mol)[0]
        orbsa = numpy.hstack((orbs[0], numpy.dot(self.orth_coeff, self.env_orb[0])))
        orbsb = numpy.hstack((orbs[1], numpy.dot(self.orth_coeff, self.env_orb[1])))
        sinva = numpy.linalg.inv(reduce(numpy.dot, (orbsa.T, s, orbsa)))
        sinvb = numpy.linalg.inv(reduce(numpy.dot, (orbsb.T, s, orbsb)))
        proja = reduce(numpy.dot, (orbsa, sinva, orbsa.T))
        projb = reduce(numpy.dot, (orbsb, sinvb, orbsb.T))
        ovlpa = reduce(numpy.dot, (mo_a.T.conj(), s, proja, s, mo_a))
        ovlpb = reduce(numpy.dot, (mo_b.T.conj(), s, projb, s, mo_b))
        # <ovlp>**2 because of the beta orbital contribution
        return numpy.linalg.det(ovlpa)*numpy.linalg.det(ovlpb)



if __name__ == '__main__':
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 1
    mol.output = 'out_hf'
    mol.atom.extend([
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ])
    mol.basis = {'H': '6-31g',
                 'O': '6-31g',}
    mol.build()
    mf = scf.RHF(mol)
    print mf.scf()

#    imp = RHF(mf)
#    imp.set_embsys([1,])
#    print imp.bath_orbital(mol)
#    print imp.bath_orbital(mol, [2,3])

    emb = RHF(mf)
    emb.imp_basidx = [1,2,3,4]
    emb.imp_scf()

#    import ci
#    h1e = emb.get_hcore()
#    rdm1 = numpy.empty_like(h1e)
#    eri = emb.eri_on_impbas(mol)
#    rec = ci.fci._run(mol, emb.nelectron, h1e, eri, 0, rdm1=rdm1)
#    e_fci = ci.fci.find_fci_key(rec, 'STATE 1 ENERGY')
#    # no "numpy.dot(rdm1.flatten(), emb._vhf_env.flatten())" because it's
#    # already included in the dot(h1e,rdm1)
#    e_tot = e_fci + emb.energy_by_env
#    print e_fci, e_tot, e_tot + mol.nuclear_repulsion()
#    print rdm1
