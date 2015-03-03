#!/usr/bin/env python

import os
import tempfile
import numpy
import scipy.linalg
import h5py

from pyscf import gto
from pyscf import scf
from pyscf import lib
import pyscf.lib.parameters as param
import pyscf.lib.logger as log
from pyscf import ao2mo
from pyscf import lo
from pyscf import tools


def select_ao_on_fragment(mol, atm_lst, bas_idx=[]):
    log.info(mol, 'atm_lst of impurity sys: %s', \
             str(atm_lst))
    log.info(mol, 'extra bas_idx of impurity sys: %s', \
             str(bas_idx))
    lbl = mol.spheric_labels()
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
    w, pre_nao = scipy.linalg.eigh(frag_dm)
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
        tools.dump_mat.dump_mo(emb.mol, bath_orb)

    proj = -numpy.dot(bath_orb, bath_orb.T)
    for i in range(bath_orb.shape[0]):
        proj[i,i] += 1
    for i in bas_on_frag:
        proj[i,i] -= 1
    dm_env = reduce(numpy.dot, (proj, dm_orth, proj))
    w, env_orb = scipy.linalg.eigh(dm_env)
    env_orb = env_orb[:,w>.1] #remove orbitals with eigenvalue ~ 0
    return numpy.eye(nimp), bath_orb, env_orb

def decompose_orbital(emb, mo_orth, bas_on_frag, num_bath=-1,
                      gen_imp_site=False):
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

    if gen_imp_site:
        imp_site = mo_bath[bas_on_frag]/w[idx]
    else:
        nimp = len(bas_on_frag)
        imp_site = numpy.eye(nimp)

    if mo_bath.shape[1] > 0:
        mo_bath[bas_on_frag] = 0
        norm = 1/numpy.sqrt(1-w[idx]**2)
        bath_orb = mo_bath * norm
    else:
        bath_orb = mo_bath

    return imp_site, bath_orb, env_orb

def padding0(mat, dims):
    mat0 = numpy.zeros(dims)
    n, m = mat.shape
    mat0[:n,:m] = mat
    return mat0

class RHF(scf.hf.RHF):
    '''Non-relativistic restricted Hartree-Fock DMET'''
    def __init__(self, entire_scf, orth_ao=None):
        self.verbose = entire_scf.verbose
        self.stdout = entire_scf.stdout

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
        self.occ_env_cutoff = 1e-8 # an MO is considered as env_orb when imp_occ < 1e-8
        self.num_bath = -1
        self.chkfile = entire_scf.chkfile

        #if not entire_scf.scf_conv:
        #    log.info(self, "SCF again before DMET.")
        #    scf.hf.kenerl(entire_scf, entire_scf.conv_tol*1e2)
        self.entire_scf = entire_scf
        self.mol = entire_scf.mol
        self.max_memory = self.entire_scf.max_memory

        self.imp_atoms = []
        self.imp_basidx = []

        if orth_ao is None:
            self.pre_orth_ao = numpy.eye(self.mol.nao_nr())
            #self.pre_orth_ao = lo.iao.pre_atm_scf_ao(mol)
            #self.pre_orth_ao = lo.iao.preiao(mol)
            self.orth_ao_method = 'lowdin'
            #self.orth_ao_method = 'meta_lowdin'
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
        self.conv_tol = entire_scf.conv_tol
        self.max_cycle = entire_scf.max_cycle
        self.init_guess = None
        self.direct_scf = entire_scf.direct_scf
        self.direct_scf_tol = entire_scf.direct_scf_tol
        self.level_shift_factor = 0
        self.damp_factor = 0
        self.scf_conv = False
        self.direct_scf = True
        self.direct_scf_tol = 1e-13
        self._eri = None
        self.energy_by_env = 0

    def dump_flags(self):
        log.info(self, '\n')
        log.info(self, '******** DMET SCF starting *************')
        log.info(self, 'bath/env cutoff = %g', self.occ_env_cutoff)
        log.info(self, 'num_bath = %g\n', self.num_bath)
        scf.hf.SCF.dump_flags(self)

    def decompose_den_mat(self, dm_orth):
        return decompose_den_mat(self, dm_orth*.5, self.bas_on_frag, self.num_bath)
    def decompose_orbital(self, mo_orth):
        return decompose_orbital(self, mo_orth, self.bas_on_frag, self.num_bath)

##################################################
# scf for impurity
    def init_dmet_scf(self):
        self.build_()
    def build_(self):
        mol = self.mol
        self.orth_coeff = self.get_orth_ao(mol)

        self.bas_on_frag = select_ao_on_fragment(mol, self.imp_atoms, \
                                                 self.imp_basidx)
        c_inv = numpy.dot(self.orth_coeff.T, self.entire_scf.get_ovlp(mol))
        mocc = self.entire_scf.mo_coeff[:,self.entire_scf.mo_occ>1e-15]
        mo_orth = numpy.dot(c_inv, mocc)

        # self.imp_site, self.bath_orb, self.env_orb are based on orth-orbitals
        self.imp_site, self.bath_orb, self.env_orb = \
                self.decompose_orbital(mo_orth)
        self.impbas_coeff = self.cons_impurity_basis()
        self.nelectron = mol.nelectron - self.env_orb.shape[1] * 2
        log.info(self, 'nelec of emb = %d', self.nelectron)

        self._eri = self.eri_on_impbas(mol)

        self.energy_by_env, self._vhf_env = self.init_vhf_env(self.env_orb)

    def init_vhf_env(self, env_orb):
        log.debug(self, 'init Hartree-Fock environment')
        env_orb = numpy.dot(self.orth_coeff, env_orb)
        dm_env = numpy.dot(env_orb, env_orb.T.conj()) * 2
        vhf_env_ao = self.entire_scf.get_veff(self.mol, dm_env)
        hcore = self.entire_scf.get_hcore(self.mol)
        energy_by_env = numpy.dot(dm_env.flatten(), hcore.flatten()) \
                      + numpy.dot(dm_env.flatten(), vhf_env_ao.flatten()) * .5
        return energy_by_env, self.mat_ao2impbas(vhf_env_ao)

    def get_init_guess(self, key=None):
        log.debug(self, 'init guess based on entire MO coefficients')
        mol = self.mol
        s = self.entire_scf.get_ovlp(mol)
        eff_scf = self.entire_scf
        entire_scf_dm = eff_scf.make_rdm1(eff_scf.mo_coeff, eff_scf.mo_occ)
        env_orb = numpy.dot(self.orth_coeff, self.env_orb)
        dm_env = numpy.dot(env_orb, env_orb.T.conj()) * 2
        cs = numpy.dot(self.impbas_coeff.T.conj(), s)
        dm = reduce(numpy.dot, (cs, entire_scf_dm-dm_env, cs.T.conj()))
        hf_energy = 0
        return dm

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
        mol = self.mol
        h1e = self.mat_ao2impbas(self.entire_scf.get_hcore(mol)) \
                + self._vhf_env
        return h1e

    def get_ovlp(self, mol=None):
        mol = self.mol
        s1e = self.mat_ao2impbas(self.entire_scf.get_ovlp(mol))
        return s1e

    def get_occ(self, mo_energy, mo_coeff=None):
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

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff_on_imp
        if mo_occ is None:
            mo_occ = self.mo_occ
        nbf = mo_coeff.shape[0]
        mo = mo_coeff[:,mo_occ>0]
        dm = numpy.dot(mo, mo.T.conj()) * 2
        log.debug1(self, 'density.diag = %s', dm.diagonal())
        return dm

    def eri_on_impbas(self, mol):
        if self.entire_scf._eri is not None:
            eri = ao2mo.incore.full(self.entire_scf._eri, self.impbas_coeff)
        else:
            eri = ao2mo.outcore.full_iofree(mol, self.impbas_coeff)
        eri = ao2mo.restore(8, eri, self.impbas_coeff.shape[1])
        return eri

    def release_eri(self):
        self._eri = None


    def get_veff(self, mol, dm, dm_last=0, vhf_last=0):
        if self._eri is None:
            self._eri = self.eri_on_impbas(mol)
        vj, vk = scf.hf.dot_eri_dm(self._eri, dm, hermi=1)
        vhf = vj - vk * .5
        return vhf

    # emb basis are assumed orthogonal
    def calc_frag_elec_energy(self, mol, vhf, dm):
        h1e = self.get_hcore(mol)
        nimp = len(self.bas_on_frag)

        nelec_frag = dm[:nimp].trace()
        log.info(self, 'number of electrons in fragment = %.15g', \
                 nelec_frag)

        e = (dm[:nimp]*(h1e-self._vhf_env)[:nimp]).sum() \
          + (dm[:nimp]*(vhf+self._vhf_env)[:nimp]).sum() * .5
        e_tot = numpy.einsum('ij,ji', dm, h1e) \
              + numpy.einsum('ij,ji', dm, vhf)*.5 \
              + self.energy_by_env
        log.info(self, 'fragment electronic energy = %.15g', e)
        log.debug(self, ' ~ total energy (non-variational) = %.15g', e_tot)
        return e_tot, e, nelec_frag

    def imp_scf(self):
        self.build_()
        self.dump_flags()

        self.scf_conv, self.hf_energy, self.mo_energy, \
                self.mo_coeff_on_imp, self.mo_occ \
                = scf.hf.kernel(self.mol, self, self.conv_tol, \
                                   dump_chk=False)

        log.info(self, 'impurity MO energy')
        for i in range(self.mo_energy.size):
            if self.mo_occ[i] > 0:
                log.info(self, 'impurity occupied MO %d energy = %.15g occ=%g', \
                         i+1, self.mo_energy[i], self.mo_occ[i])
            else:
                log.info(self, 'impurity virtual MO %d energy = %.15g occ=%g', \
                         i+1, self.mo_energy[i], self.mo_occ[i])

        #e_nuc = self.energy_nuc(self.mol)
        #log.log(self, 'impurity sys nuclear repulsion = %.15g', e_nuc)
        if self.scf_conv:
            log.log(self, 'converged impurity sys electronic energy = %.15g', \
                    self.hf_energy)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'electronic energy = %.15g after %d cycles.', \
                    self.hf_energy, self.max_cycle)

        # mo_coeff_on_imp based on embedding basis
        # mo_coeff based on AOs
        self.mo_coeff = numpy.dot(self.impbas_coeff, self.mo_coeff_on_imp)

        dm = self.make_rdm1(self.mo_coeff_on_imp, self.mo_occ)
        vhf = self.get_veff(self.mol, dm)
        self.hf_energy, self.e_frag, self.nelec_frag = \
                self.calc_frag_elec_energy(self.mol, vhf, dm)
        log.log(self, 'fragment electronic energy = %.15g', self.e_frag)
        log.log(self, 'fragment electron number = %.15g', self.nelec_frag)
        self.frag_mulliken_pop()
        return self.e_frag

    def frag_mulliken_pop(self):
        '''Mulliken M_ij = D_ij S_ji, Mulliken chg_i = \sum_j M_ij'''
        mol = self.mol
        log.info(self, ' ** Mulliken pop (on impurity basis)  **')
        s1e = mol.intor_symmetric('cint1e_ovlp_sph')
        c_inv = numpy.dot(self.orth_coeff.T, s1e)
        c_frag = numpy.dot(c_inv, self.impbas_coeff)
        dm = self.make_rdm1(self.mo_coeff_on_imp, self.mo_occ)
        nimp = len(self.bas_on_frag)
        dm[nimp:] = 0
        dm = reduce(numpy.dot, (c_frag, dm, c_frag.T.conj()))
        pop = dm.diagonal()
        label = mol.spheric_labels()

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

# scf for impurity end
##################################################

    def get_orth_ao(self, mol):
        if self.orth_coeff is None:
            log.debug(self, 'orth method = %s', self.orth_ao_method)
            return lo.orth.orth_ao(mol, self.orth_ao_method,
                                   self.pre_orth_ao,
                                   self.entire_scf)
        else:
            return self.orth_coeff

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
        t = lo.schmidt_orth_coeff(reduce(numpy.dot, (c.T.conj(), s, c)))
        off = self.env_orb.shape[1]
        impbas_coeff = numpy.dot(c, t)[:,off:]
        return impbas_coeff


##################################################
class UHF(RHF, scf.uhf.UHF):
    '''Non-relativistic unrestricted Hartree-Fock DMET'''
    def __init__(self, entire_scf, orth_ao=None):
        #assert(isinstance(entire_scf, scf.hf.UHF) \
        #      or isinstance(entire_scf, scf.hf_symm.UHF))
        RHF.__init__(self, entire_scf, orth_ao)
        self.nelectron_alpha = None
        self.nelectron_beta = None
        self.DIIS = UHF_DIIS
        self._keys = self._keys | set(['nelectron_alpha', 'nelectron_beta'])

    def decompose_den_mat(self, dm_orth):
        imp_a, bath_a, env_a = decompose_den_mat(self, dm_orth[0], \
                                                 self.bas_on_frag, self.num_bath)
        imp_b, bath_b, env_b = decompose_den_mat(self, dm_orth[1], \
                                                 self.bas_on_frag, self.num_bath)
        if bath_a.shape[1] == bath_b.shape[1]:
            cc = [bath_a,bath_b]
        elif bath_a.shape[1] > bath_b.shape[1]:
            cc = [bath_a,padding0(bath_b,bath_a.shape)]
        else:
            cc = [padding0(bath_a,bath_b.shape),bath_b]
        return (imp_a,imp_b), cc, (env_a,env_b)

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
        if bath_a.shape[1] == bath_b.shape[1]:
            cc = [bath_a,bath_b]
        elif bath_a.shape[1] > bath_b.shape[1]:
            cc = [bath_a,padding0(bath_b,bath_a.shape)]
        else:
            cc = [padding0(bath_a,bath_b.shape),bath_b]
        return (imp_a,imp_b), cc, (env_a,env_b)

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
        s = self.entire_scf.get_ovlp(self.mol)
        t_a = lo.schmidt_orth_coeff(reduce(numpy.dot, (c_a.T.conj(), s, c_a)))
        t_b = lo.schmidt_orth_coeff(reduce(numpy.dot, (c_b.T.conj(), s, c_b)))
        impbas_coeff = (numpy.dot(c_a, t_a)[:,self.env_orb[0].shape[1]:], \
                        numpy.dot(c_b, t_b)[:,self.env_orb[1].shape[1]:])
        return impbas_coeff

    def init_dmet_scf(self):
        self.build_(self)
    def build_(self):
        mol = self.mol
        self.orth_coeff = self.get_orth_ao(mol)
        self.bas_on_frag = select_ao_on_fragment(mol, self.imp_atoms, \
                                                 self.imp_basidx)
        c_inv = numpy.dot(self.orth_coeff.T, self.entire_scf.get_ovlp(self.mol))
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
        self.nelectron_alpha = self.entire_scf.nelectron_alpha \
                - self.env_orb[0].shape[1]
        self.nelectron_beta = mol.nelectron \
                - self.entire_scf.nelectron_alpha \
                - self.env_orb[1].shape[1]
        log.info(self, 'alpha / beta electrons = %d / %d', \
                 self.nelectron_alpha, self.nelectron_beta)

        self.energy_by_env, self._vhf_env = self.init_vhf_env(self.env_orb)

    def init_vhf_env(self, env_orb):
        log.debug(self, 'init Hartree-Fock environment')
        mol = self.mol
        env_a = numpy.dot(self.orth_coeff, env_orb[0])
        env_b = numpy.dot(self.orth_coeff, env_orb[1])
        dm_env = numpy.array([numpy.dot(env_a, env_a.T.conj()), \
                              numpy.dot(env_b, env_b.T.conj())])
        vhf_env_ao = scf.hf.UHF.get_veff(self.entire_scf, self.mol, dm_env)
        hcore = scf.hf.UHF.get_hcore(mol)
        energy_by_env = numpy.einsum('ij,ji', dm_env[0], hcore[0]) \
                      + numpy.einsum('ij,ji', dm_env[1], hcore[1]) \
                      + numpy.einsum('ij,ji', dm_env[0], vhf_env_ao[0]) * .5 \
                      + numpy.einsum('ij,ji', dm_env[1], vhf_env_ao[1]) * .5
        return energy_by_env, self.mat_ao2impbas(vhf_env_ao)

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
        nimp = len(self.bas_on_frag)
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


# **** impurity SCF ****
    def check_dm_converge(self, dm, dm_last, conv_tol):
        delta_dm = abs(dm[0]-dm_last[0]).sum() + abs(dm[1]-dm_last[1]).sum()
        dm_change = delta_dm/(abs(dm_last[0]).sum()+abs(dm_last[1]).sum())
        log.info(self, '          sum(delta_dm)=%g (~ %g%%)\n', \
                 delta_dm, dm_change*100)
        return dm_change < conv_tol*1e2

    def get_init_guess(self, mol):
        log.debug(self, 'init guess based on entire MO coefficients')
        s = self.entire_scf.get_ovlp(self.mol)
        eff_scf = self.entire_scf
        entire_scf_dm = eff_scf.make_rdm1(eff_scf.mo_coeff, eff_scf.mo_occ)
        env_a = numpy.dot(self.orth_coeff, self.env_orb[0])
        env_b = numpy.dot(self.orth_coeff, self.env_orb[1])
        dm_a = numpy.dot(env_a, env_a.T.conj())
        dm_b = numpy.dot(env_b, env_b.T.conj())
        cs_a = numpy.dot(self.impbas_coeff[0].T.conj(), s)
        cs_b = numpy.dot(self.impbas_coeff[1].T.conj(), s)
        dm_a = reduce(numpy.dot, (cs_a, entire_scf_dm[0]-dm_a, cs_a.T.conj()))
        dm_b = reduce(numpy.dot, (cs_b, entire_scf_dm[1]-dm_b, cs_b.T.conj()))
        hf_energy = 0
        return numpy.array((dm_a,dm_b))

#    def eri_on_impbas(self, mol):
#        if self.entire_scf._eri is not None:
#            eri = ao2mo.incore.full(self.entire_scf._eri, self.impbas_coeff)
#        else:
#            eri = ao2mo.direct.full_iofree(mol, self.impbas_coeff)
#        return eri

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0):
        dm_a = reduce(numpy.dot, (self.impbas_coeff[0], dm[0], \
                                  self.impbas_coeff[0].T))
        dm_b = reduce(numpy.dot, (self.impbas_coeff[1], dm[1], \
                                  self.impbas_coeff[1].T))
        dm_ao = numpy.array((dm_a, dm_b))
        vhf_ao = scf.hf.UHF.get_veff(self.entire_scf, self.mol, dm_ao)
        return self.mat_ao2impbas(vhf_ao)

    def get_hcore(self, mol):
        h1e = self.mat_ao2impbas(scf.hf.RHF.get_hcore(mol))
        return (h1e[0]+self._vhf_env[0], h1e[1]+self._vhf_env[1])

    def eig(self, fock, s):
        e_a, c_a = scipy.linalg.eigh(fock[0], s[0])
        e_b, c_b = scipy.linalg.eigh(fock[1], s[1])
        return (e_a,e_b), (c_a,c_b)

    def get_fock(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None):
        f = (h1e[0]+vhf[0], h1e[1]+vhf[1])
        if 0 <= cycle < self.diis_start_cycle-1:
            f = (scf.hf.damping(s1e[0], dm[0], f[0], self.damp_factor), \
                 scf.hf.damping(s1e[1], dm[1], f[1], self.damp_factor))
            f = (scf.hf.level_shift(s1e[0],dm[0],f[0],self.level_shift_factor), \
                 scf.hf.level_shift(s1e[1],dm[1],f[1],self.level_shift_factor))
        elif 0 <= cycle:
            fac = self.level_shift_factor \
                    * numpy.exp(self.diis_start_cycle-cycle-1)
            f = (scf.hf.level_shift(s[0], d[0], f[0], fac), \
                 scf.hf.level_shift(s[1], d[1], f[1], fac))

        if adiis is not None and cycle >= self.diis_start_cycle:
            f = adiis.update(s1e, dm, f)
            f = (f[:h1e[0].size].reshape(h1e[0].shape), \
                 f[h1e[0].size:].reshape(h1e[1].shape))
        return f

    def get_occ(self, mo_energy, mo_coeff=None):
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

    def make_rdm1(self, mo_coeff, mo_occ):
        mo_a = mo_coeff[0][:,mo_occ[0]>0]
        mo_b = mo_coeff[1][:,mo_occ[1]>0]
        dm_a = numpy.dot(mo_a, mo_a.T.conj())
        dm_b = numpy.dot(mo_b, mo_b.T.conj())
        #log.debug(self, 'alpha density.diag = %s', dm_a.diagonal())
        #log.debug(self, 'beta  density.diag = %s', dm_b.diagonal())
        return (dm_a,dm_b)

    def imp_scf(self):
        self.dump_flags()
        self.build_()

        self.scf_conv, self.hf_energy, self.mo_energy, \
                self.mo_coeff_on_imp, self.mo_occ \
                = scf.hf.kernel(self.mol, self, self.conv_tol, \
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
                    self.hf_energy, self.max_cycle)

        dm = self.make_rdm1(self.mo_coeff_on_imp, self.mo_occ)
        vhf = self.get_veff(self.mol, dm)
        self.hf_energy, self.e_frag, self.nelec_frag = \
                self.calc_frag_elec_energy(self.mol, vhf, dm)
        log.log(self, 'fragment electronic energy = %.15g', self.e_frag)
        log.log(self, 'fragment electron number = %.15g', self.nelec_frag)
        self.frag_mulliken_pop()
        return self.e_frag

    def calc_frag_elec_energy(self, mol, vhf, dm):
        h1e = self.get_hcore(mol)
        nimp = len(self.bas_on_frag)
        dm_frag_a = numpy.dot(dm[0], proj_a)
        dm_frag_b = numpy.dot(dm[1], proj_b)

        nelec_frag = dm[0][:nimp].trace() + dm[1][:nimp].trace()
        log.info(self, 'number of electrons in fragment = %.15g', \
                 nelec_frag)

        e = (dm[0][:nimp]*(h1e[0]-self._vhf_env[0])[:nimp]).sum() \
          + (dm[1][:nimp]*(h1e[1]-self._vhf_env[1])[:nimp]).sum() \
          + (dm[0][:nimp]*(vhf[0]+self._vhf_env[0])[:nimp]).sum()*.5 \
          + (dm[1][:nimp]*(vhf[1]+self._vhf_env[1])[:nimp]).sum()*.5
        e_tot = numpy.einsum('ij,ji', dm[0], h1e[0]) \
              + numpy.einsum('ij,ji', dm[1], h1e[1]) \
              + numpy.einsum('ij,ji', dm[0], vhf[0])*.5 \
              + numpy.einsum('ij,ji', dm[1], vhf[1])*.5 \
              + self.energy_by_env
        log.info(self, 'fragment electronic energy = %.15g', e)
        log.debug(self, ' ~ total energy (non-variational) = %.15g', e_tot)
        return e_tot, e, nelec_frag

    def frag_mulliken_pop(self):
        '''Mulliken M_ij = D_ij S_ji, Mulliken chg_i = \sum_j M_ij'''
        mol = self.mol
        log.info(self, ' ** Mulliken pop alpha/beta (on impurity basis)  **')
        s1e = self.entire_scf.get_ovlp(mol)
        c_inv = numpy.dot(self.orth_coeff.T, s1e)
        c_frag_a = numpy.dot(c_inv, self.impbas_coeff[0])
        c_frag_b = numpy.dot(c_inv, self.impbas_coeff[1])
        dm = self.make_rdm1(self.mo_coeff_on_imp, self.mo_occ)
        nimp = len(self.bas_on_frag)
        dm[0][nimp:] = 0
        dm[1][nimp:] = 0
        dm_a = reduce(numpy.dot, (c_frag_a, dm[0], c_frag_a.T.conj()))
        dm_b = reduce(numpy.dot, (c_frag_b, dm[1], c_frag_b.T.conj()))
        pop_a = dm_a.diagonal()
        pop_b = dm_b.diagonal()
        label = mol.spheric_labels()

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


class UHF_DIIS(scf.uhf.UHF_DIIS):
    def update(self, s, d, f):
        self.push_err_vec(s, d, f)
        fflat = numpy.hstack((f[0].ravel(), f[1].ravel()))
        return scf.diis.DIIS.update(self, fflat)


if __name__ == '__main__':
    from pyscf import scf
    mol = gto.Mole()
    mol.verbose = 5
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

    emb = RHF(mf)
    emb.imp_basidx = [1,2,3,4]
    print emb.imp_scf() - -17.5787474773
    print emb.nelec_frag - 3.51001441353
    emb.imp_basidx = [0] + range(5,13)
    print emb.imp_scf() - -67.5934594386
    print emb.nelec_frag - 6.48998558647

