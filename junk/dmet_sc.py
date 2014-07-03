#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import time
import pickle
import numpy
import scipy

import pyscf.lib.logger as log
from pyscf import scf
import hf
import scfopt
import scfci
import hfdm
import pyscf.lib.parameters as param


# options for fit_domain
IMP_AND_BATH  = hfdm.IMP_AND_BATH  # 1
IMP_BLK       = hfdm.IMP_BLK       # 2
IMP_BATH_DIAG = hfdm.IMP_BATH_DIAG # 3
NO_BATH_BLK   = hfdm.NO_BATH_BLK   # 4
DIAG_BLK      = hfdm.DIAG_BLK      # 5
IMP_DIAG      = hfdm.IMP_DIAG      # 6
NO_IMP_BLK    = hfdm.NO_IMP_BLK    # 7
TRACE_IMP     = hfdm.TRACE_IMP     # 8

# options for dm_fit_constraint
NO_CONSTRAINT = hfdm.NO_CONSTRAINT # 0
#IMP_DIAG      = 6
#TRACE_IMP     = 8

# options for local_fit_approx
FITTING_WITHOUT_SCF = 1
FITTING_WITH_SCF    = 2
FITTING_1SHOT       = 3
FITTING_FCI_POT     = 4

# options for global_fit_dm
# In the global fitting, fix the mean field / correlated / both density matrix
NO_FIXED_DM           = 1
FIXED_CI_DM           = 2
FIXED_MF_DM           = 3
NO_FIXED_DM_BACKWARDS = 4

# optinos for env_pot_for_fci
NO_ENV_POT   = 0
#IMP_AND_BATH  = 1
#IMP_BLK       = 2
#IMP_BATH_DIAG = 3
#NO_BATH_BLK   = 4
#DIAG_BLK      = 5
#IMP_DIAG      = 6
#NO_IMP_BLK    = 7
SCAL_ENV_POT = 11



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
    basidx_group = []
    for m, frags in enumerate(frag_group):
        if isinstance(frags[0], int):
            basidx_group.append(_remove_bas_if_not_on_frag(frags))
        else:
            tmp_group = []
            for atm_lst in frags:
                tmp_group.append(_remove_bas_if_not_on_frag(atm_lst))
            basidx_group.append(tmp_group)
    return basidx_group

def gen_frag_looper(mol, frag_group):
    frag_bas_idx = map_frag_to_bas_idx(mol, frag_group)
    all_frags = []
    uniq_frags = []
    for emb_id, frags in enumerate(frag_group):
        if isinstance(frags[0], int):
            all_frags.append((emb_id, frags, frag_bas_idx[emb_id]))
            uniq_frags.append((emb_id, frags, frag_bas_idx[emb_id]))
        else:
            uniq_frags.append((emb_id, frags[0], frag_bas_idx[emb_id][0]))
            for k, atm_lst in enumerate(frags):
                all_frags.append((emb_id, atm_lst, frag_bas_idx[emb_id][k]))
    return all_frags, uniq_frags


#TODO:
def embs_eri_ao2mo(embs, eri_ao):
    import dmet_misc
    c = []
    offsets = [0]
    off = 0
    for emb in embs:
        c.append(emb.impbas_coeff)
        off += emb.impbas_coeff.shape[1]
        offsets.append(off)
    c = numpy.array(numpy.hstack(c), order='F')
    offsets = numpy.array(offsets, dtype=numpy.int32)
    v = dmet_misc.embs_eri_ao2mo_o3(eri_ao, c, offsets)
    for n,emb in enumerate(embs):
        emb._eri = v[n]
    return embs


class EmbSys(object):
    '''DMFET/DMET_SC
    v_fit_domain
    dm_fit_domain
    '''
    def __init__(self, mol, entire_scf, frag_group, init_v=None):
        self.verbose = mol.verbose
        self.fout = mol.fout
        self.mol = mol
        self.emb_verbose = param.VERBOSE_QUITE

        self.max_iter         = 40
        self.conv_threshold   = scfopt.CONV_THRESHOLD #1e-5
        self.global_fit_dm    = NO_FIXED_DM
        self.v_fit_domain     = IMP_BLK
        self.dm_fit_domain    = IMP_BLK
        self.dm_fit_constraint = NO_CONSTRAINT
# env_fit_pot will be added to FCI solver, the fitting pot for impurity in the
# MF calculation is extracted from self.v_global on the fly
# * use NO_ENV_POT to avoid double counting on the correlation on bath, since
#   the fitting potential has already counted the correlation effects.
        self.env_pot_for_fci  = NO_ENV_POT #NO_IMP_BLK
# whether select the occupations to maximize the overlap to the previous states
        self.hf_follow_state  = False
# if > 0, scale the fitting potential, it helps convergence when
# local_vfit_method is fit_pot_without_local_scf
        self.fitpot_damp_fac  = scfopt.TUNE_FAC #.5
# when env_fit_pot covers imp+bath, with_hopping=true will transform the
# imp-bath off-diagonal block to the global potential 
        self.with_hopping     = False
        self.rand_init        = False

        self.frag_group = frag_group
        self.basidx_group = None
        self.all_frags = None
        self.uniq_frags = None
        self.v_global = None
        self.entire_scf = entire_scf
        self.embs = None
        self.orth_coeff = None
        self.bas_off_frags = []
        self.vfit_method = fit_pot_without_local_scf

        self.init_embsys(mol, init_v)

    #def __copy__(self):
    #    new = self.__class__()

    def dump_options(self):
        log.info(self, '\n')
        log.info(self, '******** DMFET/DMET_SC Options *********')
        log.info(self, 'max_iter        = %g', self.max_iter       )
        log.info(self, 'conv_threshold  = %g', self.conv_threshold )
        log.info(self, 'global_fit_dm   = %g', self.global_fit_dm  )
        log.info(self, 'v_fit_domain    = %g', self.v_fit_domain   )
        log.info(self, 'dm_fit_domain   = %g', self.dm_fit_domain  )
        log.info(self, 'dm_fit_constraint = %g', self.dm_fit_constraint)
        log.info(self, 'vfit_method     = %s', self.vfit_method.__name__)
        log.info(self, 'env_pot_for_fci = %g', self.env_pot_for_fci)
        log.info(self, 'hf_follow_state = %g', self.hf_follow_state)
        log.info(self, 'fitpot_damp_fac = %g', self.fitpot_damp_fac)
        log.info(self, 'with_hopping    = %g', self.with_hopping   )
        log.info(self, 'rand_init       = %g', self.rand_init      )


    def init_embsys(self, mol, init_v):
        #self.basidx_group = map_frag_to_bas_idx(mol, self.frag_group)
        self.all_frags, self.uniq_frags = gen_frag_looper(mol, self.frag_group)
        self.embs = self.setup_embs(mol)
        self.orth_coeff = self.embs[0].orth_coeff
        self.bas_off_frags = self.set_bas_off_frags()
        try:
            with open(init_v, 'r') as f:
                self.v_global, v_add_on_ao = pickle.load(f)
            self.entire_scf = scfci.run_hf_with_ext_pot(mol, entire_scf, \
                                                        v_add_on_ao, \
                                                        self.follow_state)
        except:
            nao = self.orth_coeff.shape[1]
            self.v_global = numpy.zeros((nao,nao))
            if self.rand_init:
                for m, atm_lst, bas_idx in self.all_frags:
                    nimp = bas_idx.__len__()
                    v = numpy.random.randn(nimp*nimp).reshape(nimp,nimp)
                    v = (v + v.T) * .1
                    for i, j in enumerate(bas_idx):
                        self.v_global[j,bas_idx] = v[i]
                self.entire_scf = scfci.run_hf_with_ext_pot(mol, entire_scf, \
                                                            self.v_global, \
                                                            self.follow_state)

    def frag_fci_solver(self, mol, emb):
        return scfci.frag_fci(mol, emb, emb.env_fit_pot)

    def meta_lowdin_orth(self, mol):
        pre_orth = hf.pre_orth_ao_atm_scf(mol)
        self.orth_coeff = hf.orthogonalize_ao(mol, self.entire_scf,
                                                   pre_orth, 'meta_lowdin')
        for emb in self.embs:
            emb.orth_coeff = self.orth_coeff
        return self.orth_coeff

    def setup_embs(self, mol):
        embs = []
        for m, atm_lst, bas_idx in self.uniq_frags:
            emb = hf.RHF(self.entire_scf)
            emb.occ_env_cutoff = 1e-14
            emb.imp_atoms = atm_lst
            emb.bas_on_frag = bas_idx
            embs.append(emb)

        orth_coeff = embs[0].get_orth_ao(mol)
        for m, emb in enumerate(embs):
            emb.orth_coeff = orth_coeff
        return embs

    def set_env_fit_pot_for_fci(self, v_global):
        for emb in self.embs:
            nimp = emb.dim_of_impurity()
            nemb = emb.num_of_impbas()
            if self.env_pot_for_fci == NO_IMP_BLK:
                emb.env_fit_pot = emb.mat_orthao2impbas(v_global)
                emb.env_fit_pot[:nimp,:nimp] = 0
#TODO:elif self.env_pot_for_fci == SCAL_ENV_POT::
            else:
                emb.env_fit_pot = numpy.zeros((nemb,nemb))

    def setup_embs_with_vglobal(self, mol, embs, v_global):
        t0 = time.clock()
        eff_scf = self.entire_scf
        verbose_bak = eff_scf.verbose
        eff_scf.verbose = 0
        sc = numpy.dot(mol.intor_symmetric('cint1e_ovlp_sph'),eff_scf.mo_coeff)
        c_inv = numpy.dot(mol.intor_symmetric('cint1e_ovlp_sph'),self.orth_coeff).T
        fock0 = numpy.dot(sc*eff_scf.mo_energy, sc.T.conj())
        hcore = scf.hf.RHF.get_hcore(mol)
        for ifrag, emb in enumerate(embs):
            emb.verbose = self.emb_verbose
            mo_orth = numpy.dot(c_inv, eff_scf.mo_coeff[:,eff_scf.mo_occ>1e-15])
            emb.imp_site, emb.bath_orb, emb.env_orb = \
                    hf.decompose_orbital(emb, mo_orth, emb.bas_on_frag)
            emb.impbas_coeff = emb.cons_impurity_basis()
            emb.nelectron = mol.nelectron - emb.env_orb.shape[1] * 2
            log.debug(emb, 'number of electrons for impurity %d = %d', \
                      ifrag, emb.nelectron)
            emb._vhf_env = emb.init_vhf_env(mol, emb.env_orb)

            f = reduce(numpy.dot, (emb.impbas_coeff.T, fock0, emb.impbas_coeff))
            emb.mo_energy, emb.mo_coeff_on_imp = numpy.linalg.eigh(f)
            emb.mo_coeff = numpy.dot(emb.impbas_coeff, emb.mo_coeff_on_imp)
            emb.hf_energy = 0
# *** just for optimizing ******
            emb._pure_hcore = emb.mat_ao2impbas(hcore)

        if eff_scf._eri is not None:
            t1 = time.clock()
            self.embs = embs_eri_ao2mo(self.embs, eff_scf._eri)
            log.debug(self, 'CPU time for embsys eri AO->MO: %.8g sec', \
                      time.clock()-t1)
            #for emb in self.embs:
            #    emb._eri = scfci.partial_eri_ao2mo(eff_scf._eri, emb.impbas_coeff)
# *** end optimizing ***********
        log.debug(self, 'CPU time for set up embsys.embs: %.8g sec', \
                  time.clock()-t0)
        eff_scf.verbose = verbose_bak
        return embs

    # should call this subroutine before dmet_scf_cycle to initialize
    # embs.*_fit_pot
    def update_embsys_vglobal(self, mol, v_add):
        v_add_ao = scfci.mat_orthao2ao(mol, v_add, self.orth_coeff)
        eff_scf = scfci.run_hf_with_ext_pot(mol, self.entire_scf, v_add_ao)

        self.v_global = v_add
        self.entire_scf = eff_scf

        for emb in self.embs:
            emb.entire_scf = eff_scf
        self.setup_embs_with_vglobal(mol, self.embs, v_add)
        self.set_env_fit_pot_for_fci(v_add)
        return eff_scf

    def set_bas_off_frags(self):
        nao = self.orth_coeff.shape[1]
        baslst = numpy.ones(nao, dtype=bool)
        for m, atm_lst, bas_idx in self.all_frags:
            baslst[bas_idx] = False
        return [i for i in range(nao) if baslst[i]]

    def off_frags_energy(self, mol, dm_mf):
        if len(self.bas_off_frags) == 0:
            return 0, 0
        nelec = dm_mf.diagonal()[self.bas_off_frags].sum()
        h1e = reduce(numpy.dot, (self.orth_coeff.T, \
                                 scf.hf.RHF.get_hcore(mol), \
                                 self.orth_coeff)) + self.v_global
        h1e = h1e[self.bas_off_frags]
        dm_ao = reduce(numpy.dot, (self.orth_coeff,dm_mf,self.orth_coeff.T))
        vhf = scf.hf.RHF.get_eff_potential(self.entire_scf, mol, dm_ao)
        vhf = reduce(numpy.dot, (self.orth_coeff.T,vhf,self.orth_coeff))
        vhf = vhf[self.bas_off_frags]
        dm_frag = dm_mf[self.bas_off_frags]
        e = numpy.dot(h1e.flatten(), dm_frag.flatten()) \
                + numpy.dot(vhf.flatten(), dm_frag.flatten())*.5
        return e, nelec

    def assemble_frag_fci_energy(self, mol, dm_ref=0):
        if len(self.bas_off_frags) == 0:
            val_tot = 0
            nelec = 0
        else:
            if dm_ref is 0:
                eff_scf = self.entire_scf
                sc = numpy.dot(mol.intor_symmetric('cint1e_ovlp_sph'), \
                               self.orth_coeff)
                mo = numpy.dot(sc.T,eff_scf.mo_coeff)
                dm_ref = eff_scf.calc_den_mat(mo, eff_scf.mo_occ)
            val_tot, nelec = self.off_frags_energy(mol, dm_ref)
        for m, emb in enumerate(self.embs):
            rdm1, rec = self.frag_fci_solver(mol, emb)
            val_frag, nelec_frag = scfci.get_emb_fci_e1_e2(emb, rdm1, rec, \
                                                           self.env_pot_for_fci)
            if isinstance(self.frag_group[m][0], int):
                val_tot += val_frag
                nelec += nelec_frag
            else: # degenerated fragments
                val_tot += val_frag * self.frag_group[m].__len__()
                nelec += nelec_frag * self.frag_group[m].__len__()
        log.info(self, 'DMET-FCI-in-HF of entire system energy = %.12g', val_tot)
        return val_tot, nelec


    def assemble_to_blockmat(self, mol, v_group):
        '''assemble matrix on impuity sites to the diagonal block'''
        nao = self.orth_coeff.shape[1]
        v_add = numpy.zeros((nao,nao))
        for m, atm_lst, bas_idx in self.all_frags:
            nimp = bas_idx.__len__()
            vfrag = v_group[m][:nimp,:nimp]
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

    def get_all_frag_fci_dm(self, mol):
        dm_group = []
        for m, emb in enumerate(self.embs):
            rdm1, rec = self.frag_fci_solver(mol, emb)
            dm_group.append(rdm1)
        return dm_group

    def set_local_fit_method(self, local_fit_approx):
        if local_fit_approx == FITTING_WITH_SCF:
            self.vfit_method = fit_pot_with_local_scf
        elif local_fit_approx == FITTING_1SHOT:
            self.vfit_method = fit_pot_1shot
        elif local_fit_approx == FITTING_FCI_POT:
            self.vfit_method = fit_vfci_fixed_mf_dm
        else: # FITTING_WITHOUT_SCF
            self.vfit_method = fit_pot_without_local_scf

    def dump_frag_prop_mat(self, mol, frag_mat_group):
        '''dump fragment potential or density matrix'''
        for m, atm_lst, bas_idx in self.uniq_frags:
            try:
                mol.fout.write('fragment %d, %s\n' % (m,str(atm_lst)))
                fmt = '    %10.5f' * frag_mat_group[m].shape[1] + '\n'
                for c in numpy.array(frag_mat_group[m]):
                    mol.fout.write(fmt % tuple(c))
            except:
                # frag_mat_group is 0-d array
                pass



def fit_without_local_scf_iter(mol, emb, embsys):
    dm_ref, rec = embsys.frag_fci_solver(mol, emb)
    log.debug(embsys, 'dm_ref = %s', dm_ref)
    nimp = emb.dim_of_impurity()
    # this fock matrix includes the pseudo potential of present fragment
    s = mol.intor_symmetric('cint1e_ovlp_sph')
    sc = reduce(numpy.dot, (emb.impbas_coeff.T.conj(), s, \
                            emb.entire_scf.mo_coeff))
    fock0 = numpy.dot(sc*emb.entire_scf.mo_energy, sc.T.conj())
    nocc = emb.nelectron/2

    # The damped potential does not minimize |dm_ref - dm(fock0+v)|^2,
    # but it may help convergence
    if 0: #old fitting function for debug
        dv = scfopt.find_emb_potential(mol, dm_ref, fock0, nocc, nimp)
    else:
        dv = hfdm.fit_solver(embsys, fock0, nocc, nimp, dm_ref*.5, \
                             embsys.v_fit_domain, embsys.dm_fit_domain, \
                             embsys.dm_fit_constraint)
    if embsys.fitpot_damp_fac > 0:
        dv *= embsys.fitpot_damp_fac
    return dv[:nimp,:nimp]

def fit_pot_without_local_scf(mol, embsys):
    '''fit potential without local SCF'''
    v_group = []
    for m, emb in enumerate(embsys.embs):
        log.debug(embsys, 'update local vfit for fragment %s', \
                  str(emb.imp_atoms))
        dv = fit_without_local_scf_iter(mol, emb, embsys)
        v_group.append(dv+embsys.v_global[emb.bas_on_frag][:,emb.bas_on_frag])
    return v_group

def fit_pot_1shot(mol, embsys, frag_id=0):
    v_group = []
    for emb in embsys.embs:
        nimp = emb.dim_of_impurity()
        v_group.append(numpy.zeros((nimp,nimp)))
    emb = embsys.embs[frag_id]
    v_group[frag_id] = \
            fit_without_local_scf_iter(mol, emb, embsys.fitpot_damp_fac)
    return v_group

def fit_pot_with_local_scf(mol, embsys):
    def fit_scfly(mol, emb):
        nimp = emb.dim_of_impurity()
        dm_ref, rec = embsys.frag_fci_solver(mol, emb)

        _vhf_env_bak = emb._vhf_env.copy()
        emb._vhf_env += emb.mat_orthao2impbas(embsys.v_global)
        # optimize initial guess to accelerate fragment-SCF convergence
        def _init_guess_method(mol):
            return 0, dm_ref
        emb.init_guess_method = _init_guess_method
#!        emb.setup_eri_on_impbas(mol)
        nocc = emb.nelectron / 2

        # use existed fitting potential as initial guess
        dv = numpy.zeros((nimp,nimp))
        icyc = 0
        dv_norm = 1
        if embsys.dm_fit_domain == IMP_AND_BATH:
            fitpot = scfopt.ImpPot4ImpBathDM()
        elif embsys.dm_fit_domain == IMP_BLK:
            fitpot = scfopt.ImpPot4ImpDM()
        elif embsys.dm_fit_domain == IMP_DIAG:
            fitpot = scfopt.ImpPot4ImpDiag()
        #elif embsys.dm_fit_domain == scfopt.FIT_DM_IMP_ONLY_DIAG_CONSTRAINT:
        #    fitpot = scfopt.ImpPot4ImpDM_DiagConstr()
        #elif embsys.dm_fit_domain == scfopt.FIT_DM_IMP_ONLY_NELE_CONSTRAINT:
        #    fitpot = scfopt.ImpPot4ImpDM_NeleConstr()
        while icyc < embsys.max_iter and dv_norm > self.conv_threshold:
            emb._vhf_env[:nimp,:nimp] += dv
            scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff_on_imp \
                    = emb.scf_cycle(mol, dump_chk=False)

            dm = emb.calc_den_mat(mol, mo_coeff_on_imp, mo_occ)
            if self.dm_fit_domain == IMP_AND_BATH:
                norm_ddm1 = numpy.linalg.norm(dm_ref - dm)
            elif self.dm_fit_domain == IMP_BLK:
                norm_ddm1 = numpy.linalg.norm((dm_ref - dm)[:nimp,:nimp])
            elif self.dm_fit_domain == IMP_DIAG:
                norm_ddm1 = numpy.linalg.norm((dm_ref-dm).diagonal()[:nimp])
            #elif self.dm_fit_domain == scfopt.FIT_DM_IMP_ONLY_DIAG_CONSTRAINT:
            #    norm_ddm1 = numpy.linalg.norm((dm_ref - dm)[:nimp,:nimp])
            #elif self.dm_fit_domain == scfopt.FIT_DM_IMP_ONLY_NELE_CONSTRAINT:
            #    norm_ddm1 = numpy.linalg.norm((dm_ref - dm)[:nimp,:nimp])
            else:
                norm_ddm1 = numpy.linalg.norm((dm_ref - dm)[:nimp])
            if icyc == 0:
                norm_ddm0 = norm_ddm1
                log.debug(embsys, 'before fitting, norm(dm_ref-dm) = %.12g', \
                          norm_ddm1)

            fock = numpy.dot(mo_coeff_on_imp*mo_energy, mo_coeff_on_imp)
            dv = fitpot.generate_pot(mol, dm_ref, fock, nocc, nimp)
            dv_norm = numpy.linalg.norm(dv)
            log.info(embsys, '  fragment-iter = %d norm(dm-dm_ref) = %.12g' \
                     'norm(v_{k+1}-v_k) = %.12g', icyc, norm_ddm1, dv_norm)
            icyc += 1
        log.debug(embsys, 'after fitting, norm(dm_ref-dm) = %.12g', norm_ddm1)

        v_acc = emb._vhf_env[:nimp,:nimp] - _vhf_env_bak[:nimp,:nimp]
        emb._vhf_env = _vhf_env_bak
#!        emb.release_eri()
        return v_acc

    v_group = []
    for m, emb in enumerate(embsys.embs):
        log.debug(embsys, 'update vfit SCFly for fragment %s', \
                  str(emb.imp_atoms))
        v_group.append(fit_scfly(mol, emb))
    return v_group


def fit_vfci_fixed_mf_dm(mol, embsys):
    assert('use numfitor to rewrite this function')
    return v_group




##################################################
# to minimize the DM difference, use mean-field analytic gradients
def dmet_sc_cycle(mol, embsys):
    import scf
    _diis = scf.diis.DIIS(mol)
    #_diis.diis_space = 6
    v_add = embsys.v_global
    e_tot = 0
    icyc = 0

    embsys.update_embsys_vglobal(mol, v_add)
    for icyc in range(embsys.max_iter):
        v_add_old = v_add
        e_tot_old = e_tot

        log.debug(embsys, '  HF energy = %.12g', embsys.entire_scf.hf_energy)
        v_group = embsys.vfit_method(mol, embsys)

        if embsys.with_hopping:
            v_add = embsys.assemble_to_fullmat(mol, v_group)
        else:
            v_add = embsys.assemble_to_blockmat(mol, v_group)
        #v_add = _diis.update(v_add)
        embsys.update_embsys_vglobal(mol, v_add)
        e_tot, nelec = embsys.assemble_frag_fci_energy(mol)

        if embsys.verbose >= param.VERBOSE_DEBUG:
            log.debug(embsys, 'fitting fragment potential')
            embsys.dump_frag_prop_mat(mol, v_group)

        dv_norm = numpy.linalg.norm(v_add_old - v_add)
        log.info(embsys, 'macro iter = %d, e_tot = %.12g, nelec = %g, dv_norm = %g', \
                 icyc, e_tot, nelec, dv_norm)
        de = abs(1-e_tot_old/e_tot)
        log.info(embsys, '                 delta_e = %g, (~ %g%%)', \
                 e_tot-e_tot_old, de * 100)

        log.debug(embsys, 'CPU time %.8g' % time.clock())

        if dv_norm < embsys.conv_threshold and de < embsys.conv_threshold*.1:
            break
        #import sys
        #if icyc > 1: sys.exit()
    return e_tot, v_group

def scdmet(mol, embsys, sav_v=None):
    log.info(embsys, '==== start DMET self-consistency ====')
    embsys.dump_options()

    if embsys.verbose >= param.VERBOSE_DEBUG:
        log.debug(embsys, '** DM of MF sys (on orthogonal AO) **')
        c = numpy.dot(numpy.linalg.inv(embsys.orth_coeff), \
                      embsys.entire_scf.mo_coeff)
        nocc = mol.nelectron / 2
        dm = numpy.dot(c[:,:nocc],c[:,:nocc].T) * 2
        fmt = '    %10.5f' * dm.shape[1] + '\n'
        for c in numpy.array(dm):
            mol.fout.write(fmt % tuple(c))

    e_tot, v_group = dmet_sc_cycle(mol, embsys)
    if embsys.with_hopping:
        v_add = embsys.assemble_to_fullmat(mol, v_group)
    else:
        v_add = embsys.assemble_to_blockmat(mol, v_group)

    log.info(embsys, '====================')
    if embsys.verbose >= param.VERBOSE_DEBUG:
        for m,emb in enumerate(embsys.embs):
            log.debug(embsys, 'vfit of frag %d = %s', m, v_group[m])

        log.debug(embsys, 'V_fitting in orth AO representation')
        fmt = '    %10.5f' * v_add.shape[1] + '\n'
        for c in numpy.array(v_add):
            mol.fout.write(fmt % tuple(c))
        log.debug(embsys, 'V_fitting in (non-orth) AO representation')
        v_add_ao = scfci.mat_orthao2ao(mol, v_add, embsys.orth_coeff)
        fmt = '    %10.5f' * v_add_ao.shape[1] + '\n'
        for c in numpy.array(v_add_ao):
            mol.fout.write(fmt % tuple(c))

        log.debug(embsys, '** mo_coeff of MF sys (on orthogonal AO) **')
        c = numpy.dot(numpy.linalg.inv(embsys.orth_coeff), \
                      embsys.entire_scf.mo_coeff)
        scf.hf.dump_orbital_coeff(mol, c)
        log.debug(embsys, '** mo_coeff of MF sys (on non-orthogonal AO) **')
        scf.hf.dump_orbital_coeff(mol, embsys.entire_scf.mo_coeff)

    e_tot, nelec = embsys.assemble_frag_fci_energy(mol)
    log.log(embsys, 'macro iter = X, e_tot = %.11g, +nuc = %.11g, nelec = %.8g', \
            e_tot, e_tot+mol.nuclear_repulsion(), nelec)
    if isinstance(sav_v, str):
        v_add_ao = scfci.mat_orthao2ao(mol, v_add, embsys.orth_coeff)
        with open(sav_v, 'w') as f:
            pickle.dump((v_add,v_add_ao), f)
    return e_tot

def dmet_1shot(mol, embsys, sav_v=None):
    log.info(embsys, '==== start DMET 1 shot ====')
    embsys.update_embsys_vglobal(mol, v_add)
    e_tot, nelec = embsys.assemble_frag_fci_energy(mol)
    log.log(embsys, 'e_tot = %.11g, +nuc = %.11g, nelec = %.8g', \
            e_tot, e_tot+mol.nuclear_repulsion(), nelec)
    return e_tot

# update the fragment frag_id in self-consistency
def scdmet_1shot(mol, embsys, frag_id=0, sav_v=None):
    embsys.vfit_method = lambda mol, embsys: \
            fit_pot_1shot(mol, embsys, frag_id)
    e_tot = scdmet(mol, embsys, sav_v)
    return e_tot

# fitting potential includes both impurity block and imp-bath block
def scdmet_hopping(mol, embsys, sav_v=None):
    dm_fit_domain_bak = embsys.dm_fit_domain
    embsys.dm_fit_domain = NO_BATH_BLK
    with_hopping_bak = embsys.with_hopping
    embsys.with_hopping = true
    e_tot = scdmet(mol, embsys, sav_v)
    embsys.dm_fit_domain = dm_fit_domain_bak
    embsys.with_hopping = with_hopping_bak
    return e_tot


# backwards fitting: MF DM fixed, add vfit on FCI to match MF DM
def scdmet_bakwards(mol, embsys, sav_v=None):
    embsys.vfit_method = fit_vfci_fixed_mf_dm
    e_tot = scdmet(mol, embsys, sav_v)
    return e_tot


def scdmet_num(mol, embsys, sav_v=None):
    import dmeft
    embsys.vfit_method = dmeft.numfit_without_local_scf
    e_tot = scdmet(mol, embsys, sav_v)
    return e_tot



# optmize potential of entire system with fixed local FCI density matrix. Then
# update local FCI density matrix.
#
#*NOTICE* the energy of embsys can be anything unless the number of electrons
# converges.  Changing the fitting potential of MF system cannot guarantee the
# convergence even with the global fitting scheme, e.g. in the h6-chain with
# the partition of [2,2,2].  It can be stuck to some fitting potentials, on
# which the MF orbitals as well as the FCI density matrix barely change, i.e.
# the density diff is roughly a constant even though the fitting potential
# becomes larger and larger.  In such case, the difference between the FCI and
# MF density matrices can never reach zero.
#
# If the fragment's electronegativities of impurity-MF and impurity-FCI
# conflict (e.g. impurity-FCI of every fragment tends to push electrons),
# minimizing the density diff might not be the best solution since the
# fragments' chemical potentials are changed simultaneously.  No electron
# transfer can occur and thereby electron number never converge to right
# value.
def scdmet_vglobal(mol, embsys, sav_v=None):
    import dmfet
    def split(v):
        v_group = []
        for m, atm_lst, bas_idx in embsys.uniq_frags:
            v_group.append(v[bas_idx][:,bas_idx])
        return v_group
    if embsys.global_fit_dm == NO_FIXED_DM:
        #def fit_fn(mol, embsys):
        #    v_add = scfci.numfit_global_pot_no_fixed_dm(mol, embsys)
        #    embsys.update_embsys_vglobal(mol, v_add)
        #    v_add = scfci.numfit_global_pot_no_fixed_dm(mol, embsys, \
        #                                                dgrad=True)
        #    return split(v_add)
        fn_fine_dm1 = dmfet.gen_global_fn_fine_dm(mol, embsys)
        walker = dmfet.gen_global_walker(mol, embsys)
        nocc = mol.nelectron / 2
        sc = numpy.dot(mol.intor_symmetric('cint1e_ovlp_sph'), embsys.orth_coeff)
        def ddm_fci_mf(v_global):
            eff_scf = embsys.update_embsys_vglobal(mol, v_global)
            mo = numpy.dot(sc.T.conj(), eff_scf.mo_coeff)
            dm_mf = numpy.dot(mo[:,:nocc], mo[:,:nocc].T)*2
            # need to set dm_mf[embsys.bas_off_frags] = 0
            dm_group = embsys.get_all_frag_fci_dm(mol)
            fci_dm = embsys.assemble_to_blockmat(mol, dm_group)
            return fn_fine_dm1(fci_dm-dm_mf)

        if 0:
            def fit_fn(mol, embsys):
                v_add, norm_ddm = \
                        scfci.numfitor(embsys, ddm_fci_mf, \
                                       walker, 0, embsys.v_global, .5e-4)
                return split(v_add)
        else:
            def mspan(x):
                v_inc = numpy.zeros_like(embsys.v_global)
                for k,(i,j) in enumerate(walker):
                    v_inc[i,j] = v_inc[j,i] = x[k]
                return v_inc
            def fit_fn(mol, embsys):
                x0 = numpy.array([embsys.v_global[i,j] for i,j in walker])
                x = scipy.optimize.leastsq(lambda x:ddm_fci_mf(mspan(x)), \
                                           x0, ftol=1e-6)[0]
                return split(mspan(x))
    elif embsys.global_fit_dm == FIXED_CI_DM:
        def fit_fn(mol, embsys):
            v_add = scfci.lobal_pot_with_fixed_fci_dm(mol, embsys)
            embsys.update_embsys_vglobal(mol, v_add)
            v_add = scfci.numfit_global_pot_with_fixed_fci_dm(mol, embsys, \
                                                              dgrad=True)
            return split(v_add)
    elif embsys.global_fit_dm == FIXED_MF_DM:
        sc = numpy.dot(mol.intor_symmetric('cint1e_ovlp_sph'), embsys.orth_coeff)
        nocc = mol.nelectron / 2
        mo = numpy.dot(sc.T.conj(), embsys.entire_scf.mo_coeff)
        dm_mf0 = numpy.dot(mo[:,:nocc], mo[:,:nocc].T) * 2
        def fit_fn(mol, embsys):
            v_add = scfci.numfit_global_pot_with_fixed_mf_dm(mol, embsys, \
                                                             dm_ref=dm_mf0)
            embsys.update_embsys_vglobal(mol, v_add)
            v_add = scfci.numfit_global_pot_with_fixed_mf_dm(mol, embsys, \
                                                             dm_ref=dm_mf0, \
                                                             dgrad=True)
            return split(v_add)
    elif embsys.global_fit_dm == NO_FIXED_DM_BACKWARDS:
        def fit_fn(mol, embsys):
            v_add = scfci.numfit_global_pot_backwards(mol, embsys)
            embsys.update_embsys_vglobal(mol, v_add)
            v_add = scfci.numfit_global_pot_backwards(mol, embsys, dgrad=True)
            return split(v_add)

    embsys.vfit_method = fit_fn
    embsys.with_hopping = False
    e_tot = scdmet(mol, embsys, sav_v)
    return e_tot


def scdmet_hybrid(mol, embsys, sav_v=None):
    _bak_max_iter = embsys.max_iter
    embsys.max_iter = 8
    e_tot, v_group = dmet_sc_cycle(mol, embsys)
    embsys.max_iter = _bak_max_iter
    return scdmet_vglobal(mol, embsys, sav_v)


def spin_square(mol, embsys):
    '''In the closed shell embedding, switch alpha and beta electrons won't
    change the total wave function (nor the wavefunction based on the
    impurity). The S^2 should be 0'''
    pass


if __name__ == '__main__':
    import scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = "out_dmet_sc"

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
    embsys = EmbSys(mol, rhf, frag_group)
    print scdmet_vglobal(mol, embsys)

