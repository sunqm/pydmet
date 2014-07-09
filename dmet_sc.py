#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

import time
import pickle
import numpy
import copy

import pyscf.lib.logger as log
from pyscf import scf
import junk.hf
import junk.fitdm
import impsolver
import pyscf.lib.parameters as param


# options for fit_domain
IMP_AND_BATH  = junk.fitdm.IMP_AND_BATH  # 1
IMP_BLK       = junk.fitdm.IMP_BLK       # 2
IMP_BATH_DIAG = junk.fitdm.IMP_BATH_DIAG # 3
NO_BATH_BLK   = junk.fitdm.NO_BATH_BLK   # 4
DIAG_BLK      = junk.fitdm.DIAG_BLK      # 5
IMP_DIAG      = junk.fitdm.IMP_DIAG      # 6
NO_IMP_BLK    = junk.fitdm.NO_IMP_BLK    # 7
TRACE_IMP     = junk.fitdm.TRACE_IMP     # 8

# options for dm_fit_constraint
NO_CONSTRAINT = junk.fitdm.NO_CONSTRAINT # 0
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
NO_IMP_BLK    = 7
SCAL_ENV_POT = 11



class EmbSys(object):
    '''DMFET/DMET_SC
    v_fit_domain
    dm_fit_domain
    '''
    def __init__(self, mol, entire_scf, frag_group=[], init_v=None,
                 orth_coeff=None):
        self.verbose = mol.verbose
        self.fout = mol.fout
        self.mol = mol
        self.emb_verbose = param.VERBOSE_QUITE
        self.OneImp = junk.hf.RHF

        self.max_iter         = 40
        self.conv_threshold   = 1e-5
        self.global_fit_dm    = NO_FIXED_DM
        self.v_fit_domain     = IMP_BLK
        self.dm_fit_domain    = IMP_BLK
        self.dm_fit_constraint = NO_CONSTRAINT
# * use NO_ENV_POT to avoid double counting on the correlation on bath, since
#   the fitting potential has already counted the correlation effects.
        self.env_pot_for_fci  = NO_ENV_POT #NO_IMP_BLK
# whether select the occupations to maximize the overlap to the previous states
        self.hf_follow_state     = False
# if > 0, scale the fitting potential, it helps convergence when
# local_vfit_method is fit_pot_without_local_scf
        self.fitpot_damp_fac  = .5
# when vfit covers imp+bath, with_hopping=true will transform the
# imp-bath off-diagonal block to the global potential 
        self.with_hopping     = False
        self.rand_init        = False

        self.orth_coeff = orth_coeff
        if orth_coeff is None:
            #self.pre_orth_ao = junk.hf.pre_orth_ao_atm_scf
            self.pre_orth_ao = lambda mol: numpy.eye(mol.num_NR_function())
            #self.orth_ao_method = 'meta_lowdin'
            self.orth_ao_method = 'lowdin'

        self.frag_group = frag_group
        self.basidx_group = None
        self.all_frags = None
        self.uniq_frags = None
        self.entire_scf = entire_scf
        self.embs = []
        self.bas_off_frags = []
        #self.vfit_mf_method = gen_all_vfit_by(fit_with_local_scf)
        #self.vfit_mf_method = gen_all_vfit_by(fit_pot_1shot)
        #self.vfit_mf_method = gen_all_vfit_by(fit_fixed_mf_dm)
        self.vfit_mf_method = gen_all_vfit_by(fit_without_local_scf)
        #self.vfit_ci_method = gen_all_vfit_by(zero_potential)
        self.vfit_ci_method = gen_all_vfit_by(fit_chemical_potential)

        self.init_v = init_v

    def dump_options(self):
        log.info(self, '\n')
        log.info(self, '******** DMFET/DMET_SC Options *********')
        log.info(self, 'max_iter        = %g', self.max_iter       )
        log.info(self, 'conv_threshold  = %g', self.conv_threshold )
        log.info(self, 'global_fit_dm   = %g', self.global_fit_dm  )
        log.info(self, 'v_fit_domain    = %g', self.v_fit_domain   )
        log.info(self, 'dm_fit_domain   = %g', self.dm_fit_domain  )
        log.info(self, 'dm_fit_constraint = %g', self.dm_fit_constraint)
        log.info(self, 'env_pot_for_fci = %g', self.env_pot_for_fci)
        log.info(self, 'hf_follow_state = %g', self.hf_follow_state)
        log.info(self, 'fitpot_damp_fac = %g', self.fitpot_damp_fac)
        log.info(self, 'with_hopping    = %g', self.with_hopping   )
        log.info(self, 'rand_init       = %g', self.rand_init      )


    def init_embsys(self, mol):
        if self.orth_coeff is None:
            self.orth_coeff = junk.hf.orthogonalize_ao(mol, self.entire_scf, \
                                                       self.pre_orth_ao(mol), \
                                                       self.orth_ao_method)
        #self.basidx_group = map_frag_to_bas_idx(mol, self.frag_group)
        self.all_frags, self.uniq_frags = \
                self.gen_frag_looper(mol, self.frag_group, self.basidx_group)
        self.bas_off_frags = self._get_bas_off_frags()

        v0_group = [0] * len(self.uniq_frags)
        try:
            with open(self.init_v, 'r') as f:
                v_add, v_add_on_ao = pickle.load(f)
            self.entire_scf = run_hf_with_ext_pot(mol, self.entire_scf,
                                                  v_add_on_ao, \
                                                  self.hf_follow_state)
            v_group = []
            for m,_,bas_idx in self.uniq_frags:
                v_group.append(v_add[bas_idx][:,bas_idx])
        except (IOError, TypeError):
            #if self.rand_init:
            #   nao = self.orth_coeff.shape[1]
            #    v_global = numpy.zeros((nao,nao))
            #    for m, atm_lst, bas_idx in self.all_frags:
            #        nimp = bas_idx.__len__()
            #        v = numpy.random.randn(nimp*nimp).reshape(nimp,nimp)
            #        v = (v + v.T) * .1
            #        for i, j in enumerate(bas_idx):
            #            v_global[j,bas_idx] = v[i]
            #    self.entire_scf = run_hf_with_ext_pot(mol, self.entire_scf,
            #                                          v_global, \
            #                                          self.hf_follow_state)
            v_group = v0_group

        embs = self.init_embs(mol, self.entire_scf, self.orth_coeff)
        embs = self.update_embs_vfit_ci(mol, embs, v0_group)
        embs = self.update_embs_vfit_mf(mol, embs, v_group)
        self.embs = embs
        return v_group, v0_group

    def init_embs(self, mol, entire_scf, orth_coeff):
        embs = []
        for m, atm_lst, bas_idx in self.uniq_frags:
            emb = self.OneImp(entire_scf, orth_ao=orth_coeff)
            emb.occ_env_cutoff = 1e-14
            emb.imp_atoms = atm_lst
            emb.bas_on_frag = bas_idx
            emb.verbose = self.emb_verbose
            embs.append(emb)
        embs = self.update_embs(mol, embs, entire_scf)
        for emb in embs:
            emb.vfit_mf = numpy.zeros_like(emb._vhf_env)
            emb.vfit_ci = numpy.zeros_like(emb._vhf_env)
        return embs

    # update the embs in terms of the given entire_scf
    def update_embs(self, mol, embs, eff_scf):
        t0 = time.clock()
        sc = numpy.dot(eff_scf.get_ovlp(mol),eff_scf.mo_coeff)
        c_inv = numpy.dot(eff_scf.get_ovlp(mol),self.orth_coeff).T
        fock0 = numpy.dot(sc*eff_scf.mo_energy, sc.T.conj())
        hcore = eff_scf.get_hcore(mol)
        for ifrag, emb in enumerate(embs):
            mo_orth = numpy.dot(c_inv, eff_scf.mo_coeff[:,eff_scf.mo_occ>1e-15])
            emb.imp_site, emb.bath_orb, emb.env_orb = \
                    junk.hf.decompose_orbital(emb, mo_orth, emb.bas_on_frag)
            emb.impbas_coeff = emb.cons_impurity_basis()
            emb.nelectron = mol.nelectron - emb.env_orb.shape[1] * 2
            log.debug(emb, 'number of electrons for impurity %d = %d', \
                      ifrag, emb.nelectron)
# OPTIMIZE ME:
            emb._vhf_env = emb.init_vhf_env(mol, emb.env_orb)

# project entire-sys SCF results to embedding-sys SCF results
            f = emb.mat_ao2impbas(fock0)
            emb.mo_energy, emb.mo_coeff_on_imp = numpy.linalg.eigh(f)
            emb.mo_coeff = numpy.dot(emb.impbas_coeff, emb.mo_coeff_on_imp)
            emb.mo_occ = numpy.zeros_like(emb.mo_energy)
            emb.mo_occ[:emb.nelectron/2] = 2
            emb.hf_energy = 0
            emb._pure_hcore = emb.mat_ao2impbas(hcore)

        if eff_scf._eri is not None:
            t1 = time.clock()
            embs_eri_ao2mo(embs, eff_scf._eri)
            log.debug(self, 'CPU time for embsys eri AO->MO: %.8g sec', \
                      time.clock()-t1)
        else:
            for emb in self.embs:
                emb._eri = emb.eri_on_impbas(mol)

# local SCF in case the bath block of vfit_mf =/= 0?
#        for emb in embs:
#            emb.imp_scf()

        log.debug(self, 'CPU time for set up embsys.embs: %.8g sec', \
                  time.clock()-t0)
        return embs

    def update_embs_vfit_ci(self, mol, embs, v_ci_group):
        for m, emb in enumerate(embs):
            if v_ci_group[m] is not 0:
                if v_ci_group[m].shape[0] < emb._vhf_env.shape[0]:
                    nd = v_ci_group[m].shape[0]
                    emb.vfit_ci[:nd,:nd] = v_ci_group[m]
                else:
                    emb.vfit_ci = v_ci_group[m]
        return embs

    def update_embs_vfit_mf(self, mol, embs, v_mf_group):
        for m, emb in enumerate(embs):
            if v_mf_group[m] is not 0:
                if v_mf_group[m].shape[0] < emb._vhf_env.shape[0]:
                    nd = v_mf_group[m].shape[0]
                    emb.vfit_mf[:nd,:nd] = v_mf_group[m]
                else:
                    emb.vfit_mf = v_mf_group[m]

#?        # should we add mean-field potentail on the impurity solver?
#?        v_mf_group = [emb.vfit_mf for emb in embs]
#?        if self.with_hopping:
#?            vglobal = self.assemble_to_fullmat(mol, v_mf_group)
#?        else:
#?            vglobal = self.assemble_to_blockmat(mol, v_mf_group)
#?
#?        for m, emb in enumerate(embs):
#?            if self.env_pot_for_fci == NO_ENV_POT:
#?                pass
#?            elif self.env_pot_for_fci == NO_IMP_BLK:
#?                nimp = len(emb.bas_on_frag)
#?                vmf = emb.mat_ao2impbas(vglobal)
#?                vmf[:nimp,:nimp] = 0
#?                emb.vfit_ci += vmf
        return embs


    def gen_frag_looper(self, mol, frag_group, basidx_group):
        if frag_group:
            if basidx_group:
                log.warn(self, 'ignore basidx_group')
# map_frag_atom_id_to_bas_index
            lbl = mol.labels_of_spheric_GTO()
            atm_basidx = [[] for i in range(mol.natm)]
            for ib, s in enumerate(lbl):
                ia = s[0]
                atm_basidx[ia].append(ib)
            def _remove_bas_if_not_on_frag(atm_lst):
                bas_on_a = []
                for ia in atm_lst:
                    bas_on_a.extend(atm_basidx[ia])
                return bas_on_a

            basidx_group = []
            for m, frags in enumerate(frag_group):
                if isinstance(frags[0], int):
                    basidx_group.append(_remove_bas_if_not_on_frag(frags))
                else:
                    basidx_group.append([_remove_bas_if_not_on_frag(atm_lst) \
                                         for atm_lst in frags])
        else:
            frag_group = []
            for m, frag_basidx in enumerate(basidx_group):
                if isinstance(frag_basidx[0], int):
                    frag_group.append([])
                else:
                    frag_group.append([[]]*len(frag_basidx))

        all_frags = []
        uniq_frags = []
        for emb_id, frag_basidx in enumerate(basidx_group):
            if isinstance(frag_basidx[0], int):
                all_frags.append((emb_id, frag_group[emb_id], frag_basidx))
                uniq_frags.append((emb_id, frag_group[emb_id], frag_basidx))
            else:
                uniq_frags.append((emb_id, frag_group[emb_id][0], frag_basidx[0]))
                for k, basidx in enumerate(frag_basidx):
                    all_frags.append((emb_id, frag_group[emb_id][k], basidx))
        return all_frags, uniq_frags

    def meta_lowdin_orth(self, mol):
        pre_orth = junk.hf.pre_orth_ao_atm_scf(mol)
        self.orth_coeff = junk.hf.orthogonalize_ao(mol, self.entire_scf, \
                                                   pre_orth, 'meta_lowdin')
        for emb in self.embs:
            emb.orth_coeff = self.orth_coeff
        return self.orth_coeff

    def mat_orthao2ao(self, mat):
        '''matrix represented on orthogonal basis to the representation on
        non-orth AOs'''
        c_inv = numpy.dot(self.orth_coeff.T, self.entire_scf.get_ovlp())
        mat_on_ao = reduce(numpy.dot, (c_inv.T.conj(), mat, c_inv))
        return mat_on_ao

    def update_embsys(self, mol, v_mf_group):
        if self.with_hopping:
            v_add = self.assemble_to_fullmat(mol, v_mf_group)
        else:
            v_add = self.assemble_to_blockmat(mol, v_mf_group)
        v_add_ao = self.mat_orthao2ao(v_add)
        eff_scf = run_hf_with_ext_pot(mol, self.entire_scf, v_add_ao, \
                                      self.hf_follow_state)
        self.entire_scf = eff_scf
        for emb in self.embs:
            emb.entire_scf = eff_scf

        embs = self.update_embs(mol, self.embs, eff_scf)
        self.embs = self.update_embs_vfit_mf(mol, embs, v_mf_group)
        return self

#?    def update_embsys_vglobal(self, mol, v_add):
#?        v_add_ao = self.mat_orthao2ao(v_add)
#?        eff_scf = run_hf_with_ext_pot(mol, self.entire_scf, v_add_ao, \
#?                                      self.hf_follow_state)
#?        self.entire_scf = eff_scf
#?        for emb in self.embs:
#?            emb.entire_scf = eff_scf
#?
#?        v_group = []
#?        for m,_,bas_idx in self.uniq_frags:
#?            v_group.append(v_add[bas_idx][:,bas_idx])
#?        embs = self.update_embs(mol, self.embs, eff_scf)
#?        self.embs = self.update_embs_vfit_mf(mol, embs, v_group)
#?        return self

    def _get_bas_off_frags(self):
        nao = self.orth_coeff.shape[1]
        baslst = numpy.ones(nao, dtype=bool)
        for m, atm_lst, bas_idx in self.all_frags:
            baslst[bas_idx] = False
        return [i for i in range(nao) if baslst[i]]

    # if fragments do not cover the whole system, the rests are treated at
    # mean-field level.  Asymmetrical energy expression is used for the rests
    def off_frags_energy(self, mol, dm_mf):
        if len(self.bas_off_frags) == 0:
            return 0, 0

        v_group = [emb.vfit_mf for emb in self.embs]
        if self.with_hopping:
            v_global = self.assemble_to_fullmat(mol, v_group)
        else:
            v_global = self.assemble_to_blockmat(mol, v_group)

        h1e = reduce(numpy.dot, (self.orth_coeff.T, \
                                 scf.entire_scf.get_hcore(mol), \
                                 self.orth_coeff)) + v_global
        h1e = h1e[self.bas_off_frags]
        dm_ao = reduce(numpy.dot, (self.orth_coeff,dm_mf,self.orth_coeff.T))
        vhf = self.entire_scf.get_eff_potential(mol, dm_ao)
        vhf = reduce(numpy.dot, (self.orth_coeff.T,vhf,self.orth_coeff))
        vhf = vhf[self.bas_off_frags]
        dm_frag = dm_mf[self.bas_off_frags]
        e = numpy.dot(h1e.flatten(), dm_frag.flatten()) \
                + numpy.dot(vhf.flatten(), dm_frag.flatten())*.5

        nelec = dm_frag.trace()
        return e, nelec

    def frag_fci_solver(self, mol, emb, v=0):
        solver = impsolver.use_local_solver(impsolver.fci)
        return solver(mol, emb, v)

    def assemble_frag_fci_energy(self, mol, dm_ref=0):
        if len(self.bas_off_frags) == 0:
            e_tot = 0
            nelec = 0
        else:
            if dm_ref is 0:
                eff_scf = self.entire_scf
                sc = numpy.dot(eff_scf.get_ovlp(mol), self.orth_coeff)
                mo = numpy.dot(sc.T,eff_scf.mo_coeff)
                dm_ref = eff_scf.calc_den_mat(mo, eff_scf.mo_occ)
            e_tot, nelec = self.off_frags_energy(mol, dm_ref)

        last_frag = -1
        for m, _, _ in self.all_frags:
            if m != last_frag:
                emb = self.embs[m]
                cires = self.frag_fci_solver(mol, emb, emb.vfit_ci)
                e_frag, nelec_frag = \
                        extract_partial_trace(emb, cires, self.env_pot_for_fci)
            e_tot += e_frag
            nelec += nelec_frag
            last_frag = m
        log.info(self, 'DMET-FCI-in-HF of entire system, e_tot = %.9g, nelec_tot = %.9g', \
                  e_tot, nelec)
        return e_tot, nelec


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

    def dump_frag_prop_mat(self, mol, frag_mat_group):
        '''dump fragment potential or density matrix'''
        for m, atm_lst, bas_idx in self.uniq_frags:
            mol.fout.write('fragment %d, %s\n' % (m,str(atm_lst)))
            try:
                fmt = '    %10.5f' * frag_mat_group[m].shape[1] + '\n'
                for c in numpy.array(frag_mat_group[m]):
                    mol.fout.write(fmt % tuple(c))
            except:
                mol.fout.write(str(frag_mat_group[m]))

    # for convergence criteria
    def diff_vfit(self, v_group, v_group_old):
        dv_mf_group = map(lambda x,y: numpy.linalg.norm(x-y), \
                          v_group[0], v_group_old[0])
        dv_ci_group = map(lambda x,y: numpy.linalg.norm(x-y), \
                          v_group[1], v_group_old[1])
        return numpy.linalg.norm(dv_mf_group+dv_ci_group)


    def scdmet(self, mol, sav_v=None):
        log.info(self, '==== start DMET self-consistency ====')
        self.dump_options()

        #if self.verbose >= param.VERBOSE_DEBUG:
        #    log.debug(self, '** DM of MF sys (on orthogonal AO) **')
        #    c = numpy.dot(numpy.linalg.inv(self.orth_coeff), \
        #                  self.entire_scf.mo_coeff)
        #    nocc = mol.nelectron / 2
        #    dm = numpy.dot(c[:,:nocc],c[:,:nocc].T) * 2
        #    fmt = '    %10.5f' * dm.shape[1] + '\n'
        #    for c in numpy.array(dm):
        #        mol.fout.write(fmt % tuple(c))

        e_tot, v_mf_group, v_ci_group = dmet_sc_cycle(mol, self)

        log.info(self, '====================')
        if self.verbose >= param.VERBOSE_DEBUG:
            for m,emb in enumerate(self.embs):
                log.debug(self, 'vfit_mf of frag %d = %s', m, v_mf_group[m])
                log.debug(self, 'vfit_ci of frag %d = %s', m, v_ci_group[m])

            if self.with_hopping:
                v_add = self.assemble_to_fullmat(mol, v_mf_group)
            else:
                v_add = self.assemble_to_blockmat(mol, v_mf_group)
            log.debug(self, 'mean-field V_fitting in orth AO representation')
            fmt = '    %10.5f' * v_add.shape[1] + '\n'
            for c in numpy.array(v_add):
                mol.fout.write(fmt % tuple(c))

            log.debug(self, '** mo_coeff of MF sys (on orthogonal AO) **')
            c = numpy.dot(numpy.linalg.inv(self.orth_coeff), \
                          self.entire_scf.mo_coeff)
            scf.hf.dump_orbital_coeff(mol, c)
            log.debug(self, '** mo_coeff of MF sys (on non-orthogonal AO) **')
            scf.hf.dump_orbital_coeff(mol, self.entire_scf.mo_coeff)

        e_tot, nelec = self.assemble_frag_fci_energy(mol)
        log.log(self, 'macro iter = X, e_tot = %.11g, +nuc = %.11g, nelec = %.8g', \
                e_tot, e_tot+mol.nuclear_repulsion(), nelec)
        if isinstance(sav_v, str):
            if self.with_hopping:
                v_add = self.assemble_to_fullmat(mol, v_mf_group)
            else:
                v_add = self.assemble_to_blockmat(mol, v_mf_group)
            v_add_ao = self.mat_orthao2ao(v_add)
            with open(sav_v, 'w') as f:
                pickle.dump((v_add,v_add_ao), f)
        return e_tot
#?
#?# update the fragment corresponding to frag_id in self-consistency
#?    def one_shot(self, mol, frag_id=0, sav_v=None):
#?        self.vfit_method = lambda mol, embsys: \
#?                fit_pot_1shot(mol, embsys, frag_id)
#?        e_tot = self.scdmet(mol, sav_v)
#?        return e_tot

    # fitting potential includes both impurity block and imp-bath block
    def scdmet_hopping(mol, sav_v=None):
        dm_fit_domain_bak = self.dm_fit_domain
        self.dm_fit_domain = NO_BATH_BLK
        with_hopping_bak = self.with_hopping
        self.with_hopping = True
        e_tot = scdmet(mol, self, sav_v)
        self.dm_fit_domain = dm_fit_domain_bak
        self.with_hopping = with_hopping_bak
        return e_tot


    # backwards fitting: MF DM fixed, add vfit on FCI to match MF DM
    def scdmet_bakwards(mol, sav_v=None):
        self.vfit_method = fit_vfci_fixed_mf_dm
        e_tot = scdmet(mol, self, sav_v)
        return e_tot


###########################################################
# fitting methods
###########################################################
##ABORT to minimize the DM difference, use mean-field analytic gradients
def fit_without_local_scf(mol, emb, embsys):
    cires = embsys.frag_fci_solver(mol, emb, emb.vfit_ci)
    dm_ref = cires['rdm1']
    log.debug(embsys, 'dm_ref = %s', dm_ref)
    nimp = emb.dim_of_impurity()
    # this fock matrix includes the previous fitting potential
    fock0 = numpy.dot(emb.mo_coeff_on_imp*emb.mo_energy, \
                      emb.mo_coeff_on_imp.T)
    nocc = emb.nelectron/2

    # The damped potential does not minimize |dm_ref - dm(fock0+v)|^2,
    # but it may help convergence
    if 0: #old fitting function for debug
        dv = scfopt.find_emb_potential(mol, dm_ref, fock0, nocc, nimp)
    else:
        dv = junk.fitdm.fit_solver(embsys, fock0, nocc, nimp, dm_ref*.5, \
                                   embsys.v_fit_domain, embsys.dm_fit_domain, \
                                   embsys.dm_fit_constraint)
    if embsys.fitpot_damp_fac > 0:
        dv *= embsys.fitpot_damp_fac
    return dv + emb.vfit_mf

#?def fit_pot_1shot(mol, embsys, frag_id=0):
#?    v_group = []
#?    for emb in embsys.embs:
#?        nimp = emb.dim_of_impurity()
#?        v_group.append(numpy.zeros((nimp,nimp)))
#?    emb = embsys.embs[frag_id]
#?    v_group[frag_id] = \
#?            fit_without_local_scf_iter(mol, emb, embsys)
#?
#?    if embsys.verbose >= param.VERBOSE_DEBUG:
#?        log.debug(embsys, 'fitting potential for fragment %d\n' % frag_id)
#?        fmt = '    %10.5f' * v_group[frag_id].shape[1] + '\n'
#?        for c in numpy.array(v_group[frag_id]):
#?            mol.fout.write(fmt % tuple(c))
#?    return v_group

def fit_with_local_scf(mol, embsys):
    # impurity SCF during local fitting
    assert(0)
    return dv + emb.vfit_mf


def fit_fixed_mf_dm(mol, embsys):
    # use numfitor to rewrite this function
    assert(0)
    return dv + emb.vfit_mf


def fit_chemical_potential(mol, emb, embsys):
    import scipy
    nimp = emb.dim_of_impurity()
    # this fock matrix includes the pseudo potential of present fragment
    mo = emb.mo_coeff_on_imp
    nocc = emb.nelectron/2
    nelec_frag = numpy.linalg.norm(mo[:nimp,:nocc])**2 * 2

# change chemical potential to get correct number of electrons
    def nelec_diff(v):
        vmat = numpy.eye(nimp) * v
        cires = embsys.frag_fci_solver(mol, emb, vmat)
        dm = cires['rdm1']
        return abs(nelec_frag - dm[:nimp].trace())
    chem_pot0 = emb.vfit_ci[0,0]
    chem_pot = scipy.optimize.leastsq(nelec_diff, chem_pot0, ftol=1e-5)[0]
    nemb = emb.impbas_coeff.shape[1]
    vmat = numpy.zeros((nemb,nemb))
    for i in range(nimp):
        vmat[i,i] = chem_pot
    return vmat


def zero_potential(mol, emb, embsys):
    nemb = emb.impbas_coeff.shape[1]
    return numpy.zeros((nemb,nemb))


def gen_all_vfit_by(local_fit_method):
    '''fit HF DM with chemical potential'''
    def fitloop(mol, embsys):
        v_group = []
        for m, emb in enumerate(embsys.embs):
            log.debug(embsys, '%s for fragment %d', local_fit_method.func_name, m)
            dv = local_fit_method(mol, emb, embsys)
            v_group.append(dv)

        if embsys.verbose >= param.VERBOSE_DEBUG:
            log.debug(embsys, 'fitting potential =')
            embsys.dump_frag_prop_mat(mol, v_group)
        return v_group
    return fitloop




##################################################
def dmet_sc_cycle(mol, embsys):
    #import scf
    #_diis = scf.diis.DIIS(mol)
    #_diis.diis_space = 6

    v_group = embsys.init_embsys(mol)
    e_tot = 0
    icyc = 0

    for icyc in range(embsys.max_iter):
        v_group_old = v_group
        e_tot_old = e_tot
        v_ci_group = embsys.vfit_ci_method(mol, embsys)
        embsys.update_embs_vfit_ci(mol, embsys.embs, v_ci_group)

        # to guarantee correct number of electrons, calculate embedded energy
        # before calling update_embsys
        e_tot, nelec = embsys.assemble_frag_fci_energy(mol)

        #log.debug(embsys, '  HF energy = %.12g', embsys.entire_scf.hf_energy)
        v_mf_group = embsys.vfit_mf_method(mol, embsys)

        v_group = (v_mf_group, v_ci_group)

        dv = embsys.diff_vfit(v_group, v_group_old)
        log.info(embsys, 'macro iter = %d, e_tot = %.12g, nelec = %g, dv = %g', \
                 icyc, e_tot, nelec, dv)
        de = abs(1-e_tot_old/e_tot)
        log.info(embsys, '                 delta_e = %g, (~ %g%%)', \
                 e_tot-e_tot_old, de * 100)

        log.debug(embsys, 'CPU time %.8g' % time.clock())

        if dv < embsys.conv_threshold and de < embsys.conv_threshold*.1:
            break
        #import sys
        #if icyc > 1: sys.exit()

        #v_group = _diis.update(v_group)
        embsys.update_embsys(mol, v_mf_group)

    return e_tot, v_mf_group, v_ci_group

def run_hf_with_ext_pot(mol, entire_scf, vext_on_ao, follow_state=False):
    def _dup_entire_scf(mol, entire_scf):
        #eff_scf = entire_scf.__class__(mol)
        eff_scf = copy.copy(entire_scf)
        eff_scf.verbose = 0#entire_scf.verbose
        eff_scf.scf_threshold = 1e-9#entire_scf.scf_threshold
        eff_scf.diis_space = 8#entire_scf.diis_space
        eff_scf.scf_conv = False
        return eff_scf
    eff_scf = _dup_entire_scf(mol, entire_scf)

    # FIXME: ground state strongly depends on initial guess.
    # when previous SCF does not converge, the initial guess will be incorrect
    # and leads to incorrect MF ground state.
    # In this case, follow old scf as initial guess.
    dm = entire_scf.calc_den_mat(entire_scf.mo_coeff, entire_scf.mo_occ)
    def _init_guess_method(mol):
        return entire_scf.hf_energy, dm
    eff_scf.init_guess_method = _init_guess_method

    def _get_hcore(mol):
        h = entire_scf.get_hcore(mol)
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

    # must release the modified get_hcore to get pure hcore
    del(eff_scf.get_hcore)
    del(eff_scf.init_guess_method)
    return eff_scf

def extract_partial_trace(emb, cires, with_env_pot=False):
    rdm1 = cires['rdm1']
    nimp = emb.dim_of_impurity()
    #log.debug(emb, 'total energy of (frag + bath) %.12g', cires['etot'])

    if emb._pure_hcore is not None:
        h1e = emb._pure_hcore
    else:
        h1e = emb.mat_ao2impbas(emb.entire_scf.get_hcore(emb.mol))

    e1_frag = numpy.dot(rdm1[:nimp,:nimp].flatten(),h1e[:nimp,:nimp].flatten())
    e1_bath = numpy.dot(rdm1[:nimp,nimp:].flatten(),h1e[:nimp,nimp:].flatten())
    if with_env_pot and emb.vfit_ci is not 0:
        e1_vfit = numpy.dot(rdm1[:nimp].flatten(), \
                            emb.vfit_ci[:nimp].flatten())
    else:
        e1_vfit = 0
    e1 = e1_frag + e1_bath + e1_vfit
    log.debug(emb, 'e1 = %.12g = fragment + bath + fitenv = %.12g + %.12g + %.12g', \
              e1, e1_frag, e1_bath, e1_vfit)

    e2env_hf = numpy.dot(rdm1[:nimp].flatten(), \
                         emb._vhf_env[:nimp].flatten()) * .5
    e2 = cires['e2frag']
    nelec_frag = rdm1[:nimp].trace()
    log.debug(emb, 'fragment e1 = %.12g, e2env_hf = %.12g, FCI pTraceSys = %.12g', \
              e1, e2env_hf, e2)
    log.debug(emb, 'fragment e2env_hf = %.12g, FCI pTraceSys = %.12g, nelec = %.12g', \
              e2env_hf, e2, nelec_frag)
    e_frag = e1 + e2env_hf + e2
    return e_frag, nelec_frag


def embs_eri_ao2mo(embs, eri_ao):
    import junk.dmet_misc
    c = []
    offsets = [0]
    off = 0
    for emb in embs:
        c.append(emb.impbas_coeff)
        off += emb.impbas_coeff.shape[1]
        offsets.append(off)
    c = numpy.array(numpy.hstack(c), order='F')
    offsets = numpy.array(offsets, dtype=numpy.int32)
    v = junk.dmet_misc.embs_eri_ao2mo_o3(eri_ao, c, offsets)
    for n,emb in enumerate(embs):
        emb._eri = v[n]
    return embs





if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = "out_dmet_sc"

    mol.atom.extend([
        ['C' , ( 0. , 0.  , 0.)],
        ['H' , ( 0.7, 0.7 , 0.7)],
        ['H' , ( 0.7,-0.7 ,-0.7)],
        ['H' , (-0.7, 0.7 ,-0.7)],
        ['H' , (-0.7,-0.7 , 0.7)] ])
    mol.basis = {'C': 'sto_3g',
                 'H': 'sto_3g',}
    mol.build()

    rhf = scf.RHF(mol)
    print "E=", rhf.scf()

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
    embsys.max_iter = 10
    print embsys.scdmet(mol)

    b1 = 1.0
    nat = 10
    mol.output = 'h%s_sz' % nat
    mol.atom = []
    r = b1/2 / numpy.sin(numpy.pi/nat)
    for i in range(nat):
        theta = i * (2*numpy.pi/nat)
        mol.atom.append((1, (r*numpy.cos(theta),
                             r*numpy.sin(theta), 0)))

    mol.basis = {'H': 'sto-3g',}
    mol.build()
    mf = scf.RHF(mol)
    print mf.scf()

    embsys = EmbSys(mol, mf)
    embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
    embsys.max_iter = 10
    print embsys.scdmet(mol)
