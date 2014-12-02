#!/usr/bin/env python

import time
import pickle
import numpy
import scipy.linalg
import copy

from pyscf import scf
import pyscf.lib.parameters as param
import pyscf.lib.logger as log
from pyscf import lo
from pyscf import tools
import pyscf.tools.dump_mat
import dmet_hf
import fitdm
import impsolver


# options for fit_domain
IMP_AND_BATH  = fitdm.IMP_AND_BATH  # 1
IMP_BLK       = fitdm.IMP_BLK       # 2
IMP_BATH_DIAG = fitdm.IMP_BATH_DIAG # 3
NO_BATH_BLK   = fitdm.NO_BATH_BLK   # 4
DIAG_BLK      = fitdm.DIAG_BLK      # 5
IMP_DIAG      = fitdm.IMP_DIAG      # 6
NO_IMP_BLK    = fitdm.NO_IMP_BLK    # 7
TRACE_IMP     = fitdm.TRACE_IMP     # 8

# options for dm_fit_constraint
NO_CONSTRAINT = fitdm.NO_CONSTRAINT # 0
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

# optinos for env_pot_for_ci
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
    def __init__(self, mol, entire_scf, frag_group=[], init_v=None,
                 orth_coeff=None):
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.mol = mol
        self.emb_verbose = log.QUIET
        self.OneImp = dmet_hf.RHF

        self.max_iter         = 40
        self.conv_threshold   = 1e-5
        self.global_fit_dm    = NO_FIXED_DM
        self.v_fit_domain     = IMP_BLK
        self.dm_fit_domain    = IMP_BLK
        self.dm_fit_constraint = NO_CONSTRAINT
# * use NO_ENV_POT to avoid double counting on the correlation on bath, since
#   the fitting potential has already counted the correlation effects.
        self.env_pot_for_ci   = NO_ENV_POT #NO_IMP_BLK
# whether select the occupations to maximize the overlap to the previous states
        self.hf_follow_state     = False
# if > 0, scale the fitting potential, it helps convergence when
# local_vfit_method is fit_pot_without_local_scf
        self.fitpot_damp_fac  = .6
# when vfit covers imp+bath, with_hopping=true will transform the
# imp-bath off-diagonal block to the global potential 
        self.with_hopping     = False
        self.rand_init        = False

        self.orth_coeff = orth_coeff
        if orth_coeff is None:
            #self.pre_orth_ao = lo.iao.pre_atm_scf_ao(mol)
            self.pre_orth_ao = numpy.eye(mol.nao_nr())
            self.orth_ao_method = 'lowdin'
            #self.orth_ao_method = 'meta_lowdin'

        self.frag_group = frag_group
        self.basidx_group = None
        self.all_frags = None
        self.uniq_frags = None
        self.entire_scf = entire_scf
        self.embs = []
        #self.vfit_mf_method = gen_all_vfit_by(fit_with_local_scf)
        #self.vfit_mf_method = gen_all_vfit_by(fit_pot_1shot)
        #self.vfit_mf_method = gen_all_vfit_by(fit_fixed_mf_dm)
        self.vfit_mf_method = gen_all_vfit_by(fit_without_local_scf)
        #self.vfit_ci_method = gen_all_vfit_by(zero_potential)
        self.vfit_ci_method = gen_all_vfit_by(fit_chemical_potential)
        self.solver = impsolver.FCI()

        self.init_v = init_v

    def dump_flags(self):
        log.info(self, '\n')
        log.info(self, '******** DMFET/DMET_SC Options *********')
        log.info(self, 'max_iter        = %g', self.max_iter       )
        log.info(self, 'conv_threshold  = %g', self.conv_threshold )
        log.info(self, 'global_fit_dm   = %g', self.global_fit_dm  )
        log.info(self, 'v_fit_domain    = %g', self.v_fit_domain   )
        log.info(self, 'dm_fit_domain   = %g', self.dm_fit_domain  )
        log.info(self, 'dm_fit_constraint = %g', self.dm_fit_constraint)
        log.info(self, 'env_pot_for_ci  = %g', self.env_pot_for_ci )
        log.info(self, 'hf_follow_state = %g', self.hf_follow_state)
        log.info(self, 'fitpot_damp_fac = %g', self.fitpot_damp_fac)
        log.info(self, 'with_hopping    = %g', self.with_hopping   )
        log.info(self, 'rand_init       = %g', self.rand_init      )


    def init_embsys(self, mol):
        return self.build_(mol)
    def build_(self, mol):
        #self.basidx_group = map_frag_to_bas_idx(mol, self.frag_group)
        self.all_frags, self.uniq_frags = \
                self.gen_frag_looper(mol, self.frag_group, self.basidx_group)

        v0_group = [0] * len(self.uniq_frags)
        try:
            with open(self.init_v, 'r') as f:
                v_add, v_add_on_ao = pickle.load(f)
            self.entire_scf = self.run_hf_with_ext_pot_(v_add_on_ao, \
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
            #    self.entire_scf = self.run_hf_with_ext_pot_(v_global, \
            #                                                self.hf_follow_state)
            v_group = v0_group

        embs = self.init_embs(mol, self.entire_scf, self.orth_coeff)
        if self.orth_coeff is None:
            self.orth_coeff = embs[0].orth_coeff
        embs = self.update_embs_vfit_ci(mol, embs, v0_group)
        embs = self.update_embs_vfit_mf(mol, embs, v_group)
        self.embs = embs
        return v_group, v0_group

    def init_embs(self, mol, entire_scf, orth_coeff):
        embs = []
        for m, atm_lst, bas_idx in self.uniq_frags:
            emb = self.OneImp(entire_scf)
            emb.occ_env_cutoff = 1e-14
            emb.imp_atoms = atm_lst
            emb.imp_basidx = bas_idx
            emb.bas_on_frag = bas_idx
            emb.pre_orth_ao = self.pre_orth_ao
            emb.orth_ao_method = self.orth_ao_method
            emb.verbose = self.emb_verbose
            embs.append(emb)

        if self.orth_coeff is None:
            orth_coeff = embs[0].get_orth_ao(mol)
        else:
            orth_coeff = self.orth_coeff
        for emb in embs:
            emb.orth_coeff = orth_coeff

        embs = self.update_embs(mol, embs, entire_scf, orth_coeff)

        for emb in embs:
            emb.vfit_mf = numpy.zeros_like(emb._vhf_env)
            emb.vfit_ci = numpy.zeros_like(emb._vhf_env)
        return embs

    # update the embs in terms of the given entire_scf
    def update_embs(self, mol, embs, eff_scf, orth_coeff=None):

# local SCF will be carried out in self.update_embs_vfit_ci
#ABORT        for emb in embs:
#ABORT            emb.imp_scf()
#ABORT        hcore = self.entire_scf.get_hcore(mol)
#ABORT        for emb in embs:
#ABORT            emb._pure_hcore = emb.mat_ao2impbas(hcore)
#ABORT        return embs

# * If entire_scf is converged, the embedding HF results can be projected from
# entire_scf as follows.
# * vaspimp.OneImpNI cannot use the enitre_scf results, since the 2e parts are
# screened.
        if orth_coeff is None:
            orth_coeff = self.orth_coeff
        t0 = time.clock()
        sc = numpy.dot(eff_scf.get_ovlp(mol), eff_scf.mo_coeff)
        c_inv = numpy.dot(eff_scf.get_ovlp(mol), orth_coeff).T
        fock0 = numpy.dot(sc*eff_scf.mo_energy, sc.T.conj())
        hcore = eff_scf.get_hcore(mol)
        nocc = int(eff_scf.mo_occ.sum()) / 2
        for ifrag, emb in enumerate(embs):
            mo_orth = numpy.dot(c_inv, eff_scf.mo_coeff[:,eff_scf.mo_occ>1e-15])
            emb.imp_site, emb.bath_orb, emb.env_orb = \
                    dmet_hf.decompose_orbital(emb, mo_orth, emb.bas_on_frag)
            emb.impbas_coeff = emb.cons_impurity_basis()
            emb.nelectron = mol.nelectron - emb.env_orb.shape[1] * 2
            log.debug(emb, 'nelec of emb %d = %d', ifrag, emb.nelectron)
#TODO: optimize ._eri and ._vhf_env, they can be generated together
            emb._eri = emb.eri_on_impbas(mol)
            emb.energy_by_env, emb._vhf_env = emb.init_vhf_env(emb.env_orb)

# project entire-sys SCF results to embedding-sys SCF results
# This is the results of embedded-HF which are projected from entire HF.
# Generally, the local fitting potential is not consistent to the global
# potential (which is not linearly transformed from the global potential), the
# embedded-HF results can be different from the projected HF results.  So the
# local impurity solver CANNOT directly use the projected HF orbitals and
# energies, local-SCF is required.
# * the relevant embedding-SCF lies in self.update_embs_vfit_ci
            emb._project_fock = emb.mat_ao2impbas(fock0)
            emb.mo_energy, emb.mo_coeff_on_imp = scipy.linalg.eigh(emb._project_fock)
            emb.mo_coeff = numpy.dot(emb.impbas_coeff, emb.mo_coeff_on_imp)
            emb.mo_occ = numpy.zeros_like(emb.mo_energy)
            emb.mo_occ[:emb.nelectron/2] = 2
            emb.hf_energy = 0
            nimp = emb.imp_site.shape[1]
            cimp = numpy.dot(emb.impbas_coeff[:,:nimp].T, sc[:,:nocc])
            emb._pure_hcore = emb.mat_ao2impbas(hcore)
            emb._project_nelec_frag = numpy.linalg.norm(cimp)**2*2

        log.debug(self, 'CPU time for set up embsys.embs: %.8g sec', \
                  time.clock()-t0)
        return embs

    def update_embs_vfit_ci(self, mol, embs, v_ci_group):
        def embscf_(emb):
            h1e = emb._pure_hcore + emb._vhf_env + emb.vfit_ci
            nemb = emb._vhf_env.shape[0]
            emb.get_hcore = lambda mol=None: h1e
            emb.get_ovlp = lambda mol=None: numpy.eye(nemb)
            emb.scf_conv, emb.hf_energy, emb.mo_energy, emb.mo_occ, \
                    emb.mo_coeff_on_imp \
                    = emb.scf_cycle(emb.mol, emb.conv_threshold, dump_chk=False)
            #ABORTemb.mo_coeff = numpy.dot(emb.impbas_coeff, emb.mo_coeff_on_imp)
            del(emb.get_hcore)

        for m, emb in enumerate(embs):
            if v_ci_group[m] is not 0:
                if v_ci_group[m].shape[0] < emb._vhf_env.shape[0]:
                    nd = v_ci_group[m].shape[0]
                    emb.vfit_ci[:nd,:nd] = v_ci_group[m]
                else:
                    emb.vfit_ci = v_ci_group[m]

# Do embedding SCF for impurity solver since the embedded HF with vfit_ci
# cannot be directly projected from the entire SCF results.
# emb.mo_coeff_on_imp will be used in solver
                #emb.scf_conv, emb.hf_energy, emb.mo_energy, emb.mo_occ, \
                #        emb.mo_coeff_on_imp \
                #        = simple_hf(emb._pure_hcore+emb._vhf_env+emb.vfit_ci,
                #                    emb._eri, emb.mo_coeff_on_imp, emb.nelectron)
                embscf_(emb)
        return embs

    # NOTE!: self.embs have not SCF against vfit_mf
    def update_embs_vfit_mf(self, mol, embs, v_mf_group):
        for m, emb in enumerate(embs):
            if v_mf_group[m] is not 0:
                if v_mf_group[m].shape[0] < emb._vhf_env.shape[0]:
                    nd = v_mf_group[m].shape[0]
                    emb.vfit_mf[:nd,:nd] = v_mf_group[m]
                else:
                    emb.vfit_mf = v_mf_group[m]

        # should we add mean-field potentail on the impurity solver?
        if self.env_pot_for_ci != NO_ENV_POT:
            if self.with_hopping:
                v_add = self.assemble_to_fullmat(v_mf_group)
            else:
                v_add = self.assemble_to_blockmat(v_mf_group)
            v_add_ao = self.mat_orthao2ao(v_add)
            for m, emb in enumerate(embs):
                if self.env_pot_for_ci == NO_IMP_BLK:
                    nimp = len(emb.bas_on_frag)
                    vmf = emb.mat_ao2impbas(v_add_ao)
                    vmf[:nimp,:nimp] = emb.vfit_ci[:nimp,:nimp]
                    emb.vfit_ci = vmf
        return embs


    def gen_frag_looper(self, mol, frag_group, basidx_group):
        if frag_group:
            if basidx_group:
                log.warn(self, 'ignore basidx_group')
# map_frag_atom_id_to_bas_index
            lbl = mol.spheric_labels()
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
        self.orth_coeff = lo.orth.orth_ao(mol, self.pre_orth_ao, 'meta_lowdin')
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
            v_add = self.assemble_to_fullmat(v_mf_group)
        else:
            v_add = self.assemble_to_blockmat(v_mf_group)
        v_add_ao = self.mat_orthao2ao(v_add)
        eff_scf = self.run_hf_with_ext_pot_(v_add_ao, self.hf_follow_state)
        self.entire_scf = eff_scf
        for emb in self.embs:
            emb.entire_scf = eff_scf

        embs = self.update_embs(mol, self.embs, eff_scf)
        self.embs = self.update_embs_vfit_mf(mol, embs, v_mf_group)
        return self

#?    def update_embsys_vglobal(self, mol, v_add):
#?        v_add_ao = self.mat_orthao2ao(v_add)
#?        eff_scf = self.run_hf_with_ext_pot_(v_add_ao, self.hf_follow_state)
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

    def run_hf_with_ext_pot_(self, vext_on_ao, follow_state=False):
        run_hf_with_ext_pot_(self.mol, self.entire_scf, vext_on_ao, follow_state)


    def assemble_frag_energy(self, mol):
        e_tot = 0
        nelec = 0

        last_frag = -1
        for m, _, _ in self.all_frags:
            if m != last_frag:
                emb = self.embs[m]
                nimp = len(emb.bas_on_frag)
                _, e2frag, dm1 = \
                        self.solver.run(emb, emb._eri, emb.vfit_ci,
                                        with_1pdm=True, with_e2frag=nimp)
                e_frag, nelec_frag = \
                        self.extract_frag_energy(emb, dm1, e2frag)

                hfdm = emb.make_rdm1(emb.mo_coeff_on_imp, emb.mo_occ)
                vhf = emb.get_veff(mol, hfdm)
                nelechf = hfdm[:nimp].trace()
                ehfinhf = (hfdm[:nimp]*(emb._pure_hcore)[:nimp]).sum() \
                        + (hfdm[:nimp]*(vhf+emb._vhf_env)[:nimp]).sum() * .5

                log.debug(self, 'fragment %3d, HF-in-HF, frag energy = %.12g, nelec = %.9g',
                          m, ehfinhf, nelechf)
                log.debug(self, '             FCI-in-HF, frag energy = %.12g, nelec = %.9g', \
                          e_frag, nelec_frag)
            e_tot += e_frag
            nelec += nelec_frag
            last_frag = m
        log.info(self, 'DMET-FCI-in-HF of entire system, e_tot = %.9g, nelec_tot = %.9g', \
                  e_tot, nelec)
        return e_tot, nelec

    def extract_frag_energy(self, emb, dm1, e2frag):
        nimp = len(emb.bas_on_frag)

        if emb._pure_hcore is not None:
            h1e = emb._pure_hcore
        else:
            h1e = emb.mat_ao2impbas(emb.entire_scf.get_hcore(emb.mol))

        e1_frag = numpy.dot(dm1[:nimp,:nimp].flatten(),h1e[:nimp,:nimp].flatten())
        e1_bath = numpy.dot(dm1[:nimp,nimp:].flatten(),h1e[:nimp,nimp:].flatten())
        if self.env_pot_for_ci and emb.vfit_ci is not 0:
            e1_vfit = numpy.dot(dm1[:nimp].flatten(), emb.vfit_ci[:nimp].flatten())
        else:
            e1_vfit = 0
        e1 = e1_frag + e1_bath + e1_vfit
        log.debug(emb, 'e1 = %.12g = fragment + bath + fitenv = %.12g + %.12g + %.12g', \
                  e1, e1_frag, e1_bath, e1_vfit)

        e2env_hf = numpy.dot(dm1[:nimp].flatten(), \
                             emb._vhf_env[:nimp].flatten()) * .5
        nelec_frag = dm1[:nimp].trace()
        e_frag = e1 + e2env_hf + e2frag
        log.debug(emb, 'fragment e1 = %.12g, e2env_hf = %.12g, FCI pTraceSys = %.12g, sum = %.12g', \
                  e1, e2env_hf, e2frag, e_frag)
        log.debug(emb, 'fragment e2env_hf = %.12g, FCI pTraceSys = %.12g, nelec = %.12g', \
                  e2env_hf, e2frag, nelec_frag)
        return e_frag, nelec_frag


    def assemble_to_blockmat(self, v_group):
        '''assemble matrix on impuity sites to the diagonal block'''
        nao = self.orth_coeff.shape[1]
        v_add = numpy.zeros((nao,nao))
        for m, atm_lst, bas_idx in self.all_frags:
            if isinstance(v_group[m], numpy.ndarray):
                nimp = bas_idx.__len__()
                vfrag = v_group[m][:nimp,:nimp]
                for i, j in enumerate(bas_idx):
                    v_add[j,bas_idx] = vfrag[i,:]
        return v_add

    def assemble_to_fullmat(self, dm_group):
        '''assemble matrix of the embsys to the full matrix'''
        nao = self.orth_coeff.shape[1]
        dm_big = numpy.zeros((nao,nao))
        for m, atm_lst, bas_idx in self.all_frags:
            if isinstance(dm_group[m], numpy.ndarray):
                emb = self.embs[m]
                nimp = len(emb.bas_on_frag)
                dm_ab = numpy.dot(dm_group[m][:nimp,nimp:], emb.bath_orb.T)
                dm_ab[:,emb.bas_on_frag] = dm_group[m][:nimp,:nimp]
                dm_big[emb.bas_on_frag] = dm_ab
        return dm_big

    def dump_frag_prop_mat(self, mol, frag_mat_group):
        '''dump fragment potential or density matrix'''
        for m, atm_lst, bas_idx in self.uniq_frags:
            mol.stdout.write('fragment %d, %s\n' % (m,str(atm_lst)))
            try:
                fmt = '    %10.5f' * frag_mat_group[m].shape[1] + '\n'
                for c in numpy.array(frag_mat_group[m]):
                    mol.stdout.write(fmt % tuple(c))
            except:
                mol.stdout.write(str(frag_mat_group[m]))

    # for convergence criteria
    def diff_vfit(self, v_group, v_group_old):
        dv_mf_group = map(lambda x,y: numpy.linalg.norm(x-y), \
                          v_group[0], v_group_old[0])
        dv_ci_group = map(lambda x,y: numpy.linalg.norm(x-y), \
                          v_group[1], v_group_old[1])
        return numpy.linalg.norm(dv_mf_group+dv_ci_group)


    def scdmet(self, sav_v=None):
        log.info(self, '==== start DMET self-consistency ====')
        self.dump_flags()
        mol = self.mol

        #if self.verbose >= param.VERBOSE_DEBUG:
        #    log.debug(self, '** DM of MF sys (on orthogonal AO) **')
        #    c = numpy.dot(numpy.linalg.inv(self.orth_coeff), \
        #                  self.entire_scf.mo_coeff)
        #    nocc = mol.nelectron / 2
        #    dm = numpy.dot(c[:,:nocc],c[:,:nocc].T) * 2
        #    fmt = '    %10.5f' * dm.shape[1] + '\n'
        #    for c in numpy.array(dm):
        #        mol.stdout.write(fmt % tuple(c))

        e_tot, v_mf_group, v_ci_group = dmet_sc_cycle(mol, self)

        log.info(self, '====================')
        if self.verbose >= param.VERBOSE_DEBUG:
            for m,emb in enumerate(self.embs):
                log.debug(self, 'vfit_mf of frag %d = %s', m, v_mf_group[m])
                log.debug(self, 'vfit_ci of frag %d = %s', m, v_ci_group[m])

            if self.with_hopping:
                v_add = self.assemble_to_fullmat(v_mf_group)
            else:
                v_add = self.assemble_to_blockmat(v_mf_group)
            log.debug(self, 'mean-field V_fitting in orth AO representation')
            fmt = '    %10.5f' * v_add.shape[1] + '\n'
            for c in numpy.array(v_add):
                mol.stdout.write(fmt % tuple(c))

        if self.verbose >= log.DEBUG2:
            log.debug(self, '** mo_coeff of MF sys (on orthogonal AO) **')
            c = numpy.dot(numpy.linalg.inv(self.orth_coeff), \
                          self.entire_scf.mo_coeff)
            label = ['%d%3s %s%-4s' % x for x in mol.spheric_labels()]
            tools.dump_mat.dump_rec(self.stdout, c, label, start=1)
            log.debug(self, '** mo_coeff of MF sys (on non-orthogonal AO) **')
            tools.dump_mat.dump_rec(self.stdout, self.entire_scf.mo_coeff, label, start=1)

        e_tot, nelec = self.assemble_frag_energy(mol)
        log.log(self, 'macro iter = X, e_tot = %.11g, +nuc = %.11g, nelec = %.8g', \
                e_tot, e_tot+mol.nuclear_repulsion(), nelec)
        if isinstance(sav_v, str):
            if self.with_hopping:
                v_add = self.assemble_to_fullmat(v_mf_group)
            else:
                v_add = self.assemble_to_blockmat(v_mf_group)
            v_add_ao = self.mat_orthao2ao(v_add)
            with open(sav_v, 'w') as f:
                pickle.dump((v_add,v_add_ao), f)
        return e_tot

#?# update the fragment corresponding to frag_id in self-consistency
#?    def one_shot(self, mol, frag_id=0, sav_v=None):
#?        self.vfit_mf_method = lambda mol, embsys: \
#?                fit_pot_1shot(mol, embsys, frag_id)
#?        e_tot = self.scdmet(sav_v)
#?        return e_tot
#?
#?    # fitting potential includes both impurity block and imp-bath block
#?    def scdmet_hopping(sav_v=None):
#?        dm_fit_domain_bak = self.dm_fit_domain
#?        self.dm_fit_domain = NO_BATH_BLK
#?        with_hopping_bak = self.with_hopping
#?        self.with_hopping = True
#?        e_tot = scdmet(self, sav_v)
#?        self.dm_fit_domain = dm_fit_domain_bak
#?        self.with_hopping = with_hopping_bak
#?        return e_tot
#?
#?
#?    # backwards fitting: MF DM fixed, add vfit on FCI to match MF DM
#?    def scdmet_bakwards(sav_v=None):
#?        self.vfit_ci_method = fit_fixed_mf_dm
#?        e_tot = scdmet(self, sav_v)
#?        return e_tot


###########################################################
# fitting methods
###########################################################
##ABORT to minimize the DM difference, use mean-field analytic gradients
def fit_without_local_scf(mol, emb, embsys):
    _, _, dm_ref = embsys.solver.run(emb, emb._eri, emb.vfit_ci, True, False)
    log.debug(embsys, 'dm_ref = %s', dm_ref)
    nimp = len(emb.bas_on_frag)
    # this fock matrix includes the previous fitting potential
    fock0 = emb._project_fock
    nocc = emb.nelectron/2

    # The damped potential does not minimize |dm_ref - dm(fock0+v)|^2,
    # but it may help convergence
    if 0: #old fitting function for debug
        dv = scfopt.find_emb_potential(mol, dm_ref, fock0, nocc, nimp)
    else:
        dv = fitdm.fit_solver(embsys, fock0, nocc, nimp, dm_ref*.5, \
                              embsys.v_fit_domain, embsys.dm_fit_domain, \
                              embsys.dm_fit_constraint)
    if embsys.fitpot_damp_fac > 0:
        dv *= embsys.fitpot_damp_fac
    if dv.size > emb.vfit_mf.size:
        nv = emb.vfit_mf.shape[0]
        dv[:nv,:nv] += emb.vfit_mf
        return dv
    else:
        nv = dv.shape[0]
        dv1 = emb.vfit_mf.copy()
        dv1[:nv,:nv] += dv
        return dv1

#?def fit_pot_1shot(mol, embsys, frag_id=0):
#?    v_group = []
#?    for emb in embsys.embs:
#?    nimp = len(emb.bas_on_frag)
#?        v_group.append(numpy.zeros((nimp,nimp)))
#?    emb = embsys.embs[frag_id]
#?    v_group[frag_id] = \
#?            fit_without_local_scf_iter(mol, emb, embsys)
#?
#?    if embsys.verbose >= param.VERBOSE_DEBUG:
#?        log.debug(embsys, 'fitting potential for fragment %d\n' % frag_id)
#?        fmt = '    %10.5f' * v_group[frag_id].shape[1] + '\n'
#?        for c in numpy.array(v_group[frag_id]):
#?            mol.stdout.write(fmt % tuple(c))
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
# correlation potential of embedded-HF is not added to correlated-solver
    import scipy.optimize
    nimp = len(emb.bas_on_frag)
    nelec_frag = emb._project_nelec_frag

# change chemical potential to get correct number of electrons
    def nelec_diff(v):
        vmat = emb.vfit_ci.copy()
        vmat[:nimp,:nimp] = numpy.eye(nimp) * v
        _, _, dm = embsys.solver.run(emb, emb._eri, vmat, True, False)
        #print 'ddm ',nelec_frag,dm[:nimp].trace(), nelec_frag - dm[:nimp].trace()
        return nelec_frag - dm[:nimp].trace()
    chem_pot0 = emb.vfit_ci[0,0]
#OPTIMIZE ME, approximate chemical potential
    sol = scipy.optimize.root(nelec_diff, chem_pot0, tol=1e-3, \
                              method='lm', options={'ftol':1e-3, 'maxiter':12})
    nemb = emb.impbas_coeff.shape[1]
    vmat = emb.vfit_ci.copy()
    for i in range(nimp):
        vmat[i,i] = sol.x
    log.debug(embsys, 'scipy.optimize summary %s', sol)
    log.debug(embsys, 'chem potential = %.11g, nelec error = %.11g', \
              sol.x, sol.fun)
    log.debug(embsys, '        ncall = %d, scipy.optimize success: %s', \
              sol.nfev, sol.success)
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

    v_mf_group,_ = embsys.init_embsys(mol)
    v_ci_group = embsys.vfit_ci_method(mol, embsys)
    embsys.update_embs_vfit_ci(mol, embsys.embs, v_ci_group)
    # to guarantee correct number of electrons, calculate embedded energy
    # before calling update_embsys
    e_tot, nelec = embsys.assemble_frag_energy(mol)
    v_group = (v_mf_group, v_ci_group)
    log.info(embsys, 'macro iter = 0, e_tot = %.12g, nelec = %g', \
             e_tot, nelec)

    for icyc in range(embsys.max_iter):
        v_group_old = v_group
        e_tot_old = e_tot

        #log.debug(embsys, '  HF energy = %.12g', embsys.entire_scf.hf_energy)
        v_mf_group = embsys.vfit_mf_method(mol, embsys)
        embsys.update_embsys(mol, v_mf_group)

        v_ci_group = embsys.vfit_ci_method(mol, embsys)
        embsys.update_embs_vfit_ci(mol, embsys.embs, v_ci_group)

        # to guarantee correct number of electrons, calculate embedded energy
        # before calling update_embsys
        e_tot, nelec = embsys.assemble_frag_energy(mol)
        v_group = (v_mf_group, v_ci_group)

        dv = embsys.diff_vfit(v_group, v_group_old)
        log.info(embsys, 'macro iter = %d, e_tot = %.12g, nelec = %g, dv = %g', \
                 icyc+1, e_tot, nelec, dv)
        de = abs(1-e_tot_old/e_tot)
        log.info(embsys, '                 delta_e = %g, (~ %g%%)', \
                 e_tot-e_tot_old, de * 100)

        log.debug(embsys, 'CPU time %.8g' % time.clock())

        if dv < embsys.conv_threshold and de < embsys.conv_threshold*.1:
            break
        #import sys
        #if icyc > 1: sys.exit()

        #v_group = _diis.update(v_group)

    return e_tot, v_mf_group, v_ci_group

def run_hf_with_ext_pot_(mol, entire_scf, vext_on_ao, follow_state=False):
    def _dup_entire_scf(mol, entire_scf):
        #eff_scf = entire_scf.__class__(mol)
        eff_scf = copy.copy(entire_scf)
        eff_scf.verbose = 0#entire_scf.verbose
        eff_scf.conv_threshold = 1e-9#entire_scf.conv_threshold
        eff_scf.diis_space = 8#entire_scf.diis_space
        eff_scf.scf_conv = False
        return eff_scf
    eff_scf = _dup_entire_scf(mol, entire_scf)

    # FIXME: ground state strongly depends on initial guess.
    # when previous SCF does not converge, the initial guess will be incorrect
    # and leads to incorrect MF ground state.
    # In this case, follow old scf as initial guess.
    dm = entire_scf.make_rdm1(entire_scf.mo_coeff, entire_scf.mo_occ)
    def _make_init_guess_method(mol):
        return entire_scf.hf_energy, dm
    eff_scf.make_init_guess = _init_guess

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
        eff_scf.set_occ = _occ_follow_state

    log.debug(mol, 'SCF for entire molecule with fitting potential')
    eff_scf.scf_conv, eff_scf.hf_energy, eff_scf.mo_energy, \
            eff_scf.mo_occ, eff_scf.mo_coeff \
            = eff_scf.scf_cycle(mol, eff_scf.conv_threshold, dump_chk=False)

    # must release the modified get_hcore to get pure hcore
    del(eff_scf.get_hcore)
    del(eff_scf.make_init_guess)
    return eff_scf




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

#    frag_group = [(0,), ((1,), (2,), (3,), (4,),) ] # -51.876399725
#    #frag_group = [(0,), (1,2,3,4,)]
#    #frag_group = [(0,1,), (2,3,4,)]
#    #frag_group = [(0,1,2,), (3,4,)]
#    #frag_group = [(0,1,2,3,), (4,)]
#    embsys = EmbSys(mol, rhf, frag_group)
#    embsys.max_iter = 10
#    print embsys.scdmet()

    b1 = 1.0
    nat = 10
    mol.atom = []
    r = b1/2 / numpy.sin(numpy.pi/nat)
    for i in range(nat):
        theta = i * (2*numpy.pi/nat)
        mol.atom.append((1, (r*numpy.cos(theta),
                             r*numpy.sin(theta), 0)))

    mol.basis = {'H': 'sto-3g',}
    mol.build(False, False)
    mf = scf.RHF(mol)
    print mf.scf()

    embsys = EmbSys(mol, mf)
    embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
    embsys.max_iter = 10
    print embsys.scdmet() # -18.0179909364

