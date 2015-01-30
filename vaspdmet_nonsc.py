#!/usr/bin/env python

import numpy
import scipy
import scipy.optimize
from pyscf import gto
from pyscf import lib
from pyscf import ao2mo
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
import impsolver
import vaspimp
import vasphf
import dmet_nonsc
from dmet_nonsc import *

# Using VASP HF results

class EmbSysPeriod(dmet_nonsc.EmbSys):
    def __init__(self, clustdump, jdump, kdump, fockdump, init_v=None):
        self.clustdump= clustdump
        self.jdump    = jdump
        self.kdump    = kdump
        self.fockdump = fockdump
        self.vasp_inpfile_pass2 = ''

        mol = gto.Mole()
        mol.verbose = 5
        mol.build(False, False)
        fake_hf = vasphf.RHF(mol, clustdump, jdump, kdump, fockdump)
        dmet_nonsc.EmbSys.__init__(self, mol, fake_hf, init_v=None)

        self.orth_coeff = numpy.eye(fake_hf.mo_coeff.shape[1])
        self.OneImp = vaspimp.OneImp
        self.solver = impsolver.Psi4CCSD()
        self.nbands = 1
        self.pwcut = 100
        self.cutri = 100
        self.verbose = 5
        self.emb_verbose = 5
        self.pot_on = 'IMP'  # or 'BATH', or 'IMP,BATH'

    def init_embsys(self, mol):
        eff_scf = self.entire_scf
        #embs = self.init_embs(mol, self.entire_scf, self.orth_coeff)
        emb = self.OneImp(eff_scf)
        emb.occ_env_cutoff = 1e-14
        emb.orth_coeff = self.orth_coeff
        emb.verbose = self.emb_verbose
        emb.imp_scf()
        embs = [emb]
        sc = numpy.dot(eff_scf.get_ovlp(mol), eff_scf.mo_coeff)
        fock0 = numpy.dot(sc*eff_scf.mo_energy, sc.T.conj())
        emb._project_fock = emb.mat_ao2impbas(fock0)
        nimp = len(emb.bas_on_frag)
        emb._pure_hcore = emb.get_hcore() \
                - emb._vhf_env # exclude HF[core] and correlation potential
        cimp = numpy.dot(emb.impbas_coeff[:,:nimp].T,
                         eff_scf.mo_coeff[:,eff_scf.mo_occ>0])
        emb._project_nelec_frag = numpy.linalg.norm(cimp)**2*2
        log.debug(emb, 'nelec of imp from lattice HF %.8g',
                  emb._project_nelec_frag)
        log.debug(emb, 'nelec of imp from embedding HF %.8g',
                  numpy.linalg.norm(emb.mo_coeff_on_imp[:nimp,:emb.nelectron/2])**2*2)
#X        embs = self.update_embs(mol, embs, self.entire_scf, self.orth_coeff)
        emb.vfit_mf = numpy.zeros_like(emb._pure_hcore)
        emb.vfit_ci = numpy.zeros_like(emb._pure_hcore)
        embs = self.update_embs_vfit_ci(mol, embs, [0])
#X        embs = self.update_embs_vfit_mf(mol, embs, [0])
        self.embs = embs
        return [0], [0]

#    def one_shot(self):
#        log.info(self, '==== one-shot ====')
#        mol = self.mol
#        self.init_embsys(mol)
#        emb = self.embs[0]
#        emb.verbose = self.verbose
#        #emb.imp_scf()
#        nimp = len(emb.bas_on_frag)
#
#        log.info(self, '')
#        log.info(self, '===== CI/CC before Fitting =====')
#        cires = self.frag_fci_solver(mol, emb)
#        e_tot = cires['etot'] + emb.energy_by_env
#        vhf = emb._project_fock - emb._pure_hcore
#        e1 = numpy.dot(cires['rdm1'][:nimp].reshape(-1),
#                       (emb._pure_hcore + .5 * vhf)[:nimp].reshape(-1))
#        dm1 = emb.make_rdm1(emb.mo_coeff_on_imp, emb.mo_occ)
#        vhf = emb.get_veff(emb.mol, dm1)
#        e2 = cires['e2frag'] - .5*numpy.dot(cires['rdm1'][:nimp].reshape(-1),
#                                            vhf[:nimp].reshape(-1))
#        e_frag = e1 + e2
#        n_frag = cires['rdm1'][:nimp].trace()
#        log.info(self, 'e_tot = %.11g, e_frag = %.11g, nelec_frag = %.11g', \
#                 e_tot, e1+e2, n_frag)
#
#        log.info(self, '')
#        log.info(self, '===== Fitting chemical potential =====')
#        if self.pot_on.upper() == 'IMP':
#            vfit_ci = fit_imp_fix_nelec(mol, emb, self)
#            #vfit_ci = fit_imp_float_nelec(mol, emb, self)
#        elif self.pot_on.upper() == 'BATH':
#            vfit_ci = fit_bath_fix_nelec(mol, emb, self)
#            #vfit_ci = fit_bath_float_nelec(mol, emb, self)
#        else:
#            vfit_ci = fit_mix_float_nelec(mol, emb, self)
#
#        cires = self.frag_fci_solver(mol, emb, vfit_ci)
#        e_tot = cires['etot'] + emb.energy_by_env
##        e1_frag = numpy.dot(cires['rdm1'][:nimp].flatten(), \
##                            emb._pure_hcore[:nimp].flatten())
##        envhf_frag = numpy.dot(cires['rdm1'][:nimp].flatten(), \
##                               emb._vhf_env[:nimp].flatten())
##        e_frag = e1_frag + envhf_frag * .5 + cires['e2frag']
##
#        vhf = emb._project_fock - emb._pure_hcore
#        e1 = numpy.dot(cires['rdm1'][:nimp].reshape(-1),
#                       (emb._pure_hcore + .5 * vhf)[:nimp].reshape(-1))
#        dm1 = emb.make_rdm1(emb.mo_coeff_on_imp, emb.mo_occ)
#        vhf = emb.get_veff(emb.mol, dm1)
#        e2 = cires['e2frag'] - .5*numpy.dot(cires['rdm1'][:nimp].reshape(-1),
#                                            vhf[:nimp].reshape(-1))
#        e_frag = e1 + e2
#        n_frag = cires['rdm1'][:nimp].trace()
#
#        log.info(self, '====================')
#        #if self.verbose >= param.VERBOSE_DEBUG:
#        #    log.debug(self, 'vfit_ci = %s', vfit_ci)
#        #    log.debug(self, 'impurity dm = %s', cires['rdm1'])
#        log.log(self, 'dmet_nonsc.one_shot: e_tot = %.11g, (+nuc=%.11g)', \
#                e_tot, e_tot+mol.energy_nuc())
#        log.log(self, 'e_frag = %.11g, nelec_frag = %.11g', \
#                e_frag, n_frag)
#        return e_frag

    def scdmet(self, init_v=None):
        return self.one_shot(mol)

    def fullsys(self, init_v=None):
        return self.one_shot(mol)

