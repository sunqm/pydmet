#!/usr/bin/env python

import subprocess
import numpy
import scipy
import scipy.optimize
from pyscf import lib
from pyscf import gto
from pyscf import ao2mo
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
import impsolver
import vaspimp
import vasp
import vasphf
import dmet_sc
from dmet_sc import *

# Using VASP HF results

class EmbSysPeriod(dmet_sc.EmbSys):
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
        dmet_sc.EmbSys.__init__(self, mol, fake_hf, init_v=None)

        self.orth_coeff = numpy.eye(fake_hf.mo_coeff.shape[1])
        self.OneImp = vaspimp.OneImp
        self.solver = impsolver.Psi4CCSD()
        self.nbands = 1
        self.pwcut = 100
        self.cutri = 100

    def init_embsys(self, mol):
        # one fragment only
        self.all_frags = self.uniq_frags = \
                [[0, [], self.entire_scf._vaspdump['ORBIND']]]

        v0_group = [0] * len(self.uniq_frags)

        embs = self.init_embs(mol, self.entire_scf, self.orth_coeff)
        if self.orth_coeff is None:
            self.orth_coeff = embs[0].orth_coeff
        embs = self.update_embs_vfit_ci(mol, embs, [0])
        embs = self.update_embs_vfit_mf(mol, embs, [0])
        self.embs = embs
        return [0], [0]

    # update the embs in terms of the given entire_scf
    def update_embs(self, mol, embs, eff_scf, orth_coeff=None):
        if orth_coeff is None:
            orth_coeff = self.orth_coeff
        t0 = time.clock()
        sc = numpy.dot(eff_scf.get_ovlp(mol), eff_scf.mo_coeff)
        c_inv = numpy.dot(eff_scf.get_ovlp(mol), orth_coeff).T
        fock0 = numpy.dot(sc*eff_scf.mo_energy, sc.T.conj())
        nocc = int(eff_scf.mo_occ.sum()) / 2
        for ifrag, emb in enumerate(embs):
            emb.build_()

            emb._project_fock = emb.mat_ao2impbas(fock0)
            emb.mo_energy, emb.mo_coeff_on_imp = numpy.linalg.eigh(emb._project_fock)
            emb.mo_coeff = numpy.dot(emb.impbas_coeff, emb.mo_coeff_on_imp)
            emb.mo_occ = numpy.zeros_like(emb.mo_energy)
            emb.mo_occ[:emb.nelectron/2] = 2
            emb.hf_energy = 0
            nimp = emb.imp_site.shape[1]
            cimp = numpy.dot(emb.impbas_coeff[:,:nimp].T, sc[:,:nocc])
            emb._pure_hcore = emb.get_hcore() \
                    - emb._vhf_env # exclude HF[core] and correlation potential
#TODO store the correlation potential on bath
            emb._project_nelec_frag = numpy.linalg.norm(cimp)**2*2
            #emb.imp_scf()

        log.debug(self, 'CPU time for set up embsys.embs: %.8g sec', \
                  time.clock()-t0)

# add the CorrPot on bath for CI solver
        if self.env_pot_for_ci != NO_ENV_POT:
            for m, emb in enumerate(embs):
                nimp = len(emb.bas_on_frag)
                vmf = self.entire_scf._vaspdump['CORRPOT'].copy()
                vmf[:nimp,:nimp] = emb.vfit_ci[:nimp,:nimp]
                emb.vfit_ci = vmf
        return embs

    def update_embs_vfit_mf(self, mol, embs, v_mf_group):
        for m, emb in enumerate(embs):
            if v_mf_group[m] is not 0:
                if v_mf_group[m].shape[0] < emb._vhf_env.shape[0]:
                    nd = v_mf_group[m].shape[0]
                    emb.vfit_mf[:nd,:nd] = v_mf_group[m]
                else:
                    emb.vfit_mf = v_mf_group[m]
        return embs


    def run_hf_with_ext_pot_(self, vext_on_ao, follow_state=False):
        with open('CorrPot', 'w') as fcorrpot:
            basidx = self.embs[0].bas_on_frag
            vdump = vext_on_ao[basidx][:,basidx]
            for v in vdump.flatten():
                fcorrpot.write('%.16g\n' % v)
        retcode = subprocess.call('bash %s' % self.vasp_inpfile_pass2, shell=True)
        if not retcode:
            mf = vasphf.RHF(self.mol, self.clustdump, self.jdump, self.kdump,
                            self.fockdump)
            return mf
        else:
            raise OSError('Failed to execute %s' % self.vasp_inpfile_pass2)
#    def run_hf_with_ext_pot_(self, vext_on_ao, follow_state=False):
#        with open('CorrPot', 'w') as fcorrpot:
#            for v in vext_on_ao.flatten():
#                fcorrpot.write('%.16g\n' % v)
#        vasp_scf = vasp.Vasp()
#        log.debug(self, 'Call Vasp HF')
#        vasp_scf.run_hf(ENCUT=self.pwcut, NBANDS=self.nbands, EDIFF=1e-9)
#        vasp_scf.run_jkdump(ENCUT=self.pwcut, ENCUTGW=self.cutri, NBANDS=self.nbands)
#        vasp_scf.run_clustdump(ENCUT=self.pwcut, ENCUTGW=self.cutri, NBANDS=self.nbands)
#        self._vaspdump = vaspimp.read_clustdump('FCIDUMP.CLUST.GTO',
#                                              'JDUMP', 'KDUMP', 'FOCKDUMP')
#        mf = vaspimp.fake_entire_scf(self._vaspdump)
#        #print numpy.linalg.norm(mf.mo_coeff), numpy.linalg.svd(mf.mo_coeff)[1]
#        return mf

def run_vasp_scf(nbands, pwcut, cutri):
    vasp_scf = vasp.Vasp()
    vasp_scf.write_kpoints()
    vasp_scf.run_wavecar(ENCUT=pwcut, NBANDS=nbands)
    vasp_scf.run_hf(ENCUT=pwcut, NBANDS=nbands, ICHARGE=2, EDIFF=1e-6)
    vasp_scf.run_hf(ENCUT=pwcut, NBANDS=nbands, EDIFF=1e-9)
    vasp_scf.run_jkdump(ENCUT=pwcut, ENCUTGW=cutri, NBANDS=nbands)
    vasp_scf.run_clustdump(ENCUT=pwcut, ENCUTGW=cutri, NBANDS=nbands)


if __name__ == '__main__':
    run_vasp_scf(48, 400, 400)
    embsys = EmbSysPeriod('FCIDUMP.CLUST.GTO', 'JDUMP', 'KDUMP', 'FOCKDUMP')
    embsys.scdmet()

