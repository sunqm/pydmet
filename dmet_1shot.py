#!/usr/bin/env python

import numpy
import lib
import lib.logger as log
import scf
import impsolver
import junk
import junk.hf

class OneImp(junk.hf.RHF):
    def __init__(self, mol, entire_scf, basidx):
        orth_ao = numpy.eye(entire_scf.mo_energy.size)
        junk.hf.RHF.__init__(self, entire_scf, orth_ao)
        self.bas_on_frag = basidx
        #self.solve_imp = \
        #        impsolver.use_local_solver(impsolver.fci, with_rdm1=True)
        self.solve_imp = \
                impsolver.use_local_solver(impsolver.cc, with_rdm1=True)

    def init_dmet_scf(self, mol=None):
        effscf = self.entire_scf
        mo_orth = effscf.mo_coeff[:,effscf.mo_occ>1e-15]
        self.imp_site, self.bath_orb, self.env_orb = \
                junk.hf.decompose_orbital(self, mo_orth, self.bas_on_frag)
        nao = mo_orth.shape[0]
        nimp = self.imp_site.shape[1]
        nemb = nimp + self.bath_orb.shape[1]
        self.impbas_coeff = numpy.zeros((nao, nemb))
        self.impbas_coeff[self.bas_on_frag,:nimp] = self.imp_site
        bas_off_frag = [i for i in range(nao) if i not in self.bas_on_frag]
        #print self.bath_orb.shape,self.impbas_coeff.shape, nao,nimp,nemb
        self.impbas_coeff[bas_off_frag,nimp:] = self.bath_orb

        self.nelectron = effscf._fcidump['NELEC'] - self.env_orb.shape[1] * 2
        log.info(self, 'number of electrons for impurity  = %d', \
                 self.nelectron)

        log.debug(self, 'init Hartree-Fock environment')
        dm_env = numpy.dot(self.env_orb, self.env_orb.T.conj()) * 2
        vhf_env_ao = effscf.get_eff_potential(self.mol, dm_env)
        hcore = effscf._fcidump['HCORE']
        self.energy_by_env = lib.trace_ab(dm_env, hcore) \
                           + lib.trace_ab(dm_env, vhf_env_ao) * .5
        self._vhf_env = self.mat_ao2impbas(vhf_env_ao)

def dmet_1shot(mol, emb):
    log.info(emb, '==== start DMET 1 shot ====')
    emb.init_dmet_scf()
    e, rdm1 = emb.solve_imp(mol, emb)
    hcore = emb.mat_ao2impbas(emb.entire_scf.get_hcore(mol))
    e_tot = e + emb.energy_by_env \
            + lib.trace_ab(rdm1, hcore) \
            + lib.trace_ab(rdm1, emb._vhf_env) * .5
    log.log(emb, 'e_tot = %.11g' % e_tot)
    return e_tot


if __name__ == '__main__':
    import gto
    import hf
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_hf'
    mol.build()

    mf = hf.RHF(mol, 'C_solid_2x2x2/test2/FCIDUMP')
    energy = mf.scf()
    print energy

    emb = OneImp(mol, mf, [0,1,2])
    print dmet_1shot(mol, emb)
