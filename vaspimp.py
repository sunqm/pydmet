#!/usr/bin/env python

import numpy
from pyscf import lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
from pyscf.lib import _vhf
from pyscf import ao2mo
import junk.hf

# AO basis of entire system are orthogonal sets
class OneImp(junk.hf.RHF):
    def __init__(self, entire_scf, basidx=[]):
        orth_ao = numpy.eye(entire_scf.mo_energy.size)
        junk.hf.RHF.__init__(self, entire_scf, orth_ao)
        self.bas_on_frag = basidx

    def init_dmet_scf(self, mol=None):
        effscf = self.entire_scf
        mo_orth = effscf.mo_coeff[:,effscf.mo_occ>1e-15]
        self.imp_site, self.bath_orb, self.env_orb = \
                junk.hf.decompose_orbital(self, mo_orth, self.bas_on_frag)
        self.impbas_coeff = self.cons_impurity_basis()

        self.nelectron = int(effscf.mo_occ.sum()) - self.env_orb.shape[1] * 2
        log.info(self, 'number of electrons for impurity  = %d', \
                 self.nelectron)
        self._vhf_env = self.init_vhf_env(mol, self.env_orb)

class OneImpNI(OneImp):
    '''Non-interacting DMET'''
    def __init__(self, entire_scf, basidx=[]):
        OneImp.__init__(self, entire_scf, basidx)

    def get_hcore(self, mol=None):
        nimp = len(self.bas_on_frag)
        effscf = self.entire_scf
        sc = reduce(numpy.dot, (self.impbas_coeff.T, \
                                self.entire_scf.get_ovlp(), effscf.mo_coeff))
        fock = numpy.dot(sc*effscf.mo_energy, sc.T)
        dmimp = effscf.calc_den_mat(mo_coeff=sc)
        dm = numpy.zeros_like(fock)
        dm[:nimp,:nimp] = dmimp[:nimp,:nimp]
        h1e = fock - self.get_eff_potential(mol, dm)
        return h1e

    def eri_on_impbas(self, mol):
        nimp = len(self.bas_on_frag)
        nemb = self.impbas_coeff.shape[1]
        mo = self.impbas_coeff[:,:nimp].copy('F')
        if self.entire_scf._eri is not None:
            eri = ao2mo.incore.full(self.entire_scf._eri, mo)
        else:
            eri = ao2mo.direct.full_iofree(self.entire_scf._eri, mo)
        npair = nemb*(nemb+1) / 2
        #eri_mo = numpy.zeros(npair*(npair+1)/2)
        npair_imp = nimp*(nimp+1) / 2
        # so only the 2e-integrals on impurity are non-zero
        #eri_mo[:npair_imp*(npair_imp+1)/2] = eri.reshape(-1)
        eri_mo = numpy.zeros((npair,npair))
        eri_mo[:npair_imp,:npair_imp] = eri
        return eri_mo

    def get_eff_potential(self, mol, dm, dm_last=0, vhf_last=0):
        if self._eri is None:
            self._eri = self.eri_on_impbas(mol)
        vj, vk = _vhf.vhf_jk_incore_o2(self._eri, dm)
        vhf = vj - vk * .5
        return vhf

