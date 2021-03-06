#!/usr/bin/env python

import re
import numpy
from pyscf import lib
import pyscf.lib.logger as log
from pyscf.lib import logger
import pyscf.lib.parameters as param
from pyscf import ao2mo
from pyscf import gto
from pyscf import scf
import dmet_hf

# NOTE:
# * When reading vasp FOCKDUMP, the fock matrix has EXCLUDED the correlation
#   potential.  It's not completely diagonalized

# AO basis of entire system are orthogonal sets

class OneImp(dmet_hf.RHF):
    def __init__(self, entire_scf):
        dmet_hf.RHF.__init__(self, entire_scf,
                             numpy.eye(entire_scf._vaspdump['NORB']))
        self.bas_on_frag = entire_scf._vaspdump['ORBIND']
        self.conv_tol = 1e-8
        assert(self.entire_scf._vaspdump['NIMP'] == len(self.bas_on_frag))

    def build_(self, mol=None):
        effscf = self.entire_scf
        self.imp_site = None
        self.bath_orb = None
        self.env_orb = None
        nemb = self.entire_scf._vaspdump['EMBASIS'].shape[1]
        self._eri = ao2mo.restore(8, self.eri_on_impbas(mol), nemb)
        mo_orth = effscf.mo_coeff[:,effscf.mo_occ>1e-15]
        self.impbas_coeff = self.entire_scf._vaspdump['EMBASIS']
        assert(numpy.linalg.norm(self.impbas_coeff) > 1e-10) # ensure embasis has been read

        self.nelectron = int(round(self.entire_scf._vaspdump['NELECEMB']))
        log.info(self, 'number of electrons = %f (%d)', \
                 self.entire_scf._vaspdump['NELECEMB'], self.nelectron)
        self.energy_by_env, self._vhf_env = self.init_vhf_env(None)

    def init_vhf_env(self, env_orb):
        nemb = self.impbas_coeff.shape[1]
        c = numpy.dot(self.impbas_coeff.T, self.entire_scf.mo_coeff)
# the correlation potential has been added to H1EMB. in vhf, it will cancel
# out the correlation potential in self.entire_scf.mo_energy.
#        vhf = numpy.dot(c*self.entire_scf.mo_energy, c.T) \
#                - self.entire_scf._vaspdump['H1EMB']
# == vhf = self.mat_ao2impbas(self.entire_scf._vaspdump['J']+self.entire_scf._vaspdump['K'])
        vhf = self.mat_ao2impbas(self.entire_scf._vaspdump['J']+self.entire_scf._vaspdump['K'])

        mocc = c[:,self.entire_scf.mo_occ>0]
        dmemb = numpy.dot(mocc, mocc.T)*2
        vemb = self.get_veff(self.mol, dmemb)
        return 0, vhf - vemb

    def get_init_guess(self, *args, **kwargs):
        log.debug(self, 'init guess based on entire MO coefficients')
        eff_scf = self.entire_scf
        c = numpy.dot(self.impbas_coeff.T, eff_scf.mo_coeff)
        dm = eff_scf.make_rdm1(c, eff_scf.mo_occ)
        return dm

    def get_hcore(self, mol=None):
# This one explicitly excluded the correlation potential
        return self.mat_ao2impbas(self.entire_scf.get_hcore(mol)) + self._vhf_env

    def get_ovlp(self, mol=None):
        return numpy.eye(self.entire_scf._vaspdump['NEMB'])

    def eri_on_impbas(self, mol):
        return ao2mo.restore(8, self.entire_scf._vaspdump['ERI'],
                             self.entire_scf._vaspdump['NEMB'])

    def kernel(self, *args, **kwargs):
        return self.imp_scf(*args, **kwargs)
    def imp_scf(self):
        self.orth_coeff = self.get_orth_ao(self.mol)
        self.dump_flags()
        self.build_(self.mol)
        self.scf_conv, self.e_tot, self.mo_energy, \
                self.mo_coeff_on_imp, self.mo_occ \
                = scf.hf.kernel(self, self.conv_tol, dump_chk=False)
        #self.mo_coeff = numpy.dot(self.impbas_coeff, self.mo_coeff_on_imp)
        self.mo_coeff = self.mo_coeff_on_imp # because the integrals are read from FCIDUMP
        if self.scf_conv:
            log.log(self, 'converged impurity sys electronic energy = %.15g', \
                    self.e_tot)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'electronic energy = %.15g after %d cycles.', \
                    self.e_tot, self.max_cycle)

        dm = self.make_rdm1(self.mo_coeff_on_imp, self.mo_occ)
        vhf = self.get_veff(self.mol, dm)
        self.e_tot, self.e_frag, self.nelec_frag = \
                self.calc_frag_elec_energy(self.mol, vhf, dm)
        return self.e_tot

    def mulliken_pop(self, mol, dm, ovlp=None, verbose=logger.DEBUG):
        if ovlp is None:
            ovlp = get_ovlp(mol)
        if isinstance(verbose, logger.Logger):
            log = verbose
        else:
            log = logger.Logger(mol.stdout, verbose)
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            pop = numpy.einsum('ij->i', dm*ovlp).real
        else: # ROHF
            pop = numpy.einsum('ij->i', (dm[0]+dm[1])*ovlp).real

        log.info(' ** Mulliken pop  **')
        for i, s in enumerate(pop):
            log.info('pop of  %d %10.5f', i, s)


##########################################################################

class OneImpNaiveNI(OneImp):
    '''Non-interacting DMET'''
    def __init__(self, entire_scf, basidx=[], orth_ao=None):
        OneImp.__init__(self, entire_scf, basidx)

    def eri_on_impbas(self, mol):
        nimp = len(self.bas_on_frag)
        nemb = self.impbas_coeff.shape[1]
        mo = self.impbas_coeff[:,:nimp].copy('F')
        eri = ao2mo.incore.full(self.entire_scf._eri, mo)
        npair = nemb*(nemb+1) / 2
        #eri_mo = numpy.zeros(npair*(npair+1)/2)
        npair_imp = nimp*(nimp+1) / 2
        # so only the 2e-integrals on impurity are non-zero
        #eri_mo[:npair_imp*(npair_imp+1)/2] = eri.reshape(-1)
        eri_mo = numpy.zeros((npair,npair))
        eri_mo[:npair_imp,:npair_imp] = eri
        return ao2mo.restore(8, eri_mo, nemb)


class OneImpNI(OneImpNaiveNI):
    def get_hcore(self, mol=None):
        nimp = len(self.bas_on_frag)
        effscf = self.entire_scf
        cs = numpy.linalg.solve(effscf.mo_coeff, self.impbas_coeff)
        fock = numpy.dot(cs.T*effscf.mo_energy, cs)
        dmimp = effscf.make_rdm1(mo_coeff=cs.T)
        dm = numpy.zeros_like(fock)
        dm[:nimp,:nimp] = dmimp[:nimp,:nimp]
        h1e = fock - self.get_veff(self.mol, dm)
        return h1e


##########################################################################




#FIXMEif __name__ == '__main__':
#FIXME    dic = read_clustdump('FCIDUMP.CLUST.GTO', 'JDUMP','KDUMP','FOCKDUMP')
#FIXME#    hcore = dic['FOCK'] - (dic['J'] - .5*dic['K'])
#FIXME#    nimp = dic['NIMP']
#FIXME#    print abs(hcore[:nimp,:nimp] - dic['H1EMB'][:nimp,:nimp]).sum()
#FIXME#    ee = reduce(numpy.dot, (dic['MO_COEFF'].T, dic['FOCK'], dic['MO_COEFF']))
#FIXME#    print abs(ee - numpy.diag(dic['MO_ENERGY'])).sum()
#FIXME    fake_hf = fake_entire_scf(dic)
#FIXME    emb = OneImpOnCLUSTDUMP(fake_hf, dic)
#FIXME    emb.verbose = 5
#FIXME    emb.imp_scf()
#FIXME
