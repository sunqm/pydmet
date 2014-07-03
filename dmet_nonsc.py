#!/usr/bin/env python

import numpy
import lib
import lib.logger as log
import scf
import impsolver
import junk
import junk.hf
import ao2mo
import lib._vhf as _vhf

class OneImp(junk.hf.RHF):
    def __init__(self, mol, entire_scf, basidx):
        orth_ao = numpy.eye(entire_scf.mo_energy.size)
        junk.hf.RHF.__init__(self, entire_scf, orth_ao)
        self.bas_on_frag = basidx
        #self.solve_imp = \
        #        impsolver.use_local_solver(impsolver.fci, with_rdm1=True)
        #self.solve_imp = \
        #        impsolver.use_local_solver(impsolver.cc, with_rdm1=True)

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
        self.impbas_coeff[:,nimp:] = self.bath_orb

        self.nelectron = int(effscf.mo_occ.sum()) - self.env_orb.shape[1] * 2
        log.info(self, 'number of electrons for impurity  = %d', \
                 self.nelectron)

        log.debug(self, 'init Hartree-Fock environment')
        dm_env = numpy.dot(self.env_orb, self.env_orb.T.conj()) * 2
        vhf_env_ao = effscf.get_eff_potential(self.mol, dm_env)
        hcore = effscf.get_hcore()
        self.energy_by_env = lib.trace_ab(dm_env, hcore) \
                           + lib.trace_ab(dm_env, vhf_env_ao) * .5
        self._vhf_env = self.mat_ao2impbas(vhf_env_ao)

class OneImpNI(OneImp):
    '''Non-interacting DMET'''
    def __init__(self, mol, entire_scf, basidx):
        OneImp.__init__(self, mol, entire_scf, basidx)

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

def dmet_1shot(mol, emb):
    log.info(emb, '==== start DMET 1 shot ====')
    #emb.init_dmet_scf()
    emb.imp_scf()
    #solve_imp = impsolver.use_local_solver(impsolver.cc, with_rdm1=True)
    solve_imp = impsolver.use_local_solver(impsolver.fci, with_rdm1=True)
    eci, rdm1 = solve_imp(mol, emb)
    print eci, emb.hf_energy
    hcore = emb.get_hcore(mol)
    e_tot = eci + emb.energy_by_env
    log.log(emb, 'e_tot = %.11g, (+nuc=%.11g)',
            e_tot, e_tot+mol.nuclear_repulsion())
    return e_tot+mol.nuclear_repulsion(), eci-emb.hf_energy


if __name__ == '__main__':
    import gto
    import hf
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_dmet_1shot'
    mol.build()

    mf = hf.RHF(mol, 'C_solid_2x2x2/test2/FCIDUMP.CLUST.GTO',
                'C_solid_2x2x2/test2/JKDUMP')
    energy = mf.scf()
    print energy

    emb = OneImp(mol, mf, [0,1,2,3])
    print dmet_1shot(mol, emb)

######################
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
    mf.scf()

    emb = OneImp(mol, mf, [0,1,2,3])
    print dmet_1shot(mol, emb)

    emb = OneImpNI(mol, mf, [0,1,2,3])
    print dmet_1shot(mol, emb)

