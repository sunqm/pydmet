#!/usr/bin/env python

import numpy
import scipy.optimize
import impsolver
from pyscf import lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
from pyscf.lib import _vhf
from pyscf import ao2mo
import junk.hf
import junk.dmet_sc
import junk.hfdm

# AO basis of entire system are orthogonal sets
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
#overlap matrix here
        self.impbas_coeff[:,nimp:] = self.bath_orb

        self.nelectron = int(effscf.mo_occ.sum()) - self.env_orb.shape[1] * 2
        log.info(self, 'number of electrons for impurity  = %d', \
                 self.nelectron)

        log.debug(self, 'init Hartree-Fock environment')
        dm_env = numpy.dot(self.env_orb, self.env_orb.T.conj()) * 2
        vhf_env_ao = effscf.get_eff_potential(self.mol, dm_env)
        hcore = effscf.get_hcore()
        self.energy_by_env = numpy.dot(dm_env.flatten(), hcore.flatten()) \
                           + numpy.dot(dm_env.flatten(), \
                                       vhf_env_ao.flatten()) * .5
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
    eci, _, rdm1 = solve_imp(mol, emb)

    print eci, emb.hf_energy
    e_tot = eci + emb.energy_by_env
    log.info(emb, 'e_tot = %.11g, (+nuc=%.11g)',
            e_tot, e_tot+mol.nuclear_repulsion())

    return e_tot+mol.nuclear_repulsion(), eci-emb.hf_energy


##################################################
# system with translation symmetry
class EmbSys(junk.dmet_sc.EmbSys):
    def __init__(self, mol, entire_scf, frag_bas_idx):
        junk.dmet_sc.EmbSys.__init__(self, mol, entire_scf, [])
        self.basidx_group = [frag_bas_idx]
        self.vfit_method = local_fit_with_chemical_pot

    def init_embsys(self, mol, init_v):
        #self.basidx_group = map_frag_to_bas_idx(mol, self.frag_group)
        #self.all_frags, self.uniq_frags = gen_frag_looper(mol, self.frag_group)
        self.embs = self.setup_embs(mol)
        self.orth_coeff = self.embs[0].orth_coeff
        nao = self.orth_coeff.shape[1]
        self.v_global = numpy.zeros((nao,nao))
        #self.bas_off_frags = self.set_bas_off_frags()
#        try:
#            with open(init_v, 'r') as f:
#                self.v_global, v_add_on_ao = pickle.load(f)
#            self.entire_scf = scfci.run_hf_with_ext_pot(mol, entire_scf, \
#                                                        v_add_on_ao, \
#                                                        self.follow_state)
#        except:
#            nao = self.orth_coeff.shape[1]
#            self.v_global = numpy.zeros((nao,nao))
#            if self.rand_init:
#                for m, atm_lst, bas_idx in self.all_frags:
#                    nimp = bas_idx.__len__()
#                    v = numpy.random.randn(nimp*nimp).reshape(nimp,nimp)
#                    v = (v + v.T) * .1
#                    for i, j in enumerate(bas_idx):
#                        self.v_global[j,bas_idx] = v[i]
#                self.entire_scf = scfci.run_hf_with_ext_pot(mol, entire_scf, \
#                                                            self.v_global, \
#                                                            self.follow_state)

    def setup_embs(self, mol):
        #emb = OneImp(mol, mf, self.basidx_group[0])
        emb = junk.hf.RHF(mf)
        emb.imp_basidx = self.basidx_group[0]
        emb.occ_env_cutoff = 1e-14
        #emb.verbose = 0
        emb.init_dmet_scf(mol)
        emb.imp_scf()
        embs = [emb]
        return embs

#    def update_embsys_vglobal(self, mol, v_add):
#        #eff_scf = junk.scfci.run_hf_with_ext_pot(mol, self.entire_scf, v_add)
#
#        #self.v_global = v_add
#        #self.entire_scf = eff_scf
#
#        #for emb in self.embs:
#        #    emb.entire_scf = eff_scf
#        #self.setup_embs_with_vglobal(mol, self.embs, v_add)
#        #self.set_env_fit_pot_for_fci(v_add)
#        #return eff_scf
#        return self.entire_scf
    def update_embsys_vglobal(self, mol, v_add):
        v_add_ao = junk.scfci.mat_orthao2ao(mol, v_add, self.orth_coeff)
        eff_scf = junk.scfci.run_hf_with_ext_pot(mol, self.entire_scf, v_add_ao)

        self.v_global = v_add
        self.entire_scf = eff_scf

        for emb in self.embs:
            emb.entire_scf = eff_scf
        self.setup_embs_with_vglobal(mol, self.embs, v_add)
        self.set_env_fit_pot_for_fci(v_add)
        return eff_scf

    def frag_fci_solver(self, mol, emb, v=0):
        solve_imp = impsolver.use_local_solver(impsolver.fci, with_rdm1=False)
        #solve_imp = impsolver.use_local_solver(impsolver.cc, with_rdm1=False)
        return solve_imp(mol, emb, v)

    def assemble_frag_fci_energy(self, mol):
        emb = embsys.embs[0]
        nimp = len(emb.bas_on_frag)
        vmat = numpy.eye(nimp) * emb._chem_pot
        cires = self.frag_fci_solver(mol, emb, vmat)
        rdm1 = cires['rdm1']
        pure_hcore = emb.mat_ao2impbas(emb.entire_scf.get_hcore(mol))
        e2env_hf = numpy.dot(rdm1[:nimp].flatten(), \
                             emb._vhf_env[:nimp].flatten()) * .5
        e_frag = numpy.dot(pure_hcore[:nimp].flatten(), rdm1[:nimp].flatten()) \
                + e2env_hf + cires['e2frag']
        nelec_frag = rdm1[:nimp].trace()
        nfrag = self.entire_scf.mo_energy.size / nimp
        log.debug(mol, 'e_frag = %.9g, nelec_frag = %.9g, e_tot = %.9g, nelec_tot = %.9g', \
                  e_frag, nelec_frag, e_frag*nfrag, nelec_frag*nfrag)
        return e_frag*nfrag, nelec_frag*nfrag

    def assemble_to_blockmat(self, mol, v_group):
        nao = self.orth_coeff.shape[1]
        v_add = numpy.zeros((nao,nao))
        nemb = v_group[0].shape[0]
        nfrag = self.entire_scf.mo_energy.size / nemb
        for m in range(nfrag):
            v_add[m*nemb:(m+1)*nemb,m*nemb:(m+1)*nemb] = v_group[0]
        return v_add

    def dump_frag_prop_mat(self, mol, frag_mat_group):
        pass

def local_fit_with_chemical_pot(mol, embsys):
    emb = embsys.embs[0]
    nimp = emb.dim_of_impurity()
    # this fock matrix includes the pseudo potential of present fragment
    mo = emb.mo_coeff_on_imp
    fock0 = numpy.dot(mo*emb.mo_energy, mo.T)
    nocc = emb.nelectron/2
    nelec_frag = numpy.linalg.norm(mo[:nimp,:nocc])**2 * 2

# change chemical potential to get correct number of electrons
    def nelec_diff(v):
        vmat = numpy.eye(nimp) * v
        cires = embsys.frag_fci_solver(mol, emb, vmat)
        dm = cires['rdm1']
        return abs(nelec_frag - dm[:nimp].trace())
    emb._chem_pot = scipy.optimize.leastsq(nelec_diff, 0, ftol=1e-5)[0]

    vmat = numpy.eye(nimp) * emb._chem_pot
    cires = embsys.frag_fci_solver(mol, emb, vmat)
    dm_ref = cires['rdm1']
    log.debug(embsys, 'dm_ref = %s', dm_ref)

    dv = junk.hfdm.fit_solver(embsys, fock0, nocc, nimp, dm_ref*.5, \
                              embsys.v_fit_domain, embsys.dm_fit_domain, \
                              embsys.dm_fit_constraint)
    if embsys.fitpot_damp_fac > 0:
        dv *= embsys.fitpot_damp_fac
    return dv[:nimp,:nimp]

def dmet_fullsys(mol, embsys, sav_v=None):
    log.info(embsys, '==== dmet_fullsys ====')
    embsys.dump_options()

#    if embsys.verbose >= param.VERBOSE_DEBUG:
#        log.debug(embsys, '** DM of MF sys (on orthogonal AO) **')
#        c = numpy.dot(numpy.linalg.inv(embsys.orth_coeff), \
#                      embsys.entire_scf.mo_coeff)
#        nocc = mol.nelectron / 2
#        dm = numpy.dot(c[:,:nocc],c[:,:nocc].T) * 2
#        fmt = '    %10.5f' * dm.shape[1] + '\n'
#        for c in numpy.array(dm):
#            mol.fout.write(fmt % tuple(c))

    embsys.max_iter = 1
    e_tot, v_group = junk.dmet_sc.dmet_sc_cycle(mol, embsys)

    log.info(embsys, '====================')
    if embsys.verbose >= param.VERBOSE_DEBUG:
        for m,emb in enumerate(embsys.embs):
            log.debug(embsys, 'vfit of frag %d = %s', m, v_group[m])

    #e_tot, nelec = embsys.assemble_frag_fci_energy(mol)
    #log.log(embsys, 'e_tot = %.11g, +nuc = %.11g, nelec = %.8g', \
    #        e_tot, e_tot+mol.nuclear_repulsion(), nelec)
    return e_tot



if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    import hf
#    mol = gto.Mole()
#    mol.verbose = 5
#    mol.output = 'out_dmet_1shot'
#    mol.build()
#
#    mf = hf.RHF(mol, 'C_solid_2x2x2/test2/FCIDUMP.CLUST.GTO',
#                'C_solid_2x2x2/test2/JKDUMP')
#    energy = mf.scf()
#    print energy
#
#    emb = OneImp(mol, mf, [0,1,2,3])
#    print dmet_1shot(mol, emb)

######################
    mol = gto.Mole()
    mol.verbose = 5
#    mol.output = 'out_dmet'
#    mol.atom = [
#        ['O' , (0. , 0.     , 0.)],
#        [1   , (0. , -0.757 , 0.587)],
#        [1   , (0. , 0.757  , 0.587)] ]
#    mol.basis = {'H': '6-31g',
#                 'O': '6-31g',}
#    mol.build()
#    mf = scf.RHF(mol)
#    mf.scf()
#
#    emb = OneImp(mol, mf, [0,1,2,3])
#    print dmet_1shot(mol, emb)
#
#    emb = OneImpNI(mol, mf, [0,1,2,3])
#    print dmet_1shot(mol, emb)

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

    embsys = EmbSys(mol, mf, [0,1])
    print dmet_fullsys(mol, embsys)
