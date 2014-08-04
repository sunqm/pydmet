#!/usr/bin/env python

import numpy
import scipy
import scipy.optimize
import impsolver
from pyscf import lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
from pyscf.lib import _vhf
from pyscf import ao2mo
import dmet_sc


##################################################
# system with translation symmetry
class EmbSys(dmet_sc.EmbSys):
    def __init__(self, mol, entire_scf, frag_group=[], init_v=None,
                 orth_coeff=None):
        dmet_sc.EmbSys.__init__(self, mol, entire_scf, frag_group, \
                                init_v, orth_coeff)
        self.max_iter = 1
        self.fitmethod_1shot = dmet_sc.fit_chemical_potential

    def scdmet(self, sav_v=None):
        log.warn(self, 'Self-consistency is not allowed in non-SC-DMET')
        self.fullsys()

    def fullsys(self):
        log.info(self, '==== fullsys ====')
        self.dump_options()
        mol = self.mol

        self.init_embsys(mol)
        v_ci_group = self.vfit_ci_method(mol, self)
        self.update_embs_vfit_ci(mol, self.embs, v_ci_group)
        e_tot, nelec = self.assemble_frag_fci_energy(mol)

        log.info(self, '====================')
        if self.verbose >= param.VERBOSE_DEBUG:
            for m,emb in enumerate(self.embs):
                log.debug(self, 'vfit_ci of frag %d = %s', m, v_ci_group[m])
                res = self.frag_fci_solver(mol, emb, emb.vfit_ci)
                log.debug(self, 'impurity dm of frag %d = %s', m, res['rdm1'])
        log.info(self, 'dmet_nonsc.fullsys: e_tot = %.12g, nelec = %g', \
                 e_tot, nelec)
        return e_tot

    def one_shot(self):
        log.info(self, '==== one-shot ====')
        mol = self.mol
        self.init_embsys(mol)
        emb = self.embs[0]
        emb.verbose = self.verbose
        #emb.imp_scf()

        vfit_ci = self.fitmethod_1shot(mol, emb, self)
        #self.update_embs_vfit_ci(mol, [emb], [vfit_ci])
        cires = self.frag_fci_solver(mol, emb, vfit_ci)
        e_tot = cires['etot'] + emb.energy_by_env
        nimp = emb.dim_of_impurity()
        e1_frag = numpy.dot(cires['rdm1'][:nimp].flatten(), \
                            emb._pure_hcore[:nimp].flatten())
        envhf_frag = numpy.dot(cires['rdm1'][:nimp].flatten(), \
                               emb._vhf_env[:nimp].flatten())
        e_frag = e1_frag + envhf_frag * .5 + cires['e2frag']
        n_frag = cires['rdm1'][:nimp].trace()
        #print etot, emb.hf_energy

        log.info(self, '====================')
        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, 'vfit_ci = %s', vfit_ci)
            log.debug(self, 'impurity dm = %s', cires['rdm1'])
        log.log(self, 'dmet_nonsc.one_shot: e_tot = %.11g, (+nuc=%.11g)', \
                e_tot, e_tot+mol.nuclear_repulsion())
        log.log(self, 'e_frag = %.11g, nelec_frag = %.11g', \
                e_frag, n_frag)
        return e_tot


##########################################################
# Using VASP HF results
import vaspimp
class EmbSysPeriod(EmbSys):
    def __init__(self, fcidump, jdump, kdump, fockdump, init_v=None):
        self._vasphf = vaspimp.read_clustdump(fcidump, jdump, kdump, fockdump)
        fake_hf = vaspimp.fake_entire_scf(self._vasphf)
        EmbSys.__init__(self, fake_hf.mol, fake_hf, init_v=None)
        self.orth_coeff = numpy.eye(self._vasphf['NORB'])
        #self.OneImp = lambda mf: vaspimp.OneImpOnCLUSTDUMP(mf, vasphf)
        self.frag_fci_solver = impsolver.use_local_solver(impsolver.cc)
        self.verbose = 5
        self.emb_verbose = 5
        self.pot_on = 'IMP'  # or 'BATH', or 'IMP,BATH'

    def init_embsys(self, mol):
        #embs = self.init_embs(mol, self.entire_scf, self.orth_coeff)
        emb = vaspimp.OneImpOnCLUSTDUMP(self.entire_scf, self._vasphf)
        emb.occ_env_cutoff = 1e-14
        emb.orth_coeff = self.orth_coeff
        emb.verbose = self.emb_verbose
        emb.imp_scf()
        embs = [emb]
        emb._project_fock = emb.mat_ao2impbas(self._vasphf['FOCK'])
        mo = self._vasphf['MO_COEFF']
        nimp = self._vasphf['NIMP']
        emb._pure_hcore = self._vasphf['H1EMB'].copy()
        cimp = numpy.dot(emb.impbas_coeff[:,:nimp].T,
                         mo[:,:self._vasphf['NELEC']/2])
        emb._project_nelec_frag = numpy.linalg.norm(cimp)**2*2
        log.debug(emb, 'nelec of imp from lattice HF %.8g',
                  emb._project_nelec_frag)
        log.debug(emb, 'nelec of imp from embedding HF %.8g',
                  numpy.linalg.norm(emb.mo_coeff_on_imp[:nimp,:emb.nelectron/2])**2*2)
        #embs = self.update_embs(mol, embs, self.entire_scf, self.orth_coeff)
        emb.vfit_mf = numpy.zeros_like(self._vasphf['H1EMB'])
        emb.vfit_ci = numpy.zeros_like(self._vasphf['H1EMB'])
        embs = self.update_embs_vfit_ci(mol, embs, [0])
        #embs = self.update_embs_vfit_mf(mol, embs, [0])
        self.embs = embs
        return [0], [0]

    def one_shot(self):
        log.info(self, '==== one-shot ====')
        mol = self.mol
        self.init_embsys(mol)
        emb = self.embs[0]
        emb.verbose = self.verbose
        #emb.imp_scf()

        log.info(self, '===== Fitting chemical potential =====')
        if self.pot_on.upper() == 'IMP':
            vfit_ci = fit_imp_fix_nelec(mol, emb, self)
            #vfit_ci = fit_imp_float_nelec(mol, emb, self)
        elif self.pot_on.upper() == 'BATH':
            vfit_ci = fit_bath_fix_nelec(mol, emb, self)
            #vfit_ci = fit_bath_float_nelec(mol, emb, self)
        else:
            vfit_ci = fit_mix_float_nelec(mol, emb, self)

        cires = self.frag_fci_solver(mol, emb, vfit_ci)
        e_tot = cires['etot'] + emb.energy_by_env
        nimp = emb.dim_of_impurity()
        e1_frag = numpy.dot(cires['rdm1'][:nimp].flatten(), \
                            emb._pure_hcore[:nimp].flatten())
        envhf_frag = numpy.dot(cires['rdm1'][:nimp].flatten(), \
                               emb._vhf_env[:nimp].flatten())
        e_frag = e1_frag + envhf_frag * .5 + cires['e2frag']
        n_frag = cires['rdm1'][:nimp].trace()

        log.info(self, '====================')
        #if self.verbose >= param.VERBOSE_DEBUG:
        #    log.debug(self, 'vfit_ci = %s', vfit_ci)
        #    log.debug(self, 'impurity dm = %s', cires['rdm1'])
        log.log(self, 'dmet_nonsc.one_shot: e_tot = %.11g, (+nuc=%.11g)', \
                e_tot, e_tot+mol.nuclear_repulsion())
        log.log(self, 'e_frag = %.11g, nelec_frag = %.11g', \
                e_frag, n_frag)
        return e_frag

    def scdmet(self, init_v=None):
        return self.one_shot(mol)

    def fullsys(self, init_v=None):
        return self.one_shot(mol)

def fit_imp_fix_nelec(mol, emb, embsys):
    log.debug(embsys, 'fit_imp_fix_nelec')
    nemb = emb.impbas_coeff.shape[1]
    nimp = emb.dim_of_impurity()
    nelec_frag = emb._project_nelec_frag # which is projected from lattice HF
    vadd = _chem_pot_on_imp
    def diff_nelec(v):
        cires = embsys.frag_fci_solver(mol, emb, vadd(emb, v))
        dm = cires['rdm1']
        #print 'ddm ',nelec_frag,dm[:nimp].trace(), nelec_frag - dm[:nimp].trace()
        return nelec_frag - dm[:nimp].trace()
    x = fit_chempot(mol, emb, embsys, diff_nelec)
    return vadd(emb, x)

def fit_bath_fix_nelec(mol, emb, embsys):
    log.debug(embsys, 'fit_bath_fix_nelec')
    nimp = emb.dim_of_impurity()
    nelec_frag = emb._project_nelec_frag
    vadd = _chem_pot_on_bath
    def diff_nelec(v):
        cires = embsys.frag_fci_solver(mol, emb, vadd(emb, v))
        dm = cires['rdm1']
        return nelec_frag - dm[:nimp].trace()
    x = fit_chempot(mol, emb, embsys, diff_nelec)
    return vadd(emb, x)

def fit_imp_float_nelec(mol, emb, embsys):
    log.debug(embsys, 'fit_imp_float_nelec')
    nimp = emb.dim_of_impurity()
    vadd = _chem_pot_on_imp
    def diff_nelec(v):
        vmat = vadd(emb, v)
        h1e = emb.get_hcore(mol) + vmat
        nocc = emb.nelectron / 2
        _,_,_,mo = impsolver._scf_energy(h1e, emb._eri, \
                                         emb.mo_coeff_on_imp, nocc)
        nelec_mf = numpy.sum(mo[:nimp,:nocc]**2)
        cires = embsys.frag_fci_solver(mol, emb, vmat)
        dm = cires['rdm1']
        return nelec_mf - dm[:nimp].trace()
    x = fit_chempot(mol, emb, embsys, diff_nelec)
    return vadd(emb, x)

def fit_bath_float_nelec(mol, emb, embsys):
    log.debug(embsys, 'fit_bath_float_nelec')
    nimp = emb.dim_of_impurity()
    vadd = _chem_pot_on_bath
    def diff_nelec(v):
        vmat = vadd(emb, v)
        h1e = emb.get_hcore(mol) + vmat
        nocc = emb.nelectron / 2
        _,_,_,mo = impsolver._scf_energy(h1e, emb._eri, \
                                         emb.mo_coeff_on_imp, nocc)
        nelec_mf = numpy.sum(mo[:nimp,:nocc]**2)
        cires = embsys.frag_fci_solver(mol, emb, vmat)
        dm = cires['rdm1']
        print 'ddm ',v,nelec_mf,dm[:nimp].trace(), nelec_mf - dm[:nimp].trace()
        return nelec_mf - dm[:nimp].trace()
    x = fit_chempot(mol, emb, embsys, diff_nelec)
    return vadd(emb, x)

# fit the electron number of
#     chemical potentail on imp for impsolver
#     chemical potentail on bath for embedding-HF
def fit_mix_float_nelec(mol, emb, embsys):
    log.debug(embsys, 'fit_bath_float_nelec')
    nimp = emb.dim_of_impurity()
    def diff_nelec(v):
        h1e = emb.get_hcore(mol) + _chem_pot_on_imp(emb, v)
        nocc = emb.nelectron / 2
        _,_,_,mo = impsolver._scf_energy(h1e, emb._eri, \
                                         emb.mo_coeff_on_imp, nocc)
        nelec_mf = numpy.sum(mo[:nimp,:nocc]**2)
        cires = embsys.frag_fci_solver(mol, emb, _chem_pot_on_bath(emb, v))
        dm = cires['rdm1']
        return nelec_mf - dm[:nimp].trace()
    x = fit_chempot(mol, emb, embsys, diff_nelec)
    return _chem_pot_on_imp(emb, x)

def _chem_pot_on_imp(emb, v):
    nemb = emb.impbas_coeff.shape[1]
    nimp = emb.dim_of_impurity()
    vmat = numpy.zeros((nemb,nemb))
    for i in range(nimp):
        vmat[i,i] = v
    return vmat

def _chem_pot_on_bath(emb, v):
    nemb = emb.impbas_coeff.shape[1]
    nimp = emb.dim_of_impurity()
    vmat = numpy.zeros((nemb,nemb))
    for i in range(nimp,nemb):
        vmat[i,i] = v
    return vmat

def fit_chempot(mol, emb, embsys, diff_nelec):
    chem_pot0 = emb.vfit_ci[0,0]
    sol = scipy.optimize.root(diff_nelec, chem_pot0, tol=1e-3, method='lm',
                              options={'ftol':1e-3, 'maxiter':10})
    log.debug(embsys, 'scipy.optimize summary %s', sol)
    log.debug(embsys, 'chem potential = %.11g, nelec error = %.11g', \
              sol.x, sol.fun)
    log.debug(embsys, '        ncall = %d, scipy.optimize success: %s', \
              sol.nfev, sol.success)
    return sol.x



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
#    emb = OneImp(mf, [0,1,2,3])
#    print dmet_1shot(mol, emb)

######################
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_dmet'
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = {'H': '6-31g',
                 'O': '6-31g',}
    mol.build()
    mf = scf.RHF(mol)
    mf.scf()

    embsys = EmbSys(mol, mf, [[0,1],[2,3],[4,5],[6,7],[8,9]])
    embsys.OneImp = oneimp.OneImpNI
    print embsys.fullsys()

    embsys = EmbSys(mol, mf, [[0,1]])
    print embsys.one_shot()

    embsys = EmbSys(mol, mf, [[0,1]])
    embsys.OneImp = oneimp.OneImpNI
    print embsys.one_shot()

#    b1 = 1.0
#    nat = 10
#    mol.output = 'h%s_sz' % nat
#    mol.atom = []
#    r = b1/2 / numpy.sin(numpy.pi/nat)
#    for i in range(nat):
#        theta = i * (2*numpy.pi/nat)
#        mol.atom.append((1, (r*numpy.cos(theta),
#                             r*numpy.sin(theta), 0)))
#
#    mol.basis = {'H': 'sto-3g',}
#    mol.build()
#    mf = scf.RHF(mol)
#    print mf.scf()
#
#    embsys = EmbSys(mol, mf, [[0,1],[2,3],[4,5],[6,7],[8,9]])
#    print embsys.fullsys()
#
#    embsys = EmbSys(mol, mf, [[0,1]])
#    print embsys.one_shot()
