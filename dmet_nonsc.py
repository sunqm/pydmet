#!/usr/bin/env python

import numpy
import scipy
import scipy.optimize
from pyscf import lib
from pyscf import ao2mo
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
import impsolver
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
        self.dump_flags()
        mol = self.mol

        self.init_embsys(mol)
        v_ci_group = self.vfit_ci_method(mol, self)
        self.update_embs_vfit_ci(mol, self.embs, v_ci_group)
        e_tot, nelec = self.assemble_frag_energy(mol)

        log.info(self, '====================')
        if self.verbose >= param.VERBOSE_DEBUG:
            for m,emb in enumerate(self.embs):
                log.debug(self, 'vfit_ci of frag %d = %s', m, v_ci_group[m])
                dm1 = self.solver.run(emb, emb._eri, emb.vfit_ci, True)[2]
                log.debug(self, 'impurity dm of frag %d = %s', m, dm1)
        log.info(self, 'dmet_nonsc.fullsys: e_tot = %.12g, nelec = %g', \
                 e_tot, nelec)
        return e_tot

    def one_shot(self):
        log.info(self, '==== one-shot ====')
        mol = self.mol
        self.init_embsys(mol)
        emb = self.embs[0]
        emb.verbose = self.verbose
        emb.imp_scf()
        nimp = len(emb.bas_on_frag)

        log.info(self, '')
        log.info(self, '===== CI/CC before Fitting =====')
        etot, e2frag, dm1 = self.solver.run(emb, emb._eri, with_1pdm=True,
                                            with_e2frag=nimp)
        e_tot = etot + emb.energy_by_env
        e_frag, nelec_frag = self.extract_frag_energy(emb, dm1, e2frag)
        log.info(self, 'before fitting, e_tot = %.11g, e_frag = %.11g, nelec_frag = %.11g',
                 e_tot, e_frag, nelec_frag)

        log.info(self, '')
        log.info(self, '===== Fitting chemical potential =====')
        vfit_ci = self.fitmethod_1shot(mol, emb, self)
        #self.update_embs_vfit_ci(mol, [emb], [vfit_ci])
        etot, e2frag, dm1 = self.solver.run(emb, emb._eri, vfit_ci,
                                            with_1pdm=True, with_e2frag=nimp)
        e_tot = etot + emb.energy_by_env
        e_frag, nelec_frag = self.extract_frag_energy(emb, dm1, e2frag)
        log.info(self, 'after fitting, e_tot = %.11g, e_frag = %.11g, nelec_frag = %.11g',
                 e_tot, e_frag, nelec_frag)

        log.info(self, '====================')
        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, 'vfit_ci = %s', vfit_ci)
            log.debug(self, 'impurity dm = %s', dm1)
        log.log(self, 'dmet_nonsc.one_shot: e_tot = %.11g, (+nuc=%.11g)', \
                e_tot, e_tot+mol.nuclear_repulsion())
        log.log(self, 'e_frag = %.11g, nelec_frag = %.11g', e_frag, nelec_frag)
        return e_tot

# emb._project_fock - emb._pure_hcore will include the fitting potential into
# the final energy expression.  Is it right?
#    def extract_frag_energy(self, emb, dm1, e2frag):
#        nimp = len(emb.bas_on_frag)
#        vhf = emb._project_fock - emb._pure_hcore
#        e1 = numpy.dot(dm1[:nimp].reshape(-1),
#                       (emb._pure_hcore + .5 * vhf)[:nimp].reshape(-1))
#        dmf = emb.make_rdm1(emb.mo_coeff_on_imp, emb.mo_occ)
#        vhf = emb.get_veff(emb.mol, dmf)
#        e2 = e2frag - .5*numpy.dot(dm1[:nimp].reshape(-1),
#                                   vhf[:nimp].reshape(-1))
#        e_frag = e1 + e2
#        n_frag = dm1[:nimp].trace()
#        log.debug(self, 'e_frag = %.11g, nelec_frag = %.11g', e_frag, n_frag)
#        return e_frag, n_frag


def fit_imp_fix_nelec(mol, emb, embsys):
    log.debug(embsys, 'fit_imp_fix_nelec')
    nemb = emb.impbas_coeff.shape[1]
    nimp = len(emb.bas_on_frag)
    nelec_frag = emb._project_nelec_frag # which is projected from lattice HF
    vadd = _chem_pot_on_imp
    def diff_nelec(v):
        dm = embsys.solver.run(emb, emb._eri, vadd(emb, v), with_1pdm=True)[2]
        print 'ddm ',nelec_frag,dm[:nimp].trace(), nelec_frag - dm[:nimp].trace()
        return nelec_frag - dm[:nimp].trace()
    x = fit_chempot(mol, emb, embsys, diff_nelec)
    return vadd(emb, x)

def fit_bath_fix_nelec(mol, emb, embsys):
    log.debug(embsys, 'fit_bath_fix_nelec')
    nimp = len(emb.bas_on_frag)
    nelec_frag = emb._project_nelec_frag
    vadd = _chem_pot_on_bath
    def diff_nelec(v):
        dm = embsys.solver.run(emb, emb._eri, vadd(emb, v), with_1pdm=True)[2]
        return nelec_frag - dm[:nimp].trace()
    x = fit_chempot(mol, emb, embsys, diff_nelec)
    return vadd(emb, x)

def fit_imp_float_nelec(mol, emb, embsys):
    log.debug(embsys, 'fit_imp_float_nelec')
    nimp = len(emb.bas_on_frag)
    vadd = _chem_pot_on_imp
    def diff_nelec(v):
        vmat = vadd(emb, v)
        h1e = emb.get_hcore(mol) + vmat
        nocc = emb.nelectron / 2
        mo = impsolver.simple_hf(h1e, emb._eri,
                                 emb.mo_coeff_on_imp, emb.nelectron)[-1]
        nelec_mf = numpy.sum(mo[:nimp,:nocc]**2)
        dm = embsys.solver.run(emb, emb._eri, vmat, with_1pdm=True)[2]
        return nelec_mf - dm[:nimp].trace()
    x = fit_chempot(mol, emb, embsys, diff_nelec)
    return vadd(emb, x)

def fit_bath_float_nelec(mol, emb, embsys):
    log.debug(embsys, 'fit_bath_float_nelec')
    nimp = len(emb.bas_on_frag)
    vadd = _chem_pot_on_bath
    def diff_nelec(v):
        vmat = vadd(emb, v)
        h1e = emb.get_hcore(mol) + vmat
        nocc = emb.nelectron / 2
        mo = impsolver.simple_hf(h1e, emb._eri,
                                 emb.mo_coeff_on_imp, emb.nelectron)[-1]
        nelec_mf = numpy.sum(mo[:nimp,:nocc]**2)
        dm = embsys.solver.run(emb, emb._eri, vmat, with_1pdm=True)[2]
        print 'ddm ',v,nelec_mf,dm[:nimp].trace(), nelec_mf - dm[:nimp].trace()
        return nelec_mf - dm[:nimp].trace()
    x = fit_chempot(mol, emb, embsys, diff_nelec)
    return vadd(emb, x)

# fit the electron number of
#     chemical potentail on imp for impsolver
#     chemical potentail on bath for embedding-HF
def fit_mix_float_nelec(mol, emb, embsys):
    log.debug(embsys, 'fit_bath_float_nelec')
    nimp = len(emb.bas_on_frag)
    def diff_nelec(v):
        h1e = emb.get_hcore(mol) + _chem_pot_on_imp(emb, v)
        nocc = emb.nelectron / 2
        mo = impsolver._scf_energy(h1e, emb._eri,
                                   emb.mo_coeff_on_imp, emb.nelectron)[-1]
        nelec_mf = numpy.sum(mo[:nimp,:nocc]**2)
        dm = embsys.solver.run(emb, emb._eri, _chem_pot_on_bath(emb, v),
                               with_1pdm=True)[2]
        return nelec_mf - dm[:nimp].trace()
    x = fit_chempot(mol, emb, embsys, diff_nelec)
    return _chem_pot_on_imp(emb, x)

def _chem_pot_on_imp(emb, v):
    nemb = emb.impbas_coeff.shape[1]
    nimp = len(emb.bas_on_frag)
    vmat = numpy.zeros((nemb,nemb))
    for i in range(nimp):
        vmat[i,i] = v
    return vmat

def _chem_pot_on_bath(emb, v):
    nemb = emb.impbas_coeff.shape[1]
    nimp = len(emb.bas_on_frag)
    vmat = numpy.zeros((nemb,nemb))
    for i in range(nimp,nemb):
        vmat[i,i] = v
    return vmat

def fit_chempot(mol, emb, embsys, diff_nelec):
    chem_pot0 = emb.vfit_ci[0,0]
    try:
        sol = scipy.optimize.root(diff_nelec, chem_pot0, tol=1e-3, method='lm',
                                  options={'ftol':1e-3, 'maxiter':10})
        log.debug(embsys, 'scipy.optimize summary %s', sol)
        log.debug(embsys, 'chem potential = %.11g, nelec error = %.11g', \
                  sol.x, sol.fun)
        log.debug(embsys, '        ncall = %d, scipy.optimize success: %s', \
                  sol.nfev, sol.success)
    except ImportError:
        sol = scipy.optimize.leastsq(diff_nelec, chem_pot0, ftol=1e-3, xtol=1e-3)
    return sol.x



if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_dmet_nonsc'
    b1 = 1.0
    nat = 10
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

    embsys = EmbSys(mol, mf, [[0,1],[2,3],[4,5],[6,7],[8,9]])
    print embsys.fullsys() # -18.0063690273

    embsys = EmbSys(mol, mf, [[0,1]])
    print embsys.one_shot() # -17.912887125

