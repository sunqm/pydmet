#!/usr/bin/env python

import time
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

class EmbSys(object):
    def __init__(self, mol, entire_scf, frag_group=[], init_v=None,
                 orth_coeff=None):
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self.mol = mol
        self.OneImp = dmet_hf.RHF

        self.max_iter         = 40
        self.conv_threshold   = 1e-5

        self.orth_coeff = orth_coeff
        #self.pre_orth_ao = lo.iao.pre_atm_scf_ao(mol)
        self.pre_orth_ao = numpy.eye(mol.nao_nr())
        self.orth_ao_method = 'lowdin'
        self.orth_coeff = lo.orth.orth_ao(mol, 'lowdin', numpy.eye(mol.nao_nr()))

        self.basidx = None
        self.entire_scf = entire_scf
        self.embs = []
        self.solver = impsolver.FCI()


    def build_(self, mol):
        embs = []
        emb = self.OneImp(self.entire_scf)
        emb.occ_env_cutoff = 1e-14
        emb.bas_on_frag = self.basidx
        emb.orth_coeff = self.orth_coeff
        emb.verbose = 0
        embs.append(emb)
        self.update_embs(mol, embs, self.entire_scf, self.orth_coeff)
        self.embs = embs

        nao = self.orth_coeff.shape[1]
        self.vfit_mf = [0]
        self.vfit_ci = [0]
        return self.vfit_mf, self.vfit_ci

    # update the embs in terms of the given entire_scf
    def update_embs(self, mol, embs, eff_scf, orth_coeff):
        sc = numpy.dot(eff_scf.get_ovlp(mol), eff_scf.mo_coeff)
        c_inv = numpy.dot(eff_scf.get_ovlp(mol), orth_coeff).T
        fock0 = numpy.dot(sc*eff_scf.mo_energy, sc.T.conj())
        hcore = eff_scf.get_hcore(mol)

        vhfwhole = eff_scf.get_veff(mol, eff_scf.make_rdm1())
        nocc = int(eff_scf.mo_occ.sum()) / 2

        for ifrag, emb in enumerate(embs):
            mo_orth = numpy.dot(c_inv, eff_scf.mo_coeff[:,eff_scf.mo_occ>1e-15])
            emb.imp_site, emb.bath_orb, emb.env_orb = \
                    dmet_hf.decompose_orbital(emb, mo_orth, emb.bas_on_frag)
            emb.impbas_coeff = emb.cons_impurity_basis()
            emb.nelectron = mol.nelectron - emb.env_orb.shape[1] * 2
            log.debug(emb, 'nelec of emb %d = %d', ifrag, emb.nelectron)
            emb._eri = emb.eri_on_impbas(mol)
            emb.energy_by_env, emb._vhf_env = emb.init_vhf_env(emb.env_orb)

            emb._project_fock = emb.mat_ao2impbas(fock0)
            emb.mo_energy, emb.mo_coeff_on_imp = scipy.linalg.eigh(emb._project_fock)
            emb.mo_coeff = numpy.dot(emb.impbas_coeff, emb.mo_coeff_on_imp)
            emb.mo_occ = numpy.zeros_like(emb.mo_energy)
            emb.mo_occ[:emb.nelectron/2] = 2
            emb.e_tot = 0
            nimp = emb.imp_site.shape[1]
            cimp = numpy.dot(emb.impbas_coeff[:,:nimp].T, sc[:,:nocc])
            emb._pure_hcore = emb.mat_ao2impbas(hcore)
            emb._project_nelec_frag = numpy.linalg.norm(cimp)**2*2

# the energy _ehfinhf is defined on emb.entire_scf, which is not the same as
# the pure SCF of the entire system, because dmet-scf will update
# emb.entire_scf with fitting potential vfit_mf
            hfdm = emb.make_rdm1(emb.mo_coeff_on_imp, emb.mo_occ)
            vhf = emb.mat_ao2impbas(vhfwhole)

        return embs


    def mat_orthao2ao(self, mat):
        '''matrix represented on orthogonal basis to the representation on
        non-orth AOs'''
        c_inv = numpy.dot(self.orth_coeff.T, self.entire_scf.get_ovlp())
        mat_on_ao = reduce(numpy.dot, (c_inv.T, mat, c_inv))
        return mat_on_ao

    def run_hf_with_ext_pot_(self, vext_on_ao):
        eff_scf = self.entire_scf
        h = eff_scf.get_hcore()
        eff_scf.get_hcore = lambda *args: h + vext_on_ao

        dm0 = eff_scf.make_rdm1(eff_scf.mo_coeff, eff_scf.mo_occ)

        eff_scf.scf_conv, eff_scf.e_tot, eff_scf.mo_energy, \
                eff_scf.mo_coeff, eff_scf.mo_occ \
                = scf.hf.kernel(eff_scf, eff_scf.conv_tol, dump_chk=False,
                                dm0=dm0)
        # must release the modified get_hcore to get pure hcore
        del(eff_scf.get_hcore)
        return eff_scf

    def update_embsys(self, mol, vfit_mf, vfit_ci):
        v_add = self.assemble_to_blockmat(vfit_mf)
        v_add_ao = self.mat_orthao2ao(v_add)
        self.entire_scf = self.run_hf_with_ext_pot_(v_add_ao)
        for emb in self.embs:
            emb.entire_scf = self.entire_scf

        embs = self.update_embs(mol, self.embs, self.entire_scf,
                                self.orth_coeff)
        self.vfit_mf = vfit_mf
        self.vfit_ci = vfit_ci


    def assemble_frag_energy(self, mol):
        e_tot = 0
        nelec = 0
        for m, emb in enumerate(self.embs):
            nimp = len(emb.bas_on_frag)
            dv = numpy.eye(nimp) * self.vfit_ci[m]
            _, e2frag, dm1 = \
                    self.solver.run(emb, emb._eri, dv,
                                    with_1pdm=True, with_e2frag=nimp)
            e_frag, nelec_frag = \
                    self.extract_frag_energy(emb, dm1, e2frag)

            log.debug(self, 'fragment %d FCI-in-HF, frag energy = %.12g, nelec = %.9g', \
                      m, e_frag, nelec_frag)
            e_tot += e_frag
            nelec += nelec_frag
        log.info(self, 'sum(e_frag), energy = %.9g, nelec = %.9g',
                  e_tot, nelec)
        return e_tot, 0, nelec

    def extract_frag_energy(self, emb, dm1, e2frag):
        nimp = len(emb.bas_on_frag)

        if emb._pure_hcore is not None:
            h1e = emb._pure_hcore
        else:
            h1e = emb.mat_ao2impbas(emb.entire_scf.get_hcore(emb.mol))

        e1_frag = numpy.dot(dm1[:nimp,:nimp].flatten(),h1e[:nimp,:nimp].flatten())
        e1_bath = numpy.dot(dm1[:nimp,nimp:].flatten(),h1e[:nimp,nimp:].flatten())
#        if self.env_pot_for_ci and emb.vfit_ci is not 0:
#            e1_vfit = numpy.dot(dm1[:nimp].flatten(), emb.vfit_ci[:nimp].flatten())
#        else:
#            e1_vfit = 0
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
        if isinstance(v_group[0], numpy.ndarray):
            nimp = v_group[0].shape[0]
            for i in range(nao//nimp):
                v_add[i*nimp:(i+1)*nimp,i*nimp:(i+1)*nimp] = v_group[0]
        return v_add

    def dump_frag_prop_mat(self, mol, frag_mat_group):
        '''dump fragment potential or density matrix'''
        for m, v in enumerate(frag_mat_group):
            try:
                pyscf.tools.dump_mat.dump_tri(self.stdout, v)
            except:
                self.stdout.write('%s\n' % str(v))


    def scdmet(self):
        log.info(self, '==== start DMET self-consistency ====')

        e_tot = dmet_sc_cycle(self)[0]
        return e_tot

    def vfit_ci_method(self, mol, embsys):
        return gen_all_vfit_by(fit_chemical_potential)(mol, embsys)

    def vfit_mf_method(self, mol, embsys):
        return gen_all_vfit_by(fit_without_local_scf)(mol, embsys)


###########################################################
# fitting methods
###########################################################
##ABORT to minimize the DM difference, use mean-field analytic gradients
def fit_without_local_scf(mol, m, embsys):
    emb = embsys.embs[m]
    dm_ref = embsys.solver.run(emb, emb._eri, embsys.vfit_ci[m], True, False)[2]
    log.debug(embsys, 'dm_ref = %s', dm_ref)
    nimp = len(emb.bas_on_frag)
    # this fock matrix includes the previous fitting potential
    fock0 = emb._project_fock.copy()
    nocc = emb.nelectron/2

    dv = fit_solver(embsys, fock0, nocc, nimp, dm_ref*.5)
    return embsys.vfit_mf[m]+dv

def gen_all_vfit_by(local_fit_method):
    '''fit HF DM with chemical potential'''
    def fitloop(mol, embsys):
        v_group = []
        for m, emb in enumerate(embsys.embs):
            dv = local_fit_method(mol, m, embsys)
            v_group.append(dv)

        if embsys.verbose >= param.VERBOSE_DEBUG:
            log.debug(embsys, 'fitting potential =')
            embsys.dump_frag_prop_mat(mol, v_group)
        return v_group
    return fitloop


def fit_chemical_potential(mol, m, embsys):
# correlation potential of embedded-HF is not added to correlated-solver
    emb = embsys.embs[m]
    import scipy.optimize
    nimp = len(emb.bas_on_frag)
    nelec_frag = emb._project_nelec_frag

# change chemical potential to get correct number of electrons
    def nelec_diff(v):
        dv = numpy.eye(nimp) * v
        dm = embsys.solver.run(emb, emb._eri, dv, True, False)[2]
        #print 'ddm ',nelec_frag,dm[:nimp].trace(), nelec_frag - dm[:nimp].trace()
        return nelec_frag - dm[:nimp].trace()
#OPTIMIZE ME, approximate chemical potential
    sol = scipy.optimize.root(nelec_diff, 0, tol=1e-3, \
                              method='lm', options={'ftol':1e-3, 'maxiter':12})
    log.debug(embsys, 'scipy.optimize summary %s', sol)
    log.debug(embsys, 'chem potential = %.11g, nelec error = %.11g', \
              sol.x, sol.fun)
    log.debug(embsys, '        ncall = %d, scipy.optimize success: %s', \
              sol.nfev, sol.success)
    return sol.x[0]




##################################################
def dmet_sc_cycle(embsys):
    mol = embsys.mol
    #import scf
#    _diis = scf.diis.DIIS(mol)
    #_diis.space = 6

    vfit_mf = embsys.build_(mol)[0]
    vfit_ci = embsys.vfit_ci_method(mol, embsys)
    # to guarantee correct number of electrons, calculate embedded energy
    # before calling update_embsys
    e_tot, e_corr, nelec = embsys.assemble_frag_energy(mol)
    log.info(embsys, 'macro iter = 0, e_tot = %.12g, nelec = %g', e_tot, nelec)
    vfit_mf_old = vfit_mf
    vfit_ci_old = vfit_ci
    e_tot_old = e_tot
    e_corr_old = e_corr

    for icyc in range(embsys.max_iter):

        #log.debug(embsys, '  HF energy = %.12g', embsys.entire_scf.e_tot)
        vfit_mf = embsys.vfit_mf_method(mol, embsys)
        embsys.update_embsys(mol, vfit_mf, vfit_ci)

        embsys.vfit_ci = embsys.vfit_ci_method(mol, embsys)

        # to guarantee correct number of electrons, calculate embedded energy
        # before calling update_embsys
        e_tot, e_corr, nelec = embsys.assemble_frag_energy(mol)

        dv = vfit_mf[0] - vfit_mf_old[0]
        dv[numpy.diag_indices(dv.shape[0])] = 0
        dv = numpy.linalg.norm(dv)

        log.info(embsys, 'macro iter = %d, e_tot = %.12g, e_tot(corr) = %.12g, nelec = %g, dv = %g', \
                 icyc+1, e_tot, e_corr, nelec, dv)
        de = abs(1-e_tot_old/e_tot)
        decorr = abs(e_corr-e_corr_old)
        log.info(embsys, '                 delta_e = %.12g, (~ %g%%), delta_e(corr) = %.12g', \
                 e_tot-e_tot_old, de * 100, decorr)

        log.debug(embsys, 'CPU time %.8g' % time.clock())

        if dv < 1e-5:
            break

        vfit_mf_old = vfit_mf
        vfit_ci_old = vfit_ci
        e_tot_old = e_tot
        e_corr_old = e_corr

    return e_tot, vfit_mf, vfit_ci




#####################################################################


def fit_solver(embsys, fock0, nocc, nimp, dm_ref_alpha):
    #fitp = DmFitObj(fock0, nocc, nimp, dm_ref_alpha, v_V, dm_V)
    def _decompress(vfit):
        idx = numpy.tril_indices(nimp)
        v1 = numpy.zeros((nimp, nimp))
        v1[idx] = vfit
        v1[idx[1],idx[0]] = vfit
        return v1

    ec = [0, 0]
    def diff_dm(vfit):
        f = fock0.copy()
        f[:nimp,:nimp] += _decompress(vfit)
        e, c = scipy.linalg.eigh(f)
        dm0 = numpy.dot(c[:nimp,:nocc], c[:nimp,:nocc].T)
        ddm = dm0 - dm_ref_alpha[:nimp,:nimp]
        ec[:] = (e, c)
        return ddm.flatten()

    def jac_ddm(vfit, *args):
        e, c = ec
        x = mat_v_to_mat_dm1(e, c, nocc, nimp, nimp)
        usymm = symm_trans_mat_for_hermit(nimp)
        nn = usymm.shape[0]
        return numpy.dot(x.reshape(-1,nn), usymm)

    x = scipy.optimize.leastsq(diff_dm, numpy.zeros(nimp*(nimp+1)/2),
                               Dfun=jac_ddm, ftol=1e-8)[0]
    log.debug(embsys, 'ddm %s', diff_dm(x))
    return _decompress(x)


def mat_v_to_mat_dm1(e, c, nocc, nd, nv):
    '''in AO representation, DM1 = X * V'''
    nmo = e.shape[0]
    nvir = nmo - nocc

    eia = 1 / (e[:nocc].reshape(nocc,1) - e[nocc:])
    tmpcc = numpy.einsum('ik,jk->kij', c[:nd], c[:nv])
    v = tmpcc.reshape(nmo,nd*nv)
    _x = reduce(numpy.dot, (v[nocc:].T, eia.T, v[:nocc]))
    _x = _x.reshape(nd,nv,nd,nv)
    x0 = _x.transpose(0,2,1,3)
    x1 = x0.transpose(1,0,3,2)

    return x0 + x1

def symm_trans_mat_for_hermit(n):
    # transformation matrix to remove the antisymmetric mode
    # usym is the symmetrized vector corresponding to symmetric component.
    usym = numpy.zeros((n*n, n*(n+1)/2))
    for i in range(n):
        for j in range(i):
            usym[i*n+j,i*(i+1)/2+j] = 1
            usym[j*n+i,i*(i+1)/2+j] = 1
        usym[i*n+i,i*(i+1)/2+i] = 1
    return usym




if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf

    mol = gto.Mole()
    mol.verbose = 5

    b1 = 1.1
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
    mf.verbose = 0
    print mf.scf()

    embsys = EmbSys(mol, mf)
    embsys.basidx = [0,1]
    print embsys.scdmet() # -18.0179909364



