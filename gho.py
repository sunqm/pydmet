#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

'''
generalized hybrid orbital
ref:
    J. Am. Chem. Soc., 114, 1606
    J. Phys. Chem. A, 102, 4714
'''

import numpy
import scipy.linalg
from pyscf import gto
from pyscf.lib import logger as log
import dmet_hf


class RHF(dmet_hf.RHF):
    def decompose_den_mat(self, dm_orth):
        return decompose_den_mat(self, dm_orth*.5, self.bas_on_frag, self.num_bath)
    def decompose_orbital(self, mo_orth):
        return decompose_orbital(self, mo_orth, self.bas_on_frag, self.num_bath)

##################################################
# scf for impurity

    def get_init_guess(self, mol):
        log.debug(self, 'init guess based on entire MO coefficients')
        s = self.entire_scf.get_ovlp(mol)
        eff_scf = self.entire_scf
        entire_scf_dm = eff_scf.make_rdm1(eff_scf.mo_coeff, eff_scf.mo_occ)
        env_orb = numpy.dot(self.orth_coeff, self.env_orb)
        dm_env = numpy.dot(env_orb, env_orb.T.conj()) * 2
        cs = numpy.dot(self.impbas_coeff.T.conj(), s)
        dm = reduce(numpy.dot, (cs, entire_scf_dm-dm_env, cs.T.conj()))
        hf_energy = 0
        return dm

    def mat_ao2impbas(self, mat):
        c = self.impbas_coeff
        mat_emb = reduce(numpy.dot, (c.T.conj(), mat, c))
        return mat_emb

    def mat_orthao2impbas(self, mat):
        a = numpy.dot(self.imp_site.T, mat[self.bas_on_frag])
        b = numpy.dot(self.bath_orb.T, mat)
        ab = numpy.vstack((a,b))
        a = numpy.dot(ab[:,self.bas_on_frag], self.imp_site)
        b = numpy.dot(ab, self.bath_orb)
        return numpy.hstack((a,b))

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        h1e = self.mat_ao2impbas(self.entire_scf.get_hcore(mol)) \
                + self._vhf_env
        return h1e

    def get_ovlp(self, mol=None):
        if mol is None:
            mol = self.mol
        s1e = self.mat_ao2impbas(self.entire_scf.get_ovlp(mol))
        return s1e

    def get_occ(self, mo_energy, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = self.nelectron / 2
        mo_occ[:nocc] = 2
        if nocc < mo_occ.size:
            log.debug(self, 'HOMO = %.12g, LUMO = %.12g,', \
                      mo_energy[nocc-1], mo_energy[nocc])
        else:
            log.debug(self, 'HOMO = %.12g,', mo_energy[nocc-1])
        log.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff_on_imp
        if mo_occ is None:
            mo_occ = self.mo_occ
        nbf = mo_coeff.shape[0]
        mo = mo_coeff[:,mo_occ>0]
        dm = numpy.dot(mo, mo.T.conj()) * 2
        #log.debug(self, 'density.diag = %s', dm.diagonal())
        return dm

    def eri_on_impbas(self, mol):
        if self.entire_scf._eri is not None:
            eri = ao2mo.incore.full(self.entire_scf._eri, self.impbas_coeff)
        else:
            eri = ao2mo.direct.full_iofree(mol, self.impbas_coeff)
        return eri

    def release_eri(self):
        self._eri = None


    def get_veff(self, mol, dm, dm_last=0, vhf_last=0):
        if self._eri is None:
            self._eri = self.eri_on_impbas(mol)
        vj, vk = _vhf.vhf_jk_incore_o2(self._eri, dm)
        vhf = vj - vk * .5
        return vhf

#    def get_veff(self, mol, dm, dm_last=0, vhf_last=0):
#        dm = reduce(numpy.dot, (self.impbas_coeff, dm, \
#                                self.impbas_coeff.T))
#        vhf_ao = scf.hf.RHF.get_veff(self.entire_scf, self.mol, dm)
#        return self.mat_ao2impbas(vhf_ao)

    def frag_non_symm_projector(self, s1e):
        '''project operator of fragment. Its definition is not unique
        Non-symmtric <bra|, |ket> leads
            P = |mu> <mu'|     where <mu'| = S^{-1}<mu|;
            P^A_{ij} = \sum_{k\in A} S_{ik} S^{-1}_kj}
        Lowdin-symmetric orthogonalized <bra|, |ket> gives rise to
            P^A_{ij} = \sum_{k\in A} S^{-1/2}_{ik} S^{-1/2}_{kj}
        The non-symmetric projection is consistent with the Mulliken pop.
            chg^A = Tr(D P^A S) = \sum_{i\in A,j} D_{ij}S_{ji}
        Here the project adopts the non-symmetric form'''
        nimp = len(self.bas_on_frag)
        s_inv = numpy.linalg.inv(s1e)
        return numpy.dot(s1e[:,:nimp], s_inv[:nimp,:])

    def calc_frag_elec_energy(self, mol, vhf, dm):
        ''' Calculate meanfiled fragment electronic energy:
            E = <\Psi|H|\Psi> = \sum_A <\Psi^A|H|\Psi> = \sum_A E^A
            E^A = <\Psi^A|H|\Psi>
                = \sum_\mu <\psi^A|h|\mu> D_{\mu\psi^A}
                + 1/2 \sum_\mu (J-K)_{\psi^A\mu} D_{\mu\psi^A}
* When the occ_env_cutoff is large, the so obtained fragment sites are not
complete basis sets for electron density. In such fragment basis sets, the
density matrix cannot represent all the electrons.  The number of electron
represented by the density matrix (= trace(D) = \sum elec_frag) would be less
then the total number of electron.  The sum of the fragment electronic energy
which does not include all electrons would be less than the total electronic
energy.
* If vhf is 0 and dm is post-HF density matrix, the "fragment energy"
will be the sum of one electron energy and the mean-filed energy of
environment two-electron part'''
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        proj = self.frag_non_symm_projector(s1e)
        dm_frag = numpy.dot(dm, proj)

        # ne = Tr(D S)
        # ne^A = Tr(D P^A S)
        nelec_frag = numpy.einsum('ij,ji', dm_frag, s1e)
        log.info(self, 'number of electrons in fragment = %.15g', \
                 nelec_frag.real)

        e = numpy.einsum('ij,ji', dm_frag, h1e-self._vhf_env) \
          + numpy.einsum('ij,ji', dm_frag, vhf+self._vhf_env) * .5
        log.info(self, 'fragment electronic energy = %.15g', e.real)
        log.debug(self, ' ~ total energy (non-variational) = %.15g', \
                  numpy.einsum('ij,ji', dm, h1e) \
                  + numpy.einsum('ij,ji', dm, vhf)*.5 \
                  + self.energy_by_env)
        return e.real, nelec_frag.real

    def imp_scf(self):
        self.orth_coeff = self.get_orth_ao(self.mol)

        self.dump_flags()
        self.init_dmet_scf(self.mol)
        dd = self.dets_ovlp(self.mol, self.impbas_coeff)
        log.info(self, 'overlap of determinants before SCF = %.15g', dd)

        self.scf_conv, self.hf_energy, self.mo_energy, self.mo_occ, \
                self.mo_coeff_on_imp \
                = scf.hf.kernel(self, self.conv_tol, dump_chk=False)

        log.info(self, 'impurity MO energy')
        for i in range(self.mo_energy.size):
            if self.mo_occ[i] > 0:
                log.info(self, 'impurity occupied MO %d energy = %.15g occ=%g', \
                         i+1, self.mo_energy[i], self.mo_occ[i])
            else:
                log.info(self, 'impurity virtual MO %d energy = %.15g occ=%g', \
                         i+1, self.mo_energy[i], self.mo_occ[i])

        #e_nuc = self.energy_nuc(self.mol)
        #log.log(self, 'impurity sys nuclear repulsion = %.15g', e_nuc)
        if self.scf_conv:
            log.log(self, 'converged impurity sys electronic energy = %.15g', \
                    self.hf_energy)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'electronic energy = %.15g after %d cycles.', \
                    self.hf_energy, self.max_cycle)

        # mo_coeff_on_imp based on embedding basis + bath
        # mo_coeff based on AOs
        self.mo_coeff = numpy.dot(self.impbas_coeff, self.mo_coeff_on_imp)
        s = self.entire_scf.get_ovlp(self.mol)
        mo0 = self.entire_scf.mo_coeff[:,self.entire_scf.mo_occ>0]
        mo1 = numpy.hstack((self.mo_coeff[:,self.mo_occ>0], \
                            numpy.dot(self.orth_coeff, self.env_orb)))
        norm = 1/numpy.sqrt(numpy.linalg.det( \
                reduce(numpy.dot, (mo1.T.conj(), s, mo1))))
        ovlp = numpy.linalg.det(reduce(numpy.dot, (mo0.T.conj(), s, mo1))) * norm
        # ovlp**2 because of the beta orbital contribution
        log.info(self, 'overlap of determinants after SCF = %.15g', (ovlp**2))

        dm = self.make_rdm1(self.mo_coeff_on_imp, self.mo_occ)
        vhf = self.get_veff(self.mol, dm)
        self.e_frag, self.n_elec_frag = \
                self.calc_frag_elec_energy(self.mol, vhf, dm)
        log.log(self, 'fragment electronic energy = %.15g', self.e_frag)
        log.log(self, 'fragment electron number = %.15g', self.n_elec_frag)
        #self.frag_mulliken_pop()
        return self.e_frag

    def energy_nuc(self, mol):
        e = 0
        for j, ja in enumerate(self.imp_atoms):
            q2 = mol.charge_of_atm(ja)
            r2 = numpy.array(mol.coord_of_atm(ja))
            for i in range(j):
                ia = self.imp_atoms[i]
                q1 = mol.charge_of_atm(ia)
                r1 = numpy.array(mol.coord_of_atm(ia))
                r = numpy.linalg.norm(r1-r2)
                e += q1 * q2 / r
        for j in range(mol.natm):
            if j not in self.imp_atoms:
                q2 = mol.charge_of_atm(j)
                r2 = numpy.array(mol.coord_of_atm(j))
                for i in self.imp_atoms:
                    q1 = mol.charge_of_atm(i)
                    r1 = numpy.array(mol.coord_of_atm(i))
                    r = numpy.linalg.norm(r1-r2)
                    e += q1 * q2 / r
        return e

    def frag_mulliken_pop(self):
        dmet_hf.RHF.frag_mulliken_pop(self)
        if self.num_bath != -1:
            self.diff_dm()

    def diff_dm(self):
        # diff between the SCF DM and DMET DM for fragment block
        mol = self.mol
        s = self.entire_scf.get_ovlp(self.mol)
        c_inv = numpy.dot(self.orth_coeff.T, s)
        eff_scf = self.entire_scf
        mo = numpy.dot(c_inv, eff_scf.mo_coeff)
        dm0 = eff_scf.make_rdm1(mo, eff_scf.mo_occ)
        # in case impurity sites are not the AO orbitals
        mo = reduce(numpy.dot, (c_inv, self.impbas_coeff, self.mo_coeff_on_imp))
        dm1 = numpy.dot(mo*self.mo_occ, mo.T)
        dm1 += numpy.dot(self.env_orb, self.env_orb.T)*2
        norm = numpy.linalg.norm((dm0-dm1)[self.bas_on_frag][:,self.bas_on_frag])
        log.info(self, 'norm(diff of imp-DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm, norm/len(self.bas_on_frag))
        norm = numpy.linalg.norm(dm0-dm1)
        log.info(self, 'norm(diff of entire DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm, norm/dm0.shape[0])

        bas_off_frag = [i for i in range(s.shape[0]) \
                        if i not in self.bas_on_frag]
        p = reduce(numpy.dot, (c_inv, self.impbas_coeff, self.impbas_coeff.T, c_inv.T))
        norm0 = numpy.linalg.norm((numpy.dot(dm0,p)-dm0)[self.bas_on_frag][:,bas_off_frag])
        norm1 = numpy.linalg.norm((dm0-dm1)[self.bas_on_frag][:,bas_off_frag])
        log.info(self, 'before SCF norm(diff off-diagonal DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm0, norm0/numpy.sqrt(len(self.bas_on_frag)*len(bas_off_frag)))
        log.info(self, 'after SCF norm(diff off-diagonal DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm1, norm1/numpy.sqrt(len(self.bas_on_frag)*len(bas_off_frag)))
        norm1 = numpy.linalg.norm((dm0-dm1)[self.bas_on_frag])
        log.info(self, 'after SCF norm(diff frag-band DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm1, norm1/numpy.sqrt(len(self.bas_on_frag)*s.shape[0]))


# scf for impurity end
##################################################

    def set_embsys(self, atm_lst):
        assert(max(atm_lst) < self.mol.natm)
        self.imp_atoms = atm_lst

    def set_bath(self, atm_lst):
        assert(max(atm_lst) < self.mol.natm)
        self.imp_atoms = filter(lambda n: n not in atm_lst, \
                                range(self.mol.natm))

    def append_embsys(self, atm_lst):
        assert(max(atm_lst) < self.mol.natm)
        self.imp_atoms = set(list(atm_lst) + list(self.imp_atoms))

    def append_bath(self, atm_lst):
        assert(max(atm_lst) < self.mol.natm)
        if self.imp_atoms == []:
            self.set_bath(atm_lst)
        else:
            self.imp_atoms = filter(lambda n: n not in atm_lst, \
                                    self.imp_atoms)


##################################################

    def cons_impurity_basis(self):
        a = numpy.dot(self.orth_coeff[:,self.bas_on_frag], self.imp_site)
        b = numpy.dot(self.orth_coeff, self.bath_orb)
        if self.orth_imp_to_env:
            return self.suborth_imp_to_env(numpy.hstack((a,b)))
        else:
            return numpy.hstack((a,b))

    def suborth_imp_to_env(self, impbas_coeff):
        c = numpy.hstack((numpy.dot(self.orth_coeff, self.env_orb), \
                          impbas_coeff))
        s = self.entire_scf.get_ovlp(self.mol)
        t = schmidt_orth_coeff(reduce(numpy.dot, (c.T.conj(), s, c)))
        off = self.env_orb.shape[1]
        impbas_coeff = numpy.dot(c, t)[:,off:]
        return impbas_coeff

## scheme 1: GHO is based on the AO of boundary atom and thereby
## non-orthogonal to the impurity sys
#    def gho_on_lowdin_aos1(self, hyb, gho_atm_lst):
#        idx = gho.gho_index(self.mol, gho_atm_lst[0])
#        nbf = self.mol.num_NR_function()
#        v = numpy.zeros((nbf, 4))
#        v[idx] = hyb
#        return numpy.mat(v)
## scheme 2: GHO is based on the orthogonalized AOs
#    def gho_on_lowdin_aos2(self, hyb, gho_atm_lst):
#        idx = gho.gho_index(self.mol, gho_atm_lst[0])
#        v = numpy.zeros((nbf, 4))
#        v[idx] = hyb
#
#        # fix the phase of s and p
#        s = self.entire_scf.get_ovlp(self.mol)
#        sh = (numpy.mat(s) * v)[idx,:]
#        if (hyb[0,0] * sh[0,0]) * (hyb[1:,0].T * sh[1:,0]) < 0:
#            hyb[0] = -hyb[0]
#            v = numpy.dot(self.orth_coeff[:,idx], hyb)
#        # based on AO basis
#        return v

    def set_gho_pseudo_bath(self, gho_atm_lst, inc_1s=False):
        assert(self.mol.pure_symbol_of_atm(gho_atm_lst[0]) == 'C')
        self.append_bath(gho_atm_lst)
        self.num_bath = 1
        if inc_1s:
            for i, s in enumerate(self.mol.spheric_labels()):
                if s[0] == gho_atm_lst[0] and s[2] == '1s':
                    self.imp_basidx = [i]
                    break

        def cons_impbas():
            import gho
            log.info(self, 'replace bath orbital with GHOs')
            g = gho.GHO()
            gho_orb = g.hybrid_coeff(self.mol, gho_atm_lst)
            gho_idx = gho.gho_index(self.mol, gho_atm_lst[0])
            ovlp = numpy.dot(self.bath_orb[gho_idx,:].T.conj(), gho_orb)
            for i,c in enumerate(ovlp):
                log.debug(self, '<bath_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
            p_hybs = numpy.dot(ovlp.T,ovlp).diagonal()
            log.debug(self, '<gho_i|bath><bath|gho_i> = %s', str(p_hybs))
            u, w, v = numpy.linalg.svd(ovlp)
            log.debug(self, 'SVD <gho|bath> = %s', str(w))

            if self.env_orb.shape[1] > 0:
                ovlp = numpy.dot(self.env_orb[gho_idx,:].T.conj(), gho_orb)
                for i,c in enumerate(ovlp):
                    log.debug(self, '<env_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
                p_hybs = numpy.dot(ovlp.T,ovlp).diagonal()
                log.debug(self, '<gho_i|env><env|gho_i> = %s', str(p_hybs))
                u, w, v = numpy.linalg.svd(ovlp)
                log.debug(self, 'SVD <gho|env> = %s', str(w))

            coord0 = self.mol.coord_of_atm(gho_atm_lst[0])
            dists = [numpy.linalg.norm(self.mol.coord_of_atm(i)-coord0) \
                     for i in self.imp_atoms]
            bondatm = self.imp_atoms[numpy.argmin(dists)]
            bath1 = self.bath_orb[gho_idx,0]/numpy.linalg.norm(self.bath_orb[gho_idx,0])
            log.debug(self, 'bath_1 hybrid = sp^%4.3f, angle to bond = %.6g', \
                      gho.sp_hybrid_level(bath1), \
                      gho.angle_to_bond(self.mol, gho_atm_lst[0], \
                                        bondatm, bath1))
            log.debug(self, 'GHO-active hybrid = sp^%4.3f, angle to bond = %.6g', \
                      gho.sp_hybrid_level(gho_orb[:,0]), \
                      gho.angle_to_bond(self.mol, gho_atm_lst[0], \
                                        bondatm, gho_orb[:,0]))
            cosovlp = numpy.dot(bath1[1:4],gho_orb[1:4,0]) \
                    / numpy.linalg.norm(bath1[1:4]) \
                    / numpy.linalg.norm(gho_orb[1:4,0])
            log.debug(self, 'angle between GHO and bath_1 = %.6g', \
                      numpy.arccos(cosovlp))

            a = numpy.dot(self.orth_coeff[:,self.bas_on_frag], self.imp_site)
            b = numpy.dot(self.orth_coeff[:,gho_idx], gho_orb[:,:1])
            impbas_coeff = numpy.hstack((a,b))
            if self.orth_imp_to_env:
                impbas_coeff = self.suborth_imp_to_env(impbas_coeff)
            return impbas_coeff
        self.cons_impurity_basis = cons_impbas


##################################################
    def bath_delta_nuc_vhf(self, mol):
        # nuclear attraction matrix not in fragment
        nbf = mol.num_NR_function()
        bnuc = numpy.zeros((nbf,nbf))
        for ia in range(mol.natm):
            if ia not in self.imp_atoms:
                mol.set_rinv_orig(mol.coord_of_atm(ia))
                chg = mol.charge_of_atm(ia)
                bnuc += -chg * self.entire_scf.get_ovlp(self.mol)

        bnuc = self.mat_ao2impbas(bnuc)
        print bnuc + self._vhf_env


    def dets_ovlp(self, mol, orbs):
        '''det(<i*|i>):  |i*> = P|i>,  P = |x>S^{-1}<x|'''
        mo0 = self.entire_scf.mo_coeff[:,self.entire_scf.mo_occ>0]
        s = self.entire_scf.get_ovlp(self.mol)
        orbs1 = numpy.hstack((orbs, numpy.dot(self.orth_coeff, self.env_orb)))
        tmp = reduce(numpy.dot, (orbs1.T.conj(), s, orbs1))
        proj = reduce(numpy.dot, (orbs1, numpy.linalg.inv(tmp), \
                                  orbs1.T.conj()))
        ovlp = reduce(numpy.dot, (mo0.T.conj(), s, proj, s, mo0))
        # <ovlp>**2 because of the beta orbital contribution
        return numpy.linalg.det(ovlp)**2


##################################################
class UHF(dmet_hf.UHF):
# **** impurity SCF ****
    def check_dm_converge(self, dm, dm_last, conv_tol):
        delta_dm = abs(dm[0]-dm_last[0]).sum() + abs(dm[1]-dm_last[1]).sum()
        dm_change = delta_dm/(abs(dm_last[0]).sum()+abs(dm_last[1]).sum())
        log.info(self, '          sum(delta_dm)=%g (~ %g%%)\n', \
                 delta_dm, dm_change*100)
        return dm_change < conv_tol*1e2

    def get_init_guess(self, mol):
        log.debug(self, 'init guess based on entire MO coefficients')
        s = self.entire_scf.get_ovlp(self.mol)
        eff_scf = self.entire_scf
        entire_scf_dm = eff_scf.make_rdm1(eff_scf.mo_coeff, eff_scf.mo_occ)
        env_a = numpy.dot(self.orth_coeff, self.env_orb[0])
        env_b = numpy.dot(self.orth_coeff, self.env_orb[1])
        dm_a = numpy.dot(env_a, env_a.T.conj())
        dm_b = numpy.dot(env_b, env_b.T.conj())
        cs_a = numpy.dot(self.impbas_coeff[0].T.conj(), s)
        cs_b = numpy.dot(self.impbas_coeff[1].T.conj(), s)
        dm_a = reduce(numpy.dot, (cs_a, entire_scf_dm[0]-dm_a, cs_a.T.conj()))
        dm_b = reduce(numpy.dot, (cs_b, entire_scf_dm[1]-dm_b, cs_b.T.conj()))
        hf_energy = 0
        return numpy.array((dm_a,dm_b))

#    def eri_on_impbas(self, mol):
#        if self.entire_scf._eri is not None:
#            eri = ao2mo.incore.full(self.entire_scf._eri, self.impbas_coeff)
#        else:
#            eri = ao2mo.direct.full_iofree(mol, self.impbas_coeff)
#        return eri

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0):
        dm_a = reduce(numpy.dot, (self.impbas_coeff[0], dm[0], \
                                  self.impbas_coeff[0].T))
        dm_b = reduce(numpy.dot, (self.impbas_coeff[1], dm[1], \
                                  self.impbas_coeff[1].T))
        dm_ao = numpy.array((dm_a, dm_b))
        vhf_ao = scf.hf.UHF.get_veff(self.entire_scf, self.mol, dm_ao)
        return self.mat_ao2impbas(vhf_ao)

    def get_hcore(self, mol):
        h1e = self.mat_ao2impbas(scf.hf.RHF.get_hcore(mol))
        return (h1e[0]+self._vhf_env[0], h1e[1]+self._vhf_env[1])

    def eig(self, fock, s):
        e_a, c_a = scipy.linalg.eigh(fock[0], s[0])
        e_b, c_b = scipy.linalg.eigh(fock[1], s[1])
        return (e_a,e_b), (c_a,c_b)

    def get_fock(self, h1e, s1e, vhf, dm, cycle=-1, adiis=None):
        f = (h1e[0]+vhf[0], h1e[1]+vhf[1])
        if 0 <= cycle < self.diis_start_cycle-1:
            f = (scf.hf.damping(s1e[0], dm[0], f[0], self.damp_factor), \
                 scf.hf.damping(s1e[1], dm[1], f[1], self.damp_factor))
            f = (scf.hf.level_shift(s1e[0],dm[0],f[0],self.level_shift_factor), \
                 scf.hf.level_shift(s1e[1],dm[1],f[1],self.level_shift_factor))
        elif 0 <= cycle:
            fac = self.level_shift_factor \
                    * numpy.exp(self.diis_start_cycle-cycle-1)
            f = (scf.hf.level_shift(s[0], d[0], f[0], fac), \
                 scf.hf.level_shift(s[1], d[1], f[1], fac))

        if adiis is not None and cycle >= self.diis_start_cycle:
            f = adiis.update(s1e, dm, f)
            f = (f[:h1e[0].size].reshape(h1e[0].shape), \
                 f[h1e[0].size:].reshape(h1e[1].shape))
        return f

    def get_occ(self, mo_energy, mo_coeff=None):
        mo_occ = [numpy.zeros_like(mo_energy[0]), \
                  numpy.zeros_like(mo_energy[1])]
        mo_occ[0][:self.nelectron_alpha] = 1
        mo_occ[1][:self.nelectron_beta]  = 1
        if self.nelectron_alpha < mo_energy[0].size:
            log.debug(self, 'alpha nocc = %d, HOMO = %.12g, LUMO = %.12g,', \
                      self.nelectron_alpha, \
                      mo_energy[0][self.nelectron_alpha-1], \
                      mo_energy[0][self.nelectron_alpha])
        else:
            log.debug(self, 'alpha nocc = %d, HOMO = %.12g, no LUMO,', \
                      self.nelectron_alpha, \
                      mo_energy[0][self.nelectron_alpha-1])
        log.debug(self, '  mo_energy = %s', mo_energy[0])
        log.debug(self, 'beta  nocc = %d, HOMO = %.12g, LUMO = %.12g,', \
                  self.nelectron_beta, \
                  mo_energy[0][self.nelectron_beta-1], \
                  mo_energy[0][self.nelectron_beta])
        log.debug(self, '  mo_energy = %s', mo_energy[1])
        return mo_occ

    def make_rdm1(self, mo_coeff, mo_occ):
        mo_a = mo_coeff[0][:,mo_occ[0]>0]
        mo_b = mo_coeff[1][:,mo_occ[1]>0]
        dm_a = numpy.dot(mo_a, mo_a.T.conj())
        dm_b = numpy.dot(mo_b, mo_b.T.conj())
        #log.debug(self, 'alpha density.diag = %s', dm_a.diagonal())
        #log.debug(self, 'beta  density.diag = %s', dm_b.diagonal())
        return (dm_a,dm_b)

    def imp_scf(self):
        self.orth_coeff = self.get_orth_ao(self.mol)

        self.dump_flags()
        self.build_()

        self.scf_conv, self.hf_energy, self.mo_energy, self.mo_occ, \
                self.mo_coeff_on_imp \
                = scf.hf.kernel(self, self.conv_tol, dump_chk=False)

        def dump_mo_energy(mo_energy, mo_occ, title=''):
            log.info(self, 'impurity %s MO energy', title)
            for i in range(mo_energy.size):
                if mo_occ[i] > 0:
                    log.info(self, 'impurity %s occupied MO %d energy = %.15g', \
                             title, i+1, mo_energy[i])
                else:
                    log.info(self, 'impurity %s virtual MO %d energy = %.15g', \
                             title, i+1, mo_energy[i])
        dump_mo_energy(self.mo_energy[0], self.mo_occ[0], 'alpha')
        dump_mo_energy(self.mo_energy[1], self.mo_occ[1], 'beta')

        if self.scf_conv:
            log.log(self, 'converged impurity sys electronic energy = %.15g', \
                    self.hf_energy)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'electronic energy = %.15g after %d cycles.', \
                    self.hf_energy, self.max_cycle)

#        # mo_coeff_on_imp based on embedding basis + bath
#        # mo_coeff based on AOs
#        c_a = numpy.dot(self.impbas_coeff[0], self.mo_coeff_on_imp[0])
#        c_b = numpy.dot(self.impbas_coeff[1], self.mo_coeff_on_imp[1])
#        self.mo_coeff = (c_a,c_b)
#        s = self.entire_scf.get_ovlp(self.mol)
#        mo0_a = self.entire_scf.mo_coeff[0][:,self.entire_scf.mo_occ[0]>0]
#        mo0_b = self.entire_scf.mo_coeff[1][:,self.entire_scf.mo_occ[1]>0]
#        mo1_a = numpy.hstack((c_a[:,self.mo_occ[0]>0], \
#                              numpy.dot(self.orth_coeff, self.env_orb[0])))
#        mo1_b = numpy.hstack((c_b[:,self.mo_occ[1]>0], \
#                              numpy.dot(self.orth_coeff, self.env_orb[1])))
#        norm = 1/numpy.sqrt( \
#                numpy.linalg.det(reduce(numpy.dot,  (mo1_a.T.conj(),s,mo1_a)))\
#                * numpy.linalg.det(reduce(numpy.dot,(mo1_b.T.conj(),s,mo1_b))))
#        ovlp = numpy.linalg.det(reduce(numpy.dot,  (mo0_a.T.conj(),s,mo1_a))) \
#               * numpy.linalg.det(reduce(numpy.dot,(mo0_b.T.conj(),s,mo1_b)))
#        log.info(self, 'overlap of determinants after SCF = %.15g', abs(ovlp * norm))

        dm = self.make_rdm1(self.mo_coeff_on_imp, self.mo_occ)
        vhf = self.get_veff(self.mol, dm)
        self.e_frag, self.n_elec_frag = \
                self.calc_frag_elec_energy(self.mol, vhf, dm)
        log.log(self, 'fragment electronic energy = %.15g', self.e_frag)
        log.log(self, 'fragment electron number = %.15g', self.n_elec_frag)
        self.frag_mulliken_pop()
        return self.e_frag

    def calc_frag_elec_energy(self, mol, vhf, dm):
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        proj_a = self.frag_non_symm_projector(s1e[0])
        proj_b = self.frag_non_symm_projector(s1e[1])
        dm_frag_a = numpy.dot(dm[0], proj_a)
        dm_frag_b = numpy.dot(dm[1], proj_b)

        # ne = Tr(D S)
        # ne^A = Tr(D P^A S)
        nelec_frag = numpy.einsum('ij,ji', dm_frag_a, s1e[0]) \
                   + numpy.einsum('ij,ji', dm_frag_b, s1e[1])
        log.info(self, 'number of electrons in fragment = %.15g', \
                 nelec_frag.real)

        e = numpy.einsum('ij,ji', dm_frag_a, h1e[0]-self._vhf_env[0]) \
          + numpy.einsum('ij,ji', dm_frag_b, h1e[1]-self._vhf_env[1]) \
          + numpy.einsum('ij,ji', dm_frag_a, vhf[0] + self._vhf_env[0]) * .5 \
          + numpy.einsum('ij,ji', dm_frag_b, vhf[1] + self._vhf_env[1]) * .5
        log.info(self, 'fragment electronic energy = %.15g', e.real)
        log.debug(self, ' ~ total energy (non-variational) = %.15g', \
                  (numpy.einsum('ij,ji', dm[0], h1e[0])
                  +numpy.einsum('ij,ji', dm[1], h1e[1]) \
                  +numpy.einsum('ij,ji', dm[0], vhf[0])*.5 \
                  +numpy.einsum('ij,ji', dm[1], vhf[1])*.5 \
                  +self.energy_by_env))
        return e.real, nelec_frag.real

    def frag_mulliken_pop(self):
        dmet_hf.UHF.frag_mulliken_pop(self)
        # diff between the SCF DM and DMET DM for fragment block
        if self.num_bath != -1:
            self.diff_dm()

    def diff_dm(self):
        mol = self.mol
        s = self.entire_scf.get_ovlp(self.mol)
        c_inv = numpy.dot(self.orth_coeff.T, s)
        eff_scf = self.entire_scf
        mo_a = numpy.dot(c_inv, eff_scf.mo_coeff[0])
        mo_b = numpy.dot(c_inv, eff_scf.mo_coeff[1])
        dm0 = eff_scf.make_rdm1((mo_a,mo_b), eff_scf.mo_occ)
        # in case impurity sites are not the AO orbitals
        mo_a = reduce(numpy.dot, (c_inv, self.impbas_coeff[0], \
                                  self.mo_coeff_on_imp[0]))
        mo_b = reduce(numpy.dot, (c_inv, self.impbas_coeff[1], \
                                  self.mo_coeff_on_imp[1]))
        dm1 = (numpy.dot(mo_a*self.mo_occ[0],mo_a.T),
               numpy.dot(mo_b*self.mo_occ[1],mo_b.T))
        dm1 = (dm1[0] + numpy.dot(self.env_orb[0], self.env_orb[0].T), \
               dm1[1] + numpy.dot(self.env_orb[1], self.env_orb[1].T))
        norm_a = numpy.linalg.norm((dm1[0]-dm0[0])[self.bas_on_frag][:,self.bas_on_frag])
        norm_b = numpy.linalg.norm((dm1[1]-dm0[1])[self.bas_on_frag][:,self.bas_on_frag])
        norm = numpy.sqrt(norm_a**2+norm_b**2)
        log.info(self, 'norm(diff of imp-DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm, norm/numpy.sqrt(2)/len(self.bas_on_frag))
        norm = numpy.sqrt(numpy.linalg.norm(dm0[0]-dm1[0])**2 \
                          +numpy.linalg.norm(dm0[1]-dm1[1])**2)
        log.info(self, 'norm(diff of entire DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm, norm/numpy.sqrt(2)/dm0[0].shape[0])

        bas_off_frag = [i for i in range(s.shape[0]) \
                        if i not in self.bas_on_frag]
        p_a = reduce(numpy.dot, (c_inv, self.impbas_coeff[0],
                                 self.impbas_coeff[0].T, c_inv.T))
        p_b = reduce(numpy.dot, (c_inv, self.impbas_coeff[1],
                                 self.impbas_coeff[1].T, c_inv.T))
        norm0a = numpy.linalg.norm((numpy.dot(dm0[0],p_a)-dm0[0])[self.bas_on_frag][:,bas_off_frag])
        norm0b = numpy.linalg.norm((numpy.dot(dm0[1],p_b)-dm0[1])[self.bas_on_frag][:,bas_off_frag])
        norm0 = numpy.sqrt((norm0a**2+norm0b**2)/2)
        norm1a = numpy.linalg.norm((dm0[0]-dm1[0])[self.bas_on_frag][:,bas_off_frag])
        norm1b = numpy.linalg.norm((dm0[1]-dm1[1])[self.bas_on_frag][:,bas_off_frag])
        norm1 = numpy.sqrt((norm1a**2+norm1b**2)/2)
        log.info(self, 'before SCF norm(diff off-diagonal DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm0, norm0/numpy.sqrt(len(self.bas_on_frag)*len(bas_off_frag)))
        log.info(self, 'after SCF norm(diff off-diagonal DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm1, norm1/numpy.sqrt(len(self.bas_on_frag)*len(bas_off_frag)))
        norm1a = numpy.linalg.norm((dm0[0]-dm1[0])[self.bas_on_frag])
        norm1b = numpy.linalg.norm((dm0[1]-dm1[1])[self.bas_on_frag])
        norm1 = numpy.sqrt((norm1a**2+norm1b**2)/2)
        log.info(self, 'after SCF norm(diff frag-band DM) on orth-AOs = %.9g, RMS = %.9g', \
                 norm1, norm1/numpy.sqrt(len(self.bas_on_frag)*s.shape[0]))


# **** GHO ****
    def set_gho_pseudo_bath(self, gho_atm_lst, inc_1s=False):
        assert(self.mol.pure_symbol_of_atm(gho_atm_lst[0]) == 'C')
        self.append_bath(gho_atm_lst)
        self.num_bath = 1
        if inc_1s:
            for i, s in enumerate(self.mol.spheric_labels()):
                if s[0] == gho_atm_lst[0] and s[2] == '1s':
                    self.imp_basidx = [i]
                    break

        def cons_impbas():
            import gho
            log.info(self, 'replace bath orbital with GHOs')
            g = gho.GHO()
            gho_orb = g.hybrid_coeff(self.mol, gho_atm_lst)
            gho_idx = gho.gho_index(self.mol, gho_atm_lst[0])

            ovlp_a = numpy.dot(self.bath_orb[0][gho_idx,:].T.conj(), gho_orb)
            for i,c in enumerate(ovlp_a):
                log.debug(self, 'alpha <bath_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
            p_hybs = numpy.dot(ovlp_a.T,ovlp_a).diagonal()
            log.debug(self, 'alpha <gho_i|bath><bath|gho_i> = %s', str(p_hybs))
            u, w, v = numpy.linalg.svd(ovlp_a)
            log.debug(self, 'alpha SVD <gho|bath> = %s', str(w))

            ovlp_b = numpy.dot(self.bath_orb[1][gho_idx,:].T.conj(), gho_orb)
            for i,c in enumerate(ovlp_b):
                log.debug(self, 'beta <bath_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
            p_hybs= numpy.dot(ovlp_b.T,ovlp_b).diagonal()
            log.debug(self, 'beta <gho_i|bath><bath|gho_i> = %s', str(p_hybs))
            u, w, v = numpy.linalg.svd(ovlp_b)
            log.debug(self, 'beta SVD <gho|bath> = %s', str(w))

            if self.env_orb[1].shape[1] > 0:
                ovlp_a = numpy.dot(self.env_orb[0][gho_idx,:].T.conj(), gho_orb)
                for i,c in enumerate(ovlp_a):
                    log.debug(self, 'alpha <env_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
                p_hybs = numpy.dot(ovlp_a.T,ovlp_a).diagonal()
                log.debug(self, 'alpha <gho_i|env><env|gho_i> = %s', str(p_hybs))
                u, w, v = numpy.linalg.svd(ovlp_a)
                log.debug(self, 'alpha SVD <gho|env> = %s', str(w))
                ovlp_b = numpy.dot(self.env_orb[1][gho_idx,:].T.conj(), gho_orb)
                for i,c in enumerate(ovlp_b):
                    log.debug(self, 'beta <env_%d|gho_i> = ' % i + ' %10.5f'*4 % tuple(c))
                p_hybs = numpy.dot(ovlp_b.T,ovlp_b).diagonal()
                log.debug(self, 'beta <gho_i|env><env|gho_i> = %s', str(p_hybs))
                u, w, v = numpy.linalg.svd(ovlp_b)
                log.debug(self, 'beta SVD <gho|env> = %s', str(w))

            coord0 = self.mol.coord_of_atm(gho_atm_lst[0])
            dists = [numpy.linalg.norm(self.mol.coord_of_atm(i)-coord0) \
                     for i in self.imp_atoms]
            bondatm = self.imp_atoms[numpy.argmin(dists)]
            bath1a = self.bath_orb[0][gho_idx,0]/numpy.linalg.norm(self.bath_orb[0][gho_idx,0])
            bath1b = self.bath_orb[1][gho_idx,0]/numpy.linalg.norm(self.bath_orb[1][gho_idx,0])
            log.debug(self, 'alpha bath_1 hybrid = sp^%4.3f, angle to bond = %.6g', \
                      gho.sp_hybrid_level(bath1a), \
                      gho.angle_to_bond(self.mol, gho_atm_lst[0], \
                                        bondatm, bath1a))
            log.debug(self, 'beta bath_1 hybrid = sp^%4.3f, angle to bond = %.6g', \
                      gho.sp_hybrid_level(bath1b), \
                      gho.angle_to_bond(self.mol, gho_atm_lst[0], \
                                        bondatm, bath1b))
            log.debug(self, 'GHO-active hybrid = sp^%4.3f, angle to bond = %.6g', \
                      gho.sp_hybrid_level(gho_orb[:,0]), \
                      gho.angle_to_bond(self.mol, gho_atm_lst[0], \
                                        bondatm, gho_orb[:,0]))
            cosovlpa = numpy.dot(bath1a[1:4],gho_orb[1:4,0]) \
                    / numpy.linalg.norm(bath1a[1:4]) \
                    / numpy.linalg.norm(gho_orb[1:4,0])
            cosovlpb = numpy.dot(bath1b[1:4],gho_orb[1:4,0]) \
                    / numpy.linalg.norm(bath1b[1:4]) \
                    / numpy.linalg.norm(gho_orb[1:4,0])
            log.debug(self, 'angle between GHO and bath_1 (alpha, beta) = %.6g, %.6g', \
                      numpy.arccos(cosovlpa), numpy.arccos(cosovlpb))

            a = numpy.dot(self.orth_coeff[:,self.bas_on_frag], self.imp_site[0])
            b = numpy.dot(self.orth_coeff[:,gho_idx], gho_orb[:,:1])
            c = numpy.hstack((a,b))
            impbas_coeff = numpy.array((c,c))
            if self.orth_imp_to_env:
                impbas_coeff = self.suborth_imp_to_env(impbas_coeff)
            return impbas_coeff
        self.cons_impurity_basis = cons_impbas

    def dets_ovlp(self, mol, orbs):
        '''det(<i*|i>):  |i*> = P|i>,  P = |x>S^{-1}<x|'''
        mo_a = self.entire_scf.mo_coeff[0][:,self.entire_scf.mo_occ[0]>0]
        mo_b = self.entire_scf.mo_coeff[1][:,self.entire_scf.mo_occ[1]>0]
        s = self.entire_scf.get_ovlp(self.mol)
        orbsa = numpy.hstack((orbs[0], numpy.dot(self.orth_coeff, self.env_orb[0])))
        orbsb = numpy.hstack((orbs[1], numpy.dot(self.orth_coeff, self.env_orb[1])))
        sinva = numpy.linalg.inv(reduce(numpy.dot, (orbsa.T, s, orbsa)))
        sinvb = numpy.linalg.inv(reduce(numpy.dot, (orbsb.T, s, orbsb)))
        proja = reduce(numpy.dot, (orbsa, sinva, orbsa.T))
        projb = reduce(numpy.dot, (orbsb, sinvb, orbsb.T))
        ovlpa = reduce(numpy.dot, (mo_a.T.conj(), s, proja, s, mo_a))
        ovlpb = reduce(numpy.dot, (mo_b.T.conj(), s, projb, s, mo_b))
        # <ovlp>**2 because of the beta orbital contribution
        return numpy.linalg.det(ovlpa)*numpy.linalg.det(ovlpb)








##################################################
# ref. Theor. Chem. Acc. 104, 336
def distance_two_dots(r0, r1):
    return numpy.linalg.norm(r0-r1)

def tetrahedron_volume_by_vertic(coords):
    '''1/6 * |det(a-b,b-c,c-d)|'''
    s = numpy.array(coords)
    s3 = numpy.reshape((s[0] - s[1], s[1] - s[2], s[2] - s[3]), (3,3))
    return abs(numpy.linalg.det(s3)) / 6

def triangle_area_by_vertice(coords):
    l1 = distance_two_dots(coords[0], coords[1])
    l2 = distance_two_dots(coords[0], coords[2])
    l3 = distance_two_dots(coords[1], coords[2])
    s = (l1 + l2 + l3) / 2
    return numpy.sqrt(s * (s - l1) * (s - l2) * (s - l3))

def tetrahedron_height(coord_top, coords):
    v = tetrahedron_volume_by_vertic([coord_top] + list(coords))
    s = triangle_area_by_vertice(coords)
    return 3 * v / s

def perp_vec(vec1, vec2):
    v = numpy.cross(vec1, vec2)
    norm = numpy.linalg.norm(v)
    if norm > 1e-12:
        return v / norm
    elif vec1[0] > 1e-12:
        return perp_vec(vec1, (vec1[0], -vec1[2], vec1[1]))
    elif vec1[1] > 1e-12:
        return perp_vec(vec1, (-vec1[2], vec1[1], vec1[0]))
    else:
        return perp_vec(vec1, (-vec1[1], vec1[0], vec1[2]))

def lowdin_orth_coeff(s):
    ''' new basis is |mu> c^{lowdin}_{mu i} '''
    e, v, info = lapack.dsyev(s)
    return numpy.dot(v/numpy.sqrt(e), v.T.conj())

class GHO:
    def s_component(self, coords):
        def stretch_vec_to_unit(r):
            norm = distance_two_dots(r, coords[0])
            return (r - coords[0]) / norm
        runit = map(stretch_vec_to_unit, coords[1:4])
        p = perp_vec(runit[1] - runit[0], runit[2] - runit[0])
        l = numpy.dot(p, runit[0])
        if l < 0:
            l = -l
        else:
            p = -p
        return numpy.sqrt(l/(l+1)), p

    def h_matrix(self, s):
        '''hyb1 ~ h_mat[:,1]'''
        p = numpy.sqrt(1 - s**2)
        s2 = 1 / numpy.sqrt(2)
        s3 = 1 / numpy.sqrt(3)
        s6 = s2 * s3
        h_b = numpy.array(((s,  p*s3,  p*s3,  p*s3), \
                           (p, -s*s3, -s*s3, -s*s3), \
                           (0, numpy.sqrt(2./3), -s6, -s6), \
                           (0, 0, s2, -s2)))
        return h_b

    def hybrid_coeff(self, mol, gho_atm_lst):
        '''determine the hybrid orbitals via geom'''
        coords = numpy.array(map(lambda i: mol.coord_of_atm(i),
                                 gho_atm_lst[:4]))
        log.debug(mol, 'gho_atm_lst %s',
                  map(lambda i: (i, mol.symbol_of_atm(i)), \
                      gho_atm_lst[:4]))
        s, x = self.s_component(coords)
        z = perp_vec(x, coords[1] - coords[0])
        y = perp_vec(z, x)
        b = numpy.array(((1,    0,    0,    0), \
                         (0, x[0], y[0], z[0]), \
                         (0, x[1], y[1], z[1]), \
                         (0, x[2], y[2], z[2])))
        h = self.h_matrix(s)
        t = numpy.dot(b, h)  # t is unitary matrix
        h = numpy.array(t)
        log.debug(mol, 'GHO matrix')
        log.debug(mol, '2s  %12.5f %12.5f %12.5f %12.5f', *h[0])
        log.debug(mol, '2px %12.5f %12.5f %12.5f %12.5f', *h[1])
        log.debug(mol, '2py %12.5f %12.5f %12.5f %12.5f', *h[2])
        log.debug(mol, '2pz %12.5f %12.5f %12.5f %12.5f', *h[3])
        return t
##################################################

##################################################
# ref. J. Am. Chem. Soc., 114, 1606
    def pre_sp(self, coords):
        '''un-normalized hybrid orbital coeffs'''
        p = perp_vec(coords[2] - coords[1], coords[3] - coords[1])
        if numpy.dot(p, coords[0]-coords[1]) < 0:
            p = -p
        return numpy.array((tetrahedron_height(coords[0], coords[1:4]), \
                           p[0], p[1], p[2]))

# coords: (center_atm, three_linked_atms, bond_atm)
    def hybrid_coeff_by_coords(self, coords):
        '''determine the hybrid orbitals via geom'''
        rs = numpy.array(coords)
        h = []
        h.append(self.pre_sp((rs[0], rs[2], rs[3], rs[4], rs[1])))
        h.append(self.pre_sp((rs[0], rs[1], rs[3], rs[4], rs[2])))
        h.append(self.pre_sp((rs[0], rs[1], rs[2], rs[4], rs[3])))
        h.append(self.pre_sp((rs[0], rs[1], rs[2], rs[3], rs[4])))
        h = numpy.array(h)
        h[:,0] = numpy.sqrt(h[:,0] / h[:,0].sum())
        for i in range(4):
            h[i,1:] *= numpy.sqrt(1 - h[i,0] ** 2)
        s = numpy.dot(h, h.T)
        c = lowdin_orth_coeff(s)
        return numpy.dot(h.T, c)

# gho_atm_lst: (center_atm, three_linked_atms, bond_atm)
    def hybrid_coeff_by_atoms(self, mol, gho_atm_lst):
        '''determine the hybrid orbitals via geom'''
        coords = map(lambda i: mol.coord_of_atm(i), gho_atm_lst)
        return self.hybrid_coeff_by_coords(coords)
##################################################

##################################################
# center_atm_id: 0-based
    def extract_valence_sto6g(self, mol, center_atm_id):
        symb = mol.symbol_of_atm(center_atm_id)
        if symb == 'H':
            basis_add = gto.basis.load('sto_6g', symb)
        elif symb in ['C', 'O', 'N', 'F']:
            # remove 1s orbital
            basis_add = gto.basis.load('sto_6g', symb)[1:]
        elif symb in ['P', 'S', 'Cl']:
            # remove 1s2s2p
            basis_add = gto.basis.load('sto_6g', symb)[3:]
        bas, env = mol.make_bas_env(basis_add, center_atm_id, len(mol.env))
        return bas, env

#FIXME
# overlap between gho space and mol space
    def gho_mol_ovlp(self, mol, gho_atm_lst):
        import copy
        pmol = mol.copy()
        bas_sto6g, env = self.extract_valence_sto6g(pmol, gho_atm_lst[0])
        nbas_sto6g = bas_sto6g.__len__()
        pmol._bas = pmol._bas + bas_sto6g
        pmol._env = pmol._env + env
        bras = range(pmol.nbas, pmol.nbas+nbas_sto6g)
        kets = range(pmol.nbas)
        s = self.gho_ovlp(pmol, gho_atm_lst)
        c = lowdin_orth_coeff(s)
        ovlp = pmol.intor_cross('cint1e_ovlp_sph', bras, kets)
        return numpy.dot(c.T.conj(), ovlp)

#FIXME
    def gho_ovlp(self, mol, gho_atm_lst):
        import copy
        pmol = copy.deepcopy(mol)
        bas_sto6g = self.extract_valence_sto6g(pmol, gho_atm_lst[0])
        nbas_sto6g = bas_sto6g.__len__()
        pmol._bas = pmol._bas + bas_sto6g
        bras = range(pmol.nbas, pmol.nbas+nbas_sto6g)
        s = pmol.intor_cross('cint1e_ovlp_sph', bras, bras)
        return s


def gho_index(mol, gho_atm):
    idx = []
    for i, s in enumerate(mol.spheric_labels()):
        if s[0] == gho_atm \
           and (s[2] == '2s' or s[2] == '2p'):
            idx.append(i)
    return numpy.array(idx)

#FIXME
def gho_dmet_bath_ovlp(mol, bath, gho_atm_lst):
    g = GHO()
    #gho = g.hybrid_coeff_by_atoms(mol, gho_atm_lst)
    gho = g.hybrid_coeff(mol, gho_atm_lst)

    s = reduce(numpy.dot, (gho.T.conj(), g.gho_mol_ovlp(mol, gho_atm_lst), bath))
    log.debug(mol, '<gho_i|bath_i> : %s', s.diagonal())
    u, w, v = numpy.linalg.svd(s)
    p_hybs = numpy.dot(s, s.T).diagonal()
    # w is the entanglement between these two space
    # p_hybs = <hyb|bath><bath|hyb>
    return w, p_hybs


########################
import scf
import dmet
# GHO SCF with auxiliary orbitals. ref. JPCA, 108, 632
# The first 4 atoms in mm_atm_lst are used to determine GHOs
MM_CHARGE = {
    'H': 0.00,
    'C':-0.00*3,
    'N': 0.,
    'O': 0.,
}

def gho_scf(mol, mm_atm_lst, mm_charge=0, mm_chg_lst=None, emb=None, \
            inc_1s=False):
    log.info(mol, '\n\n')
    log.info(mol, '================ GHO SCF ================')
    pmol = mol.copy()
    pmol.atom = [pmol.atom[i] for i in range(pmol.natm) \
                 if i not in mm_atm_lst]
    gho_atm_lst = mm_atm_lst[:4]
    pmol.atom.append(['C999', mol.atom[gho_atm_lst[0]][1]])
    # In the original GHO paper (JPCA, 102, 4714), ECP is used for the MM
    # boundary atom. We include the atom into the QM region to avoid ECP
    pmol.basis['C999'] = mol.basis[mol.symbol_of_atm(gho_atm_lst[0])]
    pmol.build(False, False)
    bound_atm_id = pmol.natm - 1 # the last one in new mol
    #print pmol.nelectron, pmol.atom
    pmol.nelectron -= 3 # 3e on auxiliary GHO is not treated variationally
    if not inc_1s:
        pmol.nelectron -= 2
    ghohf = scf.RHF(pmol)

    gho_orb = GHO().hybrid_coeff(mol, gho_atm_lst)
    sub_ovlp = pmol.intor_symmetric('cint1e_ovlp_sph')
    c = dmet.hf.pre_orth_ao_atm_scf(pmol)
    orth_coeff = dmet.hf.orthogonalize_ao(pmol, 0, c, orth_method='meta_lowdin')
    #orth_coeff = dmet.hf.orthogonalize_ao(pmol, 0, c, orth_method='lowdin')
    #orth_coeff = dmet.hf.lowdin_orth_coeff(sub_ovlp)
    gho_idx = gho_index(pmol, bound_atm_id)
    gho_aux = numpy.dot(orth_coeff[:,gho_idx],gho_orb[:,1:])
    if mm_chg_lst is not None:
        assert(len(mm_atm_lst) == len(mm_chg_lst))
        mm_charge = mm_chg_lst[0]
    dm_aux = numpy.dot(gho_aux, gho_aux.T) * (1-mm_charge/3)
    if not inc_1s:
        for i, s in enumerate(pmol.spheric_labels()):
            if s[0] == bound_atm_id and s[2] == '1s':
                c1s = orth_coeff[:,i:i+1]
                break
        if inc_1s != -1:
            dm_aux += numpy.dot(c1s, c1s.T) * 2
        else:
            # Drop 1s shell and reduce the nuclear charge as JPCA,108,632 did.
            # It completely screen 2e of nuclear charge. Thus the hybrid sp3
            # GHO feels less nuclear attraction and tends to lose more
            # electrons.
            pass
    vhf_aux = scf.hf.RHF.get_veff(ghohf, pmol, dm_aux)

    for i,ia in enumerate(mm_atm_lst[1:]):
        pmol.set_rinv_orig(mol.coord_of_atm(ia))
        if mm_chg_lst is not None:
            vhf_aux += -mm_chg_lst[i+1] * pmol.intor_symmetric('cint1e_rinv_sph')
        else:
            symb = mol.pure_symbol_of_atm(ia)
            vhf_aux += -MM_CHARGE[symb] * pmol.intor_symmetric('cint1e_rinv_sph')

    idx = []
    lbl = pmol.spheric_labels()
    for i, s in enumerate(lbl):
        if s[0] != bound_atm_id:
            idx.append(i)
        elif inc_1s: #include 1s of carbon in MM region to avoid ECP
            if s[2] == '1s':
                idx.append(i)
    orth_coeff = numpy.hstack((orth_coeff[:,idx], \
                               numpy.dot(orth_coeff[:,gho_idx],gho_orb[:,:1])))
    s1 = reduce(numpy.dot, (orth_coeff.T,sub_ovlp,orth_coeff))
    assert(abs(s1-numpy.eye(orth_coeff.shape[1])).sum()<1e-12)

    lbl = [lbl[i] for i in idx] + [(bound_atm_id, 'C', 'sp3', '')]
    pmol.spheric_labels = lambda: lbl

    vhf_aux = reduce(numpy.dot, (orth_coeff.T, vhf_aux, orth_coeff))
    def eff_hcore(pmol):
        h = pmol.intor_symmetric('cint1e_kin_sph') \
                + pmol.intor_symmetric('cint1e_nuc_sph')
        if inc_1s == -1:  # JPCA,108,632
            pmol.set_rinv_orig(pmol.coord_of_atm(bound_atm_id))
            h += 2 * pmol.intor_symmetric('cint1e_rinv_sph')
        return reduce(numpy.dot, (orth_coeff.T, h, orth_coeff)) + vhf_aux
    def eff_vhf(pmol, dm, dm_last=0, vhf_last=0):
        dm_ao = reduce(numpy.dot, (orth_coeff, dm, orth_coeff.T))
        vhf = scf.hf.RHF.get_veff(ghohf, pmol, dm_ao)
        return reduce(numpy.dot, (orth_coeff.T, vhf, orth_coeff))
    def init_guess(pmol):
        sc = numpy.dot(sub_ovlp, orth_coeff)
        dm = scf.hf.init_guess_by_minao(ghohf, pmol)[1]
        return reduce(numpy.dot, (sc.T, dm, sc))
    ghohf.get_hcore = eff_hcore
    ghohf.get_ovlp = lambda x: numpy.eye(orth_coeff.shape[1])
    ghohf.get_veff = eff_vhf
    ghohf.get_init_guess = init_guess

    res = ghohf.scf()
    if emb is not None:
        scfdm = emb.entire_scf.make_rdm1(emb.entire_scf.mo_coeff, \
                                         emb.entire_scf.mo_occ)
        ghodm = ghohf.make_rdm1(ghohf.mo_coeff, ghohf.mo_occ)
        gen_diff_dm(emb.mol, ghodm, scfdm, emb)

        v1 = pmol.intor_symmetric('cint1e_nuc_sph')
        if inc_1s == -1:  # JPCA,108,632
            pmol.set_rinv_orig(pmol.coord_of_atm(bound_atm_id))
            v1 += 2 * pmol.intor_symmetric('cint1e_rinv_sph')
        v1 = reduce(numpy.dot, (orth_coeff.T, v1, orth_coeff)) + vhf_aux
        idx = [i for i, s in enumerate(pmol.spheric_labels()) \
               if s[0] != bound_atm_id]
        vgho = v1[idx][:,idx]
        vemb = emb.mat_ao2impbas(mol.intor_symmetric('cint1e_nuc_sph')) + emb._vhf_env

        vnuc_mm = 0
        for i,ia in enumerate(mm_atm_lst):
            emb.mol.set_rinv_orig(emb.mol.coord_of_atm(ia))
            chg = emb.mol.charge_of_atm(ia)
            vnuc_mm += -chg * emb.mol.intor_symmetric('cint1e_rinv_sph')
        vnuc_mm = emb.mat_ao2impbas(vnuc_mm) + emb._vhf_env

        ndim = len(idx)
        print vemb.diagonal()
        print vgho.diagonal()
        normdv = numpy.linalg.norm(vemb[:ndim,:ndim]-vgho)
        pctdv = normdv/numpy.linalg.norm(vnuc_mm[:ndim,:ndim])
        log.info(emb, 'norm delta vext on fragment = %.9g, RMS = %.9g', \
                 normdv, normdv/ndim)
        h1emb = vemb + emb.mat_ao2impbas(emb.mol.intor_symmetric('cint1e_kin_sph'))
        log.info(emb, 'delta/RMS(vnuc_mm) = %.9g, delta/RMS(h1e) = %.9g', \
                 100*normdv/numpy.linalg.norm(vnuc_mm[:ndim,:ndim]), \
                 100*normdv/numpy.linalg.norm(h1emb[:ndim,:ndim]))
    return res

# in order to compare with the corresponding DMET, same orthogonalization
# scheme should be used in both methods
def gho_scf_compare_auxorb(emb, mm_atm_lst, mm_charge=0, mm_chg_lst=None):
    gho_atm_lst = mm_atm_lst[:4]
    assert(emb.mol.pure_symbol_of_atm(gho_atm_lst[0]) == 'C')
    import copy
    log.info(emb, '\n\n')
    log.info(emb, '================ GHO SCF ================')
    ghoemb = copy.copy(emb)
    ghoemb.append_bath(gho_atm_lst)
    ghoemb.num_bath = 1
    gho_orb = GHO().hybrid_coeff(emb.mol, gho_atm_lst)
    gho_idx = gho_index(emb.mol, gho_atm_lst[0])

    def cons_impbas():
        a = numpy.dot(emb.orth_coeff[:,ghoemb.bas_on_frag], ghoemb.imp_site)
        b = numpy.dot(emb.orth_coeff[:,gho_idx], gho_orb[:,:1])
        return numpy.hstack((a,b))
    ghoemb.cons_impurity_basis = cons_impbas

    ghoemb.init_vhf_env = lambda *args: 0

    gho_aux = numpy.dot(emb.orth_coeff[:,gho_idx],gho_orb[:,1:])
    if mm_chg_lst is not None:
        assert(len(mm_atm_lst) == len(mm_chg_lst))
        mm_charge = mm_chg_lst[0]
    dm_aux = numpy.dot(gho_aux, gho_aux.T) * (1-mm_charge/3)
    vhf_aux = scf.hf.RHF.get_veff(emb.entire_scf, emb.mol, dm_aux)
    for i,ia in enumerate(mm_atm_lst[1:]):
        emb.mol.set_rinv_orig(emb.mol.coord_of_atm(ia))
        if mm_chg_lst is not None:
            vhf_aux += -mm_chg_lst[i+1] * emb.mol.intor_symmetric('cint1e_rinv_sph')
        else:
            symb = emb.mol.pure_symbol_of_atm(ia)
            vhf_aux += -MM_CHARGE[symb] * emb.mol.intor_symmetric('cint1e_rinv_sph')
    for ia in ghoemb.imp_atoms + [gho_atm_lst[0]]:
        emb.mol.set_rinv_orig(emb.mol.coord_of_atm(ia))
        chg = emb.mol.charge_of_atm(ia)
        vhf_aux += emb.mol.intor_symmetric('cint1e_rinv_sph') * -chg

    def eff_hcore(mol):
        return ghoemb.mat_ao2impbas(mol.intor_symmetric('cint1e_kin_sph')+vhf_aux)
    ghoemb.get_hcore = eff_hcore

    def eff_vhf(mol, dm, dm_last=0, vhf_last=0):
        dm_ao = reduce(numpy.dot, (ghoemb.impbas_coeff, dm, \
                                   ghoemb.impbas_coeff.T))
        vhf = scf.hf.RHF.get_veff(emb.entire_scf, mol, dm_ao)
        return ghoemb.mat_ao2impbas(vhf)
    ghoemb.get_veff = eff_vhf

    res = ghoemb.imp_scf()

    scfdm = emb.entire_scf.make_rdm1(emb.entire_scf.mo_coeff, \
                                     emb.entire_scf.mo_occ)
    ghodm = ghoemb.make_rdm1(ghoemb.mo_coeff_on_imp, ghoemb.mo_occ)
    gen_diff_dm(emb.mol, ghodm, scfdm, emb)

    vnuc_mm = 0
    for i,ia in enumerate(mm_atm_lst):
        emb.mol.set_rinv_orig(emb.mol.coord_of_atm(ia))
        chg = emb.mol.charge_of_atm(ia)
        vnuc_mm += -chg * emb.mol.intor_symmetric('cint1e_rinv_sph')
    vnuc_mm = emb.mat_ao2impbas(vnuc_mm) + emb._vhf_env

    vemb = emb.mat_ao2impbas(emb.mol.intor_symmetric('cint1e_nuc_sph')) + emb._vhf_env
    vgho = ghoemb.mat_ao2impbas(vhf_aux)
    if ghoemb.imp_basidx:
        nimp = len(emb.bas_on_frag) - 1 # exclude 1s on linked MM carbon
    else:
        nimp = len(emb.bas_on_frag)
    normdv = numpy.linalg.norm(vemb[:ndim,:ndim]-vgho[:ndim,:ndim])
    log.info(emb, 'norm delta vext on fragment = %.9g, RMS = %.9g', \
             normdv, normdv/ndim)
    h1emb = vemb + emb.mat_ao2impbas(emb.mol.intor_symmetric('cint1e_kin_sph'))
    log.info(emb, 'delta/RMS(vnuc_mm) = %.9g%%, delta/RMS(h1e) = %.9g%%', \
             100*normdv/numpy.linalg.norm(vnuc_mm[:ndim,:ndim]), \
             100*normdv/numpy.linalg.norm(h1emb[:ndim,:ndim]))
    return res

def diff_dm(dev, dm_sets, title=[]):
    nset = len(dm_sets)
    diffs = numpy.zeros((nset,nset))
    rms   = numpy.zeros((nset,nset))
    for i, vi in enumerate(dm_sets):
        for j in range(i):
            diffs[i,j] = diffs[j,i] = numpy.linalg.norm(dm_sets[j]-vi)
            rms  [i,j] = rms  [j,i] = diffs[i,j]/vi.shape[0]
    if len(title) != nset:
        title = ['normddm-%d'%i for i in range(nset)]
        log.info(dev, 'norm diff DM')
    else:
        log.info(dev, 'norm diff DM' + ('%12s'*nset) % tuple(title))
    for i in range(nset):
        fmt = ('%12s'%title[i]) + (' %11.5f'*nset)
        log.info(dev, fmt, *diffs[i])
    if len(title) != nset:
        title = ['RMSddm-%d'%i for i in range(nset)]
        log.info(dev, 'RMS diff DM')
    else:
        log.info(dev, 'RMS diff DM' + ('%12s'*nset) % tuple(title))
    for i in range(nset):
        fmt = ('%12s'%title[i]) + (' %11.5f'*nset)
        log.info(dev, fmt, *rms[i])

def gen_diff_dm(mol, ghodm, scfdm, emb):
    if emb.imp_basidx:
        nimp = len(emb.bas_on_frag) - 1 # exclude 1s on linked MM carbon
    else:
        nimp = len(emb.bas_on_frag)
    bas_on_a = emb.bas_on_frag[:nimp]
    impidx = numpy.argsort(bas_on_a)
    c_inv = numpy.dot(emb.orth_coeff.T, \
                      mol.intor_symmetric('cint1e_ovlp_sph'))
    scfdm = reduce(numpy.dot, (c_inv, scfdm, c_inv.T.conj()))
    # in case impurity sites are not the AO orbitals
    mo = reduce(numpy.dot, (c_inv, emb.impbas_coeff, emb.mo_coeff_on_imp))
    embdm = numpy.dot(mo*emb.mo_occ, mo.T) \
            + numpy.dot(emb.env_orb,emb.env_orb.T) * 0#2
    diff_dm(mol, (scfdm[bas_on_a][:,bas_on_a], \
                  embdm[bas_on_a][:,bas_on_a], \
                  ghodm[:nimp,:nimp]), \
            ('SCF', 'GHO-DMETcore', 'GHO'))



def sp_hybrid_level(gho_coeff):
    s = gho_coeff[0] ** 2
    p = numpy.linalg.norm(gho_coeff[1:4]) ** 2
    return p/s
def angle_to_bond(mol, atm0, atm1, gho_coeff):
    coord0 = mol.coord_of_atm(atm0)
    coord1 = mol.coord_of_atm(atm1)
    vec1 = coord1 - coord0
    vec1 = vec1 / numpy.linalg.norm(vec1)
    vec2 = gho_coeff[1:4] / numpy.linalg.norm(gho_coeff[1:4])
    if gho_coeff[0] < 0: # the sign of s orbital
        vec2 = -vec2
    return numpy.arccos(numpy.dot(vec1,vec2))


if __name__ == "__main__":
#    h = GHO().hybrid_coeff_by_coords(((0.,0.,0.), \
#                                      ( 1, 1, 1), \
#                                      (-1,-1, 1), \
#                                      ( 1,-1,-1), \
#                                      (-1, 1,-1)))
#    print numpy.mat(h).T * h
#    print h
    #h = g.hybrid_coeff(((0.,0.,0.), \
    #                    ( 1, 1, 1), \
    #                    (-1,-1, 1), \
    #                    ( 1,-1,-1), \
    #                    (-1, 1,-1)))
    #print numpy.mat(h).T * h

    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_gho'
    mol.atom = [['C', (0.,0.,0.)],
                ['H', ( 1, 1, 1)],
                ['H', (-1,-1, 1)],
                ['H', ( 1,-1,-1)],
                ['H', (-1, 1,-1)], ]
    mol.basis = {
        'C': 'sto-3g',
        'H': 'sto-3g'}
    mol.build()
    m = scf.RHF(mol)
    m.scf()
    emb = dmet.hf.RHF(m)
    emb.set_ao_with_atm_scf()
    emb.init_with_meta_lowdin_ao()
    emb.set_bath([0, 1, 2, 3])
    print emb.imp_scf()
    print gho_scf(mol, (0,1,2,3), 0.0146*3)

    gho_orb = GHO().hybrid_coeff(mol, (0,1,2,3))
    print sp_hybrid_level(gho_orb[:,0])
    print angle_to_bond(mol, 0, 4, gho_orb[:,0])
