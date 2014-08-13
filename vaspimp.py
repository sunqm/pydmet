#!/usr/bin/env python

import re
import numpy
from pyscf import lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
from pyscf.lib import _vhf
from pyscf import ao2mo
from pyscf import gto
from pyscf import scf
import dmet_hf

# AO basis of entire system are orthogonal sets
class OneImp(dmet_hf.RHF):
    def __init__(self, entire_scf, basidx=[], orth_ao=None):
        dmet_hf.RHF.__init__(self, entire_scf)
        self.bas_on_frag = basidx

    def init_dmet_scf(self, mol=None):
        effscf = self.entire_scf
        c_inv = numpy.dot(self.orth_coeff.T, self.entire_scf.get_ovlp(mol))
        mo_orth = numpy.dot(c_inv, effscf.mo_coeff[:,effscf.mo_occ>1e-15])
        self.imp_site, self.bath_orb, self.env_orb = \
                dmet_hf.decompose_orbital(self, mo_orth, self.bas_on_frag)
        self.impbas_coeff = self.cons_impurity_basis()

        self.nelectron = int(effscf.mo_occ.sum()) - self.env_orb.shape[1] * 2
        log.info(self, 'number of electrons for impurity  = %d', \
                 self.nelectron)
        self._vhf_env = self.init_vhf_env(mol, self.env_orb)

    def get_orth_ao(self, mol):
        s = self.entire_scf.get_ovlp(mol)
        if abs(s-numpy.eye(s.shape[0])).sum() < 1e-12:
            return numpy.eye(self.entire_scf.mo_energy.size)
        else:
            return dmet_hf.RHF.get_orth_ao(self, mol)


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
        return eri_mo

#    def get_eff_potential(self, mol, dm, dm_last=0, vhf_last=0):
#        if self._eri is None:
#            self._eri = self.eri_on_impbas(mol)
#        vj, vk = _vhf.vhf_jk_incore_o2(self._eri, dm)
#        vhf = vj - vk * .5
#        return vhf


class OneImpNI(OneImpNaiveNI):
    def get_hcore(self, mol=None):
        nimp = len(self.bas_on_frag)
        effscf = self.entire_scf
        cs = numpy.linalg.solve(effscf.mo_coeff, self.impbas_coeff)
        fock = numpy.dot(cs.T*effscf.mo_energy, cs)
        dmimp = effscf.calc_den_mat(mo_coeff=cs.T)
        dm = numpy.zeros_like(fock)
        dm[:nimp,:nimp] = dmimp[:nimp,:nimp]
        h1e = fock - self.get_eff_potential(mol, dm)
        return h1e


##########################################################################
def decompose_orbital_with_impsite(emb, mo_orth, bas_on_frag, num_bath=-1):
    log.debug(emb, 'occupied mo shape = %d, %d', *mo_orth.shape)
    log.debug(emb, 'number of basis on fragment = %d', \
              bas_on_frag.__len__())

    log.debug(emb, '*** decompose orbitals to fragment sites, '\
              'bath, env orbitals ***')

    fmo = mo_orth[bas_on_frag]
    pre_nao, w1, pre_env_h = numpy.linalg.svd(fmo)
    mo1 = numpy.dot(mo_orth, pre_env_h.T.conj())
    w = numpy.zeros(mo_orth.shape[1])
    w[:w1.size] = w1   # when nimp < nmo, adding 0s by the end

    idx, not_idx, rest_idx = dmet_hf._pick_bath_idx(w1**2, num_bath, emb.occ_env_cutoff)
    env_idx = not_idx + range(w1.size, mo_orth.shape[1])
    mo_bath = mo1[:,idx]
    env_orb = mo1[:,env_idx]
    log.info(emb, 'number of proto bath orbital = %d', mo_bath.shape[1])
    log.info(emb, 'number of env orbitals = %d', env_orb.shape[1])
    log.debug(emb, 'entanglement weight (= sqrt(occs)),  occ')
    if emb.verbose >= param.VERBOSE_DEBUG:
        for i in idx:
            log.debug(emb, '%d th weight = %12.9f, %12.9f  => bath (%s)', \
                      i, w[i], w[i]**2, \
                      ('acceptor' if w[i]**2>0.5 else 'donor'))
        for i in env_idx:
            log.debug(emb, '%d th weight = %12.9f, %12.9f  => env', \
                      i, w[i], w[i]**2)
        for i in rest_idx:
            log.debug(emb, '%d th weight = %12.9f, %12.9f => rest/imp', \
                      i, w[i], w[i]**2)
        log.debug(emb, 'potentially change in embsys charge = %12.9f', \
                  sum(w[not_idx]**2)+sum(w[rest_idx]**2-1))
        #log.debug(emb, ' ** env orbital coefficients (on orthogonal basis)**')
        #scf.hf.dump_orbital_coeff(emb.mol, env_orb)

    imp_site = mo_bath[bas_on_frag]/w[idx]
    if mo_bath.shape[1] > 0:
        mo_bath[bas_on_frag] = 0
        norm = 1/numpy.sqrt(1-w[idx]**2)
        bath_orb = mo_bath * norm
    else:
        bath_orb = mo_bath
    #if emb.verbose >= param.VERBOSE_DEBUG:
    #    log.debug(emb, ' ** bath orbital coefficients (on orthogonal basis) **')
    #    scf.hf.dump_orbital_coeff(emb.mol, bath_orb)
    return imp_site, bath_orb, env_orb

class OneImpOnCLUSTDUMP(OneImp):
    def __init__(self, entire_scf, vasphf):
        self._vasphf = vasphf
        dmet_hf.RHF.__init__(self, entire_scf, numpy.eye(vasphf['NORB']))
        self.bas_on_frag = self._vasphf['ORBIND']
        self._eri = vasphf['ERI']

    def init_dmet_scf(self, mol=None):
        effscf = self.entire_scf
        mo_orth = effscf.mo_coeff[:,effscf.mo_occ>1e-15]
#        self.imp_site, self.bath_orb, self.env_orb = \
#                dmet_hf.decompose_orbital(self, mo_orth, self.bas_on_frag)
        self.imp_site, self.bath_orb, self.env_orb = \
                decompose_orbital_with_impsite(self, mo_orth, self.bas_on_frag)
        #self.impbas_coeff = self.cons_impurity_basis()
        self.impbas_coeff = self._vasphf['EMBASIS']
        #print abs(abs(self.cons_impurity_basis()) - abs(self._vasphf['EMBASIS'])).sum()
        #assert(0)
        #log.debug(self, 'diff of impbas_coeff to readin embasis %.8g',
        #          abs(abs(self.impbas_coeff) - abs(self._vasphf['EMBASIS'])).sum())

        self.nelectron = int(effscf.mo_occ.sum()) - self.env_orb.shape[1] * 2
        log.info(self, 'number of electrons for impurity  = %d', \
                 self.nelectron)
        self._vhf_env = self.init_vhf_env(mol, self.env_orb)

    def init_vhf_env(self, mol, env_orb):
        self.energy_by_env = 0
        nemb = self._vasphf['NEMB']
        c = numpy.dot(self.impbas_coeff.T, self._vasphf['MO_COEFF'])
#        vhf = numpy.dot(c*self._vasphf['MO_ENERGY'], c.T) \
#                - self._vasphf['H1EMB']
        vhf = self.mat_ao2impbas(self._vasphf['J'] +self._vasphf['K'])

        mocc = c[:,:self._vasphf['NELEC']/2]
        dmemb = numpy.dot(mocc, mocc.T)*2
        vemb = self.get_eff_potential(mol, dmemb)
        return vhf - vemb

    def init_guess_method(self, mol):
        log.debug(self, 'init guess based on entire MO coefficients')
        eff_scf = self.entire_scf
        c = numpy.dot(self.impbas_coeff.T, eff_scf.mo_coeff)
        dm = eff_scf.calc_den_mat(c, eff_scf.mo_occ)
        hf_energy = 0
        return hf_energy, dm

    def get_hcore(self, mol=None):
        return self._vasphf['H1EMB'] + self._vhf_env

    def get_ovlp(self, mol=None):
        return numpy.eye(self._vasphf['NEMB'])

    def eri_on_impbas(self, mol):
        return self._vasphf['ERI']

    def imp_scf(self):
        self.orth_coeff = self.get_orth_ao(self.mol)
        self.dump_options()
        self.init_dmet_scf(self.mol)
        self.scf_conv, self.hf_energy, self.mo_energy, self.mo_occ, \
                self.mo_coeff_on_imp \
                = self.scf_cycle(self.mol, self.scf_threshold, \
                                 dump_chk=False)
        self.mo_coeff = numpy.dot(self.impbas_coeff, self.mo_coeff_on_imp)
        if self.scf_conv:
            log.log(self, 'converged impurity sys electronic energy = %.15g', \
                    self.hf_energy)
        else:
            log.log(self, 'SCF not converge.')
            log.log(self, 'electronic energy = %.15g after %d cycles.', \
                    self.hf_energy, self.max_scf_cycle)

        dm = self.calc_den_mat(self.mo_coeff_on_imp, self.mo_occ)
        vhf = self.get_eff_potential(self.mol, dm)
        self.e_frag, self.n_elec_frag = \
                self.calc_frag_elec_energy(self.mol, vhf, dm)
        return self.hf_energy

    def get_eff_potential(self, mol, dm, dm_last=0, vhf_last=0):
        if self._eri is None:
            self._eri = self.eri_on_impbas(mol)
        vj, vk = _vhf.vhf_jk_incore_o2(self._eri, dm)
        vhf = vj - vk * .5
        return vhf

    def dim_of_impurity(self):
        return self._vasphf['NIMP']

def read_clustdump(fcidump, jdump, kdump, fockdump):
    dic = {}
    finp = open(jdump, 'r')
    dat = re.split('[=,]', finp.readline())
    while dat[0]:
        if 'FCI' in dat[0].upper():
            dic['NORB'] = int(dat[1])
            dic['NELEC'] = int(dat[3])
            dic['MS2'] = int(dat[5])
        elif 'END' in dat[0].upper():
            break
        dat = re.split('[=,]', finp.readline())
    norb = dic['NORB']
    vj = numpy.zeros((norb,norb))
    dat = finp.readline().split()
    while dat:
        i, j = map(int, dat[2:4])
        val = map(float, dat[:2])
        if abs(val[1]) > 1e-12:
            print i,j, val
            assert(abs(val[1]) < 1e-9)
        vj[i-1,j-1] = val[0]
        dat = finp.readline().split()

    finp = open(kdump, 'r')
    dat = finp.readline()
    while dat:
        if 'END' in dat:
            break
        dat = finp.readline()
    vk = numpy.zeros((norb,norb))
    dat = finp.readline().split()
    while dat:
        i, j = map(int, dat[2:4])
        val = map(float, dat[:2])
        #if abs(val[1]) > 1e-12:
        #    print i,j, val
        #    assert(abs(val[1]) < 1e-5)
        vk[i-1,j-1] = val[0]
        dat = finp.readline().split()

    finp = open(fockdump, 'r')
    dat = finp.readline()
    while dat:
        if 'END' in dat:
            break
        dat = finp.readline()
    fock = numpy.zeros((norb,norb))
    mo_energy = numpy.zeros(norb)
    dat = finp.readline().split()
    while dat:
        i, j = map(int, dat[2:4])
        val = map(float, dat[:2])
        #if abs(val[1]) > 1e-12:
        #    print i,j, val
        #    assert(abs(val[1]) < 1e-5)
        if j == 0:
            mo_energy[i-1] = val[0]
        else:
            fock[i-1,j-1] = val[0]
        dat = finp.readline().split()
    dic['MO_ENERGY'] = mo_energy

    finp = open(fcidump, 'r')
    dat = re.split('[=,]', finp.readline())
    while dat[0]:
        if 'FCI' in dat[0].upper():
            dic['NEMB'] = int(dat[1])
            dic['NIMP'] = int(dat[3])
            dic['NBATH'] = int(dat[5])
        elif 'ORBIND' in dat[0].upper():
            dat = re.split('[=,]', finp.readline())
            dic['ORBIND'] = map(lambda x: int(x)-1, dat[:dic['NIMP']])
        elif 'END' in dat[0].upper():
            break
        dat = re.split('[=,]', finp.readline())
    nemb = dic['NEMB']
    npair = nemb*(nemb+1)/2
    h1emb = numpy.zeros((nemb,nemb))
    mo_coeff = numpy.zeros((norb,norb))
    embasis = numpy.zeros((norb,nemb))
    eri = numpy.zeros((npair,npair))
    dat = finp.readline().split()
    while dat:
        i, j, k, l = map(int, dat[2:])
        val = map(float, dat[:2])
        if abs(val[1]) > 1e-12:
            if abs(val[1]) > 1e-9:
                print i,j,k,l
                assert(abs(val[1]) < 1e-9)
            else:
                print 'imaginary part /= 0', i,j,k,l, val
        if k == 0 and l == 0:
            h1emb[i-1,j-1] = h1emb[j-1,i-1] = val[0]
        elif l == -1:
            mo_coeff[i-1,j-1] = val[0]
        elif l == -2:
            embasis[i-1,j-1] = val[0]
        else:
            if i >= j:
                ij = (i-1)*i/2 + j-1
            else:
                ij = (j-1)*j/2 + i-1
            if k >= l:
                kl = (k-1)*k/2 + l-1
            else:
                kl = (l-1)*l/2 + k-1
            eri[ij,kl] = eri[kl,ij] = val[0]
        dat = finp.readline().split()
    dic['MO_COEFF'] = mo_coeff
    dic['FOCK'] = reduce(numpy.dot, (mo_coeff, fock, mo_coeff.T))
    dic['J'] = reduce(numpy.dot, (mo_coeff, vj, mo_coeff.T))
    dic['K'] = reduce(numpy.dot, (mo_coeff, vk, mo_coeff.T))
    dic['EMBASIS'] = numpy.dot(mo_coeff,embasis)
    dic['H1EMB'] = h1emb
    dic['ERI'] = eri
    return dic

def fake_entire_scf(vasphf):
    mol = gto.Mole()
    mol.verbose = 0
    mol.build(False, False)
    mol.nelectron = vasphf['NELEC']
    fake_hf = scf.hf.RHF(mol)
    norb = vasphf['NORB']
    hcore = vasphf['FOCK'] - (vasphf['J']+vasphf['K'])
    fake_hf.get_hcore = lambda *args: hcore
    fake_hf.get_ovlp = lambda *args: numpy.eye(norb)
    def stop(*args):
        raise RuntimeError('fake_hf')
    fake_hf.get_eff_potential = stop
    fake_hf.mo_coeff = vasphf['MO_COEFF']
    fake_hf.mo_energy = vasphf['MO_ENERGY']
    fake_hf.mo_occ = numpy.zeros(norb)
    fake_hf.mo_occ[:vasphf['NELEC']/2] = 2
    return fake_hf



if __name__ == '__main__':
    dic = read_clustdump('FCIDUMP.CLUST.GTO', 'JDUMP','KDUMP','FOCKDUMP')
#    hcore = dic['FOCK'] - (dic['J'] - .5*dic['K'])
#    nimp = dic['NIMP']
#    print abs(hcore[:nimp,:nimp] - dic['H1EMB'][:nimp,:nimp]).sum()
#    ee = reduce(numpy.dot, (dic['MO_COEFF'].T, dic['FOCK'], dic['MO_COEFF']))
#    print abs(ee - numpy.diag(dic['MO_ENERGY'])).sum()
    fake_hf = fake_entire_scf(dic)
    emb = OneImpOnCLUSTDUMP(fake_hf, dic)
    emb.verbose = 5
    emb.imp_scf()

