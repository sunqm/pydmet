#!/usr/bin/env python

import os
import tempfile
import commands
import numpy

from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf import ao2mo
from pyscf import tools
from pyscf import mcscf
import pyscf.fci

class ImpSolver(object):
    def __init__(self, solver):
        self.solver = solver

        self.escf = None
        self.etot = None
        self.e2frag = None
        self.dm1 = None

    # when with_e2frag = nimp, self.e2frag is the partially traced energy
    def run(self, emb, eri, vfit=0, with_1pdm=False, with_e2frag=None):
        h1e = emb.get_hcore()
        if isinstance(vfit, numpy.ndarray):
            nv = vfit.shape[0]
            h1e[:nv,:nv] += vfit
        nelec = emb.nelectron
        mo = emb.mo_coeff_on_imp
        self.escf, self.etot, self.e2frag, self.dm1 = \
                self.solver(emb.mol, h1e, eri, mo, nelec, \
                            with_1pdm, with_e2frag)
        return self.etot, self.e2frag, self.dm1

class Psi4CCSD(ImpSolver):
    def __init__(self):
        ImpSolver.__init__(self, psi4ccsd)

class Psi4CCSD_T(ImpSolver):
    def __init__(self):
        ImpSolver.__init__(self, psi4ccsd_t)

class FCI(ImpSolver):
    def __init__(self):
        ImpSolver.__init__(self, fci)

class CASSCF(ImpSolver):
    def __init__(self, ncas, nelecas, caslist=None):
        ImpSolver.__init__(self, None)
        def f(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag):
            return casscf(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag,
                          ncas, nelecas, caslist)
        self.solver = f



def simple_hf(h1e, eri, mo, nelec):
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.build(False, False)
    mol.nelectron = nelec
    mf = scf.RHF(mol)
    nocc = nelec / 2
    mf.make_init_guess = \
            lambda mol: (0, numpy.dot(mo[:,:nocc],mo[:,:nocc].T)*2)
    mf.get_hcore = lambda mol: h1e
    mf.get_ovlp = lambda mol: numpy.eye(mo.shape[1])
    mf._eri = eri

    scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff \
            = mf.scf_cycle(mol, 1e-9, dump_chk=False)
    return hf_energy, mo_energy, mo_occ, mo_coeff


def _psi4cc(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag, ccname='CCSD'):
    import psi4
    eri = ao2mo.restore(8, eri, mo.shape[1])
    hf_energy, mo_energy, mo_occ, mo = simple_hf(h1e, eri, mo, nelec)

    h1e = reduce(numpy.dot, (mo.T, h1e, mo))
    eri = ao2mo.incore.full(eri, mo)

    rdm1 = None
    ps = psi4.Solver(max_memory=1<<34)
    with psi4.capture_stdout():
        nmo = h1e.shape[0]
        ps.prepare('RHF', numpy.eye(nmo), h1e, eri, nelec)
        ecc = ps.energy(ccname)
        if with_1pdm or with_e2frag:
            rdm1, rdm2 = ps.density()
            rdm1 *= 2 # Psi4 gives rdm1 of alpha spin

# note the rdm1,rdm2 from psi4 solver EXCLUDES HF contributions
            for i in range(nelec/2):
                rdm1[i,i] += 2
                for j in range(nelec/2):
                    rdm2[i,j,i,j] += 4
                    rdm2[i,j,j,i] +=-2
            rdm1 = reduce(numpy.dot, (mo, rdm1, mo.T))

    e2frag = 0
    if with_e2frag:
        nmo = mo.shape[1]
        nimp = with_e2frag
        p = numpy.dot(mo[:nimp,:].T, mo[:nimp,:])
        # psi4 store DM in the order of p^+ q^+ r s
        eri1 = ao2mo.restore(1, eri, nmo).transpose(0,2,1,3)
        frag_rdm2 = numpy.dot(p, rdm2.reshape(nmo,-1))
        e2frag = .5*numpy.dot(frag_rdm2.reshape(-1), eri1.reshape(-1))
    return hf_energy, ecc+hf_energy, e2frag, rdm1

def psi4ccsd(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag):
    return _psi4cc(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag, 'CCSD')

def psi4ccsd_t(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag):
    return _psi4cc(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag, 'CCSD(T)')


def fci(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag):

# use HF as intial guess for FCI solver
    eri1 = ao2mo.restore(8, eri, mo.shape[1])
    hf_energy, mo_energy, mo_occ, mo = simple_hf(h1e, eri1, mo, nelec)
    h1e = reduce(numpy.dot, (mo.T, h1e, mo))
    eri1 = ao2mo.incore.full(eri1, mo)

    norb = h1e.shape[1]
    cis = pyscf.fci.solver(mol)
    #cis.verbose = 5
    eci, c = cis.kernel(h1e, eri1, norb, nelec)
    if with_1pdm:
        dm1 = cis.make_rdm1(c, norb, nelec)
        dm1 = reduce(numpy.dot, (mo, dm1, mo.T))
    else:
        dm1 = None
    if with_e2frag:
        eri1 = part_eri_hermi(eri, norb, with_e2frag)
        eri1 = ao2mo.incore.full(eri1, mo)
        e2frag = cis.energy(numpy.zeros_like(h1e), eri1, c, norb, nelec)
    else:
        e2frag = None
    return 0, eci, e2frag, dm1

def part_eri_hermi(eri, norb, nimp):
    eri1 = ao2mo.restore(4, eri, norb)
    for i in range(eri1.shape[0]):
        tmp = lib.unpack_tril(eri1[i])
        tmp[nimp:] = 0
        eri1[i] = lib.pack_tril(tmp+tmp.T)
    eri1 = lib.transpose_sum(eri1, inplace=True)
    return ao2mo.restore(8, eri1, norb) * .25

def casscf(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag,
           ncas, nelecas, caslist=None):
    mf = scf.RHF(mol)
    mf.get_hcore = lambda mol: h1e
    mf._eri = eri
    mc = mcscf.CASSCF(mol, mf, ncas, nelecas)
    if caslist:
        mo = mcscf.addons.sort_mo(mc, mo, caslist, 1)
    etot,ecas,civec,mo = mc.mc1step(mo)
    if with_1pdm:
        dm1a, dm1b = mcscf.addons.make_rdm1s(mc, civec, mo)
        dm1 = dm1a + dm1b
    else:
        dm1 = None
    if with_e2frag:
#TODO:        eri1 = part_eri_hermi(eri, norb, with_e2frag)
#TODO:        e2frag = fci_direct.energy(numpy.zeros_like(h1e), eri1, c, norb, nelec)
        e2frag = 0
    else:
        e2frag = 0
    return 0, eci, e2frag, dm1




class InterNormFCI(ImpSolver):
    '''<0|H|CI> with <0|CI> = 1'''
    def __init__(self):
        ImpSolver.__init__(self, internorm_fci)

def internorm_fci(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag):
# use HF as intial guess for FCI solver
    eri1 = ao2mo.restore(8, eri, mo.shape[1])
    hf_energy, mo_energy, mo_occ, mo = simple_hf(h1e, eri1, mo, nelec)
    h1e = reduce(numpy.dot, (mo.T, h1e, mo))
    eri1 = ao2mo.incore.full(eri1, mo)

    norb = h1e.shape[1]
    cis = pyscf.fci.solver(mol)
    #cis.verbose = 5
    eci, c = cis.kernel(h1e, eri1, norb, nelec)
    c0 = numpy.zeros_like(c)
    c0[0,0] = 1/c[0,0] # so that <c0|c> = 1
    if with_1pdm:
        dm1 = cis.trans_rdm1(c0, c, norb, nelec)
        dm1 = (dm1 + dm1.T) * .5
        dm1 = reduce(numpy.dot, (mo, dm1, mo.T))
    else:
        dm1 = None
    if with_e2frag:
        eri1 = part_eri_hermi(eri, norb, with_e2frag)
        eri1 = ao2mo.incore.full(eri1, mo)
        h2e = cis.absorb_h1e(numpy.zeros_like(h1e), eri1, norb, nelec, .5)
        ci1 = cis.contract_2e(h2e, c, norb, nelec)
        #e2frag = numpy.dot(c0.ravel(), ci1.ravel())
        e2frag = c0[0,0] * ci1[0,0]
    else:
        e2frag = None
    return 0, eci, e2frag, dm1
