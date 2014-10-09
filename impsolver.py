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
from pyscf.future import mcscf
import pyscf.future.fci.direct_spin0 as fci_direct
import pyscf.tools.fcidump

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


def simple_hf(h1e, eri, mo, nelec):
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = '/dev/null'
    mol.build(False, False)
    mf = scf.RHF(mol)
    nocc = nelec / 2
    mf.init_guess_method = \
            lambda mol: (0, numpy.dot(mo[:,:nocc],mo[:,:nocc].T)*2)
    mf.get_hcore = lambda mol: h1e
    mf.get_ovlp = lambda mol: numpy.eye(mo.shape[1])
    def _set_mo_occ(mo_energy, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:nocc] = 2
        return mo_occ
    mf.set_mo_occ = _set_mo_occ
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



#FCIEXE = os.path.dirname(__file__) + '/fci'
#
#def fci(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag):
#    tmpfile = tempfile.NamedTemporaryFile()
#    nmo = mo.shape[1]
#    tools.fcidump.from_integrals(tmpfile.name, h1e, eri, nmo, nelec, 0)
#
#    cmd = [FCIEXE,
#           '--subspace-dimension=16',
#           '--basis=Input']
#    if with_1pdm:
#        filedm1 = tempfile.NamedTemporaryFile()
#        cmd.append('--save-rdm1=%s' % filedm1.name)
#    if with_e2frag:
#        nimp = with_e2frag
#        cmd.append('--ptrace=%i' % with_e2frag)
#    cmd.append(tmpfile.name)
#
#    rec = commands.getoutput(' '.join(cmd))
#
#    rdm1 = []
#    if with_1pdm:
#        with open(filedm1.name, 'r') as fin:
#            n = int(fin.readline().split()[-1])
#            for d in fin.readlines():
#                rdm1.append(map(float, d.split()))
#        rdm1 = numpy.array(rdm1).reshape(n,n)
#
#    e2frag = 0
#    if with_e2frag:
#        e2frag = find_fci_key(rec, '!FCI STATE 1 pTraceSys')
#
#    return 0, find_fci_key(rec, '!FCI STATE 1 ENERGY'), e2frag, rdm1
#
#def find_fci_key(rec, key):
#    for dl in rec.splitlines():
#        if key in dl:
#            val = float(dl.split()[-1])
#            break
#    return val

def fci(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag):
    norb = h1e.shape[1]
    eci, c = fci_direct.kernel(h1e, eri, norb, nelec)
    if with_1pdm:
        dm1 = fci_direct.make_rdm1(c, norb, nelec)
    else:
        dm1 = None
    if with_e2frag:
        eri1 = part_eri_hermi(eri, norb, with_e2frag)
        e2frag = fci_direct.energy(numpy.zeros_like(h1e), eri1, c, norb, nelec)
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

