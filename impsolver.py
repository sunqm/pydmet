#!/usr/bin/env python

import os
import tempfile
import commands
import numpy

from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.lib import _vhf
from pyscf.lib import _ao2mo
from pyscf import ao2mo
from pyscf.future import mcscf
from pyscf.future import fci
import psi4

FCIEXE = os.path.dirname(__file__) + '/fci'

def _scf_energy(h1e, h2e, mo, nocc):
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = '/dev/null'
    mol.build(False, False)
    mf = scf.RHF(mol)
    mf.init_guess_method = \
            lambda mol: (0, numpy.dot(mo[:,:nocc],mo[:,:nocc].T)*2)
    mf.get_hcore = lambda mol: h1e
    mf.get_ovlp = lambda mol: numpy.eye(mo.shape[1])
    def _set_mo_occ(mo_energy, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        mo_occ[:nocc] = 2
        return mo_occ
    mf.set_mo_occ = _set_mo_occ
    def _get_veff(mol, dm, dm_last=0, vhf_last=0):
        vj, vk = _vhf.vhf_jk_incore_o2(h2e, dm)
        return vj - vk * .5
    mf.get_veff = _get_veff

    scf_conv, hf_energy, mo_energy, mo_occ, mo_coeff \
            = mf.scf_cycle(mol, 1e-9)
    return hf_energy, mo_energy, mo_occ, mo_coeff

def cc(mol, nelec, h1e, h2e, ptrace, mo, ccname='CCSD'):
    hf_energy, mo_energy, mo_occ, mo = \
            _scf_energy(h1e, h2e, mo, nelec/2)
    h1e = reduce(numpy.dot, (mo.T, h1e, mo))
    eri = _ao2mo.partial_eri_o2(h2e, mo)
    #eri1 = numpy.empty(h2e.size)
    #ij = 0
    #for i in range(h2e.shape[0]):
    #    for j in range(i+1):
    #        eri1[ij] = h2e[i,j]
    #        ij += 1
    #eri = ao2mo.incore.full(eri1, mo)

    ps = psi4.Solver(max_memory=1<<34)
    with psi4.capture_stdout():
        nmo = h1e.shape[0]
        ps.prepare('RHF', numpy.eye(nmo), h1e, eri, nelec)
        ecc = ps.energy(ccname)
        rdm1, rdm2 = ps.density()
        rdm1 *= 2 # Psi4 gives rdm1 of alpha spin

# note the rdm1,rdm2 from psi4 solver EXCLUDES HF contributions
    for i in range(nelec/2):
        rdm1[i,i] += 2
        for j in range(nelec/2):
            rdm2[i,j,i,j] += 4
            rdm2[i,j,j,i] +=-2

    nmo = mo.shape[1]
    p = numpy.dot(mo[:ptrace,:].T, mo[:ptrace,:])
    frag_rdm1 = numpy.dot(p, rdm1)
    e1_ptrace = numpy.dot(frag_rdm1.reshape(-1), h1e.reshape(-1))

    eri_full = ao2mo.restore(1, eri, nmo).transpose(0,2,1,3)
    #eri_full = numpy.empty((nmo,nmo,nmo,nmo))
    #ij = 0
    #for i in range(nmo):
    #    for j in range(i+1):
    #        kl = 0
    #        for k in range(nmo):
    #            for l in range(k+1):
    #                eri_full[i,k,j,l] = \
    #                eri_full[j,k,i,l] = \
    #                eri_full[i,l,j,k] = \
    #                eri_full[j,l,i,k] = eri[ij,kl]
    #                kl += 1
    #        ij += 1
    frag_rdm2 = numpy.dot(p, rdm2.reshape(nmo,-1))
    e2_ptrace = numpy.dot(frag_rdm2.reshape(-1), eri_full.reshape(-1))
    e_ptrace = e1_ptrace + e2_ptrace * .5

    #print ecc, hf_energy
    res = {'rdm1': reduce(numpy.dot, (mo, rdm1, mo.T)),
           #'escf': hf_energy,
           'etot': ecc+hf_energy,
           'e1frag': e1_ptrace,
           'e2frag': e2_ptrace*.5}

    return res

def ccsd(mol, nelec, h1e, h2e, ptrace, mo):
    return cc(mol, nelec, h1e, h2e, ptrace, mo, 'CCSD')

def ccsd_t(mol, nelec, h1e, h2e, ptrace, mo):
    return cc(mol, nelec, h1e, h2e, ptrace, mo, 'CCSD(T)')

def _fcidump(fout, nelec, hcore, eri):
    nmo = hcore.shape[0]
    fout.write(' &FCI NORB= %3d,NELEC=%2d,MS2= 0,\n' % (nmo, nelec))
    fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')

    ij = 0
    for i in range(nmo):
        for j in range(0, i+1):
            kl = 0
            for k in range(0, i+1):
                for l in range(0, k+1):
                    if ij >= kl:
                        fout.write(' %.16g%3d%3d%3d%3d\n' \
                                   % (eri[ij,kl], i+1, j+1, k+1, l+1))
                    kl += 1
            ij += 1

    for i in range(nmo):
        for j in range(0, i+1):
            fout.write(' %.16g%3d%3d  0  0\n' % (hcore[i,j], i+1, j+1))

def fci(mol, nelec, h1e, h2e, ptrace, mo):
    tmpfile = tempfile.mkstemp()[1]
    with open(tmpfile, 'w') as fout:
        _fcidump(fout, nelec, h1e, h2e)
        fout.write(' 0.0  0  0  0  0\n')

    cmd = [FCIEXE,
           '--subspace-dimension=16',
           '--basis=Input',
           '--ptrace=%i' % ptrace]
    filedm1 = tempfile.mkstemp()[1]
    cmd.append('--save-rdm1=%s' % filedm1)
    #if rdm2 is not None:
    #    filedm2 = tempfile.mkstemp()[1]
    #    cmd.append('--save-rdm2=%s' % filedm2)
    cmd.append(tmpfile)

    rec = commands.getoutput(' '.join(cmd))
    rdm1 = []
    with open(filedm1, 'r') as fin:
        n = int(fin.readline().split()[-1])
        for d in fin.readlines():
            rdm1.append(map(float, d.split()))
    rdm1 = numpy.array(rdm1).reshape(n,n)
    e1_ptrace = numpy.dot(rdm1[:ptrace].flatten(), h1e[:ptrace].flatten())
    e2_ptrace = find_fci_key(rec, '!FCI STATE 1 pTraceSys')
    e_ptrace = e1_ptrace + e2_ptrace

    os.remove(filedm1)
    os.remove(tmpfile)
    res = {'rdm1': rdm1, \
           'etot': find_fci_key(rec, '!FCI STATE 1 ENERGY'), \
           'e1frag': e1_ptrace, \
           'e2frag': e2_ptrace,
           'rec': rec}

    return res

def find_fci_key(rec, key):
    for dl in rec.splitlines():
        if key in dl:
            val = float(dl.split()[-1])
            break
    return val

def use_local_solver(local_solver, with_rdm1=None):
    def imp_solver(mol, emb, vfit=0):
        h1e = emb.get_hcore(mol)
        if not (isinstance(vfit, int) and vfit is 0):
            nv = vfit.shape[0]
            h1e[:nv,:nv] += vfit
        if emb._eri is None:
            eri = emb.eri_on_impbas(mol)
        else:
            eri = emb._eri
        nelec = emb.nelectron
        nimp = len(emb.bas_on_frag)
        res = local_solver(mol, nelec, h1e, eri, nimp, emb.mo_coeff_on_imp)
        return res
    return imp_solver


#FIXME
def casscf(mol, emb, vfit=0):
    mc = mcscf.CASSCF(mol, emb, ncas, nelecas, ncore)
    def get_hcore(*args):
        h1e = emb.get_hcore(mol)
        if not (isinstance(vfit, int) and vfit is 0):
            nv = vfit.shape[0]
            h1e[:nv,:nv] += vfit
        return h1e
    mc.get_hcore = get_hcore
    e_tot, e_ci, ci0, mo = mc.mc1step(mo=emb.mo_coeff_on_imp)
    import mcscf
    dm1, dm2 = mcscf.addons.make_rdm12(mc, ci0, mo)
    eri_full = ao2mo.restore(1, emb._eri, nmo)
    e1_ptrace = numpy.dot(frag_rdm1.reshape(-1), h1e.reshape(-1))

    frag_rdm2 = numpy.dot(p, rdm2.reshape(nmo,-1))
    e2_ptrace = numpy.dot(frag_rdm2.reshape(-1), eri_full.reshape(-1))
    e_ptrace = e1_ptrace + e2_ptrace * .5
    res = {'rdm1': dm1,
           #'escf': hf_energy,
           'etot': e_tot,
           'e1frag': e1_ptrace,
           'e2frag': e2_ptrace*.5}

    return res
