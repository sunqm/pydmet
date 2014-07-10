#!/usr/bin/env python

import os
import tempfile
import commands
import numpy

from pyscf import lib
from pyscf import ao2mo
import psi4

FCIEXE = os.path.dirname(__file__) + '/fci'

def scf_energy(h1e_mo, h2e_mo, nocc):
    escf = 0
    jk = 0
    for i in range(nocc):
        escf += h1e_mo[i,i]
        ii = i*(i+1)/2 + i
        for j in range(nocc):
            jj = j*(j+1)/2 + j
            if i < j:
                ij = j*(j+1)/2+i
            else:
                ij = i*(i+1)/2+j
            escf += (h2e_mo[ii,jj] - h2e_mo[ij,ij] * .5)
            jk += h2e_mo[ii,jj] - h2e_mo[ij,ij] * .5
    return escf * 2

def cc(mol, nelec, h1e, h2e, ptrace, mo):
    h1e = reduce(numpy.dot, (mo.T, h1e, mo))
    eri1 = numpy.empty(h2e.size)
    ij = 0
    for i in range(h2e.shape[0]):
        for j in range(i+1):
            eri1[ij] = h2e[i,j]
            ij += 1
    eri = ao2mo.incore.full(eri1, mo)
    escf = scf_energy(h1e, eri, nelec/2)

    ps = psi4.Solver()
    with psi4.capture_stdout():
        nmo = h1e.shape[0]
        ps.prepare('RHF', numpy.eye(nmo), h1e, eri, nelec)
        ecc = ps.energy('CCSD')
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
    e1_ptrace = lib.trace_ab(frag_rdm1.reshape(-1), h1e.reshape(-1))

    eri_full = numpy.empty((nmo,nmo,nmo,nmo))
    ij = 0
    for i in range(nmo):
        for j in range(i+1):
            kl = 0
            for k in range(nmo):
                for l in range(k+1):
                    eri_full[i,k,j,l] = \
                    eri_full[j,k,i,l] = \
                    eri_full[i,l,j,k] = \
                    eri_full[j,l,i,k] = eri[ij,kl]
                    kl += 1
            ij += 1
    frag_rdm2 = numpy.dot(p, rdm2.reshape(nmo,-1))
    e2_ptrace = lib.trace_ab(frag_rdm2.reshape(-1), eri_full.reshape(-1))
    e_ptrace = e1_ptrace + e2_ptrace * .5

    #print ecc, escf
    res = {'rdm1': reduce(numpy.dot, (mo, rdm1, mo.T)),
           #'escf': escf,
           'etot': ecc+escf,
           'e1frag': e1_ptrace,
           'e2frag': e2_ptrace*.5}

    return res

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

