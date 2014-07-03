#!/usr/bin/env python

import os
import tempfile
import commands
import numpy
import ao2mo
import psi4
import junk
import junk.molpro_fcidump

FCIEXE = os.path.dirname(__file__) + '/fci'

def cc(mol, nelec, h1e, h2e, rdm1=None, ptrace=0):
    ps = psi4.Solver()
    with psi4.capture_stdout():
        nmo = h1e.shape[0]
        ps.prepare('RHF', numpy.eye(nmo), h1e, h2e, nelec)
        ecc = ps.energy('CCSD')
        if rdm1 is not None:
            dm1, dm2 = ps.density()
    nocc = nelec / 2
    escf = 0
    jk = 0
    for i in range(nocc):
        escf += h1e[i,i]
        ii = i*(i+1)/2 + i
        for j in range(nocc):
            jj = j*(j+1)/2 + j
            if i < j:
                ij = j*(j+1)/2+i
            else:
                ij = i*(i+1)/2+j
            escf += (h2e[ii,jj] - h2e[ij,ij] * .5)
            jk += h2e[ii,jj] - h2e[ij,ij] * .5
    print ecc, escf * 2
    res = [ecc+escf*2]
    if rdm1 is not None:
        res.append(dm1)
    return res

def fci(mol, nelec, h1e, h2e, rdm1=None, ptrace=0):
    tmpfile = tempfile.mkstemp()[1]
    with open(tmpfile, 'w') as fout:
        junk.molpro_fcidump.head(h1e.shape[0], nelec, fout)
        junk.molpro_fcidump.write_eri_in_molpro_format(h2e, fout)
        junk.molpro_fcidump.write_hcore_in_molpro_format(h1e, fout)
        fout.write(' 0.0  0  0  0  0\n')

    cmd = [FCIEXE,
           '--subspace-dimension=16',
           '--basis=Input',]
    if ptrace > 0:
        cmd.append('--ptrace=%i' % ptrace)
    if rdm1 is not None:
        filedm1 = tempfile.mkstemp()[1]
        cmd.append('--save-rdm1=%s' % filedm1)
    #if rdm2 is not None:
    #    filedm2 = tempfile.mkstemp()[1]
    #    cmd.append('--save-rdm2=%s' % filedm2)
    cmd.append(tmpfile)

    rec = commands.getoutput(' '.join(cmd))
    e = find_fci_key(rec, '!FCI STATE 1 ENERGY')
    res = [e]
    if rdm1 is not None:
        dm1 = []
        with open(filedm1, 'r') as fin:
            n = int(fin.readline().split()[-1])
            for d in fin.readlines():
                dm1.append(map(float, d.split()))
        res.append(numpy.array(dm1).reshape(n,n))
        os.remove(filedm1)
    #if rdm2 is not None:
    #    dm2 = []
    #    with open(filedm1, 'r') as fin:
    #        n = int(numpy.sqrt(int(fin.readline().split()[-1])))
    #        for d in fin.readlines():
    #            dm2.append(map(float, d.split()))
    #    res.append(numpy.array(dm1).reshape(n,n,n,n))
    #    os.remove(filedm2)
    os.remove(tmpfile)
    return res

def find_fci_key(rec, key):
    for dl in rec.splitlines():
        if key in dl:
            val = float(dl.split()[-1])
            break
    return val

def use_local_solver(local_solver, with_rdm1=None):
    def imp_solver(mol, emb, vfit=0):
        h1e = reduce(numpy.dot, (emb.mo_coeff_on_imp.T, emb.get_hcore(),\
                                 emb.mo_coeff_on_imp))
        if vfit is not 0:
            nv = vfit.shape[0]
            h1e[:nv,:nv] += vfit
        if emb._eri is None:
            eri = emb.eri_on_impbas(mol)
        else:
            eri = emb._eri
        eri1 = numpy.zeros(eri.size)
        ij = 0
        for i in range(eri.shape[0]):
            for j in range(i+1):
                eri1[ij] = eri[i,j]
                ij += 1
        eri = ao2mo.incore.full(eri1, emb.mo_coeff_on_imp)
        nelec = emb.nelectron
        nimp = len(emb.bas_on_frag)
        res = local_solver(mol, nelec, h1e, eri, with_rdm1)#, ptrace=nimp)
        if with_rdm1 is not None:
            return res[0], reduce(numpy.dot, (emb.mo_coeff_on_imp, res[1], \
                                              emb.mo_coeff_on_imp.T))
        else:
            return res
    return imp_solver
