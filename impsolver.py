#!/usr/bin/env python

import os
import tempfile
import commands
import ao2mo
import psi4
import junk
import junk.molpro_fcidump

FCIEXE = os.path.dirname(__file__) + '/fci'

def cc(mol, nelec, h1e, h2e, rdm1=None, rdm2=None, ptrace=0):
    ps = psi4.Solver()
    with psi4.capture_stdout():
        nmo = h1e.shape[0]
        ps.prepare('RHF', numpy.eye(nmo), h1e, h2e, nelec)
        ecc = ps.energy('CCSD')
        if rdm1 is not None or rdm2 is not None:
            dm1, dm2 = ps.density()
    res = [ecc]
    if rdm1 is not None:
        res.append(dm1)
    if rdm2 is not None:
        res.append(dm2)
    return res

def fci(mol, nelec, h1e, h2e, rdm1=None, rdm2=None, ptrace=0):
    tmpfile = tempfile.mkstemp()
    with open(tmpfile, 'w') as fout:
        junk.molpro_fcidump.head(h1e.shape[0], nelec, fout)
        junk.molpro.write_eri_in_molpro_format(h2e, fout)
        junk.molpro.write_hcore_in_molpro_format(h1e, fout)
        fout.write(' 0.0  0  0  0  0\n')

    cmd = [FCIEXE,
           '--subspace-dimension=16',
           '--basis=Input',]
    if ptrace > 0:
        cmd.append('--ptrace=%i' % ptrace)
    if rdm1 is not None:
        filedm1 = tempfile.mkstemp()
        cmd.append('--save-rdm1=%s' % filedm1)
    if rdm2 is not None:
        filedm2 = tempfile.mkstemp()
        cmd.append('--save-rdm2=%s' % filedm2)

    rec = commands.getoutput(' '.join(cmd))
    e = find_fci_key(rec, '!%s STATE 1')
    res = [e]
    if rdm1 is not None:
        dm1 = []
        with open(filedm1, 'r') as fin:
            n = int(fin.readline().split()[-1])
            for d in fin.readlines():
                dm1.append(map(float, d.split()))
        res.append(numpy.array(dm1).reshape(n,n))
        os.remove(filedm1)
    if rdm2 is not None:
        dm2 = []
        with open(filedm1, 'r') as fin:
            n = int(numpy.sqrt(int(fin.readline().split()[-1])))
            for d in fin.readlines():
                dm2.append(map(float, d.split()))
        res.append(numpy.array(dm1).reshape(n,n,n,n))
        os.remove(filedm2)
    return res

def find_fci_key(rec, key):
    for dl in rec.splitlines():
        if key in dl:
            val = float(dl.split()[-1])
            break
    return val

def use_local_solver(local_solver, with_rdm1=None, with_rdm2=None):
    def imp_solver(mol, emb):
        h1e = emb.get_hcore()
        if vfit is not 0:
            nv = vfit.shape[0]
            h1e[:nv,:nv] += vfit
        int2e = ao2mo.incore.full(emb._eri, emb.impbas_coeff)
        nelec = emb.nelectron
        nimp = emb.dim_of_impurity()
        return local_solver(mol, nelec, h1e, int2e, \
                            with_rdm1, with_rdm2, ptrace=nimp)
    return imp_solver
