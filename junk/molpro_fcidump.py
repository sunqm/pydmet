#!/usr/bin/env python
# $Id$
# -*- coding: utf-8

'''
generate ERI_MO
'''

__author__ = "Qiming Sun <osirpt.sun@gmail.com>"
__version__ = "$ 0.1 $"

import os, sys
import tempfile
import ctypes
import cPickle as pickle
import numpy
from pyscf import scf

def head(nmo, nelec, fout):
    fout.write(' &FCI NORB= %3d,NELEC=%2d,MS2= 0,\n' % (nmo, nelec))
    fout.write('  ORBSYM=%s\n' % ('1,' * nmo))
    fout.write('  ISYM=1,\n')
    fout.write(' &END\n')

def write_eri_in_molpro_format(eri, fout):
    n = int(numpy.sqrt(eri.shape[0]*2))
    ij = 0
    for i in range(n):
        for j in range(0, i+1):
            kl = 0
            for k in range(0, i+1):
                for l in range(0, k+1):
                    if ij >= kl:
                        fout.write(' %.16g%3d%3d%3d%3d\n' \
                                   % (eri[ij,kl], i+1, j+1, k+1, l+1))
                    kl += 1
            ij += 1

def write_hcore_in_molpro_format(h, fout):
    n = h.shape[1]
    for i in range(n):
        for j in range(0, i+1):
            fout.write(' %.16g%3d%3d  0  0\n' % (h[i,j], i+1, j+1))

def eri(mol, mo_coeff, fout):
    import ao2mo
    eri_mo = ao2mo.gen_int2e_ao2mo(mol, mo_coeff)
    write_eri_in_molpro_format(eri_mo, fout)

def hcore(mol, mo_coeff, fout):
    t = mol.intor_symmetric('cint1e_kin_sph')
    v = mol.intor_symmetric('cint1e_nuc_sph')
    h = t + v
    h = reduce(numpy.dot, (mo_coeff.T.conj(), h, mo_coeff))
    write_hcore_in_molpro_format(h, fout)


def fcidump(chkfile, output):
    with open(output, 'w') as fout:
        mol, scf_rec = scf.hf.read_chkfile(chkfile)
        head(scf_rec['mo_coeff'].shape[1], mol.nelectron, fout)
        eri(mol, scf_rec['mo_coeff'], fout)
        hcore(mol, scf_rec['mo_coeff'], fout)
        fout.write(' %.16g  0  0  0  0\n' % mol.nuclear_repulsion())

if __name__ == '__main__':
    # molpro_fcidump.py chkfile output
    fcidump(sys.argv[1], sys.argv[2])
