#!/usr/bin/env python
# -*- coding: utf-8
#
# File: hf.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Non-relativistic Hartree-Fock
'''

__author__ = 'Qiming Sun <osirpt.sun@gmail.com>'

import sys
import time
import re

import numpy
#import scipy.linalg.flapack as lapack
import gto
import scf
import lib
import lib.logger as log
import lib.parameters as param

def read_fcidump(fcidump):
    sys.stdout.write('Start reading %s\n' % fcidump)
    dic = {}
    finp = open(fcidump, 'r')
    dat = re.split('[=,]', finp.readline())
    while dat[0]:
        if 'FCI' in dat[0].upper():
            dic['NORB'] = int(dat[1])
            dic['NELEC'] = int(dat[3])
            dic['MS2'] = int(dat[5])
        elif 'UHF' in dat[0].upper():
            if 'TRUE' in dat[1].upper():
                dic['UHF'] = True
            else:
                dic['UHF'] = False
        elif 'ORBSYM' in dat[0].upper():
            dic['ORBSYM'] = map(int, finp.readline().split(',')[:-1])
        elif 'END' in dat[0].upper():
            break
        dat = re.split('[=,]', finp.readline())
    norb = dic['NORB']
    mo_energy = numpy.zeros(norb)
    h1e = numpy.zeros((norb,norb))
    npair = norb*(norb+1)/2
    eri = numpy.zeros(npair*(npair+1)/2)
    dat = finp.readline().split()
    while dat:
        dat = dat + finp.readline().split()
        i, j, k, l = map(int, dat[1:])
        val = map(float, dat[0][1:-1].split(','))
        if abs(val[1]) > 1e-12:
            print i,j,k,l
        assert(abs(val[1]) < 1e-12)
        if j == 0:
            mo_energy[i-1] = val[0]
        elif k == 0:
            h1e[i-1,j-1] = h1e[j-1,i-1] = val[0]
        else:
            if i >= j:
                ij = (i-1)*i/2 + j-1
            else:
                ij = (j-1)*j/2 + i-1
            if k >= l:
                kl = (k-1)*k/2 + l-1
            else:
                kl = (l-1)*l/2 + k-1
            if ij >= kl:
                eri[ij*(ij+1)/2+kl] = val[0]
            else:
                eri[kl*(kl+1)/2+ij] = val[0]
        dat = finp.readline().split()
    dic['HCORE'] = h1e
    dic['ERI'] = eri
    dic['MO_ENERGY'] = mo_energy
    sys.stdout.write('Finish reading %s\n' % fcidump)
    return dic

class RHF(scf.hf.RHF):
    ''' RHF from vasp FCIDUMP'''
    def __init__(self, mol, fcidump):
        scf.hf.RHF.__init__(self, mol)
        self._fcidump = read_fcidump(fcidump)
        norb = self._fcidump['NORB']
        def _initguess(mol):
            dm = numpy.zeros((norb,norb))
            for i in range(self._fcidump['NELEC']/2):
                dm[i,i] = 2
            return 0, dm
        self.init_guess_method = _initguess
        self._eri = self._fcidump['ERI']
        self.eri_in_memory = True

    def get_hcore(self, mol=None):
        return self._fcidump['HCORE']

    def get_ovlp(self, mol=None):
        return numpy.eye(self._fcidump['NORB'])

    def set_mo_occ(self, mo_energy, mo_coeff=None):
        mo_occ = numpy.zeros_like(mo_energy)
        nocc = self._fcidump['NELEC'] / 2
        mo_occ[:nocc] = 2
        if nocc < mo_occ.size:
            log.debug(self, 'HOMO = %.12g, LUMO = %.12g,', \
                      mo_energy[nocc-1], mo_energy[nocc])
        else:
            log.debug(self, 'HOMO = %.12g,', mo_energy[nocc-1])
        log.debug(self, '  mo_energy = %s', mo_energy)
        return mo_occ

    def scf(self):
        self.dump_scf_option()

        self.scf_conv, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = self.scf_cycle(self.mol, self.scf_threshold)
        return self.hf_energy

if __name__ == '__main__':
    import pickle
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_hf'
    mol.build()

    mf = RHF(mol, 'C_solid_2x2x2/test2/FCIDUMP')
    energy = mf.scf()
    print energy

    mf._fcidump['HCORE']
    dm = mf.calc_den_mat()
    vj1, vk1 = scf.hf.dot_eri_dm(mf._fcidump['ERI'], dm)
    vj0, vk0 = scf.hf.dot_eri_dm(mf._fcidump['ERI'],
                               mf.init_guess_method(mol)[1])
    print abs(vj0-vj1).sum()
    print abs(vk0-vk1).sum()
