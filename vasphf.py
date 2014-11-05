#!/usr/bin/env python
# -*- coding: utf-8
#
# File: hf.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import sys
import re

import numpy
#import scipy.linalg.flapack as lapack
from pyscf import gto
from pyscf import scf
from pyscf import lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
import vasp


class RHF(scf.hf.RHF):
    ''' RHF from vasp FCIDUMP'''
    def __init__(self, mol, clustdump, jkdump):
        scf.hf.RHF.__init__(self, mol)
        self._fcidump = read_clustdump(clustdump)
        self._fcidump.update(read_jkdump(jkdump))

        # transform back to AO representation
        hcore = numpy.diag(self._fcidump['MO_ENERGY'])
        hcore -= (self._fcidump['J'] - self._fcidump['K'])
        mo = self._fcidump['MO_COEFF']
        self._hcore = reduce(numpy.dot, (mo, hcore, mo.T))

        def _initguess(mol):
            nocc = self._fcidump['NELEC'] / 2
            mo = self._fcidump['MO_COEFF']
            dm = numpy.dot(mo[:,:nocc],mo[:,:nocc].T) * 2
            return 0, dm
        self.make_init_guess = _initguess
        self._eri = self._fcidump['ERI']

    def get_hcore(self, mol=None):
        return self._hcore

    def get_ovlp(self, mol=None):
        return numpy.eye(self._fcidump['NORB'])

    def set_occ(self, mo_energy, mo_coeff=None):
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
        self.dump_flags()

        self.scf_conv, self.hf_energy, \
                self.mo_energy, self.mo_occ, self.mo_coeff \
                = self.scf_cycle(self.mol, self.conv_threshold)
        return self.hf_energy

class RHF4test(scf.hf.RHF):
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
        self.make_init_guess = _initguess
        self._eri = self._fcidump['ERI']

    def get_hcore(self, mol=None):
        return self._fcidump['HCORE']

    def get_ovlp(self, mol=None):
        return numpy.eye(self._fcidump['NORB'])

    def set_occ(self, mo_energy, mo_coeff=None):
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
                = self.scf_cycle(self.mol, self.conv_threshold)
        return self.hf_energy

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

def read_fcidump_gto(fcidump):
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
    dic['ERI'] = eri
    sys.stdout.write('Finish reading %s\n' % fcidump)
    return dic

def read_clustdump(fcidump):
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
        elif 'ORBIND' in dat[0].upper():
            dic['ORBIND'] = map(int, finp.readline().split(',')[:-1])
        elif 'END' in dat[0].upper():
            break
        dat = re.split('[=,]', finp.readline())
    norb = dic['NORB']
    mo_coeff = numpy.zeros((norb,norb))
    npair = norb*(norb+1)/2
    eri = numpy.zeros(npair*(npair+1)/2)
    dat = finp.readline().split()
    while dat:
        i, j, k, l = map(int, dat[2:])
        val = map(float, dat[:2])
        if abs(val[1]) > 1e-12:
            print i,j,k,l
        assert(abs(val[1]) < 1e-12)
        if l == -1:
            mo_coeff[i-1,j-1] = val[0]
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
    dic['MO_COEFF'] = mo_coeff
    dic['ERI'] = eri
    sys.stdout.write('Finish reading %s\n' % fcidump)
    return dic

def read_jkdump(fcidump):
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
        elif 'END' in dat[0].upper():
            break
        dat = re.split('[=,]', finp.readline())
    norb = dic['NORB']
    mo_energy = numpy.zeros(norb)
    vj = numpy.zeros((norb,norb))
    vk = numpy.zeros((norb,norb))
    npair = norb*(norb+1)/2
    eri = numpy.zeros(npair*(npair+1)/2)
    dat = finp.readline().split()
    while dat:
        i, j, k, l = map(int, dat[2:])
        val = map(float, dat[:2])
        if abs(val[1]) > 1e-12:
            print i,j,k,l
        assert(abs(val[1]) < 1e-12)
        if j == 0:
            mo_energy[i-1] = val[0]
        elif i < 0:
            vk[-i-1,-j-1] = val[0]
        else:
            vj[i-1,j-1] = val[0]
        dat = finp.readline().split()
    dic['MO_ENERGY'] = mo_energy
    dic['J'] = vj
    dic['K'] = vk
    sys.stdout.write('Finish reading %s\n' % fcidump)
    return dic


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_hf'
    mol.build()

#    mf = RHF4test(mol, 'C_solid_2x2x2/test2/FCIDUMP')
#    energy = mf.scf()
#    print energy
#
#    #print abs(mf._fcidump['HCORE'] - mf._fcidump['HCORE'].T).sum()
#    dm = mf.make_rdm1()
#    fcidump0 = read_fcidump('C_solid_2x2x2/test2/FCIDUMP')
#    vj1, vk1 = scf.hf.dot_eri_dm(fcidump0['ERI'], dm)
#    vj0, vk0 = scf.hf.dot_eri_dm(fcidump0['ERI'],
#                               mf.make_init_guess(mol)[1])
#    print abs(vj0-vj1).sum()
#    print abs(vk0-vk1).sum()
#
#    fcidump1 = read_clustdump('C_solid_2x2x2/test2/FCIDUMP.CLUST.GTO')
#    fcidump2 = read_jkdump('C_solid_2x2x2/test2/JKDUMP')
#    nocc = fcidump1['NELEC'] / 2
#    mo = fcidump1['MO_COEFF']
#    dm = numpy.dot(mo[:,:nocc], mo[:,:nocc].T) * 2
#    vj2, vk2 = scf.hf.dot_eri_dm(fcidump1['ERI'], dm)
#    print abs(fcidump2['J'] - reduce(numpy.dot, (mo.T,vj2,mo))).sum()
#    print abs(fcidump2['K'] - .5*reduce(numpy.dot, (mo.T,vk2,mo))).sum()
#    print abs(fcidump2['J'] - vj0).sum()
#    print abs(fcidump2['K'] - .5*vk0).sum()
#    print abs(vj1 - vj2).sum()
#    print abs(vj1 - vk2).sum()
#    print numpy.linalg.det(fcidump1['MO_COEFF'])

    mf = RHF(mol, 'test/C_solid_2x2x2/test2/FCIDUMP.CLUST.GTO',
             'test/C_solid_2x2x2/test2/JKDUMP')
    energy = mf.scf()
    print energy
    print mf.mo_energy
    print mf._fcidump['MO_ENERGY']
