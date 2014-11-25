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
    def __init__(self, mol, clustdump, jdump, kdump, fockdump):
        scf.hf.RHF.__init__(self, mol)
        self._vaspdump = read_vaspdump(clustdump, jdump, kdump, fockdump)
        self.mo_coeff = self._vaspdump['MO_COEFF']
        self.mo_energy = self._vaspdump['MO_ENERGY']
        self.mo_occ = numpy.zeros(self._vaspdump['NORB'])
        self.mo_occ[:self._vaspdump['NELEC']/2] = 2
        self.mol.nelectron = self._vaspdump['NELEC']

    def get_hcore(self, mol=None):
#TODO read FOCKDUMP, this Fock matrix excludes the correlation potential, it
# is not identical to the converged Fock matrix of the VASP SCF Fock
# plus CORRPOTDUMP (if exists) gives the VASP SCF Fock
        return self._vaspdump['HCORE']

    def get_ovlp(self, mol=None):
        return numpy.eye(self._vaspdump['NORB'])


def read_vaspdump(clustdump, jdump, kdump, fockdump):
#NOTE read_hfdump returns the integrals in MO representation
    hfdic = read_hfdump(jdump, kdump, fockdump)
    dic = read_clustdump(clustdump, hfdic)
    mo_coeff = dic['MO_COEFF']
    hfdic['HCORE'] = reduce(numpy.dot, (mo_coeff, hfdic['HCORE'], mo_coeff.T))
    hfdic['J'] = reduce(numpy.dot, (mo_coeff, hfdic['J'], mo_coeff.T))
    hfdic['K'] = reduce(numpy.dot, (mo_coeff, hfdic['K'], mo_coeff.T))
    dic.update(hfdic)
    return dic

def read_clustdump(clustdump, hfdic):
# ERIs on embedding basis
# 1-electron Hamiltonian on embedding basis (include correlation potential,
# which is identical to <embasis|FOCK+CORRPOT-J-K|embasis>) x x 0 0
# MO coefficients  x x 0 -1
# embedding basis (represented on MO)  x x 0 -2
# correlation potential on embedding basis x x 0 -3
    dic = {}
    sys.stdout.write('Start reading %s\n' % clustdump)
    finp = open(clustdump, 'r')
    dat = re.split('[=,]', finp.readline())
    while dat[0]:
        if 'FCI' in dat[0].upper():
            dic['NEMB'] = int(dat[1])
            dic['NIMP'] = int(dat[3])
            dic['NBATH'] = int(dat[5])
        elif 'UHF' in dat[0].upper():
            if 'TRUE' in dat[1].upper():
                dic['UHF'] = True
            else:
                dic['UHF'] = False
        elif 'ORBIND' in dat[0].upper():
            idx = map(int, finp.readline().split(',')[:-1])
            dic['ORBIND'] = [i-1 for i in idx]  # transform to 0-based indices
        elif 'END' in dat[0].upper():
            break
        dat = re.split('[=,]', finp.readline())
    nemb = dic['NEMB']
    norb = hfdic['NORB']
    npair = nemb*(nemb+1)/2
    h1emb = numpy.zeros((nemb,nemb))
    corrpot = numpy.zeros((nemb,nemb))
    mo_coeff = numpy.zeros((norb,norb))
    embasis = numpy.zeros((norb,nemb))
    eri = numpy.zeros((npair*(npair+1)/2))
    dat = finp.readline().split()
    while dat:
        i, j, k, l = map(int, dat[1:])
        if l == 0:
            h1emb[i-1,j-1] = h1emb[j-1,i-1] = float(dat[0])
        elif l == -1:
            mo_coeff[i-1,j-1] = float(dat[0])
        elif l == -2:
            embasis[i-1,j-1] = float(dat[0])
        elif l == 0:
            corrpot[i-1,j-1] = corrpot[j-1,i-1] = float(dat[0])
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
                eri[ij*(ij+1)/2+kl] = float(dat[0])
            else:
                eri[kl*(kl+1)/2+ij] = float(dat[0])
        dat = finp.readline().split()
    dic['ERI'] = eri
    dic['MO_COEFF'] = mo_coeff
    dic['EMBASIS'] = numpy.dot(mo_coeff, embasis)
    dic['H1EMB'] = h1emb
    dic['CORRPOT'] = corrpot
    return dic

def read_hfdump(jdump, kdump, fockdump):
# FOCKDUMP, JDUMP, KDUMP in MO representation
# FOCKDUMP:
# Fock matrix (exclude correlation potential)  x x 0 0
# MO orbital energy  x 0 0 0
# JDUMP:
# 2(pq|ii)  x x 0 0
# KDUMP:
# -(pi|iq)  x x 0 0
    dic = {}
    sys.stdout.write('Start reading %s\n' % fockdump)
    finp = open(fockdump, 'r')
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
    fock = numpy.zeros((norb,norb))
    mo_energy = numpy.zeros(norb)
    dat = finp.readline().split()
    while dat:
        i, j = map(int, dat[1:3])
        if j == 0:
            mo_energy[i-1] = float(dat[0])
        else:
            fock[i-1,j-1] = float(dat[0])
        dat = finp.readline().split()

    sys.stdout.write('Start reading %s\n' % jdump)
    vj = numpy.zeros((norb,norb))
    vk = numpy.zeros((norb,norb))
    finp = open(jdump, 'r')
    dat = finp.readline()
    while 'END' not in dat:
        dat = finp.readline()
    dat = finp.readline().split()
    while dat:
        i, j = map(int, dat[1:3])
        vj[i-1,j-1] = float(dat[0])
        dat = finp.readline().split()
    sys.stdout.write('Start reading %s\n' % kdump)
    finp = open(kdump, 'r')
    dat = finp.readline()
    while 'END' not in dat:
        dat = finp.readline()
    dat = finp.readline().split()
    while dat:
        i, j = map(int, dat[1:3])
        vk[i-1,j-1] = float(dat[0])
        dat = finp.readline().split()
    finp.close()

    dic['MO_ENERGY'] = mo_energy
    dic['HCORE'] = fock-(vj+vk)
    dic['J'] = vj
    dic['K'] = vk
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
