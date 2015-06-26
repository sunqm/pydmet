#!/usr/bin/env python
# -*- coding: utf-8
#
# File: hf.py
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import os, sys
import re

import numpy
import scipy.linalg
import h5py
from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf.lib import logger


class RHF(scf.hf.RHF):
    ''' RHF from vasp FCIDUMP'''
    def __init__(self, mol, path):
        scf.hf.RHF.__init__(self, mol)
        self._vaspdump = read_vaspdump(path)
        self.mo_coeff = self._vaspdump['MO_COEFF']
        self.mo_energy = self._vaspdump['MO_ENERGY']
        self.mo_occ = numpy.zeros(self._vaspdump['NORB'])
        self.mo_occ[:self._vaspdump['NELEC']/2] = 2
        self.mol.nelectron = self._vaspdump['NELEC']
        self.mol.spheric_labels = \
                lambda *args: [(i, '', '', '')
                               for i in range(self._vaspdump['NORB'])]

    def get_hcore(self, mol=None):
#TODO read FOCKDUMP, this Fock matrix excludes the correlation potential, it
# is not identical to the converged Fock matrix of the VASP SCF Fock
# plus CORRPOTDUMP (if exists) gives the VASP SCF Fock
        return self._vaspdump['HCORE']

    def get_ovlp(self, mol=None):
        return numpy.eye(self._vaspdump['NORB'])

#    def analyze(self, verbose=logger.DEBUG):
#        from pyscf.tools import dump_mat
#        mo_energy = mf.mo_energy
#        mo_occ = mf.mo_occ
#        mo_coeff = mf.mo_coeff
#        log = logger.Logger(mf.stdout, verbose)
#        log.info('**** MO energy ****')
#        for i in range(len(mo_energy)):
#            if mo_occ[i] > 0:
#                log.info('occupied MO #%d energy= %.15g occ= %g', \
#                         i+1, mo_energy[i], mo_occ[i])
#            else:
#                log.info('virtual MO #%d energy= %.15g occ= %g', \
#                         i+1, mo_energy[i], mo_occ[i])
#        if verbose >= logger.DEBUG:
#            log.debug(' ** MO coefficients **')
#            label = None
#            dump_mat.dump_rec(mf.stdout, mo_coeff, label, start=1)
#        dm = mf.make_rdm1(mo_coeff, mo_occ)
#        return mf.mulliken_pop(mf.mol, dm, mf.get_ovlp(), log)

    def mulliken_pop(self, mol, dm, ovlp=None, verbose=logger.DEBUG):
        if ovlp is None:
            ovlp = get_ovlp(mol)
        if isinstance(verbose, logger.Logger):
            log = verbose
        else:
            log = logger.Logger(mol.stdout, verbose)
        if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
            pop = numpy.einsum('ij->i', dm*ovlp).real
        else: # ROHF
            pop = numpy.einsum('ij->i', (dm[0]+dm[1])*ovlp).real

        log.info(' ** Mulliken pop  **')
        for i, s in enumerate(pop):
            log.info('pop of  %d %10.5f', i, s)


def read_vaspdump(path, h5dump=None):
#NOTE read_hfdump returns the integrals in MO representation
    clustdump = os.path.join(path, 'FCIDUMP.CLUST.GTO')
    jdump     = os.path.join(path, 'JDUMP')
    kdump     = os.path.join(path, 'KDUMP')
    fockdump  = os.path.join(path, 'FOCKDUMP')
    if h5py.is_hdf5(clustdump):
        f = h5py.File(clustdump, 'r')
        dic = {}
        for k,v in f.items():
            if v.shape: # I'm ndarray
                dic[k] = numpy.array(v)
            else:
                dic[k] = v.value
        f.close()
    else:
        hfdic = read_hfdump(jdump, kdump, fockdump)
        dic = read_clustdump(clustdump, hfdic)
        mo_coeff = dic['MO_COEFF']
        hfdic['HCORE'] = reduce(numpy.dot, (mo_coeff, hfdic['HCORE'], mo_coeff.T))
        hfdic['J'] = reduce(numpy.dot, (mo_coeff, hfdic['J'], mo_coeff.T))
        hfdic['K'] = reduce(numpy.dot, (mo_coeff, hfdic['K'], mo_coeff.T))
        dic.update(hfdic)
        if h5dump is None:
            h5dump = clustdump+'.h5'
        f = h5py.File(h5dump, 'w')
        for k,v in dic.items():
            sys.stdout.write('h5dump %s\n' % k)
            f[k] = v
        f.close()
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
    dat = finp.readline()
    head = []
    while dat:
        if 'END' in dat:
            break
        head.append(dat[:-1]) # exclude '\n'
        dat = finp.readline()
    head = ''.join(head).replace(' ','')
    head = re.split('[=,]', head[4:])
    for kv in head:
        if kv.isdigit():
            dic[klast].append(int(kv))
        else:
            try:
                float(kv)
                dic[klast].append(float(kv))
            except ValueError:
                klast = kv
                dic[klast] = []

    for k, v in dic.items():
        if k != 'ORBIND':
            dic[k] = v[0]
    dic['ORBIND'] = [i-1 for i in dic['ORBIND']]
    dic['NEMB'] = dic['NORB']

    nemb = dic['NEMB']
    nembelec = dic['NELECEMB']
    norb = hfdic['NORB']
    npair = nemb*(nemb+1)/2
    h1emb = numpy.zeros((nemb,nemb))
    corrpot = numpy.zeros((nemb,nemb))
    mo_coeff = numpy.zeros((norb,norb))
    embasis = numpy.zeros((norb,nemb))
    eri = numpy.zeros((npair*(npair+1)/2))
    dat = finp.readline().split()
    touched0 = touched1 = touched2 = 0
    while dat:
        i, j, k, l = map(int, dat[1:])
        if l == 0:
            h1emb[i-1,j-1] = h1emb[j-1,i-1] = float(dat[0])
            touched0 = 1
        elif l == -1:
            mo_coeff[i-1,j-1] = float(dat[0])
            touched1 = 1
        elif l == -2:
            embasis[i-1,j-1] = float(dat[0])
            touched2 = 1
        elif l == -3:
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
    if not (touched0 and touched1 and touched2):
        raise RuntimeError("h1emb, embasis or mo_coeff are not generated")
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
# diagonal part of Fock matrix  x 0 0 0,  they are not MO orbital energy
# orbital energy can be obtained by diagonalize FOCKDUMP+CORRPOTDUMP
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
    dat = finp.readline().split()
    while dat:
        i, j = map(int, dat[1:3])
        if j != 0:
            fock[i-1,j-1] = float(dat[0])
        dat = finp.readline().split()

    corrpot = numpy.zeros((norb,norb))
    try:
        with open('CORRPOTDUMP', 'r') as finp:
            sys.stdout.write('Start reading CORRPOTDUMP\n')
            dat = finp.readline()
            while 'END' not in dat:
                dat = finp.readline()
            dat = finp.readline().split()
            while dat:
                i, j = map(int, dat[1:3])
                corrpot[i-1,j-1] = float(dat[0])
                dat = finp.readline().split()
    except:
        sys.stdout.write('CORRPOTDUMP not found\n')

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

    dic['MO_ENERGY'] = scipy.linalg.eigh(fock+corrpot)[0]
    dic['HCORE'] = fock-(vj+vk)
    dic['J'] = vj
    dic['K'] = vk
    return dic

def convert_clustdump(path, h5name):
    clustdump = os.path.join(path, 'FCIDUMP.CLUST.GTO')
    jdump     = os.path.join(path, 'JDUMP')
    kdump     = os.path.join(path, 'KDUMP')
    fockdump  = os.path.join(path, 'FOCKDUMP')
    hfdic = read_hfdump(jdump, kdump, fockdump)
    dic = read_clustdump(clustdump, hfdic)
    mo_coeff = dic['MO_COEFF']
    hfdic['HCORE'] = reduce(numpy.dot, (mo_coeff, hfdic['HCORE'], mo_coeff.T))
    hfdic['J'] = reduce(numpy.dot, (mo_coeff, hfdic['J'], mo_coeff.T))
    hfdic['K'] = reduce(numpy.dot, (mo_coeff, hfdic['K'], mo_coeff.T))
    dic.update(hfdic)
    if h5dump is None:
        h5dump = clustdump+'.h5'
    f = h5py.File(h5dump, 'w')
    for k,v in dic.items():
        sys.stdout.write('h5dump %s\n' % k)
        f[k] = v
    f.close()


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 5
    mol.build()

    mf = RHF(mol, 'test/C_solid_2x2x2/test2/')
    energy = mf.scf()
    print energy
    print mf.mo_energy
    print mf._fcidump['MO_ENERGY']
