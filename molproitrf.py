#!/usr/bin/env python

import os, sys
import shutil
import tempfile
import commands
import re
import numpy

from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf import ao2mo
from pyscf.tools import fcidump
import impsolver
from impsolver import ImpSolver


#MOLPROEXE = os.environ['HOME'] + '/workspace/molpro-dev/bin/molpro'
MOLPROEXE = 'molpro'


Tsimple = '''!leave this line blank
memory,1000,m
aoint,c_final=0
symmetry,nosym
basis={
default=sto-3g,
he=3-21g,
h=sto-3g,
}
geometry={@GEOM}
charge=@CHARGE
{hf; maxit,1}
{matrop,read,orb,type=orbitals,file=orb.matrop
save,orb,2101.2}
!{matrop,read,s0,type=den,file=ovlp.matrop
!save,s0,}
hamiltonian,'fcidump'
{hf;start,2101.2}
@METHOD
dm,5000.2
{matrop
load,den,den,5000.2
write,den,rdm1,new
}
'''
Tsimple_no1pdm = '''!leave this line blank
memory,1000,m
aoint,c_final=0
symmetry,nosym
basis={
default=sto-3g,
he=3-21g,
h=sto-3g,
}
geometry={@GEOM}
charge=@CHARGE
{hf; maxit,1}
{matrop,read,orb,type=orbitals,file=orb.matrop
save,orb,2101.2}
!{matrop,read,s0,type=den,file=ovlp.matrop
!save,s0,}
hamiltonian,'fcidump'
{hf;start,2101.2}
@METHOD
'''

def simple_inp(method, nmo, nelec, with_1pdm=False):
    if with_1pdm:
        template = Tsimple
    else:
        template = Tsimple_no1pdm
    hchain = ['he 0. 0. %8.4f'%i for i in range(nmo/2)]
    geom = '\n'.join(hchain)
    template = re.sub('@GEOM', geom, template)
    template = re.sub('@CHARGE', str(nmo-nelec), template)
    template = re.sub('@METHOD', method, template)
    return template

def _key_multi(ncore, nocc, caslist=None):
    if caslist:
        assert(nocc-ncore == len(caslist))
        lst0 = sorted(list(set(range(ncore+1,nocc+1)) - set(caslist)))
        lst1 = sorted(list(set(caslist) - set(range(ncore+1,nocc+1))))
        map = []
        for i,k in enumerate(lst0):
            map.append('rotate,%d.1,%d.1,0' % (k, lst1[i]))
        method = '{multi;closed,%d;occ,%d\n%s}' % (ncore, nocc, '\n'.join(map))
    else:
        method = '{multi;closed,%d;occ,%d}' % (ncore, nocc)
    return method

def mr_inp(method, nmo, nelec, ncas, nelecas, with_1pdm=False,
           caslist=None):
    ncore = (nelec - nelecas) / 2
    nocc = ncore + ncas
    method = '%s\n%s' % (_key_multi(ncore, nocc, caslist), method)
    return simple_inp(method, nmo, nelec, with_1pdm)


def write_matrop(fname, mat):
    mat = mat.reshape(-1)
    with open(fname, 'w') as fin:
        fin.write('BEGIN_DATA,\n')
        for x in mat:
            fin.write('%25.15f\n' % x)
        fin.write('END_DATA,\n')

def call_molpro(h1e, eri, mo, nelec, inputstr, log=None):
    tdir = tempfile.mkdtemp(prefix='tmolpro')
    inpfile = os.path.join(tdir, 'inputs')

    open(inpfile, 'w').write(inputstr)
    write_matrop(os.path.join(tdir,'orb.matrop'), mo)
    nmo = mo.shape[1]
    fcidump.from_integrals(os.path.join(tdir, 'fcidump'),
                           h1e, eri, nmo, nelec, 0)

# note fcidump and orb.matrop should be put in the runtime dir
    cmd = ' '.join(('cd', tdir, '&& TMPDIR=`pwd`', MOLPROEXE, inpfile))
    rec = commands.getoutput(cmd)
    if 'fehler' in rec:
        sys.stderr.write('molpro tempfiles in %s\n'%tdir)
        raise RuntimeError('molpro fail as:\n' + rec)

    with open(inpfile+'.out') as fin:
        dat = fin.read()
        dat1 = dat.split('\n')
        es = dat1[-4]
        if log is not None:
            log.debug1(dat)
            log.debug('\n'.join(dat1[-5:-3]))
    eci, escf = map(float, es.split())[:2]

    if os.path.isfile(os.path.join(tdir,'rdm1')):
        with open(os.path.join(tdir,'rdm1')) as fin:
            fin.readline()
            fin.readline()
            dat = fin.read().replace(',', ' ').split()
        rdm1 = numpy.array(map(float, dat[:-1])).reshape(nmo,nmo)
# molpro will transform rdm1 back to AO representation (consistent to fcidump)
    else:
        rdm1 = None
    shutil.rmtree(tdir)
    return escf, eci, rdm1


#TODO:def part_eri_hermi(emb, eri):
#TODO:    nimp = emb.imp_site.shape[0]
#TODO:    mo = emb.mo_coeff_on_imp
#TODO:    nmo = mo.shape[1]
#TODO:    prj = numpy.dot(mo[:nimp,:].T, mo[:ptrace,:])
#TODO:
#TODO:    eri1 = ao2mo.restore(4, eri, nmo)
#TODO:    for i in range(eri1.shape[0]):
#TODO:        tmp = numpy.dot(prj, lib.unpack_tril(eri1[i]))
#TODO:        eri1[i] = lib.pack_tril(tmp+tmp.T)
#TODO:    eri1 = lib.transpose_sum(eri1, inplace=True)
#TODO:    return ao2mo.restore(8, eri1) * .25


def simple_call(method, verbose=0):
    def f(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag):
        log = lib.logger.Logger(mol.stdout, verbose)
        input = simple_inp(method, mo.shape[1], nelec, with_1pdm)
        escf, eci, rdm1 = call_molpro(h1e, eri, mo, nelec, input, log=log)
        return escf, eci, None, rdm1
    return f

def mr_call(method, ncas, nelecas, caslist=None, verbose=0):
    def f(mol, h1e, eri, mo, nelec, with_1pdm, with_e2frag):
        log = lib.logger.Logger(mol.stdout, verbose)
        input = mr_inp(method, mo.shape[1], nelec,
                       ncas, nelecas, with_1pdm, caslist)
        escf, eci, rdm1 = call_molpro(h1e, eri, mo, nelec, input, log=log)
        return escf, eci, None, rdm1
    return f

class CCSD(impsolver.ImpSolver):
    def __init__(self):
        impsolver.ImpSolver.__init__(self, simple_call('ccsd'))

class CCSD_T(impsolver.ImpSolver):
    def __init__(self):
        impsolver.ImpSolver.__init__(self, simple_call('ccsd(t)'))

# caslist: 1-based index
class CASSCF(impsolver.ImpSolver):
    def __init__(self, ncas, nelecas, caslist=None):
        impsolver.ImpSolver.__init__(self, mr_call('', ncas, nelecas, caslist))

class MRCI(impsolver.ImpSolver):
    def __init__(self, ncas, nelecas, caslist=None):
        impsolver.ImpSolver.__init__(self, mr_call('mrci', ncas, nelecas, caslist))

# rs2c cannot calculate density matrix
class CASPT2(impsolver.ImpSolver):
    def __init__(self, ncas, nelecas, caslist=None):
        impsolver.ImpSolver.__init__(self, mr_call('rs2c', ncas, nelecas, caslist))



if __name__ == '__main__':
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import numpy
    from pyscf import gto
    from pyscf import scf

    b1 = 1.2
    nat = 6
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = []
    r = b1/2 / numpy.sin(numpy.pi/nat)
    for i in range(nat):
        theta = i * (2*numpy.pi/nat)
        mol.atom.append((1, (r*numpy.cos(theta),
                             r*numpy.sin(theta), 0)))

    mol.basis = {'H': '6-31g',}
    mol.build()
    mf = scf.RHF(mol)
    print mf.scf()

    import pydmet.dmet_hf
    emb = pydmet.dmet_hf.RHF(mf)
    emb.imp_basidx = [0,1]
    emb.orth_ao_method = 'lowdin'
    print emb.imp_scf()
    print emb.hf_energy

    nemb = emb.impbas_coeff.shape[1]
    mo = emb.mo_coeff_on_imp
    e = call_molpro(emb.get_hcore(), emb._eri, mo, emb.nelectron,
                    mr_inp('rs2c', nemb, emb.nelectron, 2, 2))[1]
    print 'molpro-caspt2', e
    solver = MRCI(2, 2)
    e = solver.run(emb, emb._eri)[0]
    print 'molpro-mrci', e
    print '------------'
    _,e,dm = call_molpro(emb.get_hcore(), emb._eri, mo, emb.nelectron,
                         simple_inp('ccsd', nemb, emb.nelectron, 1))
    print 'molpro-cc', e # -4.28524318
    print dm
    print '------------'

    import pydmet.impsolver
    nimp = len(emb.bas_on_frag)
    solver = pydmet.impsolver.Psi4CCSD()
    etot, efrag, rdm1 = solver.run(emb, emb._eri, with_1pdm=True)
    print 'psi4-cc', etot
    print rdm1
