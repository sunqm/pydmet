#!/usr/bin/env python

import os
import shutil
import tempfile
import commands
import numpy

from pyscf import gto
from pyscf import scf
from pyscf import lib
from pyscf import ao2mo
from pyscf.future.tools import fcidump


#MOLPROEXE = os.environ['HOME'] + '/workspace/molpro-dev/bin/molpro'
MOLPROEXE = 'molpro'


def part_eri_hermi(emb, eri):
    nimp = emb.imp_site.shape[0]
    mo = emb.mo_coeff_on_imp
    prj = numpy.dot(mo[:nimp,:].T, mo[:ptrace,:])

    eri1 = ao2mo.restore(4, eri)
    for i in range(eri1.shape[0]):
        tmp = numpy.dot(prj, lib.unpack_tril(eri1[i]))
        eri1[i] = lib.pack_tril(tmp+tmp.T)
    eri1 = lib.transpose_sum(eri1, inplace=True)
    return ao2mo.restore(8, eri1) * .25


def simple_inp(method, nmo, nelec, do_rdm1=False):
    hchain = ['he 0. 0. %8.4f'%i for i in range(nmo/2)]
    geom = '\n'.join(hchain)
    charge = nmo - nelec
    tmpl = '''
memory,1000,m
aoint,c_final=0
symmetry,nosym
basis={
default=sto-3g,
he=3-21g,
h=sto-3g,
}
geometry={ %s }''' % geom
    tmpl += '''
charge=%d''' % charge
    tmpl += '''
{hf; maxit,1}
{matrop,read,orb,type=orbitals,file=orb.matrop
save,orb,2101.2}
!{matrop,read,s0,type=den,file=ovlp.matrop
!save,s0,}
hamiltonian,'fcidump'
{hf;start,2101.2}
'''
    tmpl += '''
{ %s;''' % method
    if do_rdm1:
        tmpl += '''
dm,5000.2 }
{matrop
load,den,den,5000.2
write,den,rdm1,new
}\n'''
    else:
        tmpl += '''}\n'''
    return tmpl

def call_molpro(emb, inputstr):
    tdir = tempfile.mkdtemp(prefix='tmolpro')
    inpfile = os.path.join(tdir, 'inputs')

    nemb = emb.impbas_coeff.shape[1]
    fcidump.from_integrals(os.path.join(tdir, 'fcidump'),
                           emb.get_hcore(), emb._eri, nemb,
                           emb.nelectron, 0)
    orbmat = numpy.eye(nemb)
    with open(os.path.join(tdir,'orb.matrop'),'w') as fin:
        fin.write('BEGIN_DATA,\n')
        for xs in orbmat:
            fin.write('%s\n' % ','.join(map(str, xs)))
        fin.write('END_DATA,\n')

    open(inpfile, 'w').write(inputstr)

    cmd = ' '.join(('cd', tdir, '&& TMPDIR=`pwd`', MOLPROEXE, inpfile))
    rec = commands.getoutput(cmd)
    if 'fehler' in rec:
        raise RuntimeError('molpro fail as:\n' + rec)

    with open(inpfile+'.out') as fin:
        dat = fin.readlines()
        es = dat[-3]
    e = float(es.split()[0])

    if os.path.isfile(os.path.join(tdir,'rdm1')):
        with open(os.path.join(tdir,'rdm1')) as fin:
            fin.readline()
            fin.readline()
            dat = fin.read().replace(',', ' ').split()
        rdm1 = numpy.array(map(float, dat[:-1])).reshape(nemb,nemb)
    else:
        rdm1 = None
    shutil.rmtree(tdir)
    return e, rdm1

#FIXME
def casscf_inp(method, nmo, nelec, do_rdm1=False):
    hchain = ['he 0. 0. %8.4f'%i for i in range(nmo/2)]
    geom = '\n'.join(hchain)
    charge = nmo - nelec
    tmpl = '''
memory,1000,m
aoint,c_final=0
symmetry,nosym
basis={
default=sto-3g,
he=3-21g,
h=sto-3g,
}
geometry={ %s }''' % geom
    tmpl += '''
charge=%d''' % charge
    tmpl += '''
{hf; maxit,1}
{matrop,read,orb,type=orbitals,file=orb.matrop
save,orb,2101.2}
!{matrop,read,s0,type=den,file=ovlp.matrop
!save,s0,}
hamiltonian,'fcidump'
{hf;start,2101.2}
'''
    tmpl += '''
{ %s;''' % method
    if do_rdm1:
        tmpl += '''
dm }
{matrop
load,den,den,2140.2
write,den,rdm1,new
}\n'''
    else:
        tmpl += '''}\n'''
    return tmpl


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
    e, dm = call_molpro(emb, simple_inp('ccsd', nemb, emb.nelectron))
    print 'molpro-cc', e, dm # -4.28524318
    e, dm = call_molpro(emb, simple_inp('ccsd', nemb, emb.nelectron, do_rdm1=1))
    print dm
    print '------------'

    import pydmet.impsolver
    nimp = emb.num_of_impbas()
    solver = pydmet.impsolver.use_local_solver(pydmet.impsolver.ccsd)
    res = solver(mol, emb)
    print 'psi4-cc', res['etot']
    print res['rdm1']
