import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy
from pyscf import gto
from pyscf import scf

b1 = 1.8
nat = 30
mol = gto.Mole()
mol.verbose = 5
mol.output = 'h%s_sz' % nat
mol.atom = []
r = b1/2 / numpy.sin(numpy.pi/nat)
for i in range(nat):
    theta = i * (2*numpy.pi/nat)
    mol.atom.append((1, (r*numpy.cos(theta),
                         r*numpy.sin(theta), 0)))

mol.basis = {'H': 'sto-3g',}
mol.build()
mf = scf.RHF(mol)
print mf.scf()

def partition(nat, size):
    group = numpy.arange(nat).reshape(-1,size)
    return [list(i) for i in group]

from pydmet import dmet_sc
from pydmet import dmet_sc_ni
from pydmet import dmet_hf

embsys = dmet_sc_ni.EmbSys(mol, mf)
size = 2
embsys.frag_group = [partition(nat, size) ]
embsys.max_iter = 40
print embsys.scdmet()


embsys = dmet_sc.EmbSys(mol, mf)
size = 2
embsys.frag_group = [partition(nat, size) ]
embsys.max_iter = 40
print embsys.scdmet()

