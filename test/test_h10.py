import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy
from pyscf import gto
from pyscf import scf
import dmet_sc

b1 = 1.0
nat = 10
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

embsys = dmet_sc.EmbSys(mol, mf)
embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
embsys.max_iter = 10
print embsys.scdmet(mol)



import dmet_nonsc
embsys = dmet_nonsc.EmbSys(mol, mf)
embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
print embsys.fullsys(mol)

embsys = dmet_nonsc.EmbSys(mol, mf, [[0,1]])
print embsys.one_shot(mol)
