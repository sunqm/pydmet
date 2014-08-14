#import os, sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy
from pyscf import gto
from pyscf import scf

b1 = 1.0
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

#from pydmet import dmet_sc
#embsys = dmet_sc.EmbSys(mol, mf)
#embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
#embsys.max_iter = 10
#print embsys.scdmet()
#
#
#
from pydmet import dmet_nonsc
embsys = dmet_nonsc.EmbSys(mol, mf, [[0,1]])
print embsys.one_shot()
#
#embsys = dmet_nonsc.EmbSys(mol, mf)
#embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
#print embsys.fullsys()
#
#
#
#from pydmet import impsolver
#def ccsolver(mol, emb, v=0):
#    solver = impsolver.use_local_solver(impsolver.cc)
#    return solver(mol, emb, v)
#
#embsys = dmet_nonsc.EmbSys(mol, mf, [[0,1]])
#embsys.frag_fci_solver = ccsolver
#print embsys.one_shot()
#
#embsys = dmet_nonsc.EmbSys(mol, mf)
#embsys.frag_fci_solver = ccsolver
#embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
#print embsys.fullsys()
