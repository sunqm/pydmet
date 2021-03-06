import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy
from pyscf import gto
from pyscf import scf

b1 = 0.74
nat = 42
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

import dmet_sc
import impsolver
size = 1
embsys = dmet_sc.EmbSys(mol, mf)
embsys.frag_group = [partition(nat, size) ]
#embsys.solver = impsolver.InterNormFCI()
#embsys.vfit_ci_method = dmet_sc.gen_all_vfit_by(lambda *args: 0)
embsys.v_fit_domain = 6
embsys.dm_fit_domain = 6
embsys.emb_verbose = 5
print embsys.scdmet()
