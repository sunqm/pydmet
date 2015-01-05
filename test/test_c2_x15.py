#import os, sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy
from pyscf import gto
from pyscf import scf
from pyscf import lib

b1 = 1.0
nat = 10
mol = gto.Mole()
mol.verbose = 5
ncopy = 15
mol.output = 'out_c2_x%d' % (ncopy*2)
b0 = 1.132
b1 = 2.395 - b0
mol.atom = []
# all atoms sit on the edge of a circle
ang = numpy.pi-(numpy.pi/ncopy)
b = numpy.sqrt(b0**2+b1**2-2*b0*b1*numpy.cos(ang))
r = b/2 / numpy.sin(numpy.pi/ncopy)
theta0 = numpy.arcsin(b0/2/r) * 2
theta1 = numpy.arcsin(b1/2/r) * 2
print theta0*180/numpy.pi, theta1*180/numpy.pi, (theta0+theta1)*180/numpy.pi
for i in range(ncopy):
    theta = i * (2*numpy.pi/ncopy)
    mol.atom.append((6, (r*numpy.cos(theta),
                         r*numpy.sin(theta), 0)))
    theta = i * (2*numpy.pi/ncopy) + theta0
    mol.atom.append((6, (r*numpy.cos(theta),
                         r*numpy.sin(theta), 0)))

mol.basis = {'C': 'sto-3g',}
mol.build()
mf = scf.RHF(mol)
print mf.scf()

from pydmet import impsolver
from pydmet import dmet_nonsc
embsys = dmet_nonsc.EmbSys(mol, mf, [[0,1]])
#embsys.basidx_group = [[1,2,3,4,6,7,8,9]]
lib.logger.debug(embsys, '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
lib.logger.debug(embsys, 'embedding HF without chemical potential')
embsys.solver = impsolver.Psi4CCSD()
embsys.fitmethod_1shot = dmet_nonsc.fit_imp_fix_nelec
print embsys.one_shot()

lib.logger.debug(embsys, '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
lib.logger.debug(embsys, 'embedding HF with chemical potential')
embsys.fitmethod_1shot = dmet_nonsc.fit_imp_float_nelec
print embsys.one_shot()
