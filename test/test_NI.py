import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy
from pyscf import gto
from pyscf import scf

b1 = 1.8
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

import dmet_sc
import vaspimp

embsys = dmet_sc.EmbSys(mol, mf)
embsys.OneImp = vaspimp.OneImp
embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
embsys.max_iter = 10
print embsys.scdmet()

embsys = dmet_sc.EmbSys(mol, mf)
embsys.OneImp = vaspimp.OneImpNaiveNI
embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
embsys.max_iter = 10
print embsys.scdmet()

embsys = dmet_sc.EmbSys(mol, mf)
embsys.OneImp = vaspimp.OneImpNI
embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
embsys.max_iter = 10
print embsys.scdmet()

class OneImpFractionNI(vaspimp.OneImpNI):
    def get_hcore(self, mol=None):
        nimp = len(self.bas_on_frag)
        effscf = self.entire_scf
        sc = reduce(numpy.dot, (self.impbas_coeff.T, \
                                self.entire_scf.get_ovlp(), effscf.mo_coeff))
        fock = numpy.dot(sc*effscf.mo_energy, sc.T)
        dmimp = effscf.calc_den_mat(mo_coeff=sc)
        dm = numpy.zeros_like(fock)
        dm[:nimp,:nimp] = dmimp[:nimp,:nimp]
        h1e = fock - self.get_eff_potential(mol, dm)
        return h1e
embsys = dmet_sc.EmbSys(mol, mf)
embsys.OneImp = OneImpFractionNI
embsys.frag_group = [[[0,1],[2,3],[4,5],[6,7],[8,9]], ]
embsys.max_iter = 10
print embsys.scdmet()

