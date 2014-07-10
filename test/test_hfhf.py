import numpy
from pyscf import gto
import hf
import dmet_nonsc

'''test HF-in-HF embedding'''

mol = gto.Mole()
mol.verbose = 5
mol.output = 'out_test_hfhf'
mol.build()

mf = hf.RHF(mol, 'C_solid_2x2x2/test2/FCIDUMP.CLUST.GTO',
            'C_solid_2x2x2/test2/JKDUMP')
escf0 = mf.scf()
#emb = OneImp(mol, mf, [0,1,2,3])
#print dmet_1shot(mol, emb)

mf = hf.RHF(mol, 'C_solid_2x2x2/test2/FCIDUMP.CLUST.GTO',
            'C_solid_2x2x2/test2/JKDUMP')
mf.mo_coeff = mf._fcidump['MO_COEFF']
mf.mo_energy = mf._fcidump['MO_ENERGY']
mf.mo_occ = numpy.zeros_like(mf.mo_energy)
nocc = mf._fcidump['NELEC']/2
mf.mo_occ[:nocc] = 2
vhf = mf._fcidump['J'] - mf._fcidump['K']
dm = numpy.dot(mf.mo_coeff, mf.mo_coeff.T) * 2
mf.hf_energy = mf.mo_energy[:nocc].sum() * 2 - vhf[:nocc].trace()

print escf0, mf.hf_energy
print mf.hf_energy - escf0

#emb = OneImp(mol, mf, [0,1,2,3])
#print dmet_1shot(mol, emb)
#
##print energy
##print mf.mo_energy
##print mf._fcidump['MO_ENERGY']
#        self.scf_conv, self.hf_energy, self.mo_energy, self.mo_occ, \
#                self.mo_coeff_on_imp \
