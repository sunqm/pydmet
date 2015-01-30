#!/usr/bin/env python
import numpy
import scipy.linalg
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo

n = 48
nefill = 48-8*2
U = 12

mol = gto.Mole()
mol.verbose = 5
mol.output = 'out6'
mol.atom = [('H', (0,0,i)) for i in range(n)]
mol.basis = 'sto-3g'
mol.build(verbose=4)
mol.nelectron = nefill

h1 = numpy.zeros((n,n))
for i in range(n-1):
    h1[i,i+1] = h1[i+1,i] = -1.0
h1[n-1,0] = h1[0,n-1] = 1.0
eri = numpy.zeros((n,n,n,n))
for i in range(n):
    eri[i,i,i,i] = U

e,c = scipy.linalg.eigh(h1)
dm = numpy.dot(c[:,:nefill/2],c[:,:nefill/2].T) * 2

class Hubbard(scf.hf.RHF):
    def get_hcore(self, *args):
        return h1
    def get_ovlp(self, *args):
        return numpy.eye(n)

mf = Hubbard(mol)
mf._eri = ao2mo.restore(8, eri, n)
mf.scf(dm)


from pydmet import dmet_hf
class HubbardEmbHF(dmet_hf.RHF):
    def __init__(self, entire_hf):
        orth_ao = numpy.eye(n)
        dmet_hf.RHF.__init__(self, entire_hf, orth_ao)

    def init_vhf_env(self, env_orb):
        nemb = self.impbas_coeff.shape[1]
        return 0, numpy.zeros((nemb,nemb))

    def eri_on_impbas(self, mol):
        nemb = self.impbas_coeff.shape[1]
        npair = nemb * (nemb+1) / 2
        eri = numpy.zeros(npair*(npair+1)/2)
        nimp = len(self.bas_on_frag)
        for i in range(nimp):
            ii = i*(i+1)/2 + i
            eri[ii*(ii+1)/2+ii] = U
        return eri

from pydmet import dmet_sc
class FakeHubbard(dmet_sc.EmbSys):
    def __init__(self, *args, **kwargs):
        dmet_sc.EmbSys.__init__(self, *args, **kwargs)
        self.OneImp = HubbardEmbHF
        self.v_fit_domain = dmet_sc.IMP_BLK
        self.dm_fit_domain = dmet_sc.IMP_AND_BATH
        self.env_pot_for_ci = dmet_sc.NO_IMP_BLK





def partition(nat, size):
    group = numpy.arange(nat).reshape(-1,size)
    return [list(i) for i in group]

size = 2
embsys = FakeHubbard(mol, mf, orth_coeff=numpy.eye(n))
embsys.frag_group = [partition(n, size) ]
embsys.vfit_ci_method = dmet_sc.gen_all_vfit_by(lambda *args: 0)
embsys.emb_verbose = 0
embsys.verbose = 5
embsys.fitpot_damp_fac  = 1.
embsys._init_v = numpy.eye(n)*(1.*U*nefill/2/n)
#embsys.v_fit_domain = 'off_diag_plus'
embsys._init_v = numpy.eye(n)*(.5*U)
import scipy.linalg
#embsys._init_v = scipy.linalg.block_diag(
#    *([numpy.array(
#[[ 2.27647161, -0.20934772,  0.03522263,  0.16199762],
# [-0.20934772,  2.27647161, -0.07691276,  0.03522272],
# [ 0.03522263, -0.07691276,  2.27647161, -0.2093476 ],
# [ 0.16199762,  0.03522272, -0.2093476 ,  2.27647161]])]*(n//size)))

print embsys.scdmet()
