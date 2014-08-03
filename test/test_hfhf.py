import numpy
from pyscf import gto
from pyscf import scf

from pydmet import vaspimp
from pydmet import dmet_nonsc

vasphf = vaspimp.read_clustdump('FCIDUMP.CLUST.GTO', 'JDUMP', 'KDUMP',
                                 'FOCKDUMP')
fake_hf = vaspimp.fake_entire_scf(vasphf)
emb = vaspimp.OneImpOnCLUSTDUMP(fake_hf, vasphf)
emb.occ_env_cutoff = 1e-14
emb.orth_coeff = numpy.eye(vasphf['NORB'])
emb.verbose = 5
emb.imp_scf()

mo = vasphf['MO_COEFF']
nimp = vasphf['NIMP']
cimp = numpy.dot(emb.impbas_coeff[:,:nimp].T, mo[:,:vasphf['NELEC']/2])
print 'NELEC from Lattice HF', numpy.linalg.norm(cimp)**2*2
cimp = emb.mo_coeff_on_imp[:nimp,:emb.nelectron/2]
print 'NELEC from embedding-HF', numpy.linalg.norm(cimp)**2*2

