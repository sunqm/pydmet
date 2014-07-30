import os
import numpy
from pyscf import gto
from pyscf import scf

from pydmet import dmet_nonsc

dir = os.environ['HOME']+'/workspace/gauss_vasp_13/docs/Polyyne/test1/'

embsys = dmet_nonsc.EmbSysPeriod(dir+'FCIDUMP.CLUST.GTO.benchmark',
                                 dir+'JDUMP.benchmark',
                                 dir+'KDUMP.benchmark',
                                 dir+'FOCKDUMP.benchmark')
embsys.one_shot()
