import os

VASPHOME = os.environ['HOME']+'/workspace/gauss_vasp_13/vasp/'
#VASPEXE = 'srun -n 1 ' + VASPHOME + '/vasp'
#VASPMPI = 'srun ' + VASPHOME + '/vasp'
VASPEXE = VASPHOME + '/vasp'
NP = 4
VASPMPI = ('mpirun -np %d '%NP) + VASPHOME + '/vasp'
