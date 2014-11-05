'''
Run vasp in python environment

modified based on ase package

https://wiki.fysik.dtu.dk/ase
'''

import os
import sys
import re
import numpy
from pyscf import gto
from pyscf import lib
import settings

# Parameters that can be set in INCAR. The values which are None
# are not written and default parameters of VASP are used for them.

float_keys = [
    'aexx',       # Fraction of exact/DFT exchange
    'aggac',      # Fraction of gradient correction to correlation
    'aggax',      # Fraction of gradient correction to exchange
    'aldac',      # Fraction of LDA correlation energy
    'amin',       #
    'amix',       #
    'amix_mag',   #
    'bmix',       # tags for mixing
    'bmix_mag',   #
    'deper',      # relative stopping criterion for optimization of eigenvalue
    'ebreak',     # absolute stopping criterion for optimization of eigenvalues (EDIFF/N-BANDS/4)
    'efield',     # applied electrostatic field
    'emax',       # energy-range for DOSCAR file
    'emin',       #
    'enaug',      # Density cutoff
    'encut',      # Planewave cutoff
    'encutgw',    # energy cutoff for response function
    'encutfock',  # FFT grid in the HF related routines
    'hfscreen',   # attribute to change from PBE0 to HSE
    'kspacing', # determines the number of k-points if the KPOINTS
                  # file is not present. KSPACING is the smallest
                  # allowed spacing between k-points in units of
                  # $\AA$^{-1}$.
    'potim',      # time-step for ion-motion (fs)
    'nelect',     # total number of electrons
    'param1',     # Exchange parameter
    'param2',     # Exchange parameter
    'pomass',     # mass of ions in am
    'sigma',      # broadening in eV
    'spring',     # spring constant for NEB
    'time',       # special control tag
    'weimin',     # maximum weight for a band to be considered empty
    'zab_vdw',    # vdW-DF parameter
    'zval',       # ionic valence
#The next keywords pertain to the VTST add-ons from Graeme Henkelman's group at UT Austin
    'jacobian',   # Weight of lattice to atomic motion
    'ddr',        # (DdR) dimer separation
    'drotmax',    # (DRotMax) number of rotation steps per translation step
    'dfnmin',     # (DFNMin) rotational force below which dimer is not rotated
    'dfnmax',     # (DFNMax) rotational force below which dimer rotation stops
    'stol',       # convergence ratio for minimum eigenvalue
    'sdr',        # finite difference for setting up Lanczos matrix and step size when translating
    'maxmove',    # Max step for translation for IOPT > 0
    'invcurve',   # Initial curvature for LBFGS (IOPT = 1)
    'timestep',   # Dynamical timestep for IOPT = 3 and IOPT = 7
    'sdalpha',    # Ratio between force and step size for IOPT = 4
#The next keywords pertain to IOPT = 7 (i.e. FIRE)
    'ftimemax',   # Max time step
    'ftimedec',   # Factor to dec. dt
    'ftimeinc',   # Factor to inc. dt
    'falpha',     # Parameter for velocity damping
    'falphadec',  # Factor to dec. alpha
]

exp_keys = [
    'ediff',      # stopping-criterion for electronic upd.
    'ediffg',     # stopping-criterion for ionic upd.
    'symprec',    # precession in symmetry routines
#The next keywords pertain to the VTST add-ons from Graeme Henkelman's group at UT Austin
    'fdstep',     # Finite diference step for IOPT = 1 or 2
]

string_keys = [
    'algo',       # algorithm: Normal (Davidson) | Fast | Very_Fast (RMM-DIIS)
    'gga',        # xc-type: PW PB LM or 91
    'prec',       # Precission of calculation (Low, Normal, Accurate)
    'system',     # name of System
    'tebeg',      #
    'teend',      # temperature during run
    'precfock',    # FFT grid in the HF related routines
]

int_keys = [
    'ialgo',      # algorithm: use only 8 (CG) or 48 (RMM-DIIS)
    'ibrion',     # ionic relaxation: 0-MD 1-quasi-New 2-CG
    'icharg',     # charge: 0-WAVECAR 1-CHGCAR 2-atom 10-const
    'idipol',     # monopol/dipol and quadropole corrections
    'images',     # number of images for NEB calculation
    'iniwav',     # initial electr wf. : 0-lowe 1-rand
    'isif',       # calculate stress and what to relax
    'ismear',     # part. occupancies: -5 Blochl -4-tet -1-fermi 0-gaus >0 MP
    'ispin',      # spin-polarized calculation
    'istart',     # startjob: 0-new 1-cont 2-samecut
    'isym',       # symmetry: 0-nonsym 1-usesym 2-usePAWsym
    'iwavpr',     # prediction of wf.: 0-non 1-charg 2-wave 3-comb
    'ldauprint',  # 0-silent, 1-occ. matrix written to OUTCAR, 2-1+pot. matrix written
    'ldautype',   # L(S)DA+U: 1-Liechtenstein 2-Dudarev 4-Liechtenstein(LDAU)
    'lmaxmix',    #
    'lorbit',     # create PROOUT
    'maxmix',     #
    'ngx',        # FFT mesh for wavefunctions, x
    'ngxf',       # FFT mesh for charges x
    'ngy',        # FFT mesh for wavefunctions, y
    'ngyf',       # FFT mesh for charges y
    'ngz',        # FFT mesh for wavefunctions, z
    'ngzf',       # FFT mesh for charges z
    'nbands',     # Number of bands
    'nblk',       # blocking for some BLAS calls (Sec. 6.5)
    'nbmod',      # specifies mode for partial charge calculation
    'nelm',       # nr. of electronic steps (default 60)
    'nelmdl',     # nr. of initial electronic steps
    'nelmin',
    'nfree',      # number of steps per DOF when calculting Hessian using finitite differences
    'nkred',      # define sub grid of q-points for HF with nkredx=nkredy=nkredz
    'nkredx',      # define sub grid of q-points in x direction for HF
    'nkredy',      # define sub grid of q-points in y direction for HF
    'nkredz',      # define sub grid of q-points in z direction for HF
    'nomega',     # number of frequency points
    'nomegar',    # number of frequency points on real axis
    'npar',       # parallelization over bands
    'nsim',       # evaluate NSIM bands simultaneously if using RMM-DIIS
    'nsw',        # number of steps for ionic upd.
    'nupdown',    # fix spin moment to specified value
    'nwrite',     # verbosity write-flag (how much is written)
    'smass',      # Nose mass-parameter (am)
    'vdwgr',      # extra keyword for Andris program
    'vdwrn',      # extra keyword for Andris program
    'voskown',    # use Vosko, Wilk, Nusair interpolation
#The next keywords pertain to the VTST add-ons from Graeme Henkelman's group at UT Austin
    'ichain',     # Flag for controlling which method is being used (0=NEB, 1=DynMat, 2=Dimer, 3=Lanczos)
                  # if ichain > 3, then both IBRION and POTIM are automatically set in the INCAR file
    'iopt',       # Controls which optimizer to use.  for iopt > 0, ibrion = 3 and potim = 0.0
    'snl',        # Maximum dimentionality of the Lanczos matrix
    'lbfgsmem',   # Steps saved for inverse Hessian for IOPT = 1 (LBFGS)
    'fnmin',      # Max iter. before adjusting dt and alpha for IOPT = 7 (FIRE)
]

bool_keys = [
    'addgrid',    # finer grid for augmentation charge density
    'kgamma',     # The generated kpoint grid (from KSPACING) is either
                  # centred at the $\Gamma$
                  # point (e.g. includes the $\Gamma$ point)
                  # (KGAMMA=.TRUE.)
    'laechg',     # write AECCAR0/AECCAR1/AECCAR2
    'lasph',      # non-spherical contributions to XC energy (and pot for VASP.5.X)
    'lasync',     # overlap communcation with calculations
    'lcharg',     #
    'lcorr',      # Harris-correction to forces
    'ldau',       # L(S)DA+U
    'ldiag',      # algorithm: perform sub space rotation
    'ldipol',     # potential correction mode
    'lelf',       # create ELFCAR
    'lepsilon',   # enables to calculate and to print the BEC tensors
    'lhfcalc',    # switch to turn on Hartree Fock calculations
    'loptics',    # calculate the frequency dependent dielectric matrix
    'lpard',      # evaluate partial (band and/or k-point) decomposed charge density
    'lplane',     # parallelisation over the FFT grid
    'lscalapack', # switch off scaLAPACK
    'lscalu',     # switch of LU decomposition
    'lsepb',      # write out partial charge of each band seperately?
    'lsepk',      # write out partial charge of each k-point seperately?
    'lthomas',    #
    'luse_vdw',   # Invoke vdW-DF implementation by Klimes et. al
    'lvdw',	  # Invoke DFT-D2 method of Grimme
    'lvhar',      # write Hartree potential to LOCPOT (vasp 5.x)
    'lvtot',      # create WAVECAR/CHGCAR/LOCPOT
    'lwave',      #
#The next keywords pertain to the VTST add-ons from Graeme Henkelman's group at UT Austin
    'lclimb',     # Turn on CI-NEB
    'ltangentold', # Old central difference tangent
    'ldneb',      # Turn on modified double nudging
    'lnebcell',   # Turn on SS-NEB
    'lglobal',    # Optmizize NEB globally for LBFGS (IOPT = 1)
    'llineopt',   # Use force based line minimizer for translation (IOPT = 1)
    'lbeefens',   # Switch on print of BEE energy contritions in OUTCAR
    'lbeefbas',   # Switch off print of all BEEs in OUTCAR
]

list_keys = [
    'dipol',      # center of cell for dipol
    'eint',       # energy range to calculate partial charge for
    'ferwe',      # Fixed band occupation (spin-paired)
    'ferdo',      # Fixed band occupation (spin-plarized)
    'iband',      # bands to calculate partial charge for
    'magmom',     # initial magnetic moments
    'kpuse',      # k-point to calculate partial charge for
    'ropt',       # number of grid points for non-local proj in real space
    'rwigs',      # Wigner-Seitz radii
    'ldauu',      # ldau parameters, has potential to redundant w.r.t. dict
    'ldaul',      # key 'ldau_luj', but 'ldau_luj' can't be read direct from
    'ldauj',      # the INCAR (since it needs to know information about atomic
                  # species. In case of conflict 'ldau_luj' gets written out
                  # when a calculation is set up
]

special_keys = [
    'lreal',      # non-local projectors in real space
]

dict_keys = [
    'ldau_luj',   # dictionary with L(S)DA+U parameters, e.g. {'Fe':{'L':2, 'U':4.0, 'J':0.9}, ...}
]

keys = [
    # 'NBLOCK' and KBLOCK       inner block; outer block
    # 'NPACO' and APACO         distance and nr. of slots for P.C.
    # 'WEIMIN, EBREAK, DEPER    special control tags
]

class Vasp:
    '''
Insert NWChem basis format for Vasp.GaussBasis , e.g.
>>> import Vasp
>>> o = Vasp()
>>> o.GaussBasis = {"C":
"""C    S
     71.6168370              0.15432897
     13.0450960              0.53532814
      3.5305122              0.44463454
C    SP
      2.9412494             -0.09996723             0.15591627
      0.6834831              0.39951283             0.60768372
      0.2222899              0.70011547             0.39195739""" }
'''
    def __init__(self):
        self.GaussBasis = {}
        self.INCAR = {}
        self.stdout = 'vasp.out'
        self.stderr = 'vasp.err'

    def write_GaussBasis(self):
        if self.GaussBasis:
            with open('GaussBasis', 'w') as fout:
                for k, v in self.GaussBasis:
                    bfmt = _format_basis(v)
                    nuc = lib.parameters.ELEMENTS_PROTON[k]
                    fout.write('C %d %d\n' % (nuc, len(bfmt)))
                    for b in bfmt:
                        fout.write('H %d %d\n' % (b.shape[0], b.shape[1]-1))
                        arr = numpy.array_str(b).replace('[',' ').replace(']',' ')
                        fout.write('%s\n' % arr)

    def run(self, vaspcmd):
        # call VASP
        cmd = '%s >> %s 2> %s' % (vaspcmd, self.stdout, self.stderr)
        exitcode = os.system(cmd)
        if exitcode != 0:
            raise RuntimeError('Vasp exited with exit code: %d.  ' % exitcode)

#        self.converged = self.read_convergence()
#        self.set_results(atoms)

    def set_results(self):
        #if self.spinpol:
        #    self.magnetic_moment = self.read_magnetic_moment()
        #    if (self.int_params['lorbit']>=10
        #        or (self.int_params['lorbit']!=None
        #            and self.list_params['rwigs'])):
        #        self.magnetic_moments = self.read_magnetic_moments(atoms)
        #    else:
        #        self.magnetic_moments = None
        self.version = self.read_version()
        self.niter = self.read_number_of_iterations()
        self.sigma = self.read_electronic_temperature()
        self.nelect = self.read_number_of_electrons()

    def clean(self):
        """Method which cleans up after a calculation.

        The default files generated by Vasp will be deleted IF this
        method is called.

        """
        files = ['CHG', 'CHGCAR', 'POSCAR', 'INCAR', 'CONTCAR',
                 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'KPOINTS', 'OSZICAR',
                 'OUTCAR', 'PCDAT', 'POTCAR', 'vasprun.xml',
                 'WAVECAR', 'XDATCAR', 'PROCAR', 'ase-sort.dat',
                 'LOCPOT', 'AECCAR0', 'AECCAR1', 'AECCAR2',
                 'WAVECAR.GTO', 'vasp.out', 'vasp.err']
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

    def read_version(self):
        version = None
        for line in open('OUTCAR'):
            if line.find(' vasp.') != -1: # find the first occurence
                version = line[len(' vasp.'):].split()[0]
                break
        return version

    def read_number_of_iterations(self):
        niter = None
        for line in open('OUTCAR'):
            if line.find('- Iteration') != -1: # find the last iteration number
                niter = int(line.split(')')[0].split('(')[-1].strip())
        return niter

    def read_electronic_temperature(self):
        sigma = None
        for line in open('OUTCAR'):
            if line.find('Fermi-smearing in eV        SIGMA') != -1:
                sigma = float(line.split('=')[1].strip())
        return sigma

    def read_default_number_of_electrons(self, filename='POTCAR'):
        nelect = []
        lines = open(filename).readlines()
        for n, line in enumerate(lines):
            if line.find('TITEL') != -1:
                symbol = line.split('=')[1].split()[1].split('_')[0].strip()
                valence = float(lines[n+4].split(';')[1].split('=')[1].split()[0].strip())
                nelect.append((symbol, valence))
        return nelect

    def read_number_of_electrons(self):
        nelect = None
        for line in open('OUTCAR'):
            if line.find('total number of electrons') != -1:
                nelect = float(line.split('=')[1].split()[0].strip())
        return nelect

    def read_stress(self):
        stress = None
        for line in open('OUTCAR'):
            if line.find(' in kB  ') != -1:
                stress = -np.array([float(a) for a in line.split()[2:]]) \
                         [[0, 1, 2, 4, 5, 3]] \
                         * 1e-1 * ase.units.GPa
        return stress

    def read_ldau(self):
        ldau_luj = None
        ldauprint = None
        ldau = None
        ldautype = None
        atomtypes = []
        # read ldau parameters from outcar
        for line in open('OUTCAR'):
            if line.find('TITEL') != -1:    # What atoms are present
                atomtypes.append(line.split()[3].split('_')[0].split('.')[0])
            if line.find('LDAUTYPE') != -1: # Is this a DFT+U calculation
                ldautype = int(line.split('=')[-1])
                ldau = True
                ldau_luj = {}
            if line.find('LDAUL') != -1:
                L = line.split('=')[-1].split()
            if line.find('LDAUU') != -1:
                U = line.split('=')[-1].split()
            if line.find('LDAUJ') != -1:
                J = line.split('=')[-1].split()
        # create dictionary
        if ldau:
            for i,symbol in enumerate(atomtypes):
                ldau_luj[symbol] = {'L': int(L[i]), 'U': float(U[i]), 'J': float(J[i])}
            self.dict_params['ldau_luj'] = ldau_luj
        return ldau, ldauprint, ldautype, ldau_luj

    def run_wavecar(self, **kwargs):
        template = \
'''ENCUT = 400
LPEAD=.TRUE.
ALGO=DMET
NELM=1
SIGMA=0.001
LHFCALC=.TRUE.
AEXX=1.0
PRECFOCK=N
NBANDS=1
ISYM=-1
'''
        with open('INCAR', 'w') as fout:
            fout.write('%s\n' % _replace(template, kwargs))
        self.run(settings.VASPEXE)
        os.system('/bin/mv WAVECAR.GTO WAVECAR')

    def run_hf(self, **kwargs):
        template = \
'''PRECFOCK=N
ICHARG=0
ISMEAR=-2
#FERWE=32*1.0 336*0.0
NBANDS=1
EDIFF=1e-6
SIGMA=0.001
LHFCALC=.TRUE.
AEXX=1.0
#NELM=1
ENCUT=400
ALGO=S
ISYM=-1
'''
        with open('INCAR', 'w') as fout:
            fout.write('%s\n' % _replace(template, kwargs))
        self.run(settings.VASPMPI)
#rotate to real
        template = \
'''PRECFOCK= N
NBANDS=1
SIGMA=0.001
LHFCALC=.TRUE.
AEXX=1.0
ENCUT = 400
ALGO=S
#ALGO=Eigenval
LORBITALREAL=.TRUE.
NELM=1
ISYM=-1
'''
        with open('INCAR', 'w') as fout:
            fout.write('%s\n' % _replace(template, kwargs))
        self.run(settings.VASPMPI)
        os.system('/bin/mv OUTCAR OUTCAR.HF')

    def run_jkdump(self, **kwargs):
        template = \
'''PRECFOCK= N
NBANDS=1
ALGO=JKDUMP
LMAXFOCKAE=4
SIGMA=0.001
LHFCALC=.TRUE.
AEXX=1.0
ENCUT=400
NELM=1
ENCUTGW=400
ISYM=-1
'''
        with open('INCAR', 'w') as fout:
            fout.write('%s\n' % _replace(template, kwargs))
        self.run(settings.VASPMPI)
        os.system('/bin/mv OUTCAR OUTCAR.JKDUMP')

    def run_clustdump(self, **kwargs):
        template = \
'''PRECFOCK=N
NBANDS=1
ALGO=CLUSTDUMP
LMAXFOCKAE=4
SIGMA=0.001
LHFCALC=.TRUE.
AEXX=1.0
ENCUT=400
NELM=1
ENCUTGW=400
ISYM=-1
'''
        with open('INCAR', 'w') as fout:
            fout.write('%s\n' % _replace(template, kwargs))
        self.run(settings.VASPEXE)

    def write_incar(self, incar_template, **kwargs):
        incar = open('INCAR', 'w')
        incar.write('%s\n' % incar_template)
        for key, val in kwargs.items():
            if key in float_keys:
                incar.write(' %s = %5.6f\n' % (key.upper(), val))
            elif key in exp_keys:
                incar.write(' %s = %5.2e\n' % (key.upper(), val))
            elif key in string_keys:
                incar.write(' %s = %s\n' % (key.upper(), val))
            elif key in int_keys:
                incar.write(' %s = %d\n' % (key.upper(), val))
                if key == 'ichain' and val > 0:
                    incar.write(' IBRION = 3\n POTIM = 0.0\n')
                    for key, val in self.int_params.items():
                        if key == 'iopt' and val == None:
                            print 'WARNING: optimization is set to LFBGS (IOPT = 1)'
                            incar.write(' IOPT = 1\n')
                    for key, val in self.exp_params.items():
                        if key == 'ediffg' and val == None:
                            RuntimeError('Please set EDIFFG < 0')
            elif key in bool_keys:
                incar.write(' %s = ' % key.upper())
                if val:
                    incar.write('.TRUE.\n')
                else:
                    incar.write('.FALSE.\n')
            elif key in list_keys:
                incar.write(' %s = ' % key.upper())
                if key in ('dipol', 'eint', 'ropt', 'rwigs'):
                    [incar.write('%.4f ' % x) for x in val]
                elif key in ('ldauu', 'ldauj', 'ldaul') and \
                    not self.dict_keys.has('ldau_luj'):
                    [incar.write('%.4f ' % x) for x in val]
                elif key in ('ferwe', 'ferdo'):
                    [incar.write('%.1f ' % x) for x in val]
                elif key in ('iband', 'kpuse'):
                    [incar.write('%i ' % x) for x in val]
                elif key == 'magmom':
                    list = [[1, val[0]]]
                    for n in range(1, len(val)):
                        if val[n] == val[n-1]:
                            list[-1][0] += 1
                        else:
                            list.append([1, val[n]])
                    [incar.write('%i*%.4f ' % (mom[0], mom[1])) for mom in list]
                incar.write('\n')
            elif key in special_keys:
                incar.write(' %s = ' % key.upper())
                if key == 'lreal':
                    if type(val) == str:
                        incar.write(val+'\n')
                    elif type(val) == bool:
                       if val:
                           incar.write('.TRUE.\n')
                       else:
                           incar.write('.FALSE.\n')
            elif key in dict_keys:
                if key == 'ldau_luj':
                    llist = ulist = jlist = ''
                    for symbol in self.symbol_count:
                        luj = val.get(symbol[0], {'L':-1, 'U': 0.0, 'J': 0.0}) # default: No +U
                        llist += ' %i' % luj['L']
                        ulist += ' %.3f' % luj['U']
                        jlist += ' %.3f' % luj['J']
                    incar.write(' LDAUL =%s\n' % llist)
                    incar.write(' LDAUU =%s\n' % ulist)
                    incar.write(' LDAUJ =%s\n' % jlist)
            else:
                raise KeyEvent('Unknown key-value %s:%s' % (key, val))
        incar.close()

    def write_kpoints(self, kx=1, ky=1, kz=1):
        temp = \
'''Automatically generated mesh
0
Gamma
%d %d %d
0 0 0
''' % (kx, ky, kz)
        with open('KPOINTS', 'w') as fout:
            fout.write('%s\n' % temp)

    def write_potcar(self,suffix = ""):
        """Writes the POTCAR file."""
        import tempfile
        potfile = open('POTCAR'+suffix,'w')
        for filename in self.ppp_list:
            if filename.endswith('R'):
                for line in open(filename, 'r'):
                    potfile.write(line)
            elif filename.endswith('.Z'):
                file_tmp = tempfile.NamedTemporaryFile()
                os.system('gunzip -c %s > %s' % (filename, file_tmp.name))
                for line in file_tmp.readlines():
                    potfile.write(line)
                file_tmp.close()
        potfile.close()

    # Methods for reading information from OUTCAR files:
    def read_energy(self, all=None):
        [energy_free, energy_zero]=[0, 0]
        if all:
            energy_free = []
            energy_zero = []
        for line in open('OUTCAR', 'r'):
            # Free energy
            if line.lower().startswith('  free  energy   toten'):
                if all:
                    energy_free.append(float(line.split()[-2]))
                else:
                    energy_free = float(line.split()[-2])
            # Extrapolated zero point energy
            if line.startswith('  energy  without entropy'):
                if all:
                    energy_zero.append(float(line.split()[-1]))
                else:
                    energy_zero = float(line.split()[-1])
        return [energy_free, energy_zero]

    def read_forces(self, atoms, all=False):
        """Method that reads forces from OUTCAR file.

        If 'all' is switched on, the forces for all ionic steps
        in the OUTCAR file be returned, in other case only the
        forces for the last ionic configuration is returned."""

        file = open('OUTCAR','r')
        lines = file.readlines()
        file.close()
        n=0
        if all:
            all_forces = []
        for line in lines:
            if line.rfind('TOTAL-FORCE') > -1:
                forces=[]
                for i in range(len(atoms)):
                    forces.append(np.array([float(f) for f in lines[n+2+i].split()[3:6]]))
                if all:
                    all_forces.append(np.array(forces)[self.resort])
            n+=1
        if all:
            return np.array(all_forces)
        else:
            return np.array(forces)[self.resort]

    def read_fermi(self):
        """Method that reads Fermi energy from OUTCAR file"""
        E_f=None
        for line in open('OUTCAR', 'r'):
            if line.rfind('E-fermi') > -1:
                E_f=float(line.split()[2])
        return E_f

    def read_dipole(self):
        dipolemoment=np.zeros([1,3])
        for line in open('OUTCAR', 'r'):
            if line.rfind('dipolmoment') > -1:
                dipolemoment=np.array([float(f) for f in line.split()[1:4]])
        return dipolemoment

    def read_magnetic_moments(self, atoms):
        magnetic_moments = np.zeros(len(atoms))
        n = 0
        lines = open('OUTCAR', 'r').readlines()
        for line in lines:
            if line.rfind('magnetization (x)') > -1:
                for m in range(len(atoms)):
                    magnetic_moments[m] = float(lines[n + m + 4].split()[4])
            n += 1
        return np.array(magnetic_moments)[self.resort]

    def read_magnetic_moment(self):
        n=0
        for line in open('OUTCAR','r'):
            if line.rfind('number of electron  ') > -1:
                magnetic_moment=float(line.split()[-1])
            n+=1
        return magnetic_moment

    def read_nbands(self):
        for line in open('OUTCAR', 'r'):
            line = self.strip_warnings(line)
            if line.rfind('NBANDS') > -1:
                return int(line.split()[-1])

    def strip_warnings(self, line):
        """Returns empty string instead of line from warnings in OUTCAR."""
        if line[0] == "|":
            return ""
        else:
            return line

    def read_convergence(self):
        """Method that checks whether a calculation has converged."""
        converged = None
        # First check electronic convergence
        for line in open('OUTCAR', 'r'):
            if 0:  # vasp always prints that!
                if line.rfind('aborting loop') > -1:  # scf failed
                    raise RuntimeError(line.strip())
                    break
            if line.rfind('EDIFF  ') > -1:
                ediff = float(line.split()[2])
            if line.rfind('total energy-change')>-1:
                # I saw this in an atomic oxygen calculation. it
                # breaks this code, so I am checking for it here.
                if 'MIXING' in line:
                    continue
                split = line.split(':')
                a = float(split[1].split('(')[0])
                b = split[1].split('(')[1][0:-2]
                # sometimes this line looks like (second number wrong format!):
                # energy-change (2. order) :-0.2141803E-08  ( 0.2737684-111)
                # we are checking still the first number so
                # let's "fix" the format for the second one
                if 'e' not in b.lower():
                    # replace last occurence of - (assumed exponent) with -e
                    bsplit = b.split('-')
                    bsplit[-1] = 'e' + bsplit[-1]
                    b = '-'.join(bsplit).replace('-e','e-')
                b = float(b)
                if [abs(a), abs(b)] < [ediff, ediff]:
                    converged = True
                else:
                    converged = False
                    continue
        # Then if ibrion in [1,2,3] check whether ionic relaxation
        # condition been fulfilled
        if (self.int_params['ibrion'] in [1,2,3]
            and self.int_params['nsw'] not in [0]) :
            if not self.read_relaxed():
                converged = False
            else:
                converged = True
        return converged

    def read_ibz_kpoints(self):
        lines = open('OUTCAR', 'r').readlines()
        ibz_kpts = []
        n = 0
        i = 0
        for line in lines:
            if line.rfind('Following cartesian coordinates')>-1:
                m = n+2
                while i==0:
                    ibz_kpts.append([float(lines[m].split()[p]) for p in range(3)])
                    m += 1
                    if lines[m]==' \n':
                        i = 1
            if i == 1:
                continue
            n += 1
        ibz_kpts = np.array(ibz_kpts)
        return np.array(ibz_kpts)

    def read_k_point_weights(self):
        file = open('IBZKPT')
        lines = file.readlines()
        file.close()
        if 'Tetrahedra\n' in lines:
            N = lines.index('Tetrahedra\n')
        else:
            N = len(lines)
        kpt_weights = []
        for n in range(3, N):
            kpt_weights.append(float(lines[n].split()[3]))
        kpt_weights = np.array(kpt_weights)
        kpt_weights /= np.sum(kpt_weights)
        return kpt_weights

    def read_eigenvalues(self, kpt=0, spin=0):
        file = open('EIGENVAL', 'r')
        lines = file.readlines()
        file.close()
        eigs = []
        for n in range(8+kpt*(self.nbands+2), 8+kpt*(self.nbands+2)+self.nbands):
            eigs.append(float(lines[n].split()[spin+1]))
        return np.array(eigs)

    def read_occupation_numbers(self, kpt=0, spin=0):
        lines = open('OUTCAR').readlines()
        nspins = self.get_number_of_spins()
        start = 0
        if nspins == 1:
            for n, line in enumerate(lines): # find it in the last iteration
                m = re.search(' k-point *'+str(kpt+1)+' *:', line)
                if m is not None:
                    start = n
        else:
            for n, line in enumerate(lines):
                if line.find(' spin component '+str(spin+1)) != -1: # find it in the last iteration
                    start = n
            for n2, line2 in enumerate(lines[start:]):
                m = re.search(' k-point *'+str(kpt+1)+' *:', line2)
                if m is not None:
                    start = start + n2
                    break
        for n2, line2 in enumerate(lines[start+2:]):
            if not line2.strip():
                break
        occ = []
        for line in lines[start+2:start+2+n2]:
            occ.append(float(line.split()[2]))
        return np.array(occ)

    def read_relaxed(self):
        for line in open('OUTCAR', 'r'):
            if line.rfind('reached required accuracy') > -1:
                return True
        return False

# The below functions are used to restart a calculation and are under early constructions

#    def read_incar(self, filename='INCAR'):
#        """Method that imports settings from INCAR file."""
#
#        self.spinpol = False
#        file=open(filename, 'r')
#        file.readline()
#        lines=file.readlines()
#        for line in lines:
#            try:
#                # Make multiplications easier to spot
#                line = line.replace("*", " * ")
#                data = line.split()
#                # Skip empty and commented lines.
#                if len(data) == 0:
#                    continue
#                elif data[0][0] in ['#', '!']:
#                    continue
#                key = data[0].lower()
#                if key in float_keys:
#                    self.float_params[key] = float(data[2])
#                elif key in exp_keys:
#                    self.exp_params[key] = float(data[2])
#                elif key in string_keys:
#                    self.string_params[key] = str(data[2])
#                elif key in int_keys:
#                    if key == 'ispin':
#                        # JRK added. not sure why we would want to leave ispin out
#                        self.int_params[key] = int(data[2])
#                        if int(data[2]) == 2:
#                            self.spinpol = True
#                    else:
#                        self.int_params[key] = int(data[2])
#                elif key in bool_keys:
#                    if 'true' in data[2].lower():
#                        self.bool_params[key] = True
#                    elif 'false' in data[2].lower():
#                        self.bool_params[key] = False
#                elif key in list_keys:
#                    list = []
#                    if key in ('dipol', 'eint', 'ferwe', 'ropt', 'rwigs',
#                               'ldauu', 'ldaul', 'ldauj'):
#                        for a in data[2:]:
#                            if a in ["!", "#"]:
#                               break
#                            list.append(float(a))
#                    elif key in ('iband', 'kpuse'):
#                        for a in data[2:]:
#                            if a in ["!", "#"]:
#                               break
#                            list.append(int(a))
#                    self.list_params[key] = list
#                    if key == 'magmom':
#                        list = []
#                        i = 2
#                        while i < len(data):
#                            if data[i] in ["#", "!"]:
#                                break
#                            if data[i] == "*":
#                                b = list.pop()
#                                i += 1
#                                for j in range(int(b)):
#                                    list.append(float(data[i]))
#                            else:
#                                list.append(float(data[i]))
#                            i += 1
#                        self.list_params['magmom'] = list
#                        list = np.array(list)
#                        if self.atoms is not None:
#                                self.atoms.set_initial_magnetic_moments(list[self.resort])
#                elif key in special_keys:
#                    if key == 'lreal':
#                        if 'true' in data[2].lower():
#                            self.special_params[key] = True
#                        elif 'false' in data[2].lower():
#                            self.special_params[key] = False
#                        else:
#                            self.special_params[key] = data[2]
#            except KeyError:
#                raise IOError('Keyword "%s" in INCAR is not known by calculator.' % key)
#            except IndexError:
#                raise IOError('Value missing for keyword "%s".' % key)

    def read_outcar(self):
        # Spin polarized calculation?
        file = open('OUTCAR', 'r')
        lines = file.readlines()
        file.close()
        for line in lines:
            if line.rfind('ISPIN') > -1:
                if int(line.split()[2])==2:
                    self.spinpol = True
                else:
                    self.spinpol = None
        self.energy_free, self.energy_zero = self.read_energy()
        self.forces = self.read_forces(self.atoms)
        self.dipole = self.read_dipole()
        self.fermi = self.read_fermi()
        self.stress = self.read_stress()
        self.nbands = self.read_nbands()
        self.read_ldau()
        p=self.int_params
        q=self.list_params
        if self.spinpol:
            self.magnetic_moment = self.read_magnetic_moment()
            if p['lorbit']>=10 or (p['lorbit']!=None and q['rwigs']):
                self.magnetic_moments = self.read_magnetic_moments(self.atoms)
            else:
                self.magnetic_moments = None
        self.set(nbands=self.nbands)

    def read_kpoints(self, filename='KPOINTS'):
        file = open(filename, 'r')
        lines = file.readlines()
        file.close()
        ktype = lines[2].split()[0].lower()[0]
        if ktype in ['g', 'm']:
            if ktype=='g':
                self.set(gamma=True)
            kpts = np.array([int(lines[3].split()[i]) for i in range(3)])
            self.set(kpts=kpts)
        elif ktype in ['c', 'k']:
            raise NotImplementedError('Only Monkhorst-Pack and gamma centered grid supported for restart.')
        else:
            raise NotImplementedError('Only Monkhorst-Pack and gamma centered grid supported for restart.')

#    def read_potcar(self):
#        """ Method that reads the Exchange Correlation functional from POTCAR file.
#        """
#        file = open('POTCAR', 'r')
#        lines = file.readlines()
#        file.close()
#
#        # Search for key 'LEXCH' in POTCAR
#        xc_flag = None
#        for line in lines:
#            key = line.split()[0].upper()
#            if key == 'LEXCH':
#                xc_flag = line.split()[-1].upper()
#                break
#
#        if xc_flag is None:
#            raise ValueError('LEXCH flag not found in POTCAR file.')
#
#        # Values of parameter LEXCH and corresponding XC-functional
#        xc_dict = {'PE':'PBE', '91':'PW91', 'CA':'LDA'}
#
#        if xc_flag not in xc_dict.keys():
#            raise ValueError(
#                'Unknown xc-functional flag found in POTCAR, LEXCH=%s' % xc_flag)
#
#        self.input_params['xc'] = xc_dict[xc_flag]

    def get_nonselfconsistent_energies(self, bee_type):
        """ Method that reads and returns BEE energy contributions
            written in OUTCAR file.
        """
        assert bee_type == 'beefvdw'
        p = os.popen('grep -32 "BEEF xc energy contributions" OUTCAR | tail -32','r')
        s = p.readlines()
        p.close()
        xc = np.array([])
        for i, l in enumerate(s):
            l_ = float(l.split(":")[-1])
            xc = np.append(xc, l_)
        assert len(xc) == 32
        return xc

def _format_basis(bstr):
    braw = gto.basis.parse_nwchem.parse_str(bstr)
    bfmt = []
    for l in set([x[0] for x in braw]):
        bl = [numpy.array(x[1:]) for x in braw if x[0] == l]
        exps = [b[:,0] for b in bl]
        cs = [b[:,1:] for b in bl]
        cs = scipy.linalg.block_diag(*cs)
        ec = numpy.hstack((exps, cs))
        bfmt.append(ec)
    return bfmt

def _count_nb(gaussbasis, poscar):
    pass

def _replace(template, dic):
    dat = template.split()
    res = []
    for line in dat:
        key = line.split('=')[0].strip()
        if key in dic:
            res.append('%s=%s' % (key, str(dic[key])))
        else:
            res.append(line)
    return '\n'.join(res)
