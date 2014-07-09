#!/usr/bin/env python

import numpy
import scipy.optimize
import impsolver
from pyscf import lib
import pyscf.lib.logger as log
import pyscf.lib.parameters as param
from pyscf.lib import _vhf
from pyscf import ao2mo
import dmet_sc


##################################################
# system with translation symmetry
class EmbSys(dmet_sc.EmbSys):
    def __init__(self, mol, entire_scf, frag_group=[], init_v=None,
                 orth_coeff=None):
        dmet_sc.EmbSys.__init__(self, mol, entire_scf, frag_group, \
                                init_v, orth_coeff)
        self.max_iter = 1

    def scdmet(self, mol, sav_v=None):
        log.warn(self, 'Self-consistency is not allowed in non-SC-DMET')
        self.fullsys(mol)

    def fullsys(self, mol):
        log.info(self, '==== fullsys ====')
        self.dump_options()

        self.init_embsys(mol)
        v_ci_group = self.vfit_ci_method(mol, self)
        self.update_embs_vfit_ci(mol, self.embs, v_ci_group)
        e_tot, nelec = self.assemble_frag_fci_energy(mol)

        log.info(self, '====================')
        if self.verbose >= param.VERBOSE_DEBUG:
            for m,emb in enumerate(self.embs):
                log.debug(self, 'vfit_ci of frag %d = %s', m, v_ci_group[m])
                res = self.frag_fci_solver(mol, emb, emb.vfit_ci)
                log.debug(self, 'impurity dm of frag %d = %s', m, res['rdm1'])
        log.info(self, 'dmet_nonsc.fullsys: e_tot = %.12g, nelec = %g', \
                 e_tot, nelec)
        return e_tot

    def one_shot(self, mol):
        log.info(self, '==== one-shot ====')
        self.init_embsys(mol)
        emb = self.embs[0]
        emb.verbose = self.verbose
        #emb.imp_scf()

        vfit_ci = dmet_sc.fit_chemical_potential(mol, emb, self)
        #self.update_embs_vfit_ci(mol, [emb], [vfit_ci])
        cires = self.frag_fci_solver(mol, emb, vfit_ci)
        e_tot = cires['etot'] + emb.energy_by_env
        #print etot, emb.hf_energy

        log.info(self, '====================')
        if self.verbose >= param.VERBOSE_DEBUG:
            log.debug(self, 'vfit_ci = %s', vfit_ci)
            log.debug(self, 'impurity dm = %s', cires['rdm1'])
        log.info(emb, 'dmet_nonsc.one_shot: e_tot = %.11g, (+nuc=%.11g)', \
                 e_tot, e_tot+mol.nuclear_repulsion())
        return e_tot


if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    import hf
#    mol = gto.Mole()
#    mol.verbose = 5
#    mol.output = 'out_dmet_1shot'
#    mol.build()
#
#    mf = hf.RHF(mol, 'C_solid_2x2x2/test2/FCIDUMP.CLUST.GTO',
#                'C_solid_2x2x2/test2/JKDUMP')
#    energy = mf.scf()
#    print energy
#
#    emb = OneImp(mf, [0,1,2,3])
#    print dmet_1shot(mol, emb)

######################
    mol = gto.Mole()
    mol.verbose = 5
    mol.output = 'out_dmet'
    mol.atom = [
        ['O' , (0. , 0.     , 0.)],
        [1   , (0. , -0.757 , 0.587)],
        [1   , (0. , 0.757  , 0.587)] ]
    mol.basis = {'H': '6-31g',
                 'O': '6-31g',}
    mol.build()
    mf = scf.RHF(mol)
    mf.scf()

    embsys = EmbSys(mol, mf, [[0,1],[2,3],[4,5],[6,7],[8,9]])
    embsys.OneImp = oneimp.OneImpNI
    print embsys.fullsys(mol)

    embsys = EmbSys(mol, mf, [[0,1]])
    print embsys.one_shot(mol)

    embsys = EmbSys(mol, mf, [[0,1]])
    embsys.OneImp = oneimp.OneImpNI
    print embsys.one_shot(mol)

#    b1 = 1.0
#    nat = 10
#    mol.output = 'h%s_sz' % nat
#    mol.atom = []
#    r = b1/2 / numpy.sin(numpy.pi/nat)
#    for i in range(nat):
#        theta = i * (2*numpy.pi/nat)
#        mol.atom.append((1, (r*numpy.cos(theta),
#                             r*numpy.sin(theta), 0)))
#
#    mol.basis = {'H': 'sto-3g',}
#    mol.build()
#    mf = scf.RHF(mol)
#    print mf.scf()
#
#    embsys = EmbSys(mol, mf, [[0,1],[2,3],[4,5],[6,7],[8,9]])
#    print embsys.fullsys(mol)
#
#    embsys = EmbSys(mol, mf, [[0,1]])
#    print embsys.one_shot(mol)
