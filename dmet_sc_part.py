#!/usr/bin/env python

import time
import pickle
import numpy
import copy

from pyscf import scf
import pyscf.lib.parameters as param
import pyscf.lib.logger as log
import dmet_sc


# when frags cannot cover the entire system, the rest of the system are
# treated in mean-field level


class EmbSys(dmet_sc.EmbSys):
    def __init__(self, mol, entire_scf, frag_group=[], init_v=None,
                 orth_coeff=None):
        dmet_sc.EmbSys.__init__(self, mol, entire_scf, frag_group, init_v,
                                orth_coeff)
        self.bas_off_frags = []

    def build_(self, mol):
        dmet_sc.EmbSys.build_(self, mol)
        nao = self.orth_coeff.shape[1]
        baslst = numpy.ones(nao, dtype=bool)
        for m, atm_lst, bas_idx in self.all_frags:
            baslst[bas_idx] = False
        self.bas_off_frags = [i for i in range(nao) if baslst[i]]

    # if fragments do not cover the whole system, the rests are treated at
    # mean-field level.  Asymmetrical energy expression is used for the rests
    def off_frags_energy(self, mol, dm_mf):
        if len(self.bas_off_frags) == 0:
            return 0, 0

        v_group = [emb.vfit_mf for emb in self.embs]
        if self.with_hopping:
            v_global = self.assemble_to_fullmat(v_group)
        else:
            v_global = self.assemble_to_blockmat(v_group)

        h1e = reduce(numpy.dot, (self.orth_coeff.T, \
                                 self.entire_scf.get_hcore(mol), \
                                 self.orth_coeff)) + v_global
        h1e = h1e[self.bas_off_frags]
        dm_ao = reduce(numpy.dot, (self.orth_coeff,dm_mf,self.orth_coeff.T))
        vhf = self.entire_scf.get_veff(mol, dm_ao)
        vhf = reduce(numpy.dot, (self.orth_coeff.T,vhf,self.orth_coeff))
        vhf = vhf[self.bas_off_frags]
        dm_frag = dm_mf[self.bas_off_frags]
        e = numpy.dot(h1e.flatten(), dm_frag.flatten()) \
                + numpy.dot(vhf.flatten(), dm_frag.flatten())*.5

        nelec = dm_frag.trace()
        return e, nelec

    def assemble_frag_fci_energy(self, mol, dm_ref=0):
        if len(self.bas_off_frags) == 0:
            e_tot = 0
            nelec = 0
        else:
            if dm_ref is 0:
                eff_scf = self.entire_scf
                sc = numpy.dot(eff_scf.get_ovlp(mol), self.orth_coeff)
                mo = numpy.dot(sc.T,eff_scf.mo_coeff)
                dm_ref = eff_scf.calc_den_mat(mo, eff_scf.mo_occ)
            e_tot, nelec = self.off_frags_energy(mol, dm_ref)

        last_frag = -1
        for m, _, _ in self.all_frags:
            if m != last_frag:
                emb = self.embs[m]
                nimp = len(emb.bas_on_frag)
                _, e2frag, dm1 = \
                        self.solver.run(emb, emb._eri, emb.vfit_ci,
                                        with_1pdm=True, with_e2frag=nimp)
                e_frag, nelec_frag = \
                        self.extract_frag_energy(emb, dm1, e2frag)
                log.debug(self, 'e_frag = %.12g, nelec_frag = %.12g', \
                          e_frag, nelec_frag)
            e_tot += e_frag
            nelec += nelec_frag
            last_frag = m
        log.info(self, 'DMET-FCI-in-HF of entire system, e_tot = %.9g, nelec_tot = %.9g', \
                  e_tot, nelec)
        return e_tot, nelec



