import numpy
import scipy.linalg
import pyscf.lib.logger as log
from pyscf import ao2mo
import dmet_hf
import dmet_sc


class OneImpNI(dmet_hf.RHF):
    '''Non-interacting DMET'''
    def __init__(self, entire_scf, orth_ao=None):
        dmet_hf.RHF.__init__(self, entire_scf, orth_ao=orth_ao)

    def eri_on_impbas(self, mol):
        nimp = len(self.bas_on_frag)
        nemb = self.impbas_coeff.shape[1]
        mo = self.impbas_coeff[:,:nimp].copy('F')
        if self.entire_scf._eri is not None:
            eri = ao2mo.incore.full(self.entire_scf._eri, mo)
        else:
            eri = ao2mo.outcore.full_iofree(mol, mo)
        npair = nemb*(nemb+1) / 2
        #eri_mo = numpy.zeros(npair*(npair+1)/2)
        npair_imp = nimp*(nimp+1) / 2
        # so only the 2e-integrals on impurity are non-zero
        #eri_mo[:npair_imp*(npair_imp+1)/2] = eri.reshape(-1)
        eri_mo = numpy.zeros((npair,npair))
        eri_mo[:npair_imp,:npair_imp] = eri
        return ao2mo.restore(8, eri_mo, nemb)

    def get_hcore(self, mol=None):
        nimp = len(self.bas_on_frag)
        effscf = self.entire_scf
        cs = numpy.linalg.solve(effscf.mo_coeff, self.impbas_coeff)
        fock = numpy.dot(cs.T*effscf.mo_energy, cs)
        dmimp = effscf.make_rdm1(mo_coeff=cs.T)
        dm = numpy.zeros_like(fock)
        dm[:nimp,:nimp] = dmimp[:nimp,:nimp]
        h1e = fock - self.get_veff(self.mol, dm)
        return h1e


#TODO: add fitting potential on bath in this case
class EmbSys(dmet_sc.EmbSys):
    def __init__(self, mol, entire_scf, frag_group=[], init_v=None,
                 orth_coeff=None):
        dmet_sc.EmbSys.__init__(self, mol, entire_scf, frag_group, init_v,
                                orth_coeff)
        self.OneImp = OneImpNI

    def extract_frag_energy(self, emb, dm1, e2frag):
        nimp = len(emb.bas_on_frag)
        hcore = emb._pure_hcore
        hfdm = self.entire_scf.make_rdm1(self.entire_scf.mo_coeff,
                                         self.entire_scf.mo_occ)
        vhf = emb.mat_ao2impbas(emb.entire_scf.get_veff(self.mol, hfdm))
        e = numpy.dot(dm1[:nimp].flatten(), hcore[:nimp].flatten()) \
          + numpy.dot(dm1[:nimp].flatten(), vhf[:nimp].flatten()) * .5

        #emb.mo_coeff_on_imp are changed in function embscf_
        #hfdm = emb.make_rdm1(emb.mo_coeff_on_imp, emb.mo_occ)
        c = scipy.linalg.eigh(emb._project_fock)[1]
        hfdm = emb.make_rdm1(c, emb.mo_occ)
        vhfemb = emb.get_veff(emb.mol, hfdm)
        ecorr = e2frag - numpy.dot(dm1[:nimp].flatten(),
                                   vhfemb[:nimp].flatten())*.5
        log.debug1(self, '  FCI pTraceSys = %.12g, ecorr = %12g', e2frag, ecorr)
        e_frag = e + ecorr
        nelec_frag = dm1[:nimp].trace()
        return e_frag, nelec_frag

