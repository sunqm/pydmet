#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

from functools import reduce
import numpy
import scipy.linalg
import pyscf.lib
from pyscf.lib import logger
from pyscf import scf
from pyscf.tools import dump_mat

def dmet_cas(casscf, dm, baslst, occ_cutoff=1e-8, baths=None, base=1,
             orth_method='meta_lowdin', verbose=logger.WARN):
    from pyscf import lo
    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2): # ROHF/UHF DM
        dm = sum(dm)
    mol = casscf.mol
    log = logger.Logger(casscf.stdout, casscf.verbose)

    mo, ncas_guess, nelecas_guess, ncore_guess = \
            _decompose_dm(casscf.mol, dm, baslst,
                          casscf.ncas, casscf.nelecas, casscf.ncore,
                          occ_cutoff=occ_cutoff, baths=baths, base=base,
                          orth_method=orth_method, s=casscf._scf.get_ovlp(),
                          verbose=log)

    mocore = mo[:,:casscf.ncore]
    mocas = mo[:,casscf.ncore:casscf.ncore+casscf.ncas]
    movir = mo[:,casscf.ncore+casscf.ncas:]
    s = casscf._scf.get_ovlp()
    sc = numpy.dot(s, casscf._scf.mo_coeff)
    fock = reduce(numpy.dot, (sc*casscf._scf.mo_energy, sc.T))

    def search_for_degeneracy(e):
        idx = numpy.where(abs(e[1:] - e[:-1]) < 1e-6)[0]
        return numpy.unique(numpy.hstack((idx, idx+1)))
    def symmetrize(e, c):
        if casscf.mol.symmetry:
            degidx = search_for_degeneracy(e)
            log.debug1('degidx %s', degidx)
            if degidx.size > 0:
                esub = e[degidx]
                csub = c[:,degidx]
                scsub = numpy.dot(s, csub)
                emin = abs(esub).min() * .5
                es = []
                cs = []
                for i,ir in enumerate(mol.irrep_id):
                    so = mol.symm_orb[i]
                    sosc = numpy.dot(so.T, scsub)
                    s_ir = reduce(numpy.dot, (so.T, s, so))
                    fock_ir = numpy.dot(sosc*esub, sosc.T)
                    e, u = scipy.linalg.eigh(fock_ir, s_ir)
                    idx = abs(e) > emin
                    es.append(e[idx])
                    cs.append(numpy.dot(mol.symm_orb[i], u[:,idx]))
                es = numpy.hstack(es)
                idx = numpy.argsort(es)
                assert(numpy.allclose(es[idx], esub))
                c[:,degidx] = numpy.hstack(cs)[:,idx]
        return c
    mo = []
    for c in (mocore, mocas, movir):
        f1 = reduce(numpy.dot, (c.T, fock, c))
        e, u = scipy.linalg.eigh(f1)
        log.debug1('Fock eig %s', e)
        mo.append(symmetrize(e, numpy.dot(c, u)))
    mo = numpy.hstack(mo)

    if ncore_guess != casscf.ncore:
        log.warn('ncore_guess %d   != casscf.ncore %d', ncore_guess, casscf.ncore)
    if ncas_guess != casscf.ncas:
        log.warn('ncas_guess %d    != casscf.ncas %d', ncas_guess, casscf.ncas)
        log.warn('nelecas_guess %s != casscf.nelecas %s',
                 str(nelecas_guess), str(casscf.nelecas))
    return mo

def _decompose_dm(mol, dm, baslst, ncas, nelecas, ncore, occ_cutoff=1e-8,
                  baths=None, base=1, orth_method='meta_lowdin', s=None,
                  verbose=logger.WARN):
    from pyscf import lo
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    assert(isinstance(dm, numpy.ndarray) and dm.ndim == 2)
    if base != 0:
        baslst = [i-base for i in baslst]
    nao = dm.shape[0]
    nocc = ncore + ncas
    nimp = len(baslst)
    log.debug('*** decompose density matrix')
    log.debug('orth AO method = %s', orth_method)
    log.debug('embedding AO list = %s', str(baslst))
    if orth_method is not None:
        if s is None:
            s = mol.intor_symmetric('cint1e_ovlp_sph')
        corth = lo.orth.orth_ao(mol, method=orth_method, s=s)
        cinv = numpy.dot(corth.T, s)
        dm = reduce(numpy.dot, (cinv, dm, cinv.T))
    else:
        corth = numpy.eye(nao)

    baslst = numpy.asarray(baslst)
    notimp = numpy.asarray([i for i in range(nao) if i not in baslst])
    occi, ui = scipy.linalg.eigh(-dm[baslst[:,None],baslst])
    occi *= -1
    idxi = numpy.argsort(abs(occi-1))
    log.debug('entanglement weight occ = %s', str(occi[idxi]))
    occb, ub = scipy.linalg.eigh(dm[notimp[:,None],notimp])
    idxb = numpy.argsort(abs(occb-1))
    log.debug('bath weight occ = %s', str(occb[idxb]))

    if log.verbose >= logger.DEBUG:
        log.debug('DMET %d impurity sites/occ', nimp)
        label = mol.spheric_labels(True)
        occ_label = ['#%d/%.5f'%(i+1,x) for i,x in enumerate(occi)]
        dump_mat.dump_rec(mol.stdout, numpy.dot(corth[:,baslst], ui),
                          label=label, label2=occ_label, start=1)
        nb = 0
        for i in idxb:
            if occ_cutoff < occb[i] < 2-occ_cutoff:
                nb += 1
            else:
                break
        log.debug('DMET %d entangled baths/occ', nb)
        occ_label = ['#%d/%.5f'%(i+1,occb[j]) for i,j in enumerate(idxb)]
        dump_mat.dump_rec(mol.stdout, numpy.dot(corth[:,notimp], ub[:,idxb[:nb]]),
                          label=label, label2=occ_label, start=1)

    if baths is not None:
        nbath = nao - nimp
        mob = numpy.dot(corth[:,notimp], ub)
        idxcas = idxb[baths]
        idxenv = numpy.asarray([i for i in idxb if i not in idxcas])
        idxenv = idxenv[numpy.argsort(-occb[idxenv])]
        mo = numpy.hstack((mob[:,idxenv[:ncore]], numpy.dot(corth[:,baslst],ui),
                           mob[:,idxcas], mob[:,idxenv[ncore:]]))
        occ = numpy.hstack((occb[idxenv[:ncore]], occi,
                            occb[idxcas], occb[idxenv[ncore:]]))
        ncore_guess = (occb[idxenv[:ncore]]>2-occ_cutoff).sum()
        nvirt_guess = (occb[idxenv[ncore:]]<  occ_cutoff).sum()
    elif 0: # The baths have largest occs
        #occ = numpy.hstack((occi, occb))
        idxb = numpy.argsort(occb)[::-1]
        occ = numpy.hstack((occb[idxb[:ncore]], occi, occb[idxb[ncore:]]))
        mob = numpy.dot(corth[:,notimp],ub[:,idxb])
        mo = numpy.hstack((mob[:,:ncore], numpy.dot(corth[:,baslst],ui), mob[:,ncore:]))
        ncore_guess = (occb>2-occ_cutoff).sum()
        nvirt_guess = (occb<occ_cutoff).sum()
    elif 1: # The baths have largest entanglement
        mob = numpy.dot(corth[:,notimp], ub)
        idxcas, idxenv = idxb[:ncas-nimp], idxb[ncas-nimp:]
        idxenv = idxenv[numpy.argsort(-occb[idxenv])]
        mo = numpy.hstack((mob[:,idxenv[:ncore]], numpy.dot(corth[:,baslst],ui),
                           mob[:,idxcas], mob[:,idxenv[ncore:]]))
        occ = numpy.hstack((occb[idxenv[:ncore]], occi,
                            occb[idxcas], occb[idxenv[ncore:]]))
        ncore_guess = (occb[idxenv[:ncore]]>2-occ_cutoff).sum()
        nvirt_guess = (occb[idxenv[ncore:]]<  occ_cutoff).sum()
    else: # truncated impurity
        occ = numpy.hstack((occi, occb))
        occidx = numpy.argsort(occ)[::-1]
        occ = occ[occidx]
        mo = numpy.hstack((numpy.dot(corth[:,baslst],ui),
                           numpy.dot(corth[:,notimp],ub)))[:,occidx]
        ncore_guess = (occ>2-occ_cutoff).sum()
        nvirt_guess = (occ<occ_cutoff).sum()

    casort = numpy.argsort(occ[ncore:nocc])[::-1] + ncore
    #mo = numpy.hstack((mo[:,:ncore], mo[:,casort], mo[:,nocc:]))
    #log.debug('active occs %s', occ[casort])
    log.debug('active occs %s  sum %s', occ[ncore:nocc], occ[ncore:nocc].sum())
    if abs(2-occ[ncore-1]) > occ_cutoff:
        log.info('Approx core space, core occ %g < 2', occ[ncore-1])
    if abs(occ[nocc]) > occ_cutoff:
        log.info('Truncate external space, occ %g > 0', occ[nocc])

    ncas_guess = nao - ncore_guess - nvirt_guess
    nelecas_guess = (ncore-ncore_guess+nelecas[0],
                     ncore-ncore_guess+nelecas[1])
    return mo, ncas_guess, nelecas_guess, ncore_guess


def dmet_decompose(casscf, mo_coeff, aolst, occ_cutoff=1e-8, base=1,
                   orth_method='meta_lowdin'):
#    if casscf.mol.symmetry:
#        raise RuntimeError('dmet_cas breaks spatial symmetry!')
    log = logger.Logger(casscf.stdout, casscf.verbose)
    mo, ncas_guess, nelecas_guess, ncore_guess = \
            _decompose_orbital(casscf.mol, mo_coeff, casscf.ncore, casscf.nelecas,
                               aolst, occ_cutoff=occ_cutoff, base=base,
                               orth_method=orth_method, verbose=log)
    mocore = mo[:,:casscf.ncore]
    mocas = mo[:,casscf.ncore:casscf.ncore+casscf.ncas]
    movir = mo[:,casscf.ncore+casscf.ncas:]
    sc = numpy.dot(casscf._scf.get_ovlp(), mo_coeff)
    fock = reduce(numpy.dot, (sc*casscf._scf.mo_energy, sc.T))
    mo = []
    for c in (mocore, mocas, movir):
        f1 = reduce(numpy.dot, (c.T, fock, c))
        e, u = scipy.linalg.eigh(f1)
        mo.append(numpy.dot(c, u))
    mo = numpy.hstack(mo)

    if ncore_guess != casscf.ncore:
        log.warn('ncas_guess %d    != casscf.ncas %d', ncas_guess, casscf.ncas)
        log.warn('nelecas_guess %s != casscf.nelecas %s',
                 str(nelecas_guess), str(casscf.nelecas))
        log.warn('ncore_guess %d   != casscf.ncore %d', ncore_guess, casscf.ncore)
    return mo

def _decompose_orbital(mol, mo, ncore, nelec, baslst, orth_method='meta_lowdin',
                       occ_cutoff=1e-8, base=1, verbose=logger.WARN):
    from pyscf import lo
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(mol.stdout, verbose)
    if base != 0:
        baslst = [i-base for i in baslst]
    log.debug('*** decompose occupied orbitals')
    log.debug('doubly occupied mo shape = (%d, %d)', mo.shape[0], ncore+nelec[1])
    log.debug('orth AO method = %s', orth_method)
    log.debug('embedding AO list = %s', str(baslst))
    if orth_method is not None:
        corth = lo.orth.orth_ao(mol, method=orth_method)
        morth = scipy.linalg.solve(corth, mo)
    else:
        corth = 1
        morth = mo

    noccb = ncore + nelec[1]
    u, w1, pre_env_h = numpy.linalg.svd(morth[baslst,:noccb])
    w1 = w1[w1>occ_cutoff]
    nw = len(w1)
    sorted_w = numpy.argsort(abs(w1**2-.5)) # most important bath comes frist
    mo1 = numpy.dot(morth[:,:noccb], pre_env_h.T)
    mo1[:,:nw] = mo1[:,:nw][:,sorted_w]
    mo1 = mo1[:,::-1]
    log.debug('entanglement weight occs^2 = %s', str(w1[sorted_w]**2))

    nocca = ncore + nelec[0]
    u, w2, vh = numpy.linalg.svd(morth[baslst,nocca:])
    w2 = w2[w2>occ_cutoff]
    sorted_w = numpy.argsort(abs(w2-.5)) # most important bath comes frist
    mo2 = numpy.dot(morth[:,nocca:], vh.T)
    log.debug('virtual weight = %s', str(w2[sorted_w]**2))

    ncas = len(w1) + len(w2) + nelec[0] - nelec[1]
    ncore1 = noccb - len(w1)
    nelecas =(nelec[0] + ncore - ncore1, nelec[1] + ncore - ncore1)
    return (numpy.dot(corth, numpy.hstack((mo1,morth[:,noccb:nocca],mo2))),
            ncas, nelecas, ncore1)

if __name__ == '__main__':
    from pyscf import scf
    from pyscf import gto
    from pyscf import mcscf
    from pyscf import symm

    b = 1.4
    mol = gto.M(
    verbose = 0,
    atom = [
        ['N',(  0.000000,  0.000000, -b/2)],
        ['N',(  0.000000,  0.000000,  b/2)], ],
    basis = {'N': 'ccpvdz', },
    symmetry= 1
    )

    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 6, 4)
    aolst = [4,5,18,19]
    #_decompose_orbital(mol, m.mo_coeff, 7, aolst, verbose=5)
    #mo = dmet_decompose(mc, m.mo_coeff, aolst)
    mo = dmet_cas(mc, m.make_rdm1(), aolst, verbose=5)
    print(symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mo))
    emc = mc.kernel(mo)[0]
    print(emc, emc - -108.922564421274)

    b = 1.4
    mol = gto.M(
    verbose = 0,
    atom = [
        ['O',(  0.000000,  0.000000, -b/2)],
        ['O',(  0.000000,  0.000000,  b/2)], ],
        basis = {'O': 'ccpvdz', },
        spin = 2,
    )

    m = scf.RHF(mol)
    m.scf()

    mc = mcscf.CASSCF(m, 8, 6)
    aolst = [4,5,18,19]
    # _decompose_orbital(mol, m.mo_coeff, 7, aolst, verbose=5)
    #mo = dmet_decompose(mc, m.mo_coeff, aolst)
    dm = m.make_rdm1()
    mo = dmet_cas(mc, dm[0]+dm[1], aolst)
    emc = mc.kernel(mo)[0]
    print(emc, emc - -149.65498663424643)
