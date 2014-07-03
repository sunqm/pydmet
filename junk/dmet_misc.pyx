#cython: boundscheck=False
#cython: wraparound=False
#cython: overflowcheck.fold=False
import numpy
cimport numpy
cimport cython

cdef extern void ao2mo_half_trans_o2(int nao, int nmo, double *eri, double *c,
                                     double *mat)
cdef extern void ao2mo_half_trans_o3(int nao, int nmo, int pair_id,
                                     double *eri, double *c, double *mat)
cdef extern void dmet_misc_embs_half_trans_o3(int nao, int nembs, int *offsets,
                                              double *eriao, double *c, double *erimo)

def embs_eri_ao2mo_o3(numpy.ndarray[double,ndim=1] eri_ao,
                      numpy.ndarray[double,ndim=2,mode='fortran'] c,
                      numpy.ndarray[int   ,ndim=1] offsets):
    cdef int nao = c.shape[0]
    cdef int nao_pair = nao*(nao+1)/2
    cdef int nembs = offsets.__len__() - 1
    cdef int n, m
    npair_sum = 0
    for k in range(nembs):
        n = offsets[k+1] - offsets[k]
        npair_sum += n*(n+1)/2
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri1 = numpy.empty((nao_pair,npair_sum))
    dmet_misc_embs_half_trans_o3(nao, nembs, &offsets[0],
                                 &eri_ao[0], &c[0,0], &eri1[0,0])

    eri1 = numpy.array(eri1.T, order='C')
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri2
    embs_eri = []
    cdef int off = 0
    cdef int ij
    for k in range(nembs):
        n = offsets[k+1] - offsets[k]
        m = offsets[k]
        npair = n*(n+1)/2
        eri2 = numpy.empty((npair,npair))
        for ij in range(npair):
            ao2mo_half_trans_o2(nao, n, &eri1[off+ij,0], &c[0,m], &eri2[ij,0])
        off += npair
        embs_eri.append(eri2)
    return embs_eri


def u_embs_eri_ao2mo_o3(numpy.ndarray[double,ndim=1] eri_ao,
                        numpy.ndarray[double,ndim=2,mode='fortran'] c_ab,
                        numpy.ndarray[int   ,ndim=1] offsets):
    cdef int nao = c_ab.shape[0]
    cdef int nao_pair = nao*(nao+1)/2
    cdef int nembs = offsets.__len__() - 1
    npair_sum = 0
    for k in range(nembs):
        n = offsets[k+1] - offsets[k]
        npair_sum += n*(n+1)/2
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri1 = numpy.empty((nao_pair,npair_sum))
    dmet_misc_embs_half_trans_o3(nao, nembs, &offsets[0],
                                 &eri_ao[0], &c_ab[0,0], &eri1[0,0])

    eri1 = numpy.array(eri1.T, order='C')
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri_aa
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri_ab
    cdef numpy.ndarray[double,ndim=2,mode='c'] eri_bb
    embs_eri = []
    cdef int na, nb, ma, mb
    cdef int ij
    cdef int off = 0
    for k in range(0, nembs, 2):
        na = offsets[k+1] - offsets[k]
        ma = offsets[k]
        nb = offsets[k+2] - offsets[k+1]
        mb = offsets[k+1]
        npair_a = na*(na+1)/2
        npair_b = nb*(nb+1)/2
        eri_aa = numpy.empty((npair_a,npair_a))
        eri_ab = numpy.empty((npair_b,npair_a)) # integral (AA|BB) in Fortran continues
        eri_bb = numpy.empty((npair_b,npair_b))

        for ij in range(npair_a):
            ao2mo_half_trans_o2(nao, na, &eri1[off+ij,0], &c_ab[0,ma], &eri_aa[ij,0])
        off += npair_a

        for ij in range(npair_b):
            ao2mo_half_trans_o2(nao, nb, &eri1[off+ij,0], &c_ab[0,mb], &eri_bb[ij,0])
            ao2mo_half_trans_o2(nao, na, &eri1[off+ij,0], &c_ab[0,ma], &eri_ab[ij,0])
        off += npair_b
        embs_eri.append(numpy.array((eri_aa,eri_bb,eri_ab)))
    return embs_eri


##############################

cdef extern void mixvhf_vj(double *dm, double *vj, int nset, int nbas_e1,
                           int *atm, int natm, int *bas, int nbas, double *env)
cdef extern void mixvhf_vk(double *dm, double *vk, int nset, int nbas_e1,
                           int *atm, int natm, int *bas, int nbas, double *env)

def mix_vhf(numpy.ndarray dm, nbas_e1, atm, bas, env):
    '''vhf_e2 = g_{e1,e2} * dm_e1'''
    cdef numpy.ndarray[int,ndim=2] c_atm = numpy.array(atm, dtype=numpy.int32)
    cdef numpy.ndarray[int,ndim=2] c_bas = numpy.array(bas, dtype=numpy.int32)
    cdef numpy.ndarray[double] c_env = numpy.array(env)
    cdef int natm = c_atm.shape[0]
    cdef int nbas = c_bas.shape[0]
    if dm.ndim == 2:
        nset = 1
        dm_shape = (dm.shape[0], dm.shape[1])
    else:
        nset = dm.shape[0]
        dm_shape = (dm.shape[0], dm.shape[1], dm.shape[2])

    cdef numpy.ndarray vj = numpy.empty(dm_shape)
    cdef numpy.ndarray vk = numpy.empty(dm_shape)

    mixvhf_vj(<double *>dm.data, <double *>vj.data, nset, nbas_e1,
                     &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    mixvhf_vk(<double *>dm.data, <double *>vk.data, nset, nbas_e1,
                     &c_atm[0,0], natm, &c_bas[0,0], nbas, &c_env[0])
    return vj, vk
