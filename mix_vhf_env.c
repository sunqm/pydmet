/*
 *
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include "cint.h"
#include "cvhf.h"
#include "fblas.h"

#define LOWERTRI_INDEX(I,J)     ((I) > (J) ? ((I)*((I)+1)/2+(J)) : ((J)*((J)+1)/2+(I)))
#define MAX(I,J)        ((I) > (J) ? (I) : (J))


void transform_mixvj(int nset, double *vj, double *tri_dm,
                     int ish_e2, int jsh_e2, int *ao_loc_e2, int *idx_tri_e1,
                     const int nbas_e1, const int *atm, const int natm,
                     const int *bas, const int nbas, const double *env,
                     CINTOpt *opt, CVHFOpt *vhfopt)
{
        const int *bas_e1 = bas;
        const int *bas_e2 = bas + nbas_e1*BAS_SLOTS;
        const int nao_e1 = CINTtot_cgto_spheric(bas_e1, nbas_e1);
        const int di = CINTcgto_spheric(ish_e2, bas_e2);
        const int dj = CINTcgto_spheric(jsh_e2, bas_e2);
        double *eribuf = (double *)malloc(sizeof(double)*di*dj*nao_e1*nao_e1);

        // shift nbas_e1 for ish_e2 and jsh_e2, then the global CINTOpt
        // can work correctly
        if (CVHFfill_nr_eri_o2(eribuf, ish_e2+nbas_e1, jsh_e2+nbas_e1, nbas_e1,
                               atm, natm, bas, nbas, env, opt, vhfopt)) {

                const int INC1;
                const int nbas_e2 = nbas - nbas_e1;
                const int nao_e2 = ao_loc_e2[nbas_e2-1]
                                 + CINTcgto_spheric(nbas_e2-1,bas_e2);
                const int npair_e1 = nao_e1*(nao_e1+1)/2;
                double *eri1 = malloc(sizeof(double)*nao_e1*nao_e1);
                int i, j, i0, j0, kl, ij0, ij1;
                int off, iset;
                double v;

                for (i0 = ao_loc_e2[ish_e2], i = 0; i < di; i++, i0++) {
                for (j0 = ao_loc_e2[jsh_e2], j = 0; j < dj; j++, j0++) {
                if (i0 >= j0) {
                        ij1 = j * di + i;
                        for (kl = 0; kl < npair_e1; kl++) {
                                off = idx_tri_e1[kl]*di*dj;
                                eri1[kl] = eribuf[off+ij1];
                        }
                        ij0 = i0 * nao_e2 + j0;
                        ij1 = j0 * nao_e2 + i0;
                        for (iset = 0; iset < nset; iset++) {
                                v = ddot_(&npair_e1, eri1, &INC1, tri_dm,&INC1);
                                vj[nao_e2*nao_e2*iset+ij0] = v;
                                vj[nao_e2*nao_e2*iset+ij1] = v;
                        }
                } } }

                free(eri1);
        }
        free(eribuf);
}

/*
 * mixvhf J_ij = (i2j2|k1l1) * dm_kl
 * nbas = nbas_e1 + nbas_e2
 */
void mixvhf_vj(double *dm, double *vj, const int nset, const int nbas_e1,
               const int *atm, const int natm,
               const int *bas, const int nbas, const double *env)
{
        const int nbas_e2 = nbas - nbas_e1;
        const int *bas_e1 = bas;
        const int *bas_e2 = bas + nbas_e1*BAS_SLOTS;
        const int nao_e1 = CINTtot_cgto_spheric(bas_e1, nbas_e1);
        const int nao_e2 = CINTtot_cgto_spheric(bas_e2, nbas_e2);
        const int npair_e1 = nao_e1*(nao_e1+1)/2;
        double *tri_dm = malloc(sizeof(double)*npair_e1*nset);
        int i, j, i0, j0, ij;
        int *ij2i = malloc(sizeof(int)*nbas_e2*nbas_e2);
        int *idx_tri_e1 = malloc(sizeof(int)*nao_e1*nao_e1);
        int *ao_loc_e1 = malloc(sizeof(int)*nbas_e1);
        int *ao_loc_e2 = malloc(sizeof(int)*nbas_e2);

        for (i = 0; i < nset; i++) {
                CVHFcompress_nr_dm(tri_dm+npair_e1*i, dm+nao_e1*nao_e1*i, nao_e1);
        }

        CVHFset_ij2i(ij2i, nbas_e2);
        CINTshells_spheric_offset(ao_loc_e1, bas_e1, nbas_e1);
        CINTshells_spheric_offset(ao_loc_e2, bas_e2, nbas_e2);
        CVHFindex_blocks2tri(idx_tri_e1, ao_loc_e1, bas_e1, nbas_e1);

        CINTOpt *opt;
        cint2e_optimizer(&opt, atm, natm, bas, nbas, env);
        CVHFOpt *vhfopt;
        CVHFnr_optimizer(&vhfopt, atm, natm, bas, nbas, env);

        const int nao = nao_e1 + nao_e2;
        double *fakedm_foropt = malloc(sizeof(double) * nao*nao*nset);
        memset(fakedm_foropt, 0, sizeof(double)*nao*nao*nset);
        double *pfakedm;
        double *pdm;
        for (i = 0; i < nset; i++) {
                pfakedm = fakedm_foropt + nao*nao*i;
                pdm = fakedm_foropt + nao_e1*nao_e1*i;
                for (i0 = 0; i0 < nao_e1; i0++) {
                for (j0 = 0; j0 < nao_e1; j0++) {
                        pfakedm[i0*nao+j0] = pdm[i0*nao_e1+j0];
                } }
        }
        CVHFset_direct_scf_dm(vhfopt, fakedm_foropt, nset, atm, natm, bas, nbas, env);
        free(fakedm_foropt);

        memset(vj, 0, sizeof(double)*nao_e2*nao_e2*nset);
        for (ij = 0; ij < nbas_e2*(nbas_e2+1)/2; ij++) {
                i = ij2i[ij]; // for e2
                j = ij - (i*(i+1)/2); // for e2
                transform_mixvj(nset, vj, tri_dm, i, j, ao_loc_e2, idx_tri_e1,
                                nbas_e1, atm, natm, bas, nbas, env,
                                opt, vhfopt);
        }

        CVHFdel_optimizer(&vhfopt);
        CINTdel_optimizer(&opt);
        free(ij2i);
        free(idx_tri_e1);
        free(ao_loc_e1);
        free(ao_loc_e2);
        free(tri_dm);
}

/*
 *************************************************
 * return eribuf[l,i,:j:,:k:]
 */
int fill_eri_for_vk(double *eri, int ish, int lsh, int nbas_e1,
                    const int *atm, const int natm,
                    const int *bas, const int nbas, const double *env,
                    CINTOpt *opt, CVHFOpt *vhfopt)
{
        const int di = CINTcgto_spheric(ish, bas);
        const int dl = CINTcgto_spheric(lsh, bas);
        int jsh, ksh, dj, dk;
        int shls[4];
        double *buf;
        int empty = 1;
        for (jsh = 0; jsh < nbas_e1; jsh++) {
        for (ksh = 0; ksh < nbas_e1; ksh++) {
                dj = CINTcgto_spheric(jsh, bas);
                dk = CINTcgto_spheric(ksh, bas);
                shls[0] = ish;
                shls[1] = jsh;
                shls[2] = ksh;
                shls[3] = lsh;
                if (!vhfopt ||
                    (*vhfopt->fprescreen)(shls, vhfopt)) {
                        buf = (double *)malloc(sizeof(double)*di*dj*dk*dl);
                        empty = !cint2e_sph(buf, shls, atm, natm, bas, nbas, env, opt)
                                && empty;
                        if (!empty) {
                                CINTdmat_transpose(eri, buf, di*dj*dk, dl);
                        }
                        free(buf);
                } else {
                        memset(eri, 0, sizeof(double)*di*dj*dk*dl);
                }
                eri += di*dj*dk*dl;
        } }
        return !empty;
}

/*
 * the output of integrals are blocks, ie [blk[sh1,sh1], blk[sh1,sh2], ...]
 * format dm to match the block form, then we can use dgemv to get K
 */
void format_dm_for_vk(double *fmtdm, double *dm,
                      const int *bas_e1, const int nbas_e1)
{
        const int nao = CINTtot_cgto_spheric(bas_e1, nbas_e1);
        int *ao_loc = malloc(sizeof(int)*(nbas_e1+1));
        CINTshells_spheric_offset(ao_loc, bas_e1, nbas_e1);
        ao_loc[nbas_e1] = nao;

        int jsh, ksh, j0, k0, dj, dk, jk;
        jk = 0;
        for (jsh = 0; jsh < nbas_e1; jsh++) {
        for (ksh = 0; ksh < nbas_e1; ksh++) {
                dj = ao_loc[jsh+1] - ao_loc[jsh];
                dk = ao_loc[ksh+1] - ao_loc[ksh];
                for (k0 = ao_loc[ksh]; k0 < ao_loc[ksh]+dk; k0++) {
                for (j0 = ao_loc[jsh]; j0 < ao_loc[jsh]+dj; j0++) {
                        fmtdm[jk] = dm[j0*nao+k0];
                        jk++;
                } }
        } }
        free(ao_loc);
}

void transform_mixvk(int nset, double *vk, double *fmtdm,
                     int ish_e2, int lsh_e2, int *ao_loc_e2,
                     const int nbas_e1, const int *atm, const int natm,
                     const int *bas, const int nbas, const double *env,
                     CINTOpt *opt, CVHFOpt *vhfopt)
{
        const int *bas_e1 = bas;
        const int *bas_e2 = bas + nbas_e1*BAS_SLOTS;
        const int nao_e1 = CINTtot_cgto_spheric(bas_e1, nbas_e1);
        const int di = CINTcgto_spheric(ish_e2, bas_e2);
        const int dl = CINTcgto_spheric(lsh_e2, bas_e2);
        double *eribuf = malloc(sizeof(double)*di*dl*nao_e1*nao_e1);

        // shift nbas_e1 for ish_e2 and lsh_e2, then the global CINTOpt
        // can work correctly
        if (fill_eri_for_vk(eribuf, ish_e2+nbas_e1, lsh_e2+nbas_e1, nbas_e1,
                            atm, natm, bas, nbas, env, opt, vhfopt)) {

                const int INC1;
                const double D0 = 0;
                const double D1 = 1;
                const char TRANS_N = 'N';
                const int nbas_e2 = nbas - nbas_e1;
                const int nao_e2 = ao_loc_e2[nbas_e2-1]
                                 + CINTcgto_spheric(nbas_e2-1,bas_e2);
                const int nil = di * dl;
                const int nao2 = nao_e1 * nao_e1;
                double *libuf = malloc(sizeof(double) * di * dl);
                int i, l, i0, l0, iset;

                for (iset = 0; iset < nset; iset++) {
                        dgemv_(&TRANS_N, &nil, &nao2, &D1, eribuf, &nil,
                               fmtdm+iset*nao2, &INC1, &D0, libuf, &INC1);
                        for (i0 = ao_loc_e2[ish_e2], i = 0; i < di; i0++, i++) {
                        for (l0 = ao_loc_e2[lsh_e2], l = 0; l < dl; l0++, l++) {
                                vk[i0*nao_e2+l0] = libuf[i*dl+l];
                                vk[l0*nao_e2+i0] = libuf[i*dl+l];
                        } }
                        vk += nao_e2*nao_e2;
                }
                free(libuf);
        }
        free(eribuf);
}


/*
 * mixvhf K_il = (i2j1|k1l2) * dm_jk
 */
void mixvhf_vk(double *dm, double *vk, const int nset, const int nbas_e1,
               const int *atm, const int natm,
               const int *bas, const int nbas, const double *env)
{
        const int nbas_e2 = nbas - nbas_e1;
        const int *bas_e1 = bas;
        const int *bas_e2 = bas + nbas_e1*BAS_SLOTS;
        const int nao_e1 = CINTtot_cgto_spheric(bas_e1, nbas_e1);
        const int nao_e2 = CINTtot_cgto_spheric(bas_e2, nbas_e2);
        double *dmk = malloc(sizeof(double)*nao_e1*nao_e1*nset);
        int *ao_loc_e2 = malloc(sizeof(int)*nbas_e2);
        int i, l, i0, j0, off;

        for (i = 0; i < nset; i++) {
                off = nao_e1*nao_e1*i;
                format_dm_for_vk(dmk+off, dm+off, bas_e1, nbas_e1);
        }

        CINTshells_spheric_offset(ao_loc_e2, bas_e2, nbas_e2);

        CINTOpt *opt;
        cint2e_optimizer(&opt, atm, natm, bas, nbas, env);
        CVHFOpt *vhfopt;
        CVHFnr_optimizer(&vhfopt, atm, natm, bas, nbas, env);
        const int nao = nao_e1 + nao_e2;
        double *fakedm_foropt = malloc(sizeof(double) * nao*nao*nset);
        memset(fakedm_foropt, 0, sizeof(double)*nao*nao*nset);
        double *pfakedm;
        double *pdm;
        for (i = 0; i < nset; i++) {
                pfakedm = fakedm_foropt + nao*nao*i;
                pdm = fakedm_foropt + nao_e1*nao_e1*i;
                for (i0 = 0; i0 < nao_e1; i0++) {
                for (j0 = 0; j0 < nao_e1; j0++) {
                        pfakedm[i0*nao+j0] = pdm[i0*nao_e1+j0];
                } }
        }
        CVHFset_direct_scf_dm(vhfopt, fakedm_foropt, nset, atm, natm, bas, nbas, env);
        free(fakedm_foropt);

        memset(vk, 0, sizeof(double)*nao_e2*nao_e2*nset);
        for (i = 0; i < nbas_e2; i++) {
        for (l = 0; l < nbas_e2; l++) {
                transform_mixvk(nset, vk, dmk, i, l, ao_loc_e2,
                                nbas_e1, atm, natm, bas, nbas, env,
                                opt, vhfopt);
        } }

        CVHFdel_optimizer(&vhfopt);
        CINTdel_optimizer(&opt);
        free(ao_loc_e2);
        free(dmk);
}

