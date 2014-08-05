#include <stdlib.h>
#include <string.h>
#include <omp.h>

#if defined SCIPY_MKL_H
typedef long FINT;
#else
typedef int FINT;
#endif

static void dmet_misc_unpack(unsigned int n, double *vec, double *mat)
{
        unsigned int i, j;
        for (i = 0; i < n; i++) {
                for (j = 0; j <= i; j++, vec++) {
                        mat[i*n+j] = *vec;
                        mat[j*n+i] = *vec;
                }
        }
}


void extract_row_from_tri_eri(double *row, unsigned int row_id,
                              double *eri, unsigned int npair)
{
        unsigned long idx;
        unsigned int i;
        idx = (unsigned long)row_id * (row_id + 1) / 2;
        memcpy(row, eri+idx, sizeof(double)*row_id);
        for (i = row_id; i < npair; i++) {
                idx += i;
                row[i] = eri[idx];
        }
}


/* eri uses 8-fold symmetry: i>=j,k>=l,ij>=kl */
static void _mos_half_trans_o3(unsigned int pair_id, int nao, int nembs, int *offsets,
                               double *eriao, double *c, double *erimo)
{
        const double D0 = 0;
        const double D1 = 1;
        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const FINT lao = nao;
        unsigned int i, j, k, nc;
        FINT nmo = offsets[nembs]; // last one in offsets is the dims of c
        unsigned int nao_pair = nao * (nao+1) / 2;
        double *row = malloc(sizeof(double)*nao_pair);
        double *tmp1 = malloc(sizeof(double)*nao*nao);
        double *tmp2 = malloc(sizeof(double)*nao*nmo);

        extract_row_from_tri_eri(row, pair_id, eriao, nao_pair);
        dmet_misc_unpack(nao, row, tmp1);
        dgemm_(&TRANS_N, &TRANS_N, &lao, &nmo, &lao,
               &D1, tmp1, &lao, c, &lao, &D0, tmp2, &lao);

        for (k = 0; k < nembs; k++) {
                nc = offsets[k+1] - offsets[k];
                dgemm_(&TRANS_T, &TRANS_N, &nc, &nc, &lao,
                       &D1, c+nao*offsets[k], &lao,
                       tmp2+nao*offsets[k], &lao, &D0, tmp1, &nc);
                for (i = 0; i < nc; i++) {
                        for (j = 0; j <= i; j++, erimo++) {
                                *erimo = tmp1[i*nc+j];
                        }
                }
        }

        free(row);
        free(tmp1);
        free(tmp2);
}
void dmet_misc_embs_half_trans_o3(int nao, int nembs, int *offsets,
                                  double *eriao, double *c, double *erimo)
{
        const unsigned int nao_pair = nao*(nao+1)/2;
        unsigned int stride = 0;
        unsigned int ij, k, n;
        for (k = 0; k < nembs; k++) {
                n = offsets[k+1] - offsets[k];
                stride += n*(n+1)/2;
        }
#pragma omp parallel default(none) \
        shared(nao, nembs, offsets, eriao, c, erimo, stride) \
        private(ij)
#pragma omp for nowait schedule(guided, 20)
        for (ij = 0; ij < nao_pair; ij++) {
                _mos_half_trans_o3(ij, nao, nembs, offsets,
                                   eriao, c, erimo+ij*stride);
        }
}

