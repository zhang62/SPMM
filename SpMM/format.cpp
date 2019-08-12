#include "spmm.h"

/**
 * @brief convert coo to csr
 * @param[in] cooidx Specify if 0 or 1 indexed
 * @param[in] coo COO matrix
 * @param[out] csr CSR matrix
 */
/*--------------------------------------------------*/
int COO2CSR(int cooidx, struct coo_t *coo, struct csr_t *csr) {
   int i, nrows, ncols, nnz;
   nrows = csr->nrow = coo->nrow;
   ncols = csr->ncol = coo->ncol;
   nnz   = csr->nnz  = coo->nnz;
   csr->ia = (int *) malloc((nrows+1)*sizeof(int));
   csr->ja = (int *) malloc(nnz*sizeof(int));
   csr->a = (REAL *) malloc(nnz*sizeof(REAL));
   /* fill (ia, ja, a) */
   for (i=0; i<nrows+1; i++) {
      csr->ia[i] = 0;
   }
   for (i=0; i<nnz; i++) {
      int row = coo->ir[i] - cooidx;
      csr->ia[row+1] ++;
   }
   for (i=0; i<nrows; i++) {
      csr->ia[i+1] += csr->ia[i];
   }
   for (i=0; i<nnz; i++) {
      int row = coo->ir[i] - cooidx;
      int col = coo->jc[i] - cooidx;
      assert(col >= 0 && col < ncols);
      REAL val = coo->val[i];
      int k = csr->ia[row];
      csr->a[k] = val;
      csr->ja[k] = col;
      csr->ia[row]++;
   }
   for (i=nrows; i>0; i--) {
      csr->ia[i] = csr->ia[i-1];
   }
   csr->ia[0] = 0;
   /* sort rows ? */
   //sortrow(csr);
   return 0;
}

