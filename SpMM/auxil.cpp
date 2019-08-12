#include "spmm.h"

double wall_timer() {
  struct timeval tim;
  gettimeofday(&tim, NULL);
  double t = tim.tv_sec + tim.tv_usec/1e6;
  return(t);
}

/*---------------------------------------------*/
void print_header() {
  if (DOUBLEPRECISION)
    printf("\nTesting SpMM, DOUBLE precision\n");
  else
    printf("\nTesting SpMM, SINGLE precision\n");
}

/*-----------------------------------------*/
double error_norm(REAL *x, REAL *y, int n) {
  int i;
  double t, normz, normx;
  normx = normz = 0.0;
  for (i=0; i<n; i++) {
    t = x[i]-y[i];
    normz += t*t;
    normx += x[i]*x[i];
  }
  return (sqrt(normz/normx));
}

/*---------------------------*/
void FreeCOO(struct coo_t *coo)
{
  free(coo->ir);
  free(coo->jc);
  free(coo->val);
}

/*---------------------------*/
void FreeCSR(struct csr_t *csr)
{
  free(csr->a);
  free(csr->ia);
  free(csr->ja);
}

/**
 * @brief convert csr to csc
 * Assume input csr is 0-based index
 * output csc 0/1 index specified by OUTINDEX      *
 * @param[in] OUTINDEX specifies if CSC should be 0/1 index
 * @param[in] nrow Number of rows
 * @param[in] ncol Number of columns
 * @param[in] job flag
 * @param[in] a Values of input matrix
 * @param[in] ia Input row pointers
 * @param[in] ja Input column indices
 * @param[out] ao Output values
 * @param[out] iao Output row pointers
 * @param[out] jao Output column indices
 */
void csrcsc(int OUTINDEX, const int nrow, const int ncol, int job,
            REAL *a, int *ja, int *ia, REAL *ao, int *jao, int *iao) {
  int i,k;
  for (i=0; i<ncol+1; i++) {
    iao[i] = 0;
  }
  // compute nnz of columns of A
  for (i=0; i<nrow; i++) {
    for (k=ia[i]; k<ia[i+1]; k++) {
      iao[ja[k]+1] ++;
    }
  }
  // compute pointers from lengths
  for (i=0; i<ncol; i++) {
    iao[i+1] += iao[i];
  }
  // now do the actual copying
  for (i=0; i<nrow; i++) {
    for (k=ia[i]; k<ia[i+1]; k++) {
      int j = ja[k];
      if (job) {
        ao[iao[j]] = a[k];
      }
      jao[iao[j]++] = i + OUTINDEX;
    }
  }
  /*---- reshift iao and leave */
  for (i=ncol; i>0; i--) {
    iao[i] = iao[i-1] + OUTINDEX;
  }
  iao[0] = OUTINDEX;
}

/**
 * @brief  Sort each row of a csr by increasing column
 * order
 * By Double transposition
 * @param[in] A Matrix to sort
 */
void sortrow(csr_t *A) {
  /*-------------------------------------------*/
  int nrows = A->nrow;
  int ncols = A->ncol;
  int nnz = A->ia[nrows];
  // work array
  REAL *b;
  int *jb, *ib;
  b = (REAL *) malloc(nnz*sizeof(REAL));
  jb = (int *) malloc(nnz*sizeof(int));
  ib = (int *) malloc((ncols+1)*sizeof(int));
  // Double transposition
  csrcsc(0, nrows, ncols, 1, A->a, A->ja, A->ia, b, jb, ib);
  csrcsc(0, ncols, nrows, 1, b, jb, ib, A->a, A->ja, A->ia);
  // free
  free(b);
  free(jb);
  free(ib);
}

void csr_spmm_symb(csr_t *A, csr_t *B, csr_t *C, int *work)
{
   int i1, i2, i3;
   int m = A->nrow, k = A->ncol, n = B->ncol;

   C->ia = (int *) malloc((m+1)*sizeof(int));
   for (i1 = 0; i1 < m; i1++)
   {
      int MARK=i1+1;
      int count = 0;
      for (i2 = A->ia[i1]; i2 < A->ia[i1+1]; i2++)
      {
         int j = A->ja[i2];
         assert(j >= 0 && j < k);
         for (i3 = B->ia[j]; i3 < B->ia[j+1]; i3++)
         {
            int col = B->ja[i3];
            assert(col >= 0 && col < n);
            if (work[col] != MARK)
            {
               count++;
               work[col] = MARK;
            }
         }
      }
      C->ia[i1+1] = count;
   }
   for (i1=0, C->ia[0]=0; i1 < m; i1++)
   {
      C->ia[i1+1] += C->ia[i1];
   }

   C->nrow = m;
   C->ncol = n;
   C->nnz = C->ia[m];
   C->ja = (int *) malloc(C->nnz*sizeof(int));
   C->a =  (REAL *) malloc(C->nnz*sizeof(REAL));
}

void csr_spmm_nume(csr_t *A, csr_t *B, csr_t *C, int *work)
{
   int i1, i2, i3;
   int m = A->nrow, k = A->ncol, n = B->ncol, pos = 0;

   for (i1 = 0; i1 < m; i1++)
   {
      int ipos=pos;
      for (i2 = A->ia[i1]; i2 < A->ia[i1+1]; i2++)
      {
         int j = A->ja[i2];
         REAL va = A->a[i2];
         assert(j >= 0 && j < k);
         for (i3 = B->ia[j]; i3 < B->ia[j+1]; i3++)
         {
            int q, col = B->ja[i3];
            REAL vb = B->a[i3];
            assert(col >= 0 && col < n);
            if ((q = work[col]) <= ipos)
            {
               C->ja[pos] = col;
               C->a[pos] = va*vb;
               work[col] = ++pos;
            }
            else
            {
               assert(C->ja[q-1] == col);
               C->a[q-1] += va*vb;
            }
         }
      }
      assert(C->ia[i1+1] == pos);
   }
}

void csr_spmm_cpu(csr_t *A, csr_t *B, csr_t *C)
{
   int *work = (int *) calloc(B->ncol, sizeof(int));
   csr_spmm_symb(A, B, C, work);
   memset(work, 0, B->ncol*sizeof(int));
   csr_spmm_nume(A, B, C, work);
   free(work);
}

/**
 * @file dumps.c
 * @brief Miscellaneous functions used for DOS based functions
 */
//-------------------- miscellaneous functions for I/O
//                     and for debugging

/**
 * @brief Saves a matrix in MatrixMarket format
 *
 * @param[in] nrow Number of rows in matrix
 * @param[in] ncol Number of cols in matrix
 * @param[in] ia Row pointers
 * @param[in] ja Column indices
 * @param[in] a Values
 * @param[in] fn filename
 */
void save_mtx_basic(int nrow, int ncol, int *ia,
                    int *ja, REAL *a, const char *fn) {
  int i,j,nnz;
  FILE *fp = fopen(fn, "w");

  nnz = ia[nrow];
  assert(ia[0] == 0);
  fprintf(fp, "%s\n", "%%MatrixMarket matrix coordinate real general");
  fprintf(fp, "%d %d %d\n", nrow, ncol, nnz);
  for (i=0; i<nrow; i++) {
    for (j=ia[i]; j<ia[i+1]; j++) {
      fprintf(fp, "%d %d %.15e\n", i+1, ja[j]+1, a[j]);
    }
  }
  fclose(fp);
}

/**
 * @brief Saves a csr matrix
 * @param[in] A csr matrix to save
 * @param[in] fn filename
 */
void saveCSR(csr_t *A, const char *fn) {
  fprintf(stdout, " * saving a matrix into %s\n", fn);
  save_mtx_basic(A->nrow, A->ncol, A->ia, A->ja, A->a, fn);
}

