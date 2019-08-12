#include "spmm.h"

/*-----------------------------------------*/
int main (int argc, char **argv)
{
   /*-----------------------------------------*
    *           C = A * B
    *     CPU CSR        kernel
    *     GPU CSR        kernel
    *-----------------------------------------*/
   int nxA=64, nyA=64, nzA=1, nptsA=5, nxB=32, nyB=16, nzB=8, nptsB=7;
   int mmA=1, mmB=1;
   char fnameA[2048], fnameB[2048];
   struct coo_t cooA, cooB;
   struct csr_t csrA, csrB, csrC;
   //double e1, e2, e3, e4;
   double t1, t2;
   /*-----------------------------------------*/
   if (findarg("help", NA, NULL, argc, argv))
   {
      printf("Usage: ./spmm?.ex -nxA [int] -nyA [int] -nzA [int] -nptsA [int] "
            "-matA [fnameA] -mmA [int]\n"
            "                  -nxB [int] -nyB [int] -nzB [int] -nptsB [int] "
            "-matB [fnameB] -mmB [int]\n");
      return 0;
   }
   srand (SEED);
   /*------------ Init GPU */
   //cuda_init(argc, argv);
   /*------------ output header */
   print_header();
   /*------------ cmd-line input */
   findarg("nxA",   INT, &nxA,   argc, argv);
   findarg("nyA",   INT, &nyA,   argc, argv);
   findarg("nzA",   INT, &nzA,   argc, argv);
   findarg("nptsA", INT, &nptsA, argc, argv);
   findarg("mmA",   INT, &mmA,   argc, argv);
   findarg("nxB",   INT, &nxB,   argc, argv);
   findarg("nyB",   INT, &nyB,   argc, argv);
   findarg("nzB",   INT, &nzB,   argc, argv);
   findarg("nptsB", INT, &nptsB, argc, argv);
   findarg("mmB",   INT, &mmB,   argc, argv);
   /*---------- Read from Martrix Market file */
   if (findarg("matA", STR, fnameA, argc, argv) == 1)
   {
      read_coo_MM(fnameA, mmA, 1, &cooA);
   }
   else
   {
      lapgen(nxA, nyA, nzA, &cooA, nptsA);
   }
   if (findarg("matB", STR, fnameB, argc, argv) == 1)
   {
      read_coo_MM(fnameB, mmB, 1, &cooB);
   }
   else
   {
      lapgen(nxB, nyB, nzB, &cooB, nptsB);
   }
   /*---------- convert to CSR */
   COO2CSR(1, &cooA, &csrA);
   COO2CSR(1, &cooB, &csrB);
   /*---------- make sure the sizes match */
   assert(csrA.ncol == csrB.nrow);
   //int m = csrA.nrow;
   //int k = csrA.ncol;
   //int n = csrB.ncol;

   t1 = wall_timer();
   csr_spmm_cpu(&csrA, &csrB, &csrC);
   t2 = wall_timer();
   printf("===== CPU =====\n"
          "Time %.2e\n", t2-t1
         );

   csr_spmm(&csrA, &csrB, &csrC);
   /*---------- save matrices for debug */
   //saveCSR(&csrA, "A.mtx");
   //saveCSR(&csrB, "B.mtx");
   //saveCSR(&csrC, "C.mtx");
   /*---------- free */
   FreeCOO(&cooA);
   FreeCOO(&cooB);
   FreeCSR(&csrA);
   FreeCSR(&csrB);
   FreeCSR(&csrC);
}

