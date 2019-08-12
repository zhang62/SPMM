void cuda_init(int argc, char **argv);
void cuda_check_err();
void print_header();
int read_coo_MM(const char *matfile, int idxin, int idxout, coo_t *coo);
int COO2CSR(int cooidx, struct coo_t *coo, struct csr_t *csr);
double wall_timer();
double error_norm(REAL *x, REAL *y, int n);
void PadJAD32(struct jad_t *jad);
void FreeCOO(struct coo_t *coo);
void FreeCSR(struct csr_t *csr);
void FreeJAD(struct jad_t *jad);
void FreeDIA(struct dia_t *dia);

int findarg(const char *argname, ARG_TYPE type, void *val, int argc, char **argv);
int lapgen(int nx, int ny, int nz, struct coo_t *Acoo, int npts);
void csr_spmm_cpu(csr_t *A, csr_t *B, csr_t *C);
void csr_spmm(csr_t *A, csr_t *B, csr_t *C);
void saveCSR(csr_t *A, const char *fn);

