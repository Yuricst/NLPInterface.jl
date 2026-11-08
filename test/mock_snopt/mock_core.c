/*
 * Mock of the core `libsnopt7` library, for testing NLPInterface without a
 * licensed SNOPT. It provides the Fortran entry points that NLPInterface calls
 * on the `sninit_` fallback path (libraries without `f_sninitx`):
 * `sninit_`, `snfileopenwrite_` and `snfileclose_` (gfortran calling
 * convention: every argument by reference, string lengths appended by value).
 *
 * It also keeps the "current output files" shared with the mock C wrapper
 * (mock_wrapper.c), which links against this library.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MOCK_NUNITS 100

static FILE *unit_files[MOCK_NUNITS];
static FILE *cur_print = NULL;
static FILE *cur_summ = NULL;
static int cur_owned = 0;  /* 1 if cur_print/cur_summ were fopen'ed by f_sninitx */
static int cur_open_failed = 0;
static int cur_initialized = 0;

static FILE *unit_file(int unit)
{
    if (unit == 6) return stdout;
    if (unit <= 0 || unit >= MOCK_NUNITS) return NULL;
    return unit_files[unit];
}

/* --- state shared with mock_wrapper.c --------------------------------- */

void mock_core_set_files(FILE *print, FILE *summ, int owned, int open_failed)
{
    cur_print = print;
    cur_summ = summ;
    cur_owned = owned;
    cur_open_failed = open_failed;
    cur_initialized = 1;
}

FILE *mock_core_print(void) { return cur_print; }
FILE *mock_core_summ(void) { return cur_summ; }
int mock_core_open_failed(void) { return cur_open_failed; }
int mock_core_initialized(void) { return cur_initialized; }

void mock_core_end(void)
{
    if (cur_owned) {
        if (cur_print && cur_print != stdout) fclose(cur_print);
        if (cur_summ && cur_summ != stdout) fclose(cur_summ);
    } else {
        if (cur_print) fflush(cur_print);
        if (cur_summ) fflush(cur_summ);
    }
    cur_print = cur_summ = NULL;
    cur_owned = 0;
    cur_open_failed = 0;
    cur_initialized = 0;
}

/* --- Fortran entry points ---------------------------------------------- */

void snfileopenwrite_(int *iunit, const char *name, int *inform, size_t len)
{
    char buf[4096];
    size_t n = len < sizeof(buf) - 1 ? len : sizeof(buf) - 1;
    memcpy(buf, name, n);
    buf[n] = '\0';
    while (n > 0 && buf[n - 1] == ' ') buf[--n] = '\0';  /* Fortran trims trailing blanks */
    if (*iunit <= 0 || *iunit >= MOCK_NUNITS || *iunit == 6 || unit_files[*iunit] != NULL) {
        *inform = 1;
        return;
    }
    FILE *f = fopen(buf, "w");
    if (f == NULL) {
        *inform = 2;
        return;
    }
    unit_files[*iunit] = f;
    *inform = 0;
}

void snfileclose_(int *iunit)
{
    if (*iunit > 0 && *iunit < MOCK_NUNITS && unit_files[*iunit] != NULL) {
        fclose(unit_files[*iunit]);
        unit_files[*iunit] = NULL;
    }
}

void sninit_(int *iprint, int *isumm, char *cw, int *lencw,
             int *iw, int *leniw, double *rw, int *lenrw)
{
    (void)cw; (void)lencw; (void)iw; (void)leniw; (void)rw; (void)lenrw;
    FILE *p = unit_file(*iprint);
    FILE *s = unit_file(*isumm);
    if (p) fprintf(p, "MOCK SNOPT print file (sninit_, unit %d)\n", *iprint);
    if (s) fprintf(s, "MOCK SNOPT summary file (sninit_, unit %d)\n", *isumm);
    mock_core_set_files(p, s, 0, 0);
}
