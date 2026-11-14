/*
 * Mock of the SNOPT C wrapper library (`libsnopt7_cpp` / `libsnopt7_c`), for
 * testing NLPInterface's ccall plumbing without a licensed SNOPT. Signatures
 * follow SNOPT 7.7's snopt.h.
 *
 * f_snopta does not implement SNOPT. It runs projected gradient descent on the
 * objective row (the constraints are evaluated but ignored), which solves the
 * box-constrained convex test problems, and follows SNOPT's user-function
 * protocol: Status = 1 on the first call, 2 on the last one, Status = -1 from
 * the user means "undefined here" (shorten the step), Status <= -2 means stop.
 *
 * Built twice: with -DMOCK_HAS_SNINITX it exports f_sninitx (the normal path);
 * without it, it exports only f_sninit, so NLPInterface uses the `sninit_`
 * fallback in the mock core library.
 *
 * Behaviour switches (environment variables, read when used):
 *   MOCK_SNOPT_MEMINFO  INFO returned by f_snmema (default 104 = success)
 *   MOCK_SNOPT_INFORM   override the INFO returned by f_snopta
 *   MOCK_SNOPT_PERTURB  if nonzero, evaluate F at a perturbed point after the
 *                       last iterate and return that F (as SNOPT's derivative
 *                       check can), so F[1] does not belong to x
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

typedef void (*snFunA)(int *Status, int *n, double x[], int *needF, int *neF, double F[],
                       int *needG, int *neG, double G[], char cu[], int *lencu,
                       int iu[], int *leniu, double ru[], int *lenru);

/* from mock_core.c */
void mock_core_set_files(FILE *print, FILE *summ, int owned, int open_failed);
FILE *mock_core_print(void);
FILE *mock_core_summ(void);
int mock_core_open_failed(void);
int mock_core_initialized(void);
void mock_core_end(void);

static int opt_major_iterations_limit = 1000;
static int opt_total_iw = 0;
static int opt_total_rw = 0;

static int env_int(const char *name, int dflt)
{
#ifdef _WIN32
    /* Julia's `withenv` uses SetEnvironmentVariableW, which the C runtime's
     * getenv (a copy of the environment taken at startup) does not see. */
    char buf[64];
    DWORD n = GetEnvironmentVariableA(name, buf, sizeof(buf));
    return (n > 0 && n < sizeof(buf)) ? atoi(buf) : dflt;
#else
    const char *v = getenv(name);
    return (v && *v) ? atoi(v) : dflt;
#endif
}

/* Copy a (pointer, length) string into buf without trailing blanks. */
static void trimmed(char *buf, size_t size, const char *s, int len)
{
    size_t n = (len > 0 && s) ? (size_t)len : 0;
    if (n > size - 1) n = size - 1;
    if (n) memcpy(buf, s, n);
    buf[n] = '\0';
    while (n > 0 && buf[n - 1] == ' ') buf[--n] = '\0';
}

static void log_line(const char *fmt, const char *a, double v)
{
    FILE *p = mock_core_print();
    if (p) { fprintf(p, fmt, a, v); fputc('\n', p); }
}

static void reset_options(void)
{
    opt_major_iterations_limit = 1000;
    opt_total_iw = 0;
    opt_total_rw = 0;
}

#ifdef MOCK_HAS_SNINITX
void f_sninitx(const char *print, int plen, const char *summ, int slen,
               int iw[], int leniw, double rw[], int lenrw)
{
    (void)iw; (void)leniw; (void)rw; (void)lenrw;
    char pbuf[4096], sbuf[4096];
    trimmed(pbuf, sizeof(pbuf), print, plen);
    trimmed(sbuf, sizeof(sbuf), summ, slen);
    FILE *p = NULL, *s = NULL;
    int failed = 0;
    if (*pbuf) {
        p = strcmp(pbuf, "screen") == 0 ? stdout : fopen(pbuf, "w");
        if (!p) failed = 1;
    }
    if (*sbuf) {
        s = strcmp(sbuf, "screen") == 0 ? stdout : fopen(sbuf, "w");
        if (!s) failed = 1;
    }
    if (p) fprintf(p, "MOCK SNOPT print file (f_sninitx)\n");
    if (s) fprintf(s, "MOCK SNOPT summary file (f_sninitx)\n");
    mock_core_set_files(p, s, 1, failed);
    reset_options();
}
#else
/* Present so that NLPInterface accepts the library; it calls sninit_ instead. */
void f_sninit(const char *print, int plen, int iprint, const char *summ, int slen, int isumm,
              int iw[], int leniw, double rw[], int lenrw)
{
    (void)print; (void)plen; (void)iprint; (void)summ; (void)slen; (void)isumm;
    (void)iw; (void)leniw; (void)rw; (void)lenrw;
    abort();  /* NLPInterface must not call the (possibly broken) f_sninit */
}
#endif

static int starts_with(const char *s, const char *prefix)
{
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

/* Unknown keywords are accepted, except those starting with "Bogus". */
void f_snset(const char *buffer, int len, int *errors,
             int iw[], int leniw, double rw[], int lenrw)
{
    (void)iw; (void)leniw; (void)rw; (void)lenrw;
    char buf[1024];
    trimmed(buf, sizeof(buf), buffer, len);
    *errors = starts_with(buf, "Bogus") ? 1 : 0;
    log_line("option: %s%.0f", buf, 0.0);
}

void f_snseti(const char *buffer, int len, int iopt, int *errors,
              int iw[], int leniw, double rw[], int lenrw)
{
    (void)iw; (void)leniw; (void)rw; (void)lenrw;
    char buf[1024];
    trimmed(buf, sizeof(buf), buffer, len);
    *errors = starts_with(buf, "Bogus") ? 1 : 0;
    if (strcmp(buf, "Major iterations limit") == 0) opt_major_iterations_limit = iopt;
    if (strcmp(buf, "Total integer workspace") == 0) opt_total_iw = iopt;
    if (strcmp(buf, "Total real workspace") == 0) opt_total_rw = iopt;
    log_line("option: %s = %.0f", buf, (double)iopt);
}

void f_snsetr(const char *buffer, int len, double rvalue, int *errors,
              int iw[], int leniw, double rw[], int lenrw)
{
    (void)iw; (void)leniw; (void)rw; (void)lenrw;
    char buf[1024];
    trimmed(buf, sizeof(buf), buffer, len);
    *errors = starts_with(buf, "Bogus") ? 1 : 0;
    log_line("option: %s = %g", buf, rvalue);
}

static int mock_miniw(int nf, int n, int neG) { return 1000 + 10 * (nf + n) + neG; }
static int mock_minrw(int nf, int n, int neG) { return 2000 + 20 * (nf + n) + neG; }

void f_snmema(int *info, int nf, int n, int neA, int neG, int *miniw, int *minrw,
              int iw[], int leniw, double rw[], int lenrw)
{
    (void)neA; (void)iw; (void)leniw; (void)rw; (void)lenrw;
    *info = env_int("MOCK_SNOPT_MEMINFO", 104);
    if (*info == 104) {
        *miniw = mock_miniw(nf, n, neG);
        *minrw = mock_minrw(nf, n, neG);
    } else {
        *miniw = -123456;  /* garbage, as the TODO feared */
        *minrw = -123456;
    }
}

void f_snend(int iw[], int leniw, double rw[], int lenrw)
{
    (void)iw; (void)leniw; (void)rw; (void)lenrw;
    mock_core_end();
    reset_options();  /* options belong to one solve, on both init paths */
}

static double clamp(double v, double lo, double hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Evaluate at x; returns the user's Status. */
static int call_user(snFunA usrfun, int status, int n, double *x, int needF, int nf,
                     double *F, int needG, int neG, double *G,
                     int iu[], int leniu, double ru[], int lenru)
{
    char cu[8] = {0};
    int lencu = 1;
    int st = status;
    int nn = n, nF = nf, nG = neG, nF_need = needF, nG_need = needG;
    usrfun(&st, &nn, x, &nF_need, &nF, F, &nG_need, &nG, G, cu, &lencu, iu, &leniu, ru, &lenru);
    return st;
}

void f_snopta(int start, const char *name, int nf, int n, double objadd, int objrow,
              snFunA usrfun, int iAfun[], int jAvar[], int neA, double A[],
              int iGfun[], int jGvar[], int neG,
              double xlow[], double xupp[], double flow[], double fupp[],
              double x[], int xstate[], double xmul[],
              double f[], int fstate[], double fmul[],
              int *inform, int *ns, int *ninf, double *sinf,
              int *miniw, int *minrw,
              int iu[], int leniu, double ru[], int lenru,
              int iw[], int leniw, double rw[], int lenrw)
{
    (void)start; (void)name; (void)objadd; (void)iAfun; (void)jAvar; (void)neA; (void)A;
    (void)flow; (void)fupp; (void)xstate; (void)fstate; (void)fmul;
    (void)iw; (void)rw;
    *ns = 0; *ninf = 0; *sinf = 0.0;
    *miniw = mock_miniw(nf, n, neG);
    *minrw = mock_minrw(nf, n, neG);

    if (!mock_core_initialized()) { *inform = 999; return; }
    if (mock_core_open_failed()) { *inform = 91; return; }
    if (leniw < *miniw || lenrw < *minrw) { *inform = 84; return; }
    /* NLPInterface must announce resized workspaces with these options. */
    if ((opt_total_iw && opt_total_iw != leniw) || (opt_total_rw && opt_total_rw != lenrw)) {
        *inform = 92; return;
    }
    if (objrow != 1) { *inform = 91; return; }

    double *F = calloc((size_t)nf, sizeof(double));
    double *G = calloc((size_t)(neG > 0 ? neG : 1), sizeof(double));
    double *xt = calloc((size_t)n, sizeof(double));
    double *g = calloc((size_t)n, sizeof(double));
    int code = 0, st, status = 1, iter = 0;
    double step = 0.25;

    for (int j = 0; j < n; j++) x[j] = clamp(x[j], xlow[j], xupp[j]);
    st = call_user(usrfun, status, n, x, 1, nf, F, 1, neG, G, iu, leniu, ru, lenru);
    status = 0;
    if (st <= -2) { code = 71; goto done; }
    if (st == -1) { code = 61; goto done; }

    for (;;) {
        double pg = 0.0;
        for (int j = 0; j < n; j++) g[j] = 0.0;
        for (int k = 0; k < neG; k++)
            if (iGfun[k] == objrow) g[jGvar[k] - 1] = G[k];
        for (int j = 0; j < n; j++) {
            double d = clamp(x[j] - g[j], xlow[j], xupp[j]) - x[j];
            pg = fmax(pg, fabs(d));
        }
        if (pg < 1e-10) {
            /* Reduced costs, as SNOPT reports them in xmul: for a box-constrained
               problem the objective gradient (>= 0 at an active lower bound,
               <= 0 at an active upper bound, ~0 elsewhere). Constraint rows are
               ignored by this mock, so fmul stays as passed in. */
            for (int j = 0; j < n; j++) xmul[j] = g[j];
            code = 1; break;
        }
        if (iter >= opt_major_iterations_limit) { code = 32; break; }
        iter++;
        /* Shorten the step until the user function is defined at the trial point. */
        for (int tries = 0;; tries++) {
            if (tries == 30) { code = 63; goto done; }
            for (int j = 0; j < n; j++) xt[j] = clamp(x[j] - step * g[j], xlow[j], xupp[j]);
            st = call_user(usrfun, status, n, xt, 1, nf, F, 1, neG, G, iu, leniu, ru, lenru);
            if (st <= -2) { code = 71; goto done; }
            if (st == 0) break;
            step *= 0.5;
        }
        memcpy(x, xt, (size_t)n * sizeof(double));
    }

    if (env_int("MOCK_SNOPT_PERTURB", 0)) {
        for (int j = 0; j < n; j++) xt[j] = x[j] + 1e-3 * (1.0 + fabs(x[j]));
        st = call_user(usrfun, status, n, xt, 1, nf, F, 0, neG, G, iu, leniu, ru, lenru);
        if (st <= -2) { code = 71; goto done; }
    }

done:
    if (code != 71) {
        /* Final call, as SNOPT makes it: Status = 2, nothing needed. */
        call_user(usrfun, 2, n, x, 0, nf, F, 0, neG, G, iu, leniu, ru, lenru);
    }
    memcpy(f, F, (size_t)nf * sizeof(double));
    *inform = env_int("MOCK_SNOPT_INFORM", code);
    {
        FILE *p = mock_core_print();
        FILE *s = mock_core_summ();
        if (p) fprintf(p, "MOCK SNOPT finished: INFO %d after %d iterations\n", *inform, iter);
        if (s) fprintf(s, "MOCK SNOPT summary: INFO %d\n", *inform);
    }
    free(F); free(G); free(xt); free(g);
}
