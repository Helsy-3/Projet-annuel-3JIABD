#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NOYAU_LINEAIRE 0
#define NOYAU_RBF 1

double noyau(double *a, double *b, int n_features, int type_noyau, double gamma)
{
    double somme = 0.0;
    if (type_noyau == NOYAU_RBF)
    {
        for (int j = 0; j < n_features; j++)
            somme += (a[j] - b[j]) * (a[j] - b[j]);
        return exp(-gamma * somme) + 1.0;
    }
    for (int j = 0; j < n_features; j++)
        somme += a[j] * b[j];
    return somme + 1.0;
}

int svm_entrainer(double *vecteurs, int *labels, int n_exemples, int n_features,
                  int type_noyau, double gamma, double C, int max_iter, double *coefs)
{
    double *K = malloc((size_t)n_exemples * n_exemples * sizeof(double));
    double *alpha = calloc(n_exemples, sizeof(double));
    double *score = calloc(n_exemples, sizeof(double));

    for (int i = 0; i < n_exemples; i++)
        for (int j = i; j < n_exemples; j++)
        {
            double k = noyau(&vecteurs[i * n_features], &vecteurs[j * n_features],
                             n_features, type_noyau, gamma);
            K[(size_t)i * n_exemples + j] = k;
            K[(size_t)j * n_exemples + i] = k;
        }

    for (int iter = 0; iter < max_iter; iter++)
    {
        double plus_grand_changement = 0.0;
        for (int i = 0; i < n_exemples; i++)
        {
            double *K_i = &K[(size_t)i * n_exemples];

            double marge = labels[i] * score[i];

            double nouveau = alpha[i] + (1.0 - marge) / K_i[i];
            if (nouveau < 0.0)
                nouveau = 0.0;
            if (nouveau > C)
                nouveau = C;

            double changement = nouveau - alpha[i];
            if (changement != 0.0)
            {
                alpha[i] = nouveau;
                for (int j = 0; j < n_exemples; j++)
                    score[j] += changement * labels[i] * K_i[j];
                if (fabs(changement) > plus_grand_changement)
                    plus_grand_changement = fabs(changement);
            }
        }
        if (plus_grand_changement < 1e-6)
            break;
    }

    int n_supports = 0;
    for (int i = 0; i < n_exemples; i++)
    {
        coefs[i] = alpha[i] * labels[i];
        if (alpha[i] > 0.0)
            n_supports++;
    }

    free(K);
    free(alpha);
    free(score);
    return n_supports;
}

double svm_score(double *vecteurs, double *coefs, int n_exemples, int n_features,
                 int type_noyau, double gamma, double *x)
{
    double score = 0.0;
    for (int i = 0; i < n_exemples; i++)
        if (coefs[i] != 0.0)
            score += coefs[i] * noyau(&vecteurs[i * n_features], x, n_features, type_noyau, gamma);
    return score;
}

int svm_predire(double *vecteurs, double *coefs, int n_exemples, int n_features,
                int type_noyau, double gamma, double *x)
{
    return svm_score(vecteurs, coefs, n_exemples, n_features, type_noyau, gamma, x) >= 0 ? 1 : -1;
}

int svm_sauvegarder(const char *chemin, double *vecteurs, double *coefs, int n_exemples,
                    int n_features, int type_noyau, double gamma)
{
    FILE *f = fopen(chemin, "w");
    if (f == NULL)
        return -1;

    int n_supports = 0;
    for (int i = 0; i < n_exemples; i++)
        if (coefs[i] != 0.0)
            n_supports++;

    fprintf(f, "%d %d %d %.17g\n", n_supports, n_features, type_noyau, gamma);
    for (int i = 0; i < n_exemples; i++)
    {
        if (coefs[i] == 0.0)
            continue;
        fprintf(f, "%.17g", coefs[i]);
        for (int j = 0; j < n_features; j++)
            fprintf(f, " %.17g", vecteurs[i * n_features + j]);
        fprintf(f, "\n");
    }
    fclose(f);
    return n_supports;
}

int svm_lire_entete(const char *chemin, int *n_supports, int *n_features, int *type_noyau, double *gamma)
{
    FILE *f = fopen(chemin, "r");
    if (f == NULL)
        return -1;
    int lus = fscanf(f, "%d %d %d %lf", n_supports, n_features, type_noyau, gamma);
    fclose(f);
    return lus == 4 ? 0 : -1;
}

int svm_charger(const char *chemin, double *vecteurs, double *coefs)
{
    FILE *f = fopen(chemin, "r");
    if (f == NULL)
        return -1;
    int n_supports, n_features, type_noyau;
    double gamma;
    if (fscanf(f, "%d %d %d %lf", &n_supports, &n_features, &type_noyau, &gamma) != 4)
    {
        fclose(f);
        return -1;
    }
    for (int i = 0; i < n_supports; i++)
    {
        if (fscanf(f, "%lf", &coefs[i]) != 1)
        {
            fclose(f);
            return -1;
        }
        for (int j = 0; j < n_features; j++)
            if (fscanf(f, "%lf", &vecteurs[i * n_features + j]) != 1)
            {
                fclose(f);
                return -1;
            }
    }
    fclose(f);
    return n_supports;
}

static int compter_erreurs(double *vecteurs, int *labels, double *coefs, int n_exemples,
                           int n_features, int type_noyau, double gamma)
{
    int erreurs = 0;
    for (int i = 0; i < n_exemples; i++)
        if (svm_predire(vecteurs, coefs, n_exemples, n_features, type_noyau, gamma,
                        &vecteurs[i * n_features]) != labels[i])
            erreurs++;
    return erreurs;
}

double X_lineaire[3][2] = {{1, 1}, {2, 3}, {3, 3}};
int Y_lineaire[3] = {1, -1, -1};

double X_xor[4][2] = {{1, 0}, {0, 1}, {0, 0}, {1, 1}};
int Y_xor[4] = {1, 1, -1, -1};

int run_svm_lineaire(void)
{
    double coefs[3];
    svm_entrainer(&X_lineaire[0][0], Y_lineaire, 3, 2, NOYAU_LINEAIRE, 0.0, 1000.0, 10000, coefs);
    return compter_erreurs(&X_lineaire[0][0], Y_lineaire, coefs, 3, 2, NOYAU_LINEAIRE, 0.0);
}

int run_svm_xor_lineaire(void)
{
    double coefs[4];
    svm_entrainer(&X_xor[0][0], Y_xor, 4, 2, NOYAU_LINEAIRE, 0.0, 1000.0, 10000, coefs);
    return compter_erreurs(&X_xor[0][0], Y_xor, coefs, 4, 2, NOYAU_LINEAIRE, 0.0);
}

int run_svm_xor_rbf(void)
{
    double coefs[4];
    svm_entrainer(&X_xor[0][0], Y_xor, 4, 2, NOYAU_RBF, 1.0, 1000.0, 10000, coefs);
    return compter_erreurs(&X_xor[0][0], Y_xor, coefs, 4, 2, NOYAU_RBF, 1.0);
}

int main(void)
{
    printf("Linear Simple - noyau lineaire : erreurs = %d/3\n", run_svm_lineaire());
    printf("XOR           - noyau lineaire : erreurs = %d/4\n", run_svm_xor_lineaire());
    printf("XOR           - noyau RBF      : erreurs = %d/4\n", run_svm_xor_rbf());
    return 0;
}