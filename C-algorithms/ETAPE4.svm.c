#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NOYAU_LINEAIRE 0
#define NOYAU_RBF 1

// Mesure la "ressemblance" entre 2 points. Noyau lineaire = produit scalaire classique.
// Noyau RBF = gaussienne : vaut 1 si a et b sont identiques, tend vers 0 s'ils sont loin.
// C'est ce qui projette implicitement les donnees dans un espace ou elles sont separables
// par une droite, sans jamais calculer cet espace explicitement (le "kernel trick").
double noyau(double *a, double *b, int n_features, int type_noyau, double gamma)
{
    double somme = 0.0;
    if (type_noyau == NOYAU_RBF)
    {
        for (int j = 0; j < n_features; j++)
            somme += (a[j] - b[j]) * (a[j] - b[j]);   // distance au carre entre a et b
        return exp(-gamma * somme) + 1.0;              // distance -> similarite (gamma = sensibilite)
    }
    for (int j = 0; j < n_features; j++)
        somme += a[j] * b[j];
    return somme + 1.0;
}

// Entraine le SVM : au lieu de chercher directement une frontiere, on apprend un poids
// (alpha[i]) par point du dataset, qui dit a quel point ce point compte pour definir la
// frontiere. coefs[] (le tableau de sortie) recoit alpha[i] * labels[i] a la fin.
int svm_entrainer(double *vecteurs, int *labels, int n_exemples, int n_features,
                  int type_noyau, double gamma, double C, int max_iter, double *coefs)
{
    // K = matrice de similarite entre chaque paire de points (calculee une seule fois,
    // on la reutilise a chaque iteration au lieu de rappeler noyau() sans arret)
    double *K = malloc((size_t)n_exemples * n_exemples * sizeof(double));
    double *alpha = calloc(n_exemples, sizeof(double));   // poids appris, tous a 0 au depart
    double *score = calloc(n_exemples, sizeof(double));   // score actuel de chaque point

    for (int i = 0; i < n_exemples; i++)
        for (int j = i; j < n_exemples; j++)
        {
            double k = noyau(&vecteurs[i * n_features], &vecteurs[j * n_features],
                             n_features, type_noyau, gamma);
            K[(size_t)i * n_exemples + j] = k;
            K[(size_t)j * n_exemples + i] = k;
        }

    // Boucle d'entrainement : on ajuste les poids un par un, jusqu'a ce qu'ils arretent
    // de changer significativement (convergence) ou que max_iter soit atteint.
    for (int iter = 0; iter < max_iter; iter++)
    {
        double plus_grand_changement = 0.0;
        for (int i = 0; i < n_exemples; i++)
        {
            double *K_i = &K[(size_t)i * n_exemples];

            // marge > 0 : le point i est bien classe (et avec de la marge). marge < 0 : mal classe.
            double marge = labels[i] * score[i];

            // Si la marge est insuffisante, on augmente alpha[i] (ce point devient plus
            // important pour la frontiere). On borne entre 0 et C pour qu'un seul point
            // ne puisse pas dominer tout le modele.
            double nouveau = alpha[i] + (1.0 - marge) / K_i[i];
            if (nouveau < 0.0)
                nouveau = 0.0;
            if (nouveau > C)
                nouveau = C;

            double changement = nouveau - alpha[i];
            if (changement != 0.0)
            {
                alpha[i] = nouveau;
                // Le score de TOUS les points doit etre recalcule, puisque le poids de i a change
                for (int j = 0; j < n_exemples; j++)
                    score[j] += changement * labels[i] * K_i[j];
                if (fabs(changement) > plus_grand_changement)
                    plus_grand_changement = fabs(changement);
            }
        }
        if (plus_grand_changement < 1e-6)   // les poids ne bougent presque plus : converge
            break;
    }

    // Seuls les points avec alpha[i] > 0 sont des "vecteurs de support" : ce sont les
    // seuls qui comptent reellement pour la prediction (les autres ont un poids de 0).
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

// Score d'un nouveau point x : somme ponderee de sa similarite avec chaque vecteur de
// support deja appris. Les points non-support (coefs[i] == 0) ne comptent pas.
double svm_score(double *vecteurs, double *coefs, int n_exemples, int n_features,
                 int type_noyau, double gamma, double *x)
{
    double score = 0.0;
    for (int i = 0; i < n_exemples; i++)
        if (coefs[i] != 0.0)
            score += coefs[i] * noyau(&vecteurs[i * n_features], x, n_features, type_noyau, gamma);
    return score;
}

// Classe x selon le signe du score : positif = classe 1, negatif = classe -1.
int svm_predire(double *vecteurs, double *coefs, int n_exemples, int n_features,
                int type_noyau, double gamma, double *x)
{
    return svm_score(vecteurs, coefs, n_exemples, n_features, type_noyau, gamma, x) >= 0 ? 1 : -1;
}

// Sauvegarde seulement les vecteurs de support (coefs != 0) dans un fichier texte simple :
// une ligne d'en-tete (tailles + parametres), puis un vecteur de support par ligne.
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