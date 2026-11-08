#include <stdio.h>
#include <stdlib.h>

// Perceptron lineaire classique : apprend des poids + un biais par descente d'erreur.
// Reutilise par entrainer_perceptron_transforme() ci-dessous, sur des donnees transformees.
int entrainer_perceptron(double *vecteurs, int *labels, int n_exemples, int n_features,
                         double learning_rate, int max_iter, double *weights, double *bias)
{
    // On repart toujours de zero (pas de poids herites d'un appel precedent).
    for (int j = 0; j < n_features; j++)
        weights[j] = 0.0;
    *bias = 0.0;

    // Boucle d'entrainement : on repete jusqu'a convergence (0 erreur) ou max_iter atteint.
    for (int rep = 0; rep < max_iter; rep++)
    {
        int total_errors = 0;
        for (int i = 0; i < n_exemples; i++)
        {
            // Prediction actuelle du modele pour ce point (somme ponderee + biais).
            double linear_output = *bias;
            for (int j = 0; j < n_features; j++)
                linear_output += vecteurs[i * n_features + j] * weights[j];
            int pred = (linear_output >= 0) ? 1 : -1;

            // Si on s'est trompe, on corrige les poids dans la direction de la bonne reponse.
            if (pred != labels[i])
            {
                for (int j = 0; j < n_features; j++)
                    weights[j] += learning_rate * labels[i] * vecteurs[i * n_features + j];
                *bias += learning_rate * labels[i];
                total_errors++;
            }
        }
        if (total_errors == 0)  // plus aucune erreur : le modele a converge, inutile de continuer
            break;
    }

    // Une fois l'entrainement termine, on recompte les erreurs avec les poids finaux.
    int erreurs_finales = 0;
    for (int i = 0; i < n_exemples; i++)
    {
        double sortie = *bias;
        for (int j = 0; j < n_features; j++)
            sortie += vecteurs[i * n_features + j] * weights[j];
        int pred = (sortie >= 0) ? 1 : -1;
        if (pred != labels[i])
            erreurs_finales++;
    }
    return erreurs_finales;
}

// Nombre de caracteristiques apres transformation : les n_features d'origine, plus un
// produit croise (x_a * x_b) pour chaque paire differente de caracteristiques.
// Pour XOR (n_features=2), ca donne 2 + 1 = 3 (x1, x2, x1*x2).
int nb_features_transformees(int n_features)
{
    return n_features + n_features * (n_features - 1) / 2;
}

// Ajoute les produits croises (x_a * x_b) a chaque exemple du dataset. C'est cette seule
// caracteristique en plus (x1*x2 pour XOR) qui rend le probleme lineairement separable,
// sans changer l'algorithme du perceptron lui-meme.
void transformer(double *vecteurs, int n_exemples, int n_features, double *transformes)
{
    int n_out = nb_features_transformees(n_features);
    for (int i = 0; i < n_exemples; i++)
    {
        double *x = &vecteurs[i * n_features];   // exemple d'origine (n_features valeurs)
        double *t = &transformes[i * n_out];     // exemple transforme (n_out valeurs)
        int k = 0;
        for (int a = 0; a < n_features; a++)
            t[k++] = x[a];                        // on garde d'abord les valeurs d'origine
        for (int a = 0; a < n_features; a++)
            for (int b = a + 1; b < n_features; b++)
                t[k++] = x[a] * x[b];              // puis on ajoute chaque produit croise
    }
}

// Transforme les donnees puis entraine un perceptron classique dessus - pas un nouvel
// algorithme, juste un pretraitement avant d'appeler entrainer_perceptron().
int entrainer_perceptron_transforme(double *vecteurs, int *labels, int n_exemples, int n_features,
                                    double learning_rate, int max_iter, double *weights, double *bias)
{
    int n_out = nb_features_transformees(n_features);
    double *transformes = malloc(n_exemples * n_out * sizeof(double));
    transformer(vecteurs, n_exemples, n_features, transformes);
    int erreurs = entrainer_perceptron(transformes, labels, n_exemples, n_out,
                                       learning_rate, max_iter, weights, bias);
    free(transformes);
    return erreurs;
}

// XOR : "vrai" (1) si les deux entrees sont differentes, "faux" (-1) si elles sont identiques.
// Les 4 lignes couvrent les 4 combinaisons possibles de deux entrees binaires (0 ou 1).
double X_xor[4][2] = {{1, 0}, {0, 1}, {0, 0}, {1, 1}};
int Y_xor[4] = {1, 1, -1, -1};

int run_xor_brut(void)
{
    double weights[2], bias;
    return entrainer_perceptron(&X_xor[0][0], Y_xor, 4, 2, 0.1, 1000, weights, &bias);
}

int run_xor_transforme(void)
{
    double weights[3], bias;
    return entrainer_perceptron_transforme(&X_xor[0][0], Y_xor, 4, 2, 0.1, 1000, weights, &bias);
}

int main(void)
{
    double weights[3], bias;

    int erreurs = entrainer_perceptron(&X_xor[0][0], Y_xor, 4, 2, 0.1, 1000, weights, &bias);
    printf("XOR - donnees brutes       : erreurs = %d/4\n", erreurs);

    erreurs = entrainer_perceptron_transforme(&X_xor[0][0], Y_xor, 4, 2, 0.1, 1000, weights, &bias);
    printf("XOR - transformation x1*x2 : erreurs = %d/4\n", erreurs);
    printf("  poids appris : w1 = %.2f, w2 = %.2f, w(x1*x2) = %.2f, biais = %.2f\n",
           weights[0], weights[1], weights[2], bias);
    return 0;
}