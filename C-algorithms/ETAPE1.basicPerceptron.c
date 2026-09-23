#include <stdio.h>
#include <string.h>

// EXEMPLE
//int X_data[3][2] = {{1, 1}, {2, 3}, {3, 3}};
//int Y_pred[3] = {1, -1, -1};

int entrainer_perceptron(double learning_rate, double *bias, int epochs, int X_data_lignes, int X_data_colonnes, double X_data[X_data_lignes][X_data_colonnes], double Y_pred[])
{
    size_t dataset_size = X_data_lignes; // taille du dataset
    size_t poids_nb = X_data_colonnes;

    // Dans le data set, chaque donnée est un vecteur avec des caractéristiques [1, 2, 3, 4]
    // A chaque vecteur correspond un poids. Il y a autant de poids qu'il y a de caractéristiques par donnée du dataset.

    // Il faut qu'il y ait autant de poids qu'il y a de caractéristiques pour une donnée.
    // Exemple 1: une maison avec deux caractéristiques: la superficie et le nombre d'étages : 2 caractéristiques => donc il faut 2 poids.
    // Exemple 2: une photo de bouteille avec 32 pixels : 32 caractéristiques => donc il faut 32 poids.

    double poids[poids_nb];
    memset(poids, 0, sizeof(poids)); // on initialise le tableau de poids à 0.
    int total_errors = 0;

    for (int i = 0; i < epochs; i++) // on entraine le modèle autant de fois qu'il y a de répétitions
    {
        int total_errors = 0; // on ca comptabiliser le nombre d'erreurs à chaque fois en comparant Y_pred (le résultat attendu)
                              // avec le résultat obtenu avec les poids initialisés à 0.

        for (int i = 0; i < dataset_size; i++) // on veut itérer sur toutes les données du data_set
        {
            double linear_output = 0; // un seul neurone

            for (int y = 0; y < poids_nb; y++)
            {
                linear_output += X_data[i][y] * poids[y]; // un seul neurone donc on aditionne chaque membre du vecteur et ça donne la sortie de l'unique neurone
            }

            linear_output = linear_output + *bias; // sortie de l'unique neurone
            
            int pred = (linear_output >= 0) ? 1 : -1; // prediction du modèle
            if (pred != Y_pred[i])
                { // si la prédiction obtenue est différente de la prédiction attendue
                    for(int y = 0; y < poids_nb ; y++){
                    poids[y] += learning_rate * Y_pred[i] * X_data[i][y];
                    }
                    *bias += learning_rate * Y_pred[i];
                    total_errors++;
                }
            
        }
    }
    return total_errors;
}

// Entraine un perceptron sur des donnees envoyees depuis Python (vraies photos du dataset)
// vecteurs : n_exemples * 12 valeurs a la suite (chaque exemple = 12 valeurs, une photo reduite en 2x2 pixels)
// labels   : n_exemples valeurs, -1 ou 1
int run_perceptron_dataset(double *vecteurs, int *labels, int n_exemples)
{
    double weights[12] = {0};
    double bias = 0.0;
    double learning_rate = 0.1;

    for (int rep = 0; rep < 1000; rep++)
    {
        int total_errors = 0;
        for (int i = 0; i < n_exemples; i++)
        {
            double linear_output = bias;
            for (int j = 0; j < 12; j++)
                linear_output += vecteurs[i * 12 + j] * weights[j];
            int pred = (linear_output >= 0) ? 1 : -1;

            if (pred != labels[i])
            {
                for (int j = 0; j < 12; j++)
                    weights[j] += learning_rate * labels[i] * vecteurs[i * 12 + j];
                bias += learning_rate * labels[i];
                total_errors++;
            }
        }
        if (total_errors == 0)
            break;
    }

    int erreurs_finales = 0;
    for (int i = 0; i < n_exemples; i++)
    {
        double sortie = bias;
        for (int j = 0; j < 12; j++)
            sortie += vecteurs[i * 12 + j] * weights[j];
        int test_pred = (sortie >= 0) ? 1 : -1;
        if (test_pred != labels[i])
            erreurs_finales++;
    }
    return erreurs_finales;
}

int main(void)
{
    printf("Cas de test - erreurs finales : %d/3\n", run_perceptron());
    return 0;
}
