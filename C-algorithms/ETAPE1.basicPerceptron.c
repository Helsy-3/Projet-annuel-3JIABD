#include <stdio.h>
#include <stdlib.h>

// Cas de test officiel du prof ("Linear Simple")
int X_data[3][2] = {{1, 1}, {2, 3}, {3, 3}};
int Y_pred[3] = {1, -1, -1};

// Entraine un perceptron simple sur X_data et renvoie le nombre d'erreurs finales
int run_perceptron(void)
{
    double weights[2] = {0.0, 0.0};
    double bias = 0.0;
    double learning_rate = 0.1;

    for (int rep = 0; rep < 1000; rep++)
    {
        int total_errors = 0;
        for (int i = 0; i < 3; i++)
        {
            double linear_output = X_data[i][0] * weights[0] + X_data[i][1] * weights[1] + bias;
            int pred = (linear_output >= 0) ? 1 : -1;

            if (pred != Y_pred[i])
            {
                weights[0] += learning_rate * Y_pred[i] * X_data[i][0];
                weights[1] += learning_rate * Y_pred[i] * X_data[i][1];
                bias += learning_rate * Y_pred[i];
                total_errors++;
            }
        }
        if (total_errors == 0)
            break;
    }

    int erreurs_finales = 0;
    for (int i = 0; i < 3; i++)
    {
        double sortie = X_data[i][0] * weights[0] + X_data[i][1] * weights[1] + bias;
        int test_pred = (sortie >= 0) ? 1 : -1;
        if (test_pred != Y_pred[i])
            erreurs_finales++;
    }
    return erreurs_finales;
}

// Entraine un perceptron sur des donnees envoyees depuis Python (vraies photos du dataset)
// vecteurs : n_exemples * 12 valeurs a la suite (chaque exemple = 12 valeurs, une photo reduite en 2x2 pixels)
// labels   : n_exemples valeurs, -1 ou 1
int run_perceptron_dataset(
    double *vecteurs,
    int *labels,
    int n_exemples,
    int n_features,
    double learning_rate,
    int max_iter,
    double *weights,
    double *bias)
{
    for (int rep = 0; rep < max_iter; rep++)
    {
        int total_errors = 0;
        for (int i = 0; i < n_exemples; i++)
        {
            double linear_output = *bias;
            for (int j = 0; j < n_features; j++)
                linear_output += vecteurs[i * n_features + j] * weights[j];
            int pred = (linear_output >= 0) ? 1 : -1;

            if (pred != labels[i])
            {
                for (int j = 0; j < n_features; j++)
                    weights[j] += learning_rate * labels[i] * vecteurs[i * n_features + j];
                *bias += learning_rate * labels[i];
                total_errors++;
            }
        }
        if (total_errors == 0)
            break;
    }

    int erreurs_finales = 0;
    for (int i = 0; i < n_exemples; i++)
    {
        double sortie = *bias;
        for (int j = 0; j < n_features; j++)
            sortie += vecteurs[i * n_features + j] * weights[j];
        int test_pred = (sortie >= 0) ? 1 : -1;
        if (test_pred != labels[i])
            erreurs_finales++;
    }
    return erreurs_finales;
}
