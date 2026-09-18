#include <stdio.h>

// Cas de test officiel du prof ("XOR")
int Y[4] = {1, 1, -1, -1}; // sorties attendues pour XOR
double learning_rate = 0.1;

// XOR brut (2 caracteristiques utiles, 3e colonne mise a 0 pour garder la meme taille)
double X_brut[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 0}, {1, 1, 0}};

// XOR + caracteristique x1*x2 (3e colonne) : devient lineairement separable
double X_transforme[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 0}, {1, 1, 1}};

// Entraine un perceptron sur les donnees X et renvoie le nombre d'erreurs finales
int train(double X[4][3])
{
    double w[3] = {0.0, 0.0, 0.0};
    double bias = 0.0;

    for (int rep = 0; rep < 1000; rep++)
    {
        int errors = 0;
        for (int i = 0; i < 4; i++)
        {
            double out = bias;
            for (int j = 0; j < 3; j++)
                out += X[i][j] * w[j];
            int pred = (out >= 0) ? 1 : -1;

            if (pred != Y[i])
            {
                for (int j = 0; j < 3; j++)
                    w[j] += learning_rate * Y[i] * X[i][j];
                bias += learning_rate * Y[i];
                errors++;
            }
        }
        if (errors == 0)
            break;
    }

    int erreurs_finales = 0;
    for (int i = 0; i < 4; i++)
    {
        double out = bias;
        for (int j = 0; j < 3; j++)
            out += X[i][j] * w[j];
        int pred = (out >= 0) ? 1 : -1;
        if (pred != Y[i])
            erreurs_finales++;
    }
    return erreurs_finales;
}

int run_xor_brut(void)
{
    return train(X_brut);
}

int run_xor_transforme(void)
{
    return train(X_transforme);
}

int main(void)
{
    printf("XOR - donnees brutes        : erreurs = %d/4\n", run_xor_brut());
    printf("XOR - transformation x1*x2  : erreurs = %d/4\n", run_xor_transforme());
    return 0;
}
