#include <stdio.h>

// Cas de test officiel du prof ("Linear Simple")
// X_data = les points (x1, x2), Y_pred = la classe attendue pour chaque point (1 ou -1)
int X_data[3][2] = {{1, 1}, {2, 3}, {3, 3}};
int Y_pred[3] = {1, -1, -1};

// Entraine un perceptron simple sur X_data. Remplit poids[0], poids[1] et bias
// avec ce que le perceptron a appris (utile pour tracer la droite de separation),
// et renvoie le nombre d'erreurs finales
int entrainer_perceptron(double *poids, double *bias)
{
    // On part de poids et biais a zero : le perceptron ne sait encore rien
    poids[0] = 0.0;
    poids[1] = 0.0;
    *bias = 0.0;
    double learning_rate = 0.1; // a quelle vitesse le perceptron corrige ses erreurs

    // Boucle d'entrainement : on repete jusqu'a 1000 fois, ou jusqu'a ce qu'il n'y ait plus d'erreur
    for (int rep = 0; rep < 1000; rep++)
    {
        int total_errors = 0;
        for (int i = 0; i < 3; i++)
        {
            // Le neurone calcule : (x1 * poids1) + (x2 * poids2) + biais
            double linear_output = X_data[i][0] * poids[0] + X_data[i][1] * poids[1] + *bias;

            // Si le resultat est positif ou nul, le perceptron predit la classe 1, sinon -1
            int pred = (linear_output >= 0) ? 1 : -1;

            // Si la prediction est fausse, on ajuste les poids et le biais
            // dans la direction qui aurait donne la bonne reponse
            if (pred != Y_pred[i])
            {
                poids[0] += learning_rate * Y_pred[i] * X_data[i][0];
                poids[1] += learning_rate * Y_pred[i] * X_data[i][1];
                *bias += learning_rate * Y_pred[i];
                total_errors++;
            }
        }
        // Si un tour complet s'est fait sans aucune erreur, le perceptron a fini d'apprendre
        if (total_errors == 0)
            break;
    }

    // Une fois l'entrainement termine, on recompte les erreurs avec les poids finaux
    int erreurs_finales = 0;
    for (int i = 0; i < 3; i++)
    {
        double sortie = X_data[i][0] * poids[0] + X_data[i][1] * poids[1] + *bias;
        int test_pred = (sortie >= 0) ? 1 : -1;
        if (test_pred != Y_pred[i])
            erreurs_finales++;
    }
    return erreurs_finales;
}

// Cas de test : entraine le perceptron et ne renvoie que le nombre d'erreurs
// (les poids sont calcules mais pas utilises ici, juste pour un test simple)
int run_perceptron(void)
{
    double poids[2];
    double bias;
    return entrainer_perceptron(poids, &bias);
}

// Meme perceptron que ci-dessus, mais entraine sur des donnees envoyees depuis Python
// (de vraies photos du dataset, pas les 3 points fixes de test)
// vecteurs    : n_exemples * 12 valeurs a la suite (chaque exemple = 12 valeurs, une photo reduite en 2x2 pixels)
// labels      : n_exemples valeurs, -1 ou 1
// predictions : rempli avec ce que le perceptron predit pour chaque exemple (utile pour savoir
//               quelle photo precise est bien/mal classee, pas juste le total)
int predire_perceptron_dataset(double *vecteurs, int *labels, int n_exemples, int *predictions)
{
    double weights[12] = {0};
    double bias = 0.0;
    double learning_rate = 0.1;

    for (int rep = 0; rep < 1000; rep++)
    {
        int total_errors = 0;
        for (int i = 0; i < n_exemples; i++)
        {
            // Meme calcul que plus haut, mais sur 12 valeurs au lieu de 2
            // (vecteurs[i * 12 + j] = la valeur j de l'exemple i, puisque tout est stocke a la suite)
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
        predictions[i] = (sortie >= 0) ? 1 : -1;
        if (predictions[i] != labels[i])
            erreurs_finales++;
    }
    return erreurs_finales;
}

// Cas de test : entraine le perceptron sur le dataset reel et ne renvoie que le nombre d'erreurs
int run_perceptron_dataset(double *vecteurs, int *labels, int n_exemples)
{
    int predictions[50]; // taille max largement suffisante pour notre dataset
    return predire_perceptron_dataset(vecteurs, labels, n_exemples, predictions);
}

int main(void)
{
    printf("Cas de test - erreurs finales : %d/3\n", run_perceptron());
    return 0;
}
