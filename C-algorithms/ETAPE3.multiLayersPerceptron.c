#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void init_poids(double *poids, int taille)
{
    for (int i = 0; i < taille; i++)
    {
        poids[i] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
    }
}

void init_biais(double *biais, int taille)
{
    for (int i = 0; i < taille; i++)
    {
        biais[i] = 0.0;
    }
}

// Calcule la sortie du reseau pour une entree x (une seule couche cachee)
void forward(double *pixel_array, double *hidden_layer, double *output,
             double *W1, double *bias_1_layer, double *W2, double *bias_2_layer,
             int pixel_array_length, int hidden_length, int output_length)
{

    // calcul de la couche cachée
    for (int i = 0; i < hidden_length; i++)
    {
        hidden_layer[i] = bias_1_layer[i];
        for (int j = 0; j < pixel_array_length; j++)
        {
            hidden_layer[j] += pixel_array[j] * W1[i * hidden_length + j];
        }
    }

    // calcul de la couche de sortie
    for(int j = 0; j < output_length; j++){
        output[j] = bias_2_layer[j];
        for( int i = 0; i < hidden_length ; i++){
            output[j] += hidden_layer[i] * W2[i * hidden_length + j];
        }
    }

    // conversion en probabilités
    double somme = 0.0;
    for(int j = 0; j < output_length; j++){
        somme += output[j];
    }
    for(int j = 0; j < output_length; j++){
        output[j] = output[j] / somme;
    }
}

// Prédiction (classe avec la plus grande probabilité)
int predire(double *output, int output_length) {
    int meilleur = 0;
    for (int j = 1; j < output_length; j++) {
        if (output[j] > output[meilleur]) {
            meilleur = j;
        }
    }
    return meilleur;
}

// Rétropropagation 
void retropropager(double *pixel_array, int classe_attendue, double learning_rate,
                   double *W1, double *b1, double *W2, double *b2,
                   int pixel_array_length, int hidden_layer_length, int output_length) {
    
    double *hidden_layer = malloc(hidden_layer_length * sizeof(double));
    double *output = malloc(output_length * sizeof(double));
    forward(pixel_array, hidden_layer, output, W1, b1, W2, b2, pixel_array_length, hidden_layer_length, output_length);

    double *cible = calloc(output_length, sizeof(double));
    cible[classe_attendue] = 1.0;

    // rétropropagation: on part de la couche 2 et on avance vers la couche 1
    double *derivée_couche_2 = malloc(output_length * sizeof(double)); // erreur couche 2
    for (int j = 0; j < output_length; j++) {
        derivée_couche_2[j] = output[j] - cible[j]; // on calcule la dérivée de la couche 2
    }

    // Erreur couche cachée : la dérivée de f(x)=x est 1, donc pas de multiplication par relu_deriv
    double *derivée_couche_1 = malloc(hidden_layer_length * sizeof(double)); // erreur couche 1
    for (int i = 0; i < hidden_layer_length; i++) {
        double somme = 0.0;
        for (int j = 0; j < output_length; j++) {
            somme += derivée_couche_2[j] * W2[i * output_length + j];
        }
        derivée_couche_1[i] = somme; // 
    }

    // Mise à jour Weight Couche 2 et biais Couche 2
    for (int i = 0; i < hidden_layer_length; i++) {
        for (int j = 0; j < output_length; j++) {
            W2[i * output_length + j] -= learning_rate * derivée_couche_2[j] * hidden_layer[i];
        }
    }
    for (int j = 0; j < output_length; j++) {
        b2[j] -= learning_rate * derivée_couche_2[j];
    }

    // Mise à jour W1 et b1
    for (int i = 0; i < pixel_array_length; i++) {
        for (int j = 0; j < hidden_layer_length; j++) {
            W1[i * hidden_layer_length + j] -= learning_rate * derivée_couche_1[j] * pixel_array[i];
        }
    }
    for (int j = 0; j < hidden_layer_length; j++) {
        b1[j] -= learning_rate * derivée_couche_1[j];
    }

    free(hidden_layer);
    free(output);
    free(cible);
    free(derivée_couche_2);
    free(derivée_couche_1);
}

// Entraine le reseau sur des donnees envoyees depuis Python (vraies photos du dataset)
// vecteurs : n_exemples * IN valeurs a la suite (chaque exemple = IN valeurs)
// classes  : n_exemples valeurs (0, 1 ou 2)

// Entraînement complet
int run_pmc_dataset(double *vecteurs, int *classes, int n_exemples,
                    int in_dim, int h_dim, int out_dim) {
    
    srand(42);

    double *W1 = malloc(in_dim * h_dim * sizeof(double));
    double *b1 = malloc(h_dim * sizeof(double));
    double *W2 = malloc(h_dim * out_dim * sizeof(double));
    double *b2 = malloc(out_dim * sizeof(double));

    init_poids(W1, in_dim * h_dim);
    init_biais(b1, h_dim);
    init_poids(W2, h_dim * out_dim);
    init_biais(b2, out_dim);

    int epochs = 10000;
    double lr = 0.05;

    for (int e = 0; e < epochs; e++) {
        for (int n = 0; n < n_exemples; n++) {
            retropropager(&vecteurs[n * in_dim], classes[n], lr,
                          W1, b1, W2, b2, in_dim, h_dim, out_dim);
        }
    }

    int bonnes_predictions = 0;
    double *h = malloc(h_dim * sizeof(double));
    double *out = malloc(out_dim * sizeof(double));

    for (int n = 0; n < n_exemples; n++) {
        forward(&vecteurs[n * in_dim], h, out, W1, b1, W2, b2, in_dim, h_dim, out_dim);
        if (predire(out, out_dim) == classes[n]) {
            bonnes_predictions++;
        }
    }

    free(W1); free(b1);
    free(W2); free(b2);
    free(h);  free(out);

    return bonnes_predictions;
}

int main(void) {
    int in_dim = 12;
    int h_dim = 8;
    int out_dim = 3;
    int n_exemples = 3;

    double X[3][12] = {
        {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1},
        {1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0}
    };
    int vraies_classes[3] = {2, 0, 1};

    int bonnes = run_pmc_dataset((double*)X, vraies_classes, n_exemples, in_dim, h_dim, out_dim);

    printf("Bonnes predictions : %d/%d\n", bonnes, n_exemples);
    return 0;
}
