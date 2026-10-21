#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

void calculer_sortie_reseau(double *pixel_array, double *hidden_layer, double *output_layer,
             double *W1, double *bias_layer_1, double *W2, double *bias_layer_2,
             int pixel_array_length, int hidden_layer_length, int output_length)
{

    // calcul de la couche cachée
    for (int i = 0; i < hidden_layer_length; i++)
    {
        hidden_layer[i] = bias_layer_1[i];
        for (int j = 0; j < pixel_array_length; j++)
        {
            hidden_layer[i] += pixel_array[j] * W1[j * hidden_layer_length + i];
        }
    }

    // calcul de la couche de sortie
    for(int i = 0; i < output_length; i++)
    {
        output_layer[i] = bias_layer_2[i];
        for( int j = 0; j < hidden_layer_length ; j++)
        {
            output_layer[i] += hidden_layer[j] * W2[j * output_length + i];
        }
    }

    // conversion en probabilités (softmax : il faut exponentier avant de diviser par la somme)
    double somme = 0.0;
    for(int j = 0; j < output_length; j++){
        output_layer[j] = exp(output_layer[j]);
        somme += output_layer[j];
    }
    for(int j = 0; j < output_length; j++){
        output_layer[j] = output_layer[j] / somme;
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
    calculer_sortie_reseau(pixel_array, hidden_layer, output, W1, b1, W2, b2, pixel_array_length, hidden_layer_length, output_length);

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

// Entraînement complet. W1_sortie/b1_sortie/W2_sortie/b2_sortie sont des tableaux deja
// alloues par Python : on y recopie les poids appris, pour que Python puisse les reutiliser
// ensuite avec predire_pmc() sur des donnees jamais vues pendant l'entrainement (split train/test).
int run_pmc_dataset(double *vecteurs, int *classes, int n_exemples,
                    int in_dim, int h_dim, int out_dim, int epochs,
                    double *W1_sortie, double *b1_sortie, double *W2_sortie, double *b2_sortie) {

    srand(42);

    double *W1 = malloc(in_dim * h_dim * sizeof(double));
    double *b1 = malloc(h_dim * sizeof(double));
    double *W2 = malloc(h_dim * out_dim * sizeof(double));
    double *b2 = malloc(out_dim * sizeof(double));

    init_poids(W1, in_dim * h_dim);
    init_biais(b1, h_dim);
    init_poids(W2, h_dim * out_dim);
    init_biais(b2, out_dim);

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
        calculer_sortie_reseau(&vecteurs[n * in_dim], h, out, W1, b1, W2, b2, in_dim, h_dim, out_dim);
        if (predire(out, out_dim) == classes[n]) {
            bonnes_predictions++;
        }
    }

    memcpy(W1_sortie, W1, in_dim * h_dim * sizeof(double));
    memcpy(b1_sortie, b1, h_dim * sizeof(double));
    memcpy(W2_sortie, W2, h_dim * out_dim * sizeof(double));
    memcpy(b2_sortie, b2, out_dim * sizeof(double));

    free(W1); free(b1);
    free(W2); free(b2);
    free(h);  free(out);

    return bonnes_predictions;
}

// Predit la classe d'un seul exemple a partir de poids DEJA entraines (par run_pmc_dataset
// ci-dessus). Ne fait aucun entrainement : utile pour evaluer sur des donnees de test separees.
int predire_pmc(double *pixel_array, double *W1, double *b1, double *W2, double *b2,
                int in_dim, int h_dim, int out_dim) {
    double *h = malloc(h_dim * sizeof(double));
    double *out = malloc(out_dim * sizeof(double));

    calculer_sortie_reseau(pixel_array, h, out, W1, b1, W2, b2, in_dim, h_dim, out_dim);
    int classe = predire(out, out_dim);

    free(h); free(out);
    return classe;
}

// Sauvegarde les poids appris dans un fichier texte, pour les reutiliser plus tard avec
// predire_pmc() sans avoir a ré-entrainer (meme principe que svm_sauvegarder() dans ETAPE4.svm.c).
int pmc_sauvegarder(const char *chemin, double *W1, double *b1, double *W2, double *b2,
                    int in_dim, int h_dim, int out_dim) {
    FILE *f = fopen(chemin, "w");
    if (f == NULL)
        return -1;

    fprintf(f, "%d %d %d\n", in_dim, h_dim, out_dim);
    for (int i = 0; i < in_dim * h_dim; i++) fprintf(f, "%.17g ", W1[i]);
    fprintf(f, "\n");
    for (int i = 0; i < h_dim; i++) fprintf(f, "%.17g ", b1[i]);
    fprintf(f, "\n");
    for (int i = 0; i < h_dim * out_dim; i++) fprintf(f, "%.17g ", W2[i]);
    fprintf(f, "\n");
    for (int i = 0; i < out_dim; i++) fprintf(f, "%.17g ", b2[i]);
    fprintf(f, "\n");

    fclose(f);
    return 0;
}

int pmc_charger(const char *chemin, double *W1, double *b1, double *W2, double *b2,
               int *in_dim, int *h_dim, int *out_dim) {
    FILE *f = fopen(chemin, "r");
    if (f == NULL)
        return -1;

    if (fscanf(f, "%d %d %d", in_dim, h_dim, out_dim) != 3) { fclose(f); return -1; }
    for (int i = 0; i < (*in_dim) * (*h_dim); i++)
        if (fscanf(f, "%lf", &W1[i]) != 1) { fclose(f); return -1; }
    for (int i = 0; i < *h_dim; i++)
        if (fscanf(f, "%lf", &b1[i]) != 1) { fclose(f); return -1; }
    for (int i = 0; i < (*h_dim) * (*out_dim); i++)
        if (fscanf(f, "%lf", &W2[i]) != 1) { fclose(f); return -1; }
    for (int i = 0; i < *out_dim; i++)
        if (fscanf(f, "%lf", &b2[i]) != 1) { fclose(f); return -1; }

    fclose(f);
    return 0;
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

    double W1[12 * 8], b1[8], W2[8 * 3], b2[3];
    int bonnes = run_pmc_dataset((double*)X, vraies_classes, n_exemples, in_dim, h_dim, out_dim, 10000,
                                  W1, b1, W2, b2);

    printf("Bonnes predictions : %d/%d\n", bonnes, n_exemples);
    return 0;
}
