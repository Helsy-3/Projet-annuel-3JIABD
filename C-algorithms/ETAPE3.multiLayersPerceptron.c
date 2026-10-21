#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

void init_biais(double *biais, int taille){
    for (int i = 0; i < taille; i++){
        biais[i] = 0.0;
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


// Entraine le reseau sur des donnees envoyees depuis Python (vraies photos du dataset)
// vecteurs : n_exemples * IN valeurs a la suite (chaque exemple = IN valeurs)
// classes  : n_exemples valeurs (0, 1 ou 2)

// Entraînement complet. W1_sortie/b1_sortie/W2_sortie/b2_sortie sont des tableaux deja
// alloues par Python : on y recopie les poids appris, pour que Python puisse les reutiliser
// ensuite avec predire_pmc() sur des donnees jamais vues pendant l'entrainement (split train/test).

int init_couches_cachees(int nb_couches, int nb_neurones_par_couches[]){
    double *hidden_neurones_by_layer_array;

    for(int couche = 0; couche < nb_couches; couche ++){
        double hidden_layer = malloc(nb_neurones_par_couches[couche] * sizeof(double));
        hidden_neurones_by_layer_array[couche] = hidden_layer;
    }
}

int init_weights_by_hidden_layers(int nb_couches, int nb_neurones_par_couche, double **couches_cachees_array[nb_couches][nb_neurones_par_couche]){
    double poids_par_couche[nb_couches][nb_neurones_par_couche]; 

    for (int couche_index = 1; couche_index < nb_couches; couche_index++){
        int nb_neurones_current_layer = couches_cachees_array[couche_index];
        double *weights = malloc(nb_neurones_current_layer * sizeof(double));
        double biais_couche[nb_neurones_current_layer];
        init_poids(weights, nb_neurones_current_layer);
        init_biais(biais_couche, nb_neurones_current_layer);
        **couches_cachees_array[couche_index] = weights;
    }
}


void calculer_layer_output(int layer_before_length, int current_layer_length, double *current_layer_neurones, double *neurone_layer_before, double *output_layer, double *before_layer_weight, double *before_layer_bias) {
    
    // Calcul de la couche actuelle (couche cachée)
    for (int neurone_index = 0; neurone_index < current_layer_length; neurone_index++) {
        
        double current_neurone_output = before_layer_bias[neurone_index];
        current_layer_neurones[neurone_index] = before_layer_bias[neurone_index];
        
        // calculer neurone output (layer[1] * w[1] + layer[2] * w[2] + layer[3] * w[3], etc...)
        double output = 0.0;
        for(int neurone_before_index = 0; neurone_before_index < layer_before_length; neurone_before_index ++){
            current_neurone_output+= neurone_layer_before[neurone_before_index] * before_layer_weight[neurone_before_index];
        }
        current_layer_neurones[neurone_index] = current_neurone_output;
    }
}


// Rétropropagation 
void backpropagation(double *neurones_by_layer_array, int neurone_length, int nb_couches, int nb_neurones_par_couches[], int weights[], int bias_array[]) {
    
    for(int couche_index = 1; couche_index < nb_couches; couche_index ++){
        
        double *neurone_layer_before =  &neurones_by_layer_array[couche_index -1];
        double *before_layer_weight = weights[couche_index -1];
        double *before_layer_bias = bias_array[couche_index -1];
        int neurone_layer_before_length = sizeof(&neurones_by_layer_array[couche_index-1]) / sizeof(neurone_length);
        double *current_neurones_layer = &neurones_by_layer_array[couche_index];
        int current_layer_length = nb_neurones_par_couches[couche_index];
        double *output_layer = malloc(current_layer_length * sizeof(double));
        calculer_layer_output(neurone_layer_before_length, current_layer_length, current_neurones_layer, neurone_layer_before, output_layer, before_layer_weight, before_layer_bias);
        
        double *expected_target = calloc(current_layer_length, sizeof(double));
        double derivee_current_couche = malloc(current_layer_length * sizeof(double));
        for (int i = 0; i < current_layer_length; i++){
            derivee_current_couche = output_layer[i] - expected_target[i];
        }
    }

}

int run_pmc_dataset(double *vecteurs, int *classes, int n_exemples, int in_dim, int h_dim, int out_dim, int nb_couches, int nb_neurones_par_couche[], double *couches_cachees[nb_couches][nb_neurones_par_couche], int epochs, int learning_rate, int seed) {
    srand(seed);

    // step 1: initialiser le tableau de couches cachées avec les neurones
    init_couches_cachees(nb_couches, nb_neurones_par_couche);

    // step 2: initialiser le nombre de poids par couches cachées. 
    init_weights_by_hidden_layers(nb_couches, nb_neurones_par_couche, *couches_cachees);

    for (int epoch = 0; epoch < epochs; epoch ++) {
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
    int bonnes = run_pmc_dataset((double*)X, vraies_classes, n_exemples, in_dim, h_dim, out_dim,
                                  W1, b1, W2, b2);

    printf("Bonnes predictions : %d/%d\n", bonnes, n_exemples);
    return 0;
}
