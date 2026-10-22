#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

/*  INIT PHASE */
void init_biais(double *biais, int layer_length){
    for (int neurone = 0; neurone < layer_length; neurone++){
        biais[neurone] = 0.0;
    }
}

int init_couches_cachees(int nb_couches, int nb_neurones_par_couches[], double **hidden_neurones_by_layer_array){
    for (int couche = 0; couche < nb_couches; couche++){
        hidden_neurones_by_layer_array[couche] = malloc(nb_neurones_par_couches[couche] * sizeof(double));

        if (hidden_neurones_by_layer_array[couche] == NULL){
            for (int couche_a_liberer = 0; couche_a_liberer < couche; couche_a_liberer++){
                free(hidden_neurones_by_layer_array[couche_a_liberer]);
            }

            return -1;
        }
    }
    return 0;
}

int init_weights_by_hidden_layers(int nb_couches, int nb_neurones_par_couche[], double **weights, double **biases){
    for (int couche_index = 0; couche_index < nb_couches - 1; couche_index++){
        int nb_neurones_before = nb_neurones_par_couche[couche_index];
        int nb_neurones_current_layer = nb_neurones_par_couche[couche_index + 1];
        int nb_poids = nb_neurones_before * nb_neurones_current_layer;
        init_poids(weights[couche_index], nb_poids);
        init_biais(biases[couche_index], nb_neurones_current_layer);
    }
    return 0;
}


/*  PMC - CORE FUNCTIONS  */
int find_strongest_class_by_neurone_score(double *final_output, int output_length){
    if (final_output == NULL || output_length <= 0)
        return -1;

    int promising_neurone_index = 0;

    for (int neurone = 1; neurone < output_length; neurone++){
        if (final_output[neurone] > final_output[promising_neurone_index]){
            promising_neurone_index = neurone;
        }
    }

    return promising_neurone_index;
}


void calculer_single_layer_output(int layer_before_length, int current_layer_length, double *current_layer_neurones, double *neurone_layer_before, double *output_layer, double *before_layer_weight, double *before_layer_bias){
    
    for (int neurone_index = 0; neurone_index < current_layer_length; neurone_index++){
        double current_neurone_output = before_layer_bias[neurone_index];

        // Calcul de la somme pondérée
        for (int neurone_before_index = 0; neurone_before_index < layer_before_length; neurone_before_index++){

            int poids_index = neurone_index * layer_before_length + neurone_before_index;

            current_neurone_output +=  neurone_layer_before[neurone_before_index]  * before_layer_weight[poids_index];
        }

        current_layer_neurones[neurone_index] =   current_neurone_output;

        if (output_layer != NULL){
            output_layer[neurone_index] = current_neurone_output;
        }
    }
}

double calculer_derivee_partielle_poids(double neurone_layer_before){
    return neurone_layer_before;
}

void update_before_layer_weights(int before_layer_index,double **all_weights,int nb_poids,double learning_rate,double *erreurs,double *neurones_layer_before,int layer_before_length){
    
    for (int neurone = 0; neurone < nb_poids; neurone++){
        int neurone_index = neurone / layer_before_length;
        int before_neurone_index = neurone % layer_before_length;
        double before_weight = all_weights[before_layer_index][neurone];
        double derivee_partielle = calculer_derivee_partielle_poids(neurones_layer_before[before_neurone_index]);
        double new_before_weight = before_weight - learning_rate * erreurs[neurone_index] * derivee_partielle;
        all_weights[before_layer_index][neurone] = new_before_weight;
    }

}

/* Rétropropagation : calcul des erreurs et mise à jour des poids. */
int backpropagation_to_every_layer(double **neurones_by_layer_array, int nb_couches, int nb_neurones_par_couches[], double **weights, double **bias_array, int target_class, double learning_rate){
    
    if (neurones_by_layer_array == NULL || nb_neurones_par_couches == NULL || weights == NULL || bias_array == NULL || nb_couches < 2 || learning_rate <= 0.0){
        return -1;
    }

    int output_length = nb_neurones_par_couches[nb_couches - 1];

    if (target_class < 0 || target_class >= output_length){
        return -1;
    }

    /* Étape 1 : allouer les erreurs de chaque couche. */
    double **erreurs_by_layer = malloc(nb_couches * sizeof(double *));

    if (erreurs_by_layer == NULL)
        return -1;

    for (int couche = 0; couche < nb_couches; couche++){
        erreurs_by_layer[couche] = NULL;
    }

    for (int couche = 1; couche < nb_couches; couche++){
        erreurs_by_layer[couche] =
            malloc(nb_neurones_par_couches[couche] * sizeof(double));

        if (erreurs_by_layer[couche] == NULL){
            for (int couche_a_liberer = 0;
                 couche_a_liberer < nb_couches;
                 couche_a_liberer++){
                free(erreurs_by_layer[couche_a_liberer]);
            }

            free(erreurs_by_layer);
            return -1;
        }
    }

    /* Étape 2 : calculer les erreurs de la couche de sortie. */
    int derniere_couche = nb_couches - 1;

    for (int neurone = 0; neurone < output_length; neurone++){
        double cible = (neurone == target_class) ? 1.0 : 0.0;
        erreurs_by_layer[derniere_couche][neurone] = neurones_by_layer_array[derniere_couche][neurone] - cible;
    }

    /* Étape 3 : propager les erreurs vers les couches cachées. */
    for (int current_couche = nb_couches - 2; current_couche >= 1; current_couche--){

        int current_layer_length = nb_neurones_par_couches[current_couche];
        int next_layer_length = nb_neurones_par_couches[current_couche + 1];
        for (int neurone = 0;  neurone < current_layer_length; neurone++){

            double erreur = 0.0;
            for (int neurone_suivant = 0; neurone_suivant < next_layer_length; neurone_suivant++){
                int poids_index = neurone_suivant * current_layer_length + neurone;
                erreur += weights[current_couche][poids_index] * erreurs_by_layer[current_couche + 1][neurone_suivant];
            }

            erreurs_by_layer[current_couche][neurone] = erreur;
        }
    }

    /*
     * Étape 4 : mettre à jour les poids et les biais.
     * Les erreurs ont été calculées avant toute mise à jour.
     */
    for (int couche = 0; couche < nb_couches - 1; couche++){
        int layer_before_length = nb_neurones_par_couches[couche];
        int current_layer_length = nb_neurones_par_couches[couche + 1];

        for (int neurone = 0; neurone < current_layer_length; neurone++){
            double erreur = erreurs_by_layer[couche + 1][neurone];

            for (int neurone_before = 0; neurone_before < layer_before_length; neurone_before++){
                int poids_index = neurone * layer_before_length + neurone_before;
                double derivee_partielle = calculer_derivee_partielle_poids(neurones_by_layer_array[couche][neurone_before]);
                weights[couche][poids_index] -= learning_rate * erreur * derivee_partielle;
            }
            bias_array[couche][neurone] -= learning_rate * erreur;
        }
    }

    /* Étape 5 : libérer la mémoire des erreurs. */
    for (int couche = 0; couche < nb_couches; couche++){
        free(erreurs_by_layer[couche]);
    }

    free(erreurs_by_layer);

    return 0;
}

int exec_MCP(double *dataset, int dataset_length, int *predictions_by_categories, int nb_couches, int nb_neurones_par_couches[], double **weights, double **biases, int epochs, double learning_rate){
    
    if (dataset == NULL || predictions_by_categories == NULL || nb_neurones_par_couches == NULL || weights == NULL || biases == NULL || dataset_length <= 0 || nb_couches < 2 || epochs <= 0 || learning_rate <= 0.0) {
        return -1;
    }

    /* Étape 1 : initialiser les poids et les biais. */
    for (int couche_index = 0; couche_index < nb_couches - 1; couche_index++){
        int nb_neurones_before = nb_neurones_par_couches[couche_index];
        int nb_neurones_current = nb_neurones_par_couches[couche_index + 1];

        if (nb_neurones_before <= 0 || nb_neurones_current <= 0 || weights[couche_index] == NULL ||   biases[couche_index] == NULL){
            return -1;
        }

        int nb_poids = nb_neurones_before * nb_neurones_current;
        init_poids(weights[couche_index], nb_poids);
        init_biais(biases[couche_index], nb_neurones_current);
    }

    /* Étape 2 : allouer les neurones de chaque couche. */
    double **neurones_by_layer_array = malloc(nb_couches * sizeof(double *));

    if (neurones_by_layer_array == NULL)
        return -1;

    if (init_couches_cachees(nb_couches, nb_neurones_par_couches, neurones_by_layer_array) != 0){
        free(neurones_by_layer_array);
        return -1;
    }

    int input_length = nb_neurones_par_couches[0];
    int output_length = nb_neurones_par_couches[nb_couches - 1];

    /* Étape 3 : entraînement. */
    for (int epoch = 0; epoch < epochs; epoch++){
        for (int picture = 0; picture < dataset_length; picture++) {
            for (int neurone = 0; neurone < input_length; neurone++){
                neurones_by_layer_array[0][neurone] =
                    dataset[picture * input_length + neurone];
            }

            for (int couche_index = 1; couche_index < nb_couches; couche_index++){
                calculer_single_layer_output(
                    nb_neurones_par_couches[couche_index - 1],
                    nb_neurones_par_couches[couche_index],
                    neurones_by_layer_array[couche_index],
                    neurones_by_layer_array[couche_index - 1],
                    NULL,
                    weights[couche_index - 1],
                    biases[couche_index - 1]);
            }

            if (backpropagation_to_every_layer(
                    neurones_by_layer_array,
                    nb_couches,
                    nb_neurones_par_couches,
                    weights,
                    biases,
                    predictions_by_categories[picture],
                    learning_rate) != 0)
            {
                for (int couche = 0; couche < nb_couches; couche++)
                    free(neurones_by_layer_array[couche]);

                free(neurones_by_layer_array);
                return -1;
            }
        }
    }

    /* Étape 4 : compter les bonnes prédictions après l'entraînement. */
    int bonnes_predictions = 0;

    for (int picture = 0; picture < dataset_length; picture++){
        for (int neurone = 0; neurone < input_length; neurone++){
            neurones_by_layer_array[0][neurone] =
                dataset[picture * input_length + neurone];
        }

        for (int couche_index = 1; couche_index < nb_couches; couche_index++){
            calculer_single_layer_output(nb_neurones_par_couches[couche_index - 1], nb_neurones_par_couches[couche_index], neurones_by_layer_array[couche_index], neurones_by_layer_array[couche_index - 1], NULL, weights[couche_index - 1], biases[couche_index - 1]);
        }

        if (find_strongest_class_by_neurone_score(neurones_by_layer_array[nb_couches - 1], output_length) == predictions_by_categories[picture]){
            bonnes_predictions++;
        }
    }

    /* Étape 5 : libérer la mémoire. */
    for (int couche = 0; couche < nb_couches; couche++)
        free(neurones_by_layer_array[couche]);

    free(neurones_by_layer_array);

    return bonnes_predictions;
}


/* Sauvegarde les poids appris dans un fichier texte. */
int pmc_sauvegarder(const char *chemin, double *W1, double *b1, double *W2, double *b2, int in_dim, int h_dim, int out_dim){
    if (chemin == NULL || W1 == NULL || b1 == NULL ||
        W2 == NULL || b2 == NULL)
        return -1;

    FILE *f = fopen(chemin, "w");

    if (f == NULL)
        return -1;

    fprintf(f, "%d %d %d\n", in_dim, h_dim, out_dim);

    for (int i = 0; i < in_dim * h_dim; i++)
        fprintf(f, "%.17g ", W1[i]);

    fprintf(f, "\n");

    for (int i = 0; i < h_dim; i++)
        fprintf(f, "%.17g ", b1[i]);

    fprintf(f, "\n");

    for (int i = 0; i < h_dim * out_dim; i++)
        fprintf(f, "%.17g ", W2[i]);

    fprintf(f, "\n");

    for (int i = 0; i < out_dim; i++)
        fprintf(f, "%.17g ", b2[i]);

    fprintf(f, "\n");

    fclose(f);
    return 0;
}


int pmc_charger(const char *chemin, double *W1, double *b1, double *W2, double *b2, int *in_dim, int *h_dim, int *out_dim){
    if (chemin == NULL || W1 == NULL || b1 == NULL ||
        W2 == NULL || b2 == NULL ||
        in_dim == NULL || h_dim == NULL || out_dim == NULL)
        return -1;

    FILE *f = fopen(chemin, "r");

    if (f == NULL)
        return -1;

    if (fscanf(f, "%d %d %d", in_dim, h_dim, out_dim) != 3 ||
        *in_dim <= 0 || *h_dim <= 0 || *out_dim <= 0)
    {
        fclose(f);
        return -1;
    }

    for (int i = 0; i < (*in_dim) * (*h_dim); i++)
    {
        if (fscanf(f, "%lf", &W1[i]) != 1)
        {
            fclose(f);
            return -1;
        }
    }

    for (int i = 0; i < *h_dim; i++)
    {
        if (fscanf(f, "%lf", &b1[i]) != 1)
        {
            fclose(f);
            return -1;
        }
    }

    for (int i = 0; i < (*h_dim) * (*out_dim); i++)
    {
        if (fscanf(f, "%lf", &W2[i]) != 1)
        {
            fclose(f);
            return -1;
        }
    }

    for (int i = 0; i < *out_dim; i++)
    {
        if (fscanf(f, "%lf", &b2[i]) != 1)
        {
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}


int main(void)
{
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

    int bonnes = run_pmc_dataset(
        (double *)X, vraies_classes, n_exemples,
        in_dim, h_dim, out_dim,
        W1, b1, W2, b2);

    printf("Bonnes predictions : %d/%d\n", bonnes, n_exemples);

    return 0;
}
