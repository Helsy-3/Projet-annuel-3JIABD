#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "utils.h"

/* INIT PHASE */
double *init_biais_by_layer(int nb_neurones_current_layer)
{
    if (nb_neurones_current_layer <= 0)
        return NULL;

    double *bias_array = malloc(nb_neurones_current_layer * sizeof(double));

    if (bias_array == NULL)
        return NULL;

    for (int i = 0; i < nb_neurones_current_layer; i++)
    {
        bias_array[i] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
    }

    return bias_array;
}

/* INIT PHASE */

double *init_poids_by_layer(int before_layer_neurones_number, int next_layer_neurones_number)
{
    if (before_layer_neurones_number <= 0 || next_layer_neurones_number <= 0)
        return NULL;

    int weight_number = before_layer_neurones_number * next_layer_neurones_number;

    double *weight_array = malloc(weight_number * sizeof(double));

    if (weight_array == NULL)
        return NULL;

    for (int i = 0; i < weight_number; i++)
    {
        weight_array[i] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
    }

    return weight_array;
}

/* INIT PHASE */

int init_couches_cachees_by_layer(int nb_couches, int nb_neurones_par_couches[], double **neurones_by_layer_array)
{
    if (nb_couches <= 0 || nb_neurones_par_couches == NULL || neurones_by_layer_array == NULL)
        return -1;

    for (int couche = 0; couche < nb_couches; couche++)
    {
        neurones_by_layer_array[couche] = NULL;
    }

    for (int couche = 0; couche < nb_couches; couche++)
    {
        if (nb_neurones_par_couches[couche] <= 0)
            goto erreur;

        neurones_by_layer_array[couche] = malloc(nb_neurones_par_couches[couche] * sizeof(double));

        if (neurones_by_layer_array[couche] == NULL)
            goto erreur;
    }

    return 0;

erreur:
    for (int couche = 0; couche < nb_couches; couche++)
    {
        free(neurones_by_layer_array[couche]);
        neurones_by_layer_array[couche] = NULL;
    }

    return -1;
}

/* INIT PHASE */

int init_weights_and_biases_by_hidden_layers(int nb_couches, int nb_neurones_par_couche[], double **weights, double **biases)
{
    if (nb_couches < 2 || nb_neurones_par_couche == NULL || weights == NULL || biases == NULL)
        return -1;

    for (int couche = 0; couche < nb_couches - 1; couche++)
    {
        weights[couche] = NULL;
        biases[couche] = NULL;
    }

    for (int couche = 0; couche < nb_couches - 1; couche++)
    {
        int nb_neurones_before = nb_neurones_par_couche[couche];
        int nb_neurones_next = nb_neurones_par_couche[couche + 1];

        if (nb_neurones_before <= 0 || nb_neurones_next <= 0)
            goto erreur;

        weights[couche] = init_poids_by_layer(nb_neurones_before, nb_neurones_next);
        biases[couche] = init_biais_by_layer(nb_neurones_next);

        if (weights[couche] == NULL || biases[couche] == NULL)
            goto erreur;
    }

    return 0;

erreur:
    for (int i = 0; i < nb_couches - 1; i++)
    {
        free(weights[i]);
        free(biases[i]);

        weights[i] = NULL;
        biases[i] = NULL;
    }

    return -1;
}

/* PROPAGATION PHASE */
void calculer_single_layer_output(int layer_before_length, int current_layer_length,
                                  double *current_layer_neurones,
                                  double *neurone_layer_before,
                                  double *output_layer,
                                  double *before_layer_weight,
                                  double *before_layer_bias)
{
    for (int current_neurone_index = 0; current_neurone_index < current_layer_length; current_neurone_index++){
        double current_neurone_output = before_layer_bias[current_neurone_index];
        // Un neurone reçoit la somme pondérée de tous les neurones de la couche précédente.
        for (int neurone_before_index = 0; neurone_before_index < layer_before_length; neurone_before_index++){
            int poids_index = current_neurone_index * layer_before_length + neurone_before_index;
            current_neurone_output += neurone_layer_before[neurone_before_index] * before_layer_weight[poids_index];
        }

        current_layer_neurones[current_neurone_index] = current_neurone_output;

        if (output_layer != NULL)
        {
            output_layer[current_neurone_index] = current_neurone_output;
        }
    }
}

/* BACK-PROPAGATION PHASE */

double calculer_derivee_partielle_loss_par_rapport_au_poids(double derivee_loss_par_rapport_sortie_neurone, double neurone_before)
{
    // y^ = w * x + b
    // dy/dw = x

    double derivee_neurone_par_rapport_au_poids = neurone_before;

    // dL/dw = dL/dy * dy/dw

    double derivee_LOSS_par_rapport_au_poids =
        derivee_loss_par_rapport_sortie_neurone * derivee_neurone_par_rapport_au_poids;

    return derivee_LOSS_par_rapport_au_poids;
}

void update_before_layer_weights(int before_layer_index, double **weights_matrice,
                                 int nb_poids, double learning_rate,
                                 double *derivees_loss_sortie_neurones,
                                 double *neurones_layer_before,
                                 int layer_before_length)
{
    if (layer_before_length <= 0 || weights_matrice == NULL ||
        derivees_loss_sortie_neurones == NULL || neurones_layer_before == NULL)
        return;

    for (int poids = 0; poids < nb_poids; poids++)
    {
        int current_neurone_index = poids / layer_before_length;
        int before_neurone_index = poids % layer_before_length;

        double derivee_L = derivees_loss_sortie_neurones[current_neurone_index];
        double neurone_before = neurones_layer_before[before_neurone_index];

        double gradient = calculer_derivee_partielle_loss_par_rapport_au_poids(derivee_L, neurone_before);

        // Descente de gradient.
        weights_matrice[before_layer_index][poids] -= learning_rate * gradient;
    }
}

void update_before_layer_bias(int before_layer_index, double **all_biases,
                              int nb_biases, double learning_rate,
                              double *erreurs_by_layer)
{
    for (int i = 0; i < nb_biases; i++)
    {
        all_biases[before_layer_index][i] -= learning_rate * erreurs_by_layer[i];
    }
}

int backpropagation_to_every_layer(double **neurones_by_layer_array,
                                   int nb_couches,
                                   int nb_neurones_par_couches[],
                                   double **weights,
                                   double **biases_array,
                                   int target_class,
                                   double learning_rate)
{
    if (neurones_by_layer_array == NULL ||
        nb_neurones_par_couches == NULL ||
        weights == NULL ||
        biases_array == NULL ||
        nb_couches < 2 ||
        !isfinite(learning_rate) ||
        learning_rate <= 0.0)
    {
        return -1;
    }

    for (int couche = 0; couche < nb_couches; couche++)
    {
        if (nb_neurones_par_couches[couche] <= 0 || neurones_by_layer_array[couche] == NULL)
            return -1;
    }

    for (int couche = 0; couche < nb_couches - 1; couche++)
    {
        if (weights[couche] == NULL || biases_array[couche] == NULL)
            return -1;
    }

    int output_layer = nb_couches - 1;
    int output_length = nb_neurones_par_couches[output_layer];

    if (target_class < 0 || target_class >= output_length)
        return -1;

    // Étape 1 : allocation des erreurs par couche.
    double **erreurs_by_layer = calloc(nb_couches, sizeof(double *));

    if (erreurs_by_layer == NULL)
        return -1;

    for (int couche = 1; couche < nb_couches; couche++){
        erreurs_by_layer[couche] = malloc(nb_neurones_par_couches[couche] * sizeof(double));
        if (erreurs_by_layer[couche] == NULL)
            goto erreur;
    }

    // Étape 2 : calculer l'erreur de sortie.
    // Loss = 1/2 * (prediction - cible)^2.
    // dLoss/dPrediction = prediction - cible.

    for (int neurone = 0; neurone < output_length; neurone++){
        double cible = (neurone == target_class) ? 1.0 : 0.0;
        erreurs_by_layer[output_layer][neurone] =
            neurones_by_layer_array[output_layer][neurone] - cible;
    }

    // Étape 3 : propager les erreurs vers les couches cachées.
    for (int couche = nb_couches - 2; couche >= 1; couche--){ // on commence à l'avant-dernière couche et on s'arrête à la première couche cachée 
        int current_layer_length = nb_neurones_par_couches[couche];
        int next_layer_length = nb_neurones_par_couches[couche + 1];

        for (int neurone = 0; neurone < current_layer_length; neurone++){
            double erreur = 0.0;
            // l'erreur d'un neurone dépend des erreurs des neurones auxquels il est connecté dans la couche suivante
            for (int next_neurone = 0; next_neurone < next_layer_length; next_neurone++){
                int poids_index = next_neurone * current_layer_length + neurone;
                erreur += weights[couche][poids_index] * erreurs_by_layer[couche + 1][next_neurone];
            }

            erreurs_by_layer[couche][neurone] = erreur;
        }
    }

    // Étape 4 : mettre à jour les poids et les biais.
    for (int couche = 0; couche < nb_couches - 1; couche++){
        int layer_before_length = nb_neurones_par_couches[couche];
        int current_layer_length = nb_neurones_par_couches[couche + 1];
        int weight_number = layer_before_length * current_layer_length;

        update_before_layer_weights(
            couche,
            weights,
            weight_number,
            learning_rate,
            erreurs_by_layer[couche + 1],
            neurones_by_layer_array[couche],
            layer_before_length);

        update_before_layer_bias(
            couche,
            biases_array,
            current_layer_length,
            learning_rate,
            erreurs_by_layer[couche + 1]);
    }

    // Étape 5 : libérer la mémoire.

    for (int couche = 0; couche < nb_couches; couche++)
        free(erreurs_by_layer[couche]);

    free(erreurs_by_layer);

    return 0;

erreur:
    for (int couche = 0; couche < nb_couches; couche++)
        free(erreurs_by_layer[couche]);

    free(erreurs_by_layer);

    return -1;
}

/* TRAINING PHASE */
int exec_PMC(double *dataset, int dataset_length,
             int *predictions_by_categories,
             int nb_couches,
             int nb_neurones_par_couches[],
             double **weights,
             double **biases,
             int epochs,
             double learning_rate)
{
    if (dataset == NULL ||
        predictions_by_categories == NULL ||
        nb_neurones_par_couches == NULL ||
        weights == NULL ||
        biases == NULL ||
        dataset_length <= 0 ||
        nb_couches < 2 ||
        epochs <= 0 ||
        !isfinite(learning_rate) ||
        learning_rate <= 0.0)
    {
        return -1;
    }

    for (int couche = 0; couche < nb_couches; couche++)
    {
        if (nb_neurones_par_couches[couche] <= 0)
            return -1;
    }

    for (int couche = 0; couche < nb_couches - 1; couche++)
    {
        if (weights[couche] == NULL || biases[couche] == NULL)
            return -1;
    }

    int output_length = nb_neurones_par_couches[nb_couches - 1];

    for (int picture = 0; picture < dataset_length; picture++)
    {
        if (predictions_by_categories[picture] < 0 ||
            predictions_by_categories[picture] >= output_length)
        {
            return -1;
        }
    }

    // Étape 1 : allouer les neurones de chaque couche.
    double **neurones_by_layer_array = malloc(nb_couches * sizeof(double *));

    if (neurones_by_layer_array == NULL)
        return -1;

    if (init_couches_cachees_by_layer(nb_couches, nb_neurones_par_couches, neurones_by_layer_array) != 0)
    {
        free(neurones_by_layer_array);
        return -1;
    }

    int input_length = nb_neurones_par_couches[0];

    // Étape 2 : entraînement.

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int picture = 0; picture < dataset_length; picture++)
        {
            for (int neurone = 0; neurone < input_length; neurone++)
            {
                neurones_by_layer_array[0][neurone] =
                    dataset[picture * input_length + neurone];
            }

            for (int couche_index = 1; couche_index < nb_couches; couche_index++)
            {
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

    // Étape 3 : compter les bonnes prédictions après l'entraînement.

    int bonnes_predictions = 0;

    for (int picture = 0; picture < dataset_length; picture++)
    {
        for (int neurone = 0; neurone < input_length; neurone++)
        {
            neurones_by_layer_array[0][neurone] =
                dataset[picture * input_length + neurone];
        }

        for (int couche_index = 1; couche_index < nb_couches; couche_index++)
        {
            calculer_single_layer_output(
                nb_neurones_par_couches[couche_index - 1],
                nb_neurones_par_couches[couche_index],
                neurones_by_layer_array[couche_index],
                neurones_by_layer_array[couche_index - 1],
                NULL,
                weights[couche_index - 1],
                biases[couche_index - 1]);
        }

        if (find_strongest_category_by_neurone_score(
                neurones_by_layer_array[nb_couches - 1],
                output_length) == predictions_by_categories[picture])
        {
            bonnes_predictions++;
        }
    }

    // Étape 4 : libérer la mémoire.

    for (int couche = 0; couche < nb_couches; couche++)
        free(neurones_by_layer_array[couche]);

    free(neurones_by_layer_array);

    return bonnes_predictions;
}

/* SAVE PHASE */

int pmc_sauvegarder(const char *chemin, int nb_couches, int nb_neurones_par_couches[], double **weights, double **biases)
{
    if (chemin == NULL ||
        nb_neurones_par_couches == NULL ||
        weights == NULL ||
        biases == NULL ||
        nb_couches < 2)
    {
        return -1;
    }

    FILE *f = fopen(chemin, "w");

    if (f == NULL)
        return -1;

    fprintf(f, "%d\n", nb_couches);

    /* Sauvegarder la taille de chaque couche. */
    for (int couche = 0; couche < nb_couches; couche++)
    {
        if (nb_neurones_par_couches[couche] <= 0)
        {
            fclose(f);
            return -1;
        }

        fprintf(f, "%d ", nb_neurones_par_couches[couche]);
    }

    fprintf(f, "\n");

    /* Sauvegarder les poids et les biais de chaque couche. */
    for (int couche = 0; couche < nb_couches - 1; couche++)
    {

        int nb_neurones_before = nb_neurones_par_couches[couche];
        int nb_neurones_current = nb_neurones_par_couches[couche + 1];

        if (weights[couche] == NULL || biases[couche] == NULL)
        {
            fclose(f);
            return -1;
        }

        int nb_poids = nb_neurones_before * nb_neurones_current;

        for (int i = 0; i < nb_poids; i++)
        {
            fprintf(f, "%.17g ", weights[couche][i]);
        }

        fprintf(f, "\n");

        for (int i = 0; i < nb_neurones_current; i++)
        {
            fprintf(f, "%.17g ", biases[couche][i]);
        }

        fprintf(f, "\n");
    }

    if (fclose(f) != 0)
        return -1;

    return 0;
}

int pmc_charger(const char *chemin, int nb_couches, int nb_neurones_par_couches[], double **weights, double **biases)
{
    if (chemin == NULL ||
        nb_neurones_par_couches == NULL ||
        weights == NULL ||
        biases == NULL ||
        nb_couches < 2)
    {
        return -1;
    }

    FILE *f = fopen(chemin, "r");
    if (f == NULL)
        return -1;
    int nb_couches_fichier;
    if (fscanf(f, "%d", &nb_couches_fichier) != 1 || nb_couches_fichier != nb_couches)
    {
        fclose(f);
        return -1;
    }

    for (int couche = 0; couche < nb_couches; couche++)
    {
        int dimension_fichier;
        if (fscanf(f, "%d", &dimension_fichier) != 1 || dimension_fichier != nb_neurones_par_couches[couche] || dimension_fichier <= 0)
        {
            fclose(f);
            return -1;
        }
    }

    /* Charger les poids et les biais de chaque couche. */
    for (int couche = 0; couche < nb_couches - 1; couche++)
    {

        int nb_neurones_before = nb_neurones_par_couches[couche];
        int nb_neurones_current = nb_neurones_par_couches[couche + 1];

        int nb_poids = nb_neurones_before * nb_neurones_current;

        if (weights[couche] == NULL || biases[couche] == NULL)
        {
            fclose(f);
            return -1;
        }

        for (int i = 0; i < nb_poids; i++)
        {
            if (fscanf(f, "%lf", &weights[couche][i]) != 1)
            {
                fclose(f);
                return -1;
            }
        }

        for (int i = 0; i < nb_neurones_current; i++)
        {
            if (fscanf(f, "%lf", &biases[couche][i]) != 1)
            {
                fclose(f);
                return -1;
            }
        }
    }

    fclose(f);
    return 0;
}


int main(void)
{
    srand((unsigned int)time(NULL));
    int input_dimension = 12;
    int hidden_inside_dimension = 8;
    int output_dim = 3;
    int nb_couches = 3;

    int nb_neurones_par_couches[] = {
        input_dimension,
        hidden_inside_dimension,
        output_dim
    };

    int dataset_length = 3;

    double X[3][12] = {
        {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, // X[0] = premier exemple
        {0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1},
        {1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0}
    };

    int dataset_classes[3] = {2, 0, 1};

    /* Allocation des poids et des biais. */
    double **weights = calloc(nb_couches - 1, sizeof(double *));
    double **biases = calloc(nb_couches - 1, sizeof(double *));

    if (weights == NULL || biases == NULL)
    {
        free(weights);
        free(biases);
        return 1;
    }

    /* Initialisation des poids et des biais. */

    if (init_weights_and_biases_by_hidden_layers(
            nb_couches,
            nb_neurones_par_couches,
            weights,
            biases) != 0)
    {
        for (int i = 0; i < nb_couches - 1; i++)
        {
            free(weights[i]);
            free(biases[i]);
        }

        free(weights);
        free(biases);

        return 1;
    }

    /* Entraînement du perceptron multicouche. */

    int predictions = exec_PMC(
        (double *)X,
        dataset_length,
        dataset_classes,
        nb_couches,
        nb_neurones_par_couches,
        weights,
        biases,
        100,
        0.01
    );

    printf("Bonnes predictions : %d/%d\n",
           predictions < 0 ? 0 : predictions,
           dataset_length);

    /* Sauvegarde du réseau entraîné. */

    if (predictions >= 0)
    {
        if (pmc_sauvegarder(
                "pmc_weights.txt",
                nb_couches,
                nb_neurones_par_couches,
                weights,
                biases) != 0)
        {
            printf("Erreur lors de la sauvegarde du réseau.\n");
        }
        else
        {
            printf("Reseau sauvegarde avec succes.\n");
        }
    }

    /* Libération de la mémoire. */

    for (int i = 0; i < nb_couches - 1; i++)
    {
        free(weights[i]);
        free(biases[i]);
    }

    free(weights);
    free(biases);

    return predictions < 0 ? 1 : 0;
}
