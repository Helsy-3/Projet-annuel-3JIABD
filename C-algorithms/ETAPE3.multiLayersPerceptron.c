#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

/*  INIT PHASE */
double *init_biais(int input_length, int neurones_layer_number)
{
    int bias_number = input_length * neurones_layer_number;
    double *bias_array = malloc(bias_number * sizeof(double));
    if (bias_array == NULL)
        return NULL;

    for (int i = 0; i < bias_number; i++)
    {
        bias_array[i] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
    }
    return bias_array;
}

double *init_poids_by_layer(int input_length, int neurones_layer_number)
{
    int weight_number = input_length * neurones_layer_number;
    double *weight_array = malloc(weight_number * sizeof(double));
    if (weight_array == NULL)
        return NULL;

    for (int i = 0; i < weight_number; i++)
    {
        weight_array[i] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
    }
    return weight_array;
}

// créer une matrice lignes x colonnes pour recevoir les couches cachées et le nombre de neurones par couches. allouer de l'espace en mémoire
int init_couches_cachees_by_layer(int nb_couches, int nb_neurones_par_couches[], double **hidden_neurones_by_layer_array)
{
    for (int couche = 0; couche < nb_couches; couche++)
    {
        hidden_neurones_by_layer_array[couche] = malloc(nb_neurones_par_couches[couche] * sizeof(double));
        if (hidden_neurones_by_layer_array[couche] == NULL)
        {
            for (int couche_a_liberer = 0; couche_a_liberer < couche; couche_a_liberer++)
            {
                free(hidden_neurones_by_layer_array[couche_a_liberer]);
            }
            return -1;
        }
    }
    return 0;
}

int init_weights_and_biases_by_hidden_layers(int nb_couches, int nb_neurones_par_couche[], double **weights, double **biases)
{
    for (int couche_index = 0; couche_index < nb_couches - 1; couche_index++)
    {
        weights[couche_index] = NULL;
        biases[couche_index] = NULL;
        int nb_neurones_before_layer = nb_neurones_par_couche[couche_index];
        int nb_neurones_current_layer = nb_neurones_par_couche[couche_index + 1];
        weights[couche_index] = init_poids_by_layer(nb_neurones_before_layer, nb_neurones_current_layer);
        biases[couche_index] = init_biais_by_layer(nb_neurones_current_layer);
        if (weights[couche_index] == NULL || biases[couche_index] == NULL)
            return 1;
    }
    return 0;
}

/*  PMC - CORE FUNCTIONS  */
int find_strongest_category_by_neurone_score(double *couche_sortie, int output_length)
{
    if (couche_sortie == NULL || output_length <= 0)
        return -1;
    int promising_neurone_index = 0;

    for (int neurone = 1; neurone < output_length; neurone++)
    {
        if (couche_sortie[neurone] > couche_sortie[promising_neurone_index])
        {
            promising_neurone_index = neurone;
        }
    }
    return promising_neurone_index;
}

void calculer_single_layer_output(int layer_before_length, int current_layer_length, double *current_layer_neurones, double *neurone_layer_before, double *output_layer, double *before_layer_weight, double *before_layer_bias)
{

    for (int neurone_index = 0; neurone_index < current_layer_length; neurone_index++)
    {
        double current_neurone_output = before_layer_bias[neurone_index];

        // un neurone reçoit la somme pondérée de tous les autres neurones multipliés par leur poids
        for (int neurone_before_index = 0; neurone_before_index < layer_before_length; neurone_before_index++)
        {
            int poids_index = neurone_index * layer_before_length + neurone_before_index;
            current_neurone_output += neurone_layer_before[neurone_before_index] * before_layer_weight[poids_index];
        }
        current_layer_neurones[neurone_index] = current_neurone_output;
        if (output_layer != NULL)
        {
            output_layer[neurone_index] = current_neurone_output;
        }
    }
}

double calculer_derivee_partielle(double neurone_layer_before)
{
    return neurone_layer_before;
}

void update_before_layer_weights(int before_layer_index, double **all_weights, int nb_poids, double learning_rate, double *error_by_neurone_array, double *neurones_layer_before, int layer_before_length)
{
    for (int neurone = 0; neurone < nb_poids; neurone++)
    {
        int neurone_index = neurone / layer_before_length;
        int before_neurone_index = neurone % layer_before_length;
        double before_weight = all_weights[before_layer_index][neurone];
        double derivee_partielle = calculer_derivee_partielle(neurones_layer_before[before_neurone_index]);
        double new_before_weight = before_weight - learning_rate * error_by_neurone_array[neurone_index] * derivee_partielle;
        all_weights[before_layer_index][neurone] = new_before_weight;
    }
}

void update_before_layer_bias(int before_layer_index, double **all_biases, int nb_biases, double learning_rate, double *erreurs_by_layer)
{
    for (int i = 0; i < nb_biases; i++)
    {
        all_biases[before_layer_index][i] -= learning_rate * erreurs_by_layer[i];
    }
}

int backpropagation_to_every_layer(double **neurones_by_layer_array, int nb_couches, int nb_neurones_par_couches[], double **weights, double **biases_array, int target_class, double learning_rate)
{

    if (neurones_by_layer_array == NULL || nb_neurones_par_couches == NULL || weights == NULL || biases_array == NULL || nb_couches < 2 || learning_rate <= 0.0)
    {
        return -1;
    }
    int output_length = nb_neurones_par_couches[nb_couches - 1];
    if (target_class < 0 || target_class >= output_length)
    {
        return -1;
    }

    double **erreurs_by_layer = malloc(nb_couches * sizeof(double *));
    if (erreurs_by_layer == NULL)
        return -1;
    for (int couche = 0; couche < nb_couches; couche++)
    {
        erreurs_by_layer[couche] = NULL;
    }

    for (int couche = 1; couche < nb_couches; couche++)
    {
        erreurs_by_layer[couche] = malloc(nb_neurones_par_couches[couche] * sizeof(double));

        if (erreurs_by_layer[couche] == NULL)
        {
            for (int couche_a_liberer = 0;
                 couche_a_liberer < nb_couches;
                 couche_a_liberer++)
            {
                free(erreurs_by_layer[couche_a_liberer]);
            }

            free(erreurs_by_layer);
            return -1;
        }
    }

    /* Étape 2 : Calculer l'erreur de la couche de sortie. */
    int derniere_couche = nb_couches - 1;
    for (int neurone = 0; neurone < output_length; neurone++)
    {
        double cible = (neurone == target_class) ? 1.0 : 0.0;
        erreurs_by_layer[derniere_couche][neurone] = neurones_by_layer_array[derniere_couche][neurone] - cible;
    }

    /* Étape 3 : Propager les erreurs vers les couches cachées. À chaque itération, on calcule l'erreur de la couche actuelle (current_couche)
    en utilisant l'erreur de la couche suivante (current_couche + 1). */

    for (int current_couche = nb_couches - 2; current_couche >= 1; current_couche--)
    {
        int current_layer_length = nb_neurones_par_couches[current_couche];
        int next_layer_length = nb_neurones_par_couches[current_couche + 1];
        for (int neurone = 0; neurone < current_layer_length; neurone++)
        {
            double current_layer_error = 0.0;
            for (int neurone_suivant = 0; neurone_suivant < next_layer_length; neurone_suivant++)
            {
                int poids_index = neurone_suivant * current_layer_length + neurone;
                int next_layer = current_couche + 1;
                current_layer_error += weights[current_couche][poids_index] * erreurs_by_layer[next_layer][neurone_suivant];
            }

            erreurs_by_layer[current_couche][neurone] = current_layer_error;
        }
    }

    /* Étape 4 : mettre à jour les poids et les biais. 
    On commence de la couche de départ.
    */
    for (int before_layer = 0; before_layer < nb_couches - 1; before_layer++)
    {
        int next_layer_index = before_layer +1;
        int before_layer_index = before_layer;
        int layer_before_length = nb_neurones_par_couches[before_layer];
        int current_layer_length = nb_neurones_par_couches[next_layer_index];
        int nb_poids = layer_before_length * current_layer_length;

        /* Mise à jour de tous les poids entre les deux couches à l'aide de l'erreur de la couche d'arrivée (et non l'erreur de la couche de départ). */
        update_before_layer_weights(before_layer_index, weights, nb_poids, learning_rate, erreurs_by_layer[next_layer_index], neurones_by_layer_array[next_layer], layer_before_length);

        /* Mise à jour des biais entre les deux couches à l'aide de l'erreur de la couche d'arrivée (et non l'erreur de la couche de départ). */
        update_before_layer_bias(before_layer_index, biases_array, current_layer_length, learning_rate, erreurs_by_layer[next_layer]);
    }

    /* Étape 5 : libérer la mémoire des erreurs. */
    for (int couche = 0; couche < nb_couches; couche++)
    {
        free(erreurs_by_layer[couche]);
    }

    free(erreurs_by_layer);

    return 0;
}

int exec_PMC(double *dataset, int dataset_length, int *predictions_by_categories, int nb_couches, int nb_neurones_par_couches[], double **weights, double **biases, int epochs, double learning_rate)
{

    if (dataset == NULL || predictions_by_categories == NULL || nb_neurones_par_couches == NULL || weights == NULL || biases == NULL || dataset_length <= 0 || nb_couches < 2 || epochs <= 0 || learning_rate <= 0.0)
    {
        return -1;
    }

    /* Étape 1 : initialiser les poids et les biais. */
    for (int couche_index = 0; couche_index < nb_couches - 1; couche_index++)
    {
        int nb_neurones_before = nb_neurones_par_couches[couche_index];
        int nb_neurones_current = nb_neurones_par_couches[couche_index + 1];

        if (nb_neurones_before <= 0 || nb_neurones_current <= 0 || weights[couche_index] == NULL || biases[couche_index] == NULL)
        {
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

    if (init_couches_cachees(nb_couches, nb_neurones_par_couches, neurones_by_layer_array) != 0)
    {
        free(neurones_by_layer_array);
        return -1;
    }

    int input_length = nb_neurones_par_couches[0];
    int output_length = nb_neurones_par_couches[nb_couches - 1];

    /* Étape 3 : entraînement. */
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

            if (backpropagation_to_every_layer(neurones_by_layer_array, nb_couches, nb_neurones_par_couches, weights, biases, predictions_by_categories[picture], learning_rate) != 0)
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

    for (int picture = 0; picture < dataset_length; picture++)
    {
        for (int neurone = 0; neurone < input_length; neurone++)
        {
            neurones_by_layer_array[0][neurone] =
                dataset[picture * input_length + neurone];
        }

        for (int couche_index = 1; couche_index < nb_couches; couche_index++)
        {
            calculer_single_layer_output(nb_neurones_par_couches[couche_index - 1], nb_neurones_par_couches[couche_index], neurones_by_layer_array[couche_index], neurones_by_layer_array[couche_index - 1], NULL, weights[couche_index - 1], biases[couche_index - 1]);
        }

        if (find_strongest_category_by_neurone_score(neurones_by_layer_array[nb_couches - 1], output_length) == predictions_by_categories[picture])
        {
            bonnes_predictions++;
        }
    }

    /* Étape 5 : libérer la mémoire. */
    for (int couche = 0; couche < nb_couches; couche++)
        free(neurones_by_layer_array[couche]);

    free(neurones_by_layer_array);

    return bonnes_predictions;
}

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

    if (fscanf(f, "%d", &nb_couches_fichier) != 1 ||
        nb_couches_fichier != nb_couches)
    {
        fclose(f);
        return -1;
    }

    /* Vérifier les dimensions de toutes les couches. */
    for (int couche = 0; couche < nb_couches; couche++)
    {

        int dimension_fichier;

        if (fscanf(f, "%d", &dimension_fichier) != 1 ||
            dimension_fichier != nb_neurones_par_couches[couche] ||
            dimension_fichier <= 0)
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
    int input_dimension = 12;
    int hidden_inside_dimension = 8;
    int output_dim = 3;
    int dataset_length = 3;

    double X[3][12] = {
        {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1},
        {1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0}};

    int vraies_classes[3] = {2, 0, 1};

    double W1[12 * 8], b1[8], W2[8 * 3], b2[3];

    int bonnes = run_pmc_dataset(
        (double *)X, vraies_classes, dataset_length,
        input_dimension, hidden_inside_dimension, output_dim,
        W1, b1, W2, b2);

    printf("Bonnes predictions : %d/%d\n", bonnes, dataset_length);

    return 0;
}
