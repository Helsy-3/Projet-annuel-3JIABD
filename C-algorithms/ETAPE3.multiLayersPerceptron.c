#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utils.h"

/*  INIT PHASE */
double *init_biais_by_layer(int nb_neurones_current_layer){
    if (nb_neurones_current_layer <= 0)
        return NULL;

    double *bias_array =
        malloc(nb_neurones_current_layer * sizeof(double));

    if (bias_array == NULL)
        return NULL;

    for (int i = 0; i < nb_neurones_current_layer; i++)
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

int init_weights_and_biases_by_hidden_layers(int nb_couches, int nb_neurones_par_couche[], double **weights, double **biases){
    if (nb_couches < 2 || nb_neurones_par_couche == NULL ||    weights == NULL || biases == NULL)
        return -1;

    for (int couche = 0; couche < nb_couches - 1; couche++){
        weights[couche] = NULL;
        biases[couche] = NULL;

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

// PMC - CORE FUNCTIONS  
int find_strongest_category_by_neurone_score(double *couche_sortie, int output_length){
    if (couche_sortie == NULL || output_length <= 0)
        return -1;
    int promising_neurone_index = 0;

    for (int neurone_index = 1; neurone_index < output_length; neurone_index++)
    {
        if (couche_sortie[neurone_index] > couche_sortie[promising_neurone_index])
        {
            promising_neurone_index = neurone_index;
        }
    }
    return promising_neurone_index;
}

void calculer_single_layer_output(int layer_before_length, int current_layer_length, double *current_layer_neurones, double *neurone_layer_before, double *output_layer, double *before_layer_weight, double *before_layer_bias){
    for (int neurone_index = 0; neurone_index < current_layer_length; neurone_index++){
        double current_neurone_output = before_layer_bias[neurone_index];

        // un neurone reçoit la somme pondérée de tous les autres neurones multipliés par leur poids
        for (int neurone_before_index = 0; neurone_before_index < layer_before_length; neurone_before_index++){
            int poids_index = neurone_index * layer_before_length + neurone_before_index;
            current_neurone_output += neurone_layer_before[neurone_before_index] * before_layer_weight[poids_index];
        }
        current_layer_neurones[neurone_index] = current_neurone_output;
        if (output_layer != NULL){
            output_layer[neurone_index] = current_neurone_output;
        }
    }
}

// ETAPE 1 : calculer la loss (écart entre la prédiction du réseau et la valeur attendue).
// Formule = 1/2 * (prediction - cible)^2

double calculer_loss(double prediction, double cible){
    double difference = prediction - cible;
    double loss = 0.5 * difference * difference;
    return loss;
}

// ETAPE 2 : calculer la dérivée partielle de la loss par rapport à un poids.
// next_neurone = poids * neurone_before + biais
// On veut calculer : dérivée de la loss / dérivée du poids. Pour cela, on utilise la règle de dérivation en chaîne :
// dérivée de la loss par rapport au poids =  dérivée de la loss par rapport au résultat du neurone * dérivée du résultat du neurone par rapport au poids
// Or, dans notre calcul : dérivée du résultat du neurone par rapport au poids = activation_precedente
// On obtient donc : gradient = dérivée_loss_par_rapport_au_resultat_neurone * activation_precedente
// ATTENTION : la première valeur passée à cette fonction doit déjà être la dérivée de la loss par rapport au résultat du neurone. Ce n'est pas nécessairement la loss elle-même.

double calculer_derivee_partielle_loss_par_rapport_au_poids(double derivee_L, double neurone_before) {
    
    // ETAPE 2A : calculer l'influence du poids sur le résultat du neurone.
    // ^y = next_neurone = poids * neurone_before + biais
    // ^y = w * x + b
    // la dérivée de w * x par rapport à w vaut x, car x est considéré comme une constante.
    // la dérivée de b par rapport à w vaut 
    // Donc dérivée(y^) = x
    // Donc dérivée (y¨) = neurone_before. 
    double derivee_y_ = neurone_before;

    // ETAPE 2B : calculer le gradient en appliquant la règle de dérivation en chaîne.
    // Le gradient indique comment la loss varie localement lorsque l'on modifie ce poids.
    double gradient = derivee_L * derivee_y_;
    return gradient;
}

void update_before_layer_weights(int before_layer_index, double **weights_matrice, int nb_poids, double learning_rate, double *derivees_loss_sortie_neurones, double *neurones_layer_before, int layer_before_length){
    
    for (int poids = 0; poids < nb_poids; poids++){
        int current_neurone_index = poids / layer_before_length;
        int before_neurone_index = poids % layer_before_length;
        // la dérivée de la Loss vaut = y^ - y

        double derivee_L = derivees_loss_sortie_neurones[current_neurone_index];
        double neurone_before = neurones_layer_before[before_neurone_index];

        double gradient = calculer_derivee_partielle_loss_par_rapport_au_poids(derivee_L, neurone_before);
        weights_matrice[before_layer_index][poids] -= learning_rate * gradient;
    }

}

void update_before_layer_bias(int before_layer_index, double **all_biases, int nb_biases, double learning_rate, double *erreurs_by_layer){
    for (int i = 0; i < nb_biases; i++){
        all_biases[before_layer_index][i] -= learning_rate * erreurs_by_layer[i];
    }
}

int backpropagation_to_every_layer(double **neurones_by_layer_array,int nb_couches,int nb_neurones_par_couches[],double **weights, double **biases_array, int target_class,double learning_rate){
    if (neurones_by_layer_array == NULL || nb_neurones_par_couches == NULL || weights == NULL || biases_array == NULL || nb_couches < 2 || !isfinite(learning_rate) || learning_rate <= 0.0){
        return -1;
    }
    int output_layer = nb_couches - 1;
    int output_length = nb_neurones_par_couches[output_layer];
    if (target_class < 0 || target_class >= output_length)
        return -1;

    // Étape 1 : allouer un tableau d'erreurs par couche.
    // La couche d'entrée n'a pas besoin d'un tableau d'erreurs pour mettre à jour les poids.
    double **erreurs_by_layer = calloc(nb_couches, sizeof(double *));
    if (erreurs_by_layer == NULL)
        return -1;
    for (int couche = 1; couche < nb_couches; couche++){
        if (nb_neurones_par_couches[couche] <= 0)
            goto erreur;
        erreurs_by_layer[couche] = malloc(nb_neurones_par_couches[couche] * sizeof(double));
        if (erreurs_by_layer[couche] == NULL)
            goto erreur;
    }
    
    // Étape 2 : calculer l'erreur de sortie. Loss = 1/2 * (prediction - cible)^2
    // dLoss/dPrediction = prediction - cible. Cette formule suppose une sortie linéaire.
    for (int neurone = 0; neurone < output_length; neurone++){
        double cible = (neurone == target_class) ? 1.0 : 0.0;
        erreurs_by_layer[output_layer][neurone] = neurones_by_layer_array[output_layer][neurone] - cible;
    }
    
    // Étape 3 : propager les erreurs vers les couches cachées.
    // delta[j] = somme_k (poids[j -> k] * delta_suivant[k]). Le réseau étant linéaire, la dérivée de l'activation vaut 1. 
    for (int couche = nb_couches - 2; couche >= 1; couche--){
        int current_layer_length = nb_neurones_par_couches[couche];
        int next_layer_length = nb_neurones_par_couches[couche + 1];

        for (int neurone = 0; neurone < current_layer_length; neurone++){
            double erreur = 0.0;
            for (int next_neurone = 0; next_neurone < next_layer_length; next_neurone++){
                // Les poids sont indexés ainsi : index = neurone_destination * taille_source + neurone_source
                int poids_index = next_neurone * current_layer_length + neurone;
                erreur += weights[couche][poids_index] * erreurs_by_layer[couche + 1][next_neurone];
            }
            erreurs_by_layer[couche][neurone] = erreur;
        }
    }

    
     // Étape 4 : mettre à jour les poids et les biais.
     // Important : toutes les erreurs ont été calculées AVANT de modifier les poids. On peut donc mettre à jour ici.
    for (int couche = 0; couche < nb_couches - 1; couche++){
        int layer_before_length = nb_neurones_par_couches[couche];
        int current_layer_length = nb_neurones_par_couches[couche + 1];
        int nb_poids = layer_before_length * current_layer_length;
        if (weights[couche] == NULL || biases_array[couche] == NULL)
            goto erreur;

        update_before_layer_weights(couche, weights, nb_poids, learning_rate, erreurs_by_layer[couche + 1], neurones_by_layer_array[couche], layer_before_length);
        update_before_layer_bias(couche,biases_array,current_layer_length,learning_rate,erreurs_by_layer[couche + 1]);
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

int exec_PMC(double *dataset, int dataset_length, int *predictions_by_categories, int nb_couches, int nb_neurones_par_couches[], double **weights, double **biases, int epochs, double learning_rate){
    if (dataset == NULL || predictions_by_categories == NULL || nb_neurones_par_couches == NULL || weights == NULL || biases == NULL || dataset_length <= 0 || nb_couches < 2 || epochs <= 0 || learning_rate <= 0.0){
        return -1;
    }

    /* Étape 1 : initialiser les poids et les biais. */
    for (int couche_index = 0; couche_index < nb_couches - 1; couche_index++){
        int nb_neurones_before = nb_neurones_par_couches[couche_index];
        int nb_neurones_current = nb_neurones_par_couches[couche_index + 1];

        if (nb_neurones_before <= 0 || nb_neurones_current <= 0 || weights[couche_index] == NULL || biases[couche_index] == NULL){
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
        for (int picture = 0; picture < dataset_length; picture++){
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

            if (backpropagation_to_every_layer(neurones_by_layer_array, nb_couches, nb_neurones_par_couches, weights, biases, predictions_by_categories[picture], learning_rate) != 0){
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

int pmc_charger(const char *chemin, int nb_couches, int nb_neurones_par_couches[], double **weights, double **biases){
    if (chemin == NULL ||
        nb_neurones_par_couches == NULL ||
        weights == NULL ||
        biases == NULL ||
        nb_couches < 2){
        return -1;
    }

    FILE *f = fopen(chemin, "r");
    if (f == NULL)
        return -1;
    int nb_couches_fichier;
    if (fscanf(f, "%d", &nb_couches_fichier) != 1 || nb_couches_fichier != nb_couches){
        fclose(f);
        return -1;
    }

    for (int couche = 0; couche < nb_couches; couche++){
        int dimension_fichier;
        if (fscanf(f, "%d", &dimension_fichier) != 1 || dimension_fichier != nb_neurones_par_couches[couche] || dimension_fichier <= 0){
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


int main(void){
    int input_dimension = 12; // Nombre de neurones dans la couche d'entrée.
    int hidden_inside_dimension = 8;  // Nombre de neurones dans la couche cachée.
    int output_dim = 3;    // Nombre de neurones dans la couche de sortie (3 classes).
    int nb_couches = 3;  // Nombre total de couches : entrée, cachée et sortie.
    
    // Tableau qui indique le nombre de neurones de chaque couche. Couche 0 : 12 neurones ; couche 1 : 8 ; couche 2 : 3.
    int nb_neurones_par_couches[] = {
        input_dimension,
        hidden_inside_dimension,
        output_dim
    };

    int dataset_length = 3;    // Nombre d'exemples présents dans le dataset.
    double X[3][12] = {
        {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1},
        {0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1},
        {1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0}
    };
    int dataset_classes[3] = {2, 0, 1};  // Classe correcte associée à chaque exemple du dataset. Le premier exemple appartient à la classe 2, le deuxième à la classe 0, et le troisième à la classe 1.

    double **weights = malloc((nb_couches - 1) * sizeof(double *));
    double **biases = malloc((nb_couches - 1) * sizeof(double *));

    if (weights == NULL || biases == NULL) {
        free(weights);
        free(biases);
        return 1;
    }

    // Alloue et initialise les poids et les biais de chaque connexion. La fonction utilise les dimensions des couches pour calculer les tailles.
    if (init_weights_and_biases_by_hidden_layers(
            nb_couches,                  // Nombre total de couches.
            nb_neurones_par_couches,      // Nombre de neurones dans chaque couche.
            weights,                      // Tableaux destinés à recevoir les poids.
            biases) != 0) {               // Tableaux destinés à recevoir les biais.

        for (int i = 0; i < nb_couches - 1; i++) {
            free(weights[i]);
            free(biases[i]);
        }
        free(weights);
        free(biases);
        return 1;
    }

    // Lance l'entraînement du perceptron multicouche et évalue ses prédictions.
    int predictions = exec_PMC(
        (double *)X,                 // Exemples d'entrée, convertis en tableau 1D.
        dataset_length,               // Nombre d'exemples à traiter.
        dataset_classes,               // Classes attendues pour les exemples.
        nb_couches,                   // Nombre total de couches du réseau.
        nb_neurones_par_couches,       // Architecture du réseau : 12, 8, 3.
        weights,                      // Poids qui seront utilisés et modifiés.
        biases,                       // Biais qui seront utilisés et modifiés.
        100,                          // Nombre d'époques d'entraînement.
        0.01                          // Taux d'apprentissage.
    );

    // Affiche le nombre de prédictions correctes parmi les 3 exemples.
    printf("Bonnes predictions : %d/%d\n", predictions < 0 ? 0 : predictions, dataset_length);          

    // Parcourt les connexions pour libérer les tableaux de poids et de biais.
    for (int i = 0; i < nb_couches - 1; i++) {
        free(weights[i]);
        free(biases[i]);
    }
    free(weights);
    free(biases);

    // Termine le programme : 0 signifie succès, 1 signifie erreur.
    return predictions < 0 ? 1 : 0;
}
