#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

// ETAPE 1 : Initialiser les centroïdes de façon aléatoire.
void find_centroids(int K, int dataset_rows, int vector_dimensions, double dataset_array[dataset_rows][vector_dimensions], double centroids_array[K][vector_dimensions])
{
    // K est le nombre de centroïdes que l'on souhaite obtenir.
    // On tire les points SANS REMISE (mélange de Fisher-Yates) : si le même point était choisi deux fois,
    // deux centroïdes seraient identiques, l'un resterait vide et update_centroids() diviserait par 0 (NaN).
    int *indices = malloc(dataset_rows * sizeof(int));
    for (int i = 0; i < dataset_rows; i++)
    {
        indices[i] = i;
    }
    for (int i = 0; i < K; i++)
    {
        int r = i + rand() % (dataset_rows - i); // Le nombre aléatoire sert à choisir quel point du dataset devient le centre initial.
        int tmp = indices[i];
        indices[i] = indices[r];
        indices[r] = tmp;
        int random_centroid = indices[i];
        for (int j = 0; j < vector_dimensions; j++){
            centroids_array[i][j] = dataset_array[random_centroid][j];
        }
    }
    free(indices);
}

// ETAPE 2 : Assigner les points du dataset au centroïde le plus proche en calculant la distance euclidienne de chaque point à chacun des centroïdes.
void assign_clusters(int K, int dataset_rows, int vector_dimensions, double dataset_array[dataset_rows][vector_dimensions], double centroids_array[K][vector_dimensions], int clusters_array[K][dataset_rows], int cluster_sizes[K])
{
    for (int i = 0; i < K; i++)
    {
        cluster_sizes[i] = 0;
    }
    for (int i = 0; i < dataset_rows; i++)
    {
        int closest_centroid_index = calculer_closest_centroid(dataset_array[i], K, vector_dimensions, centroids_array);
        int dot_cluster_index = cluster_sizes[closest_centroid_index]; // On regarde combien de points sont déjà dans ce cluster
        clusters_array[closest_centroid_index][dot_cluster_index] = i; // On ajoute le point au cluster
        cluster_sizes[closest_centroid_index]++;                       // Le cluster contient maintenant un point de plus
    }
}

// ETAPE 3 : Pour chaque cluster, la moyenne de tous les points devient le nouveau centroïde.
void update_centroids(int K, int dataset_rows, int vector_dimensions, double dataset_array[dataset_rows][vector_dimensions], int clusters_array[K][dataset_rows], int cluster_sizes[K], double centroids_array[K][vector_dimensions])
{
    for (int cluster_index = 0; cluster_index < K; cluster_index++)
    { // Pour chaque cluster
        if (cluster_sizes[cluster_index] == 0)
            continue; // Cluster vide : on garde l'ancien centroïde (sinon 0/0 = NaN)

        for (int dimension = 0; dimension < vector_dimensions; dimension++)
        { // Pour chaque dimension d'un point
            double sum = 0.0;
            // Parcourir tous les points du cluster
            for (int dot_index = 0; dot_index < cluster_sizes[cluster_index]; dot_index++)
            {                                                                     // Récupérer l'indice du point dans le dataset
                int dot_index_dataset = clusters_array[cluster_index][dot_index]; // Ajouter la coordonnée du point pour cette dimension
                double dot_dimension = dataset_array[dot_index_dataset][dimension];
                sum += dot_dimension;
            }

            double mean = sum / cluster_sizes[cluster_index]; // Calculer la moyenne de cette dimension
            centroids_array[cluster_index][dimension] = mean; // Mettre à jour la coordonnée du centroïde
        }
    }
}


/* ETAPE 4 : Calculer Gamma.
   On calcule la distance moyenne entre chaque centroïde
   et son voisin le plus proche.
   Puis on convertit cette distance en Gamma :
   gamma = 1 / (2 * sigma * sigma).
*/

double calculer_gamma(
    int K,
    int vector_dimensions,
    double centroids_array[K][vector_dimensions])
{
    // Il faut au moins deux centroïdes.
    if (K < 2)
        return -1.0;

    double distances_sum = 0.0;

    for (int current_centroid = 0;
         current_centroid < K;
         current_centroid++)
    {
        double min_neighbor_distance = INFINITY;

        for (int other_centroid = 0;
             other_centroid < K;
             other_centroid++)
        {
            if (other_centroid == current_centroid)
                continue;

            double distance_carre = 0.0;

            // Calcul de la distance euclidienne au carré.
            for (int dimension = 0;
                 dimension < vector_dimensions;
                 dimension++)
            {
                double difference =
                    centroids_array[current_centroid][dimension]
                    - centroids_array[other_centroid][dimension];

                distance_carre += difference * difference;
            }

            double neighbor_distance = sqrt(distance_carre);

            if (neighbor_distance < min_neighbor_distance)
            {
                min_neighbor_distance = neighbor_distance;
            }
        }

        distances_sum += min_neighbor_distance;
    }

    // Distance moyenne entre les centroïdes et leurs voisins.
    double sigma = distances_sum / K;

    if (sigma <= 0.0)
        return -1.0;

    // Conversion de Sigma vers Gamma.
    double gamma = 1.0 / (2.0 * sigma * sigma);

    return gamma;
}

/* ETAPE 5 : Calculer la proximité d'un point
   par rapport à chaque centroïde avec RBF.
*/
void compute_dot_proximity_to_centroids(
    int vector_dimensions,
    double dot[vector_dimensions],
    int K,
    double centroids_array[K][vector_dimensions],
    double gamma,
    double dot_proximity_to_centroids[K])
{
    for (int centroid = 0; centroid < K; centroid++)
    {
        double distance_carre = 0.0;

        for (int dimension = 0;
             dimension < vector_dimensions;
             dimension++)
        {
            double difference =
                dot[dimension] - centroids_array[centroid][dimension];

            distance_carre += difference * difference;
        }

        // Application de la fonction gaussienne du cours.
        dot_proximity_to_centroids[centroid] =
            exp(-gamma * distance_carre);
    }
}


/* ETAPE 6 : Entrainer les poids rétro-activement */
void recursive_train_weights(int K, int dataset_length, int vector_dimensions, double dataset_array[dataset_length][vector_dimensions], double target_predictions[dataset_length], double centroids_array[K][vector_dimensions], double gamma, double weights[K], double learning_rate,int epochs){
    double dot_proximity_to_centroids[K];

    for (int epoch = 0; epoch < epochs; epoch++){
        double total_error = 0.0;

        for (int i = 0; i < dataset_length; i++){
            compute_dot_proximity_to_centroids(
                vector_dimensions,
                dataset_array[i],
                K,
                centroids_array,
                gamma,
                dot_proximity_to_centroids
            );

            double output = calculer_rbf_dot_output(
                K,
                dot_proximity_to_centroids,
                weights
            );

            double error = output - target_predictions[i];

            total_error += error * error;

            for (int centroid = 0; centroid < K; centroid++){
                weights[centroid] -=
                    learning_rate * 2.0 * error
                    * dot_proximity_to_centroids[centroid];
            }
        }

        double mse = total_error / dataset_length;
        printf("Epoch %d - MSE = %f\n", epoch, mse);
    }
}

void RBF(int dataset_rows, int vector_dimensions, double dataset_array[dataset_rows][vector_dimensions], double targets[dataset_rows],int cluster_size){
    
    // STEP 0 : Initialisation
    size_t dataset_size = dataset_rows;
    int K = dataset_size / cluster_size;
    size_t weight_nb = K;
    double weights[weight_nb];
    memset(weights, 0, sizeof(weights));
    double centroids_array[K][vector_dimensions];
    int clusters_array[K][dataset_rows];
    int cluster_sizes[K];

    // STEP 1 : Trouver les centroïdes avec K-means / Lloyd.
    find_centroids(K, dataset_rows, vector_dimensions, dataset_array, centroids_array);
    int max_iterations = 100;
    for (int iteration = 0; iteration < max_iterations; iteration++){
        assign_clusters(
            K, dataset_rows, vector_dimensions,
            dataset_array, centroids_array,
            clusters_array, cluster_sizes
        );
        update_centroids(
            K, dataset_rows, vector_dimensions,
            dataset_array, clusters_array,
            cluster_sizes, centroids_array
        );
    }

    // STEP 2 : Calculer Gamma.
    double gamma = calculer_gamma(
        K, vector_dimensions, centroids_array
    );
    if (gamma <= 0.0){
        printf("Erreur : impossible de calculer Gamma.\n");
        return;
    }
    printf("Gamma = %f\n", gamma);

    // STEP 3 : Apprentissage des poids.
    double learning_rate = 0.01;
    int epochs = 1000;
    recursive_train_weights(
        K,
        dataset_rows,
        vector_dimensions,
        dataset_array,
        targets,
        centroids_array,
        gamma,
        weights,
        learning_rate,
        epochs
    );

    // STEP 4 : Effectuer les prédictions.
    double dot_proximity_to_centroids[K];
    printf("\nResultats :\n");
    for (int i = 0; i < dataset_rows; i++){
        compute_dot_proximity_to_centroids(
            vector_dimensions,
            dataset_array[i],
            K,
            centroids_array,
            gamma,
            dot_proximity_to_centroids
        );
        double output = calculer_rbf_dot_output(
            K,
            dot_proximity_to_centroids,
            weights
        );
        int prediction = (output >= 0.0) ? 1 : -1;
        printf("Point %d : cible = %f, prediction = %f, classe = %d\n", i, targets[i], output, prediction);
    }
}

// Sauvegarde les centroides, le sigma et les poids appris dans un fichier texte, pour les reutiliser plus tard sans avoir a ré-entrainer (meme principe que svm_sauvegarder() dans ETAPE4.svm.c).
int rbf_sauvegarder(const char *chemin, double *centroids, double sigma, double *weights, int K, int vector_dimensions)
{
    FILE *f = fopen(chemin, "w");
    if (f == NULL)
        return -1;
    fprintf(f, "%d %d %.17g\n", K, vector_dimensions, sigma);
    for (int i = 0; i < K * vector_dimensions; i++)
        fprintf(f, "%.17g ", centroids[i]);
    fprintf(f, "\n");
    for (int i = 0; i < K; i++)
        fprintf(f, "%.17g ", weights[i]);
    fprintf(f, "\n");
    fclose(f);
    return 0;
}

int rbf_charger(const char *chemin, double *centroids, double *weights, int *K, int *vector_dimensions, double *sigma)
{
    FILE *f = fopen(chemin, "r");
    if (f == NULL)
        return -1;

    if (fscanf(f, "%d %d %lf", K, vector_dimensions, sigma) != 3)
    {
        fclose(f);
        return -1;
    }
    for (int i = 0; i < (*K) * (*vector_dimensions); i++)
        if (fscanf(f, "%lf", &centroids[i]) != 1)
        {
            fclose(f);
            return -1;
        }
    for (int i = 0; i < *K; i++)
        if (fscanf(f, "%lf", &weights[i]) != 1)
        {
            fclose(f);
            return -1;
        }

    fclose(f);
    return 0;
}
