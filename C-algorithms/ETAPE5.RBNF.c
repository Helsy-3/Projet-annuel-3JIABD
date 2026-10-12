#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

// ETAPE 1 : Initialiser les centroïdes de façon aléatoire.
void find_centroids(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double centroids_array[K][dataset_columns])
{
    // K est le nombre de centroïdes que l'on souhaite obtenir.
    for (int i = 0; i < K; i++)
    {
        int random_centroid = rand() % dataset_rows; // Le nombre aléatoire sert à choisir quel point du dataset devient le centre initial.
        for (int j = 0; j < dataset_columns; j++)
        {
            centroids_array[i][j] = dataset_array[random_centroid][j];
        }
    }
}

// ETAPE 2 : Assigner les points du dataset au centroïde le plus proche.
// en calculant la distance euclidienne de chaque point à chacun des centroïdes.
void assign_clusters(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double centroids_array[K][dataset_columns], int clusters_array[K][dataset_rows], int cluster_sizes[K]){
    for (int i = 0; i < K; i++){
        cluster_sizes[i] = 0;
    }
    for (int i = 0; i < dataset_rows; i++){
        int closest_centroid = calculer_closest_centroid(dataset_array[i], K, dataset_columns, centroids_array);
        int position = cluster_sizes[closest_centroid]; // On regarde combien de points sont déjà dans ce cluster
        clusters_array[closest_centroid][position] = i; // On ajoute le point au cluster
        cluster_sizes[closest_centroid]++;              // Le cluster contient maintenant un point de plus
    }
}

// ETAPE 3 : Mettre à jour les centroïdes.
// Pour chaque cluster, on calcule la moyenne de tous les points. 
// Cette moyenne devient le nouveau centroïde.
void update_centroids_mean(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], int clusters_array[K][dataset_rows], int cluster_sizes[K], double centroids_array[K][dataset_columns])
{
    // int clusters_array[K][dataset_rows] : taille totale K, dataset_rows: nombre maximal de points que chaque cluster peut contenir
    int clusters_number = K;
    for (int i = 0; i < clusters_number; i++){ // on calcule la moyenne du cluster à l'index i
        for (int a = 0; a < dataset_columns; a++){ // on parcourt en largeur chaque dimension des vecteurs qui constituent le cluster
            double sum = 0.0;
            for (int j = 0; j < cluster_sizes[i]; j++){
                int cluster_dot = clusters_array[i][j]; // on récupère le point du cluster, point par point
                int dot_index = cluster_dot;
                sum += dataset_array[dot_index][a]; // on additionne le vecteur d'index i du point
            }
            double average = sum / cluster_sizes[i]; // pour chaque dimension du vecteur, on calcule la moyenne
            centroids_array[i][a] = average;         // on met à jour le centre à l'index K avec la nouvelle valeur pour sa dimension.
        }
    }
}

// ETAPE 4: calculer les sigma. Le sigma, basé sur la distance moyenne entre les centroïdes, sert de largeur aux fonctions gaussiennes.
// Sigma représente la distance moyenne entre toutes les pairs de centroïdes.
double calculer_sigma(int K, int dataset_columns, double centroids_array[K][dataset_columns]){
    double sum_distance = 0.0;
    int number_of_distances = 0;
    for (int i = 0; i < K; i++){
        for (int j = i + 1; j < K; j++){
            double squared_distance = 0.0;
            for (int vecteur_index = 0; vecteur_index < dataset_columns; vecteur_index++){
                double difference = centroids_array[i][vecteur_index] - centroids_array[j][vecteur_index];
                squared_distance += difference * difference;
            }
            double distance = sqrt(squared_distance);
            sum_distance += distance;
            number_of_distances++;
        }
    }
    double sigma = sum_distance / number_of_distances;
    return sigma;
}

// ETAPE 5 : Calculer la distance d'un point par rapport au centre avec RBF
void calculer_rbf_closeness_to_centroid(int dataset_columns, double dot[dataset_columns], int K, double centroids_array[K][dataset_columns], double sigma, double rbf_closeness[K]){
    for (int i = 0; i < K; i++){ // On regarde chaque centroïde
        double distance_carre = 0;
        for (int j = 0; j < dataset_columns; j++){
            double difference = dot[j] - centroids_array[i][j]; // On calcule la distance entre le point et le centroïde
            distance_carre += difference * difference;
        }
        double distance = sqrt(distance_carre);           // On obtient la distance
        rbf_closeness[i] = rbf_distance(distance, sigma); // On transforme la distance en activation RBF
    }
}

// ETAPE 6 : Entrainer les poids rétro-activement
// La fonction train_weights() va modifier ces poids petit à petit pour que le réseau fasse de meilleures prédictions.
void train_weights(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double targets[dataset_rows], double centroids_array[K][dataset_columns], double sigma, double weights[K], double learning_rate, int epochs){
    double rbf_closeness[K]; // rbf_closeness est réécrit pour chaque point
    for (int epoch = 0; epoch < epochs; epoch++){
        double total_error = 0.0;
        for (int i = 0; i < dataset_rows; i++){
            calculer_rbf_closeness_to_centroid(dataset_columns, dataset_array[i], K, centroids_array, sigma, rbf_closeness);
            double output = calculer_output(K, rbf_closeness, weights);
            double error = targets[i] - output;
            total_error += error * error;
            for (int j = 0; j < K; j++){
                weights[j] += learning_rate * error * rbf_closeness[j];
            }
        }
        double mse = total_error / dataset_rows; // Erreur moyenne
        printf("Epoch %d - MSE = %f\n", epoch, mse);
    }
}

void RBNF(int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double targets[dataset_rows], int cluster_size){
    size_t dataset_size = dataset_rows;
    size_t weight_nb = dataset_columns;
    double weights[weight_nb];
    memset(weights, 0, sizeof(weights)); // on initialise le tableau de poids à 0.

    // COUCHE 1

    // COUCHE 2

    int K = dataset_size / cluster_size;  // K = nombre de clusters. 
    double centroids_array[K][dataset_columns];   // centroids_array = K centroids, then for each point in the centroïd vector, we use dataset_columns dimensions
    int clusters_array[K][dataset_rows]; // NOTE:  clusters_array ne stocke pas les vecteurs. Il stocke seulement les indices des vecteurs.
    int cluster_sizes[K]; // cluster_sizes = nombre de points dans chaque cluster. Par exemple, si K = 3, alors cluster_sizes pourrait être : [4 | 2 | 3]. Ce qui signifie : cluster 0 contient 4 points, cluster 1 contient 2 points, cluster 2 contient 3 points

    // step 1: on trouve les centroïds à partir des données du dataset (dataset_array)
    find_centroids(K, dataset_rows, dataset_columns, dataset_array, centroids_array);

    // step 2: K-MEANS. 
    // On stabilise les centroïds sur plusieurs itérations. Le modèle trouve lui même les similitudes entre les données et les groupe en k groupes distincts.
    int max_iterations = 100;
    for (int iteration = 0; iteration < max_iterations; iteration++){
        
        // Mettre chaque point dans son cluster
        assign_clusters(K, dataset_rows, dataset_columns, dataset_array, centroids_array, clusters_array, cluster_sizes);

        // Recalculer les centroïdes
        update_centroids_mean(K, dataset_rows, dataset_columns, dataset_array, clusters_array, cluster_sizes, centroids_array);
    }


    // step 3: RBF

    // Calcul de sigma
    double sigma = compute_sigma(K, dataset_columns, centroids_array);

    // apprentissage des poids
    double learning_rate = 0.01;
    int epochs = 1000;
    train_weights(K, dataset_rows, dataset_columns, dataset_array, targets, centroids_array, sigma, weights, learning_rate, epochs);

    double dot_proximity_to_centroids[K]; //ce tableau contient, pour chaque centroïde, une valeur qui indique à quel point le point étudié lui ressemble ou en est proche.
    printf("\nResultats :\n");
    for (int i = 0; i < dataset_rows; i++){
        calculer_rbf_closeness_to_centroid(K, dataset_columns, dataset_array[i], centroids_array, sigma, dot_proximity_to_centroids); 
        double output = compute_output(K, dot_proximity_to_centroids, weights);
        printf("Point %d : cible = %f, prediction = %f\n", i, targets[i], output);
    }
}
