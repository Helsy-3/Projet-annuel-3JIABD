#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "utils.h"

// ETAPE 1 : Initialiser les centroïdes de façon aléatoire.
void find_centroids(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double centroids_array[K][dataset_columns])
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
        for (int j = 0; j < dataset_columns; j++)
        {
            centroids_array[i][j] = dataset_array[random_centroid][j];
        }
    }
    free(indices);
}

// ETAPE 2 : Assigner les points du dataset au centroïde le plus proche en calculant la distance euclidienne de chaque point à chacun des centroïdes.
void assign_clusters(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double centroids_array[K][dataset_columns], int clusters_array[K][dataset_rows], int cluster_sizes[K]){
    for (int i = 0; i < K; i++){
        cluster_sizes[i] = 0;
    }
    for (int i = 0; i < dataset_rows; i++){
        int closest_centroid_index = calculer_closest_centroid(dataset_array[i], K, dataset_columns, centroids_array);
        int cluster_dot_index = cluster_sizes[closest_centroid_index]; // On regarde combien de points sont déjà dans ce cluster
        clusters_array[closest_centroid_index][cluster_dot_index] = i; // On ajoute le point au cluster
        cluster_sizes[closest_centroid_index]++;              // Le cluster contient maintenant un point de plus
    }
}

// ETAPE 3 : Mettre à jour les centroïdes. Pour chaque cluster, on calcule la moyenne de tous les points. 
// Cette moyenne devient le nouveau centroïde.
void update_centroids(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], int clusters_array[K][dataset_rows], int cluster_sizes[K], double centroids_array[K][dataset_columns]) {
    for (int cluster_index = 0; cluster_index < K; cluster_index++) {// Pour chaque cluster
        if (cluster_sizes[cluster_index] == 0) continue; // Cluster vide : on garde l'ancien centroïde (sinon 0/0 = NaN)
        for (int vector_index = 0; vector_index < dataset_columns; vector_index++) { // Pour chaque dimension d'un point
            double sum = 0.0;
            // Parcourir tous les points du cluster
            for (int dot_index = 0; dot_index < cluster_sizes[cluster_index]; dot_index++) {  // Récupérer l'indice du point dans le dataset
                int dot_index_dataset = clusters_array[cluster_index][dot_index];  // Ajouter la coordonnée du point pour cette dimension
                double dot_dimension = dataset_array[dot_index_dataset][vector_index];
                sum += dot_dimension;
            }
            double mean = sum / cluster_sizes[cluster_index]; // Calculer la moyenne de cette dimension
            centroids_array[cluster_index][vector_index] = mean;   // Mettre à jour la coordonnée du centroïde
        }
    }
}

// ETAPE 4: Le sigma sert de largeur aux fonctions gaussiennes.
// Sigma représente la distance moyenne entre chaque centroïde et son voisin le PLUS PROCHE.
 double calculer_sigma(int K, int dataset_columns, double centroids_array[K][dataset_columns]){
    if (K < 2)
        return 1.0; // un seul centroïde : pas de voisin, on prend une largeur par défaut

    double sum_distance = 0.0; // somme des distances de chaque centroïde à son plus proche voisin

    for (int current_centroïd_index = 0; current_centroïd_index < K; current_centroïd_index++){ // on parcourt tous les centroïdes
        double min_distance = INFINITY;
        for (int other_centroïd_index = 0; other_centroïd_index < K; other_centroïd_index++){
            if (other_centroïd_index == current_centroïd_index) continue;
            double squared_distance = 0.0;
            for (int vecteur_dimension = 0; vecteur_dimension < dataset_columns; vecteur_dimension++){ // parcourir toutes les dimensions des centroïdes
                double difference = centroids_array[current_centroïd_index][vecteur_dimension] - centroids_array[other_centroïd_index][vecteur_dimension]; // Calculer la différence entre les coordonnées des deux centroïdes pour la dimension actuelle
                squared_distance += difference * difference; // Élever cette différence au carré et l'ajouter à la somme des carrés des différences
            }
            double distance_to_centroid = sqrt(squared_distance); // Calculer la distance euclidienne entre les deux centroïdes
            if (distance_to_centroid < min_distance)
                min_distance = distance_to_centroid; // On garde la plus petite distance : celle du plus proche voisin
        }
        sum_distance += min_distance;
    }
    double sigma = sum_distance / K;
    return sigma;
}

// ETAPE 5 : Calculer la distance d'un point par rapport au centre avec RBF
void calculer_rbf_closeness_to_centroid(int dataset_columns, double dot[dataset_columns], int K, double centroids_array[K][dataset_columns], double sigma, double dot_proximity_to_centroids[K]){
    for (int i = 0; i < K; i++){ // On parcourt tous les centroïdes
        double distance_carre = 0;
        for (int dimension = 0; dimension < dataset_columns; dimension++){
            double difference = dot[dimension] - centroids_array[i][dimension]; // Pour chaque centroïde, on calcule la distance entre le point et le centroïde
            distance_carre += difference * difference;
        }
        double distance = sqrt(distance_carre); // On obtient la distance
        dot_proximity_to_centroids[i] = rbf_distance(distance, sigma); // On transforme la distance en activation RBF
    }
}

// ETAPE 6 : Entrainer les poids rétro-activement
// La fonction train_weights() va modifier ces poids petit à petit pour que le réseau fasse de meilleures prédictions.
void recursive_train_weights(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double targets[dataset_rows], double centroids_array[K][dataset_columns], double sigma, double weights[K], double learning_rate, int epochs){
    double dot_proximity_to_centroids[K]; // rbf_closeness est réécrit pour chaque point
    for (int epoch = 0; epoch < epochs; epoch++){
        double total_error = 0.0;
        for (int i = 0; i < dataset_rows; i++){ // pour un point dans le dataset (dataset_array[i], on calcule sa proximité avec les centroides, puis on l'utilise pour calculer l'output)
            calculer_rbf_closeness_to_centroid(dataset_columns, dataset_array[i], K, centroids_array, sigma, dot_proximity_to_centroids);
            double output = calculer_dot_output(K, dot_proximity_to_centroids, weights);
            double error = targets[i] - output;
            total_error += error * error;
            for (int j = 0; j < K; j++){
                weights[j] += learning_rate * error * dot_proximity_to_centroids[j];
            }
        }
        double mse = total_error / dataset_rows; // Erreur moyenne
        printf("Epoch %d - MSE = %f\n", epoch, mse);
    }
}

void RBNF(int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double targets[dataset_rows], int cluster_size){
    size_t dataset_size = dataset_rows;

    // COUCHE 1

    // COUCHE 2
    int K = dataset_size / cluster_size;  // K = nombre de clusters.
    size_t weight_nb = K; // un poids par centroide, pas par caracteristique d'entree
    double weights[weight_nb];
    memset(weights, 0, sizeof(weights)); // on initialise le tableau de poids à 0.
    double centroids_array[K][dataset_columns];   // centroids_array = K centroids, then for each point in the centroïd vector, we use dataset_columns dimensions
    int clusters_array[K][dataset_rows]; // NOTE:  clusters_array ne stocke pas les vecteurs. Il stocke seulement les indices des vecteurs.
    int cluster_sizes[K]; // cluster_sizes = nombre de points dans chaque cluster. Par exemple, si K = 3, alors cluster_sizes pourrait être : [4 | 2 | 3]. Ce qui signifie : cluster 0 contient 4 points, cluster 1 contient 2 points, cluster 2 contient 3 points

    // step 1: on trouve les centroïds à partir des données du dataset (dataset_array)
    find_centroids(K, dataset_rows, dataset_columns, dataset_array, centroids_array);

    // step 2: K-MEANS. 
    // On stabilise les centroïdes sur plusieurs itérations. Le modèle trouve lui même les similitudes entre les données et les groupe en k groupes distincts.
    int max_iterations = 100;
    for (int iteration = 0; iteration < max_iterations; iteration++){
        // Mettre chaque point dans son cluster
        assign_clusters(K, dataset_rows, dataset_columns, dataset_array, centroids_array, clusters_array, cluster_sizes);
        // Recalculer les centroïdes
        update_centroids(K, dataset_rows, dataset_columns, dataset_array, clusters_array, cluster_sizes, centroids_array);
    }


    // step 3: RBF

    // Calcul de sigma
    double sigma = calculer_sigma(K, dataset_columns, centroids_array);

    // apprentissage des poids
    double learning_rate = 0.01;
    int epochs = 1000;
    recursive_train_weights(K, dataset_rows, dataset_columns, dataset_array, targets, centroids_array, sigma, weights, learning_rate, epochs);

    double dot_proximity_to_centroids[K]; //ce tableau contient, pour chaque centroïde, une valeur qui indique à quel point le point étudié lui ressemble ou en est proche.
    printf("\nResultats :\n");
    for (int i = 0; i < dataset_rows; i++){
        calculer_rbf_closeness_to_centroid(dataset_columns, dataset_array[i], K, centroids_array, sigma, dot_proximity_to_centroids);
        double output = calculer_dot_output(K, dot_proximity_to_centroids, weights);
        printf("Point %d : cible = %f, prediction = %f\n", i, targets[i], output);
    }
}

// Sauvegarde les centroides, le sigma et les poids appris dans un fichier texte, pour les reutiliser plus tard sans avoir a ré-entrainer (meme principe que svm_sauvegarder() dans ETAPE4.svm.c).
int rbf_sauvegarder(const char *chemin, double *centroids, double sigma, double *weights, int K, int dataset_columns){
    FILE *f = fopen(chemin, "w");
    if (f == NULL)
        return -1;
    fprintf(f, "%d %d %.17g\n", K, dataset_columns, sigma);
    for (int i = 0; i < K * dataset_columns; i++) fprintf(f, "%.17g ", centroids[i]);
    fprintf(f, "\n");
    for (int i = 0; i < K; i++) fprintf(f, "%.17g ", weights[i]);
    fprintf(f, "\n");
    fclose(f);
    return 0;
}

int rbf_charger(const char *chemin, double *centroids, double *weights, int *K, int *dataset_columns, double *sigma){
    FILE *f = fopen(chemin, "r");
    if (f == NULL)
        return -1;

    if (fscanf(f, "%d %d %lf", K, dataset_columns, sigma) != 3) { fclose(f); return -1; }
    for (int i = 0; i < (*K) * (*dataset_columns); i++)
        if (fscanf(f, "%lf", &centroids[i]) != 1) { fclose(f); return -1; }
    for (int i = 0; i < *K; i++)
        if (fscanf(f, "%lf", &weights[i]) != 1) { fclose(f); return -1; }

    fclose(f);
    return 0;
}
