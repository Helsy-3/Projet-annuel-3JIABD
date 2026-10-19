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

// ETAPE 3 : Mettre à jour les centroïdes. Pour chaque cluster, on calcule la moyenne de tous les points. 
// Cette moyenne devient le nouveau centroïde.
void update_centroids(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], int clusters_array[K][dataset_rows], int cluster_sizes[K], double centroids_array[K][dataset_columns]) {
    for (int cluster_index = 0; cluster_index < K; cluster_index++) {// Pour chaque cluster
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

// ETAPE 4: Le sigma, basé sur la distance moyenne entre les centroïdes, sert de largeur aux fonctions gaussiennes.
// Sigma représente la distance moyenne entre toutes les paires de centroïdes.
double calculer_sigma(int K, int dataset_columns, double centroids_array[K][dataset_columns]){
    double sum_distance = 0.0; // somme de toutes les distances calculées entre tous les centroïdes
    int number_of_distances = 0; // nombre total de distances calculées entre les paires de centroïdes
    
    for (int centroïd_index = 0; centroïd_index < K; centroïd_index++){ // on parcourt tous les centroïdes
        for (int j = centroïd_index + 1; j < K; j++){
            double squared_distance = 0.0;
            for (int vecteur_index = 0; vecteur_index < dataset_columns; vecteur_index++){ // parcourir toutes les dimensions des centroïdes
                double difference = centroids_array[centroïd_index][vecteur_index] - centroids_array[j][vecteur_index]; // Calculer la différence entre les coordonnées des deux centroïdes pour la dimension actuelle
                squared_distance += difference * difference; // Élever cette différence au carré et l'ajouter à la somme des carrés des différences
            }
            double distance = sqrt(squared_distance); // Calculer la distance euclidienne entre les deux centroïdes
            sum_distance += distance;  // Ajouter cette distance à la somme de toutes les distances
            number_of_distances++;
        }
    }
    double sigma = sum_distance / number_of_distances;
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
void train_weights(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double targets[dataset_rows], double centroids_array[K][dataset_columns], double sigma, double weights[K], double learning_rate, int epochs){
    double dot_proximity_to_centroids[K]; // rbf_closeness est réécrit pour chaque point
    for (int epoch = 0; epoch < epochs; epoch++){
        double total_error = 0.0;
        for (int i = 0; i < dataset_rows; i++){
            calculer_rbf_closeness_to_centroid(dataset_columns, dataset_array[i], K, centroids_array, sigma, dot_proximity_to_centroids);
            double output = calculer_output(K, dot_proximity_to_centroids, weights);
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
    // On stabilise les centroïds sur plusieurs itérations. Le modèle trouve lui même les similitudes entre les données et les groupe en k groupes distincts.
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
    train_weights(K, dataset_rows, dataset_columns, dataset_array, targets, centroids_array, sigma, weights, learning_rate, epochs);

    double dot_proximity_to_centroids[K]; //ce tableau contient, pour chaque centroïde, une valeur qui indique à quel point le point étudié lui ressemble ou en est proche.
    printf("\nResultats :\n");
    for (int i = 0; i < dataset_rows; i++){
        calculer_rbf_closeness_to_centroid(dataset_columns, dataset_array[i], K, centroids_array, sigma, dot_proximity_to_centroids);
        double output = calculer_output(K, dot_proximity_to_centroids, weights);
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
