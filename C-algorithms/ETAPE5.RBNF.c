#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "distance.h"

// ETAPE 1 : Initialiser les centroïdes.

// La meilleure méthode pour trouver K (nombre souhaité) centroïds est de les sélectionner de façon aléatoire dans le dataset et de les désigner
// comme les centroïdes initiaux.

void find_centroids(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double centroids_array[K][dataset_columns]){
     // K est le nombre de centroïdes que l'on souhaite obtenir.
    for (int i = 0; i < K; i++){
        int random_centroid = rand() % dataset_rows; // Le nombre aléatoire sert à choisir quel point du dataset devient le centre initial.
            for (int j = 0; j < dataset_columns; j++)

        {
            centroids_array[i][j] = dataset_array[random_centroid][j];
        }
    }
}

// ETAPE 2 : Assigner les points du dataset au centroïde le plus proche.

// Pour assigner les points du dataset au centroïde le plus proche, on va calculer la distance de chaque point à chacun des centroïdes.
// Pour calculer la distance, on utilise la distance euclidienne.

void assign_clusters(
    int K,
    int dataset_rows,
    int dataset_columns,
    double dataset_array[dataset_rows][dataset_columns],
    double centroids_array[K][dataset_columns],
    int clusters_array[K][dataset_rows],
    int cluster_sizes[K])
{
    for (int i = 0; i < K; i++)

    {
        cluster_sizes[i] = 0;
    }

    for (int i = 0; i < dataset_rows; i++)
    {
        int closest_centroid = compute_closest_centroid(
            dataset_array[i], // point dans le dataset
            K,
            dataset_columns,
            centroids_array);
        int position = cluster_sizes[closest_centroid]; // On regarde combien de points sont déjà dans ce cluster
        clusters_array[closest_centroid][position] = i; // On ajoute le point au cluster
        cluster_sizes[closest_centroid]++; // Le cluster contient maintenant un point de plus
    }
}

// ETAPE 3 : Mettre à jour les centroïdes.
// Pour chaque cluster, on calcule la moyenne de tous les points. Cette moyenne devient le nouveau centroïde.

void update_centroids(
    int K,
    int dataset_rows,
    int dataset_columns,
    double dataset_array[dataset_rows][dataset_columns],
    int clusters_array[K][dataset_rows],
    int cluster_sizes[K],
    double centroids_array[K][dataset_columns])
{
    for (int cluster = 0; cluster < K; cluster++)
    {
        for (int dimension = 0; dimension < dataset_columns; dimension++)
        {
            double sum = 0;
            for (int point = 0; point < cluster_sizes[cluster]; point++)
            {
                int point_number = clusters_array[cluster][point];
                double coordinate = dataset_array[point_number][dimension];
                sum = sum + coordinate;
            }
            double average = sum / cluster_sizes[cluster];
            centroids_array[cluster][dimension] = average;
        }
    }
}

// ETAPE 4: calculer les sigma. Le sigma, basé sur la distance moyenne entre les centroïdes, sert de largeur aux fonctions gaussiennes. 
double calculer_sigma(int K, int dataset_columns, double centroids_array[K][dataset_columns]){
    double sum_distance = 0.0;
    int number_of_distances = 0;
    for (int i = 0; i < K; i++){
        for (int j = i + 1; j < K; j++){
            double squared_distance = 0.0;
            for (int dimension = 0; dimension < dataset_columns; dimension++){
                double difference = centroids_array[i][dimension]- centroids_array[j][dimension];
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

// ETAPE 5: FONCTION RBF GAUSSIENNE
double rbf(double distance, double sigma){
    return exp(-(distance * distance)/ (2.0 * sigma * sigma));
}



void RBNF(int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], int cluster_size)
{
    size_t dataset_size = dataset_rows;
    size_t weight_nb = dataset_columns;

    double weight[weight_nb];
    memset(weight, 0, sizeof(weight));// on initialise le tableau de poids à 0.

    // COUCHE 1
    
    // COUCHE 2
    
    // K = nombre de clusters. On le trouve en divisant la taille totale du dataset par la taille d'un cluster.
    int K = dataset_size / cluster_size;
    // centroids_array = K centroids, then for each point in the centroïd vector, we use dataset_columns dimensions
    double centroids_array[K][dataset_columns];
    // clusters_array = de dimension K (nombre de clusters) et d'indice dataset_rows. 
    // NOTE:  clusters_array ne stocke pas les vecteurs. Il stocke seulement les indices des vecteurs.
    int clusters_array[K][dataset_rows];
    // cluster_sizes = nombre de points dans chaque cluster. Par exemple, si K = 3, alors cluster_sizes pourrait être : [4 | 2 | 3]
    // cluster 0 contient 4 points, cluster 1 contient 2 points, cluster 2 contient 3 points
    int cluster_sizes[K];

    // step 1: on trouve les centroïds à partir des données du dataset (dataset_array)
        find_centroids(
            K,
            dataset_rows,
            dataset_columns,
            dataset_array,
            centroids_array
        );

    // step 2: K-MEANS. On stabilise les centroïds sur plusieurs itérations. En gros, c'est de l'apprentissage non-supervisé. 
    // Le modèle trouve lui même les similitudes entre les données et les groupe en k groupes distincts. 
    int max_iterations = 100;
    for (int iteration = 0; iteration < max_iterations; iteration++)
    {
        // Mettre chaque point dans son cluster
            assign_clusters(
                K,
                dataset_rows,
                dataset_columns,
                dataset_array,
                centroids_array,
                clusters_array,
                cluster_sizes);

        // Recalculer les centroïdes
            update_centroids(
                K,
                dataset_rows,
                dataset_columns,
                dataset_array,
                clusters_array,
                cluster_sizes,
                centroids_array);
    }

    // When the algorithm stops, the centroids represent the centers of the final clusters, and each data point belongs to the cluster associated with the nearest final centroid.
    
    // Calcul de sigma
    double sigma = compute_sigma(
        K,
        dataset_columns,
        centroids_array
    );

    printf("Sigma = %f\n", sigma);


    // APPRENTISSAGE DES POIDS
    double learning_rate = 0.01;
    int epochs = 1000;
    train_weights(
        K,
        dataset_rows,
        dataset_columns,
        dataset_array,
        targets,
        centroids_array,
        sigma,
        weights,
        learning_rate,
        epochs
    );


    double activations[K];
    printf("\nResultats :\n");
    for (int i = 0; i < dataset_rows; i++)
    {
        // Calcul des activations
        compute_rbf_activations(
            K,
            dataset_columns,
            dataset_array[i],
            centroids_array,
            sigma,
            activations
        );
        // Calcul de la sortie
        double output = compute_output(
            K,
            activations,
            weights
        );
        printf(
            "Point %d : cible = %f, prediction = %f\n",
            i,
            targets[i],
            output
        );
    }
}

