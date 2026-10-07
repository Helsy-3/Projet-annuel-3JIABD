/*
Un Radial Basis Function Network est un type de réseau de neurones artificiels qui utilise
des fonctions de base radiale comme fonctions d'activation.
Un RBFN se compose généralement de trois couches :

• Couche d'entrée : Elle reçoit les données brutes et les transmet au réseau sans modification.
• Couche cachée : Elle transforme les données de manière non linéaire.
Chaque neurone calcule la distance entre l'entrée et un point central (appelé centre),
en appliquant une fonction radiale (souvent une fonction gaussienne).
Le neurone s'active fortement si l'entrée est proche de son centre.
• Couche de sortie : Elle calcule une combinaison linéaire des activations de la
couche cachée pour produire la prédiction finale.

DEFINITIONS

A. Transformer des données de manière non linéaire:
- En informatique et en mathématiques, transformer les données de manière non linéaire
signifie modifier la structure des données afin que la relation entre l'entrée et la sortie ne forme plus une simple ligne droite

-> EN EFFET, la transformation linéaire est insuffisante. Si vos points rouges entourent les points bleus (comme une cible), aucune ligne droite ne pourra les séparer.
Une transformation linéaire échouera toujours à résoudre ce problème.

CONSEQUENCE: La transformation non linéaire est la solution.
• Le principe : Elle courbe, tord ou projette les données dans un espace totalement différent (souvent avec plus de dimensions).
• L'effet visuel : C'est comme si vous preniez la feuille de papier et que vous la déformiez pour créer une colline. Les points bleus se retrouvent au sommet de la colline et les points rouges en bas.
• Le résultat : Il devient maintenant très facile de séparer les deux groupes (par exemple, en passant un coup de couteau horizontal pour couper le sommet de la colline).
Comment le RBFN fait cela ?
Dans un réseau RBFN, la couche cachée utilise une courbe en cloche (gaussienne). Au lieu de regarder la valeur brute de l'entrée, elle calcule : "À quelle distance se trouve cette entrée du centre de mon neurone ?".
Cette notion de distance et de courbe en cloche brise la rigidité des lignes droites et permet au réseau de comprendre des motifs complexes, circulaires ou entremêlés.

B. Fonction gaussienne
Une fonction gaussienne est une fonction mathématique qui dessine une courbe en forme de cloche
inversée, parfaitement symétrique.
Dans le réseau de neurones RBFN, la fonction gaussienne sert de détecteur de proximité.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// STEP 1. Initialization: First, you must decide how many clusters you want to find, which is the value K. Then, the algorithm initializes
// K centroids. A common way to do this is to randomly select K data points from your dataset and designate them as the initial
// centroids. Other initialization methods exist, but random selection is a straightforward starting point.
void find_centroids(int K, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], double centroids_array[K][dataset_columns])
{
    // K is the number of desired centroids
    for (int i = 0; i < K; i++)
    {
        int centroïd = rand() % dataset_rows; // Le nombre aléatoire sert à choisir quel point du dataset devient le centre initial.
        for (int j = 0; j < dataset_columns; j++)
        {
            centroids_array[i][j] = dataset_array[centroïd][j];
        }
    }
}

// STEP 2. Assignement step: For each data point in your dataset, calculate its distance to each of the  centroids.
// Assign the data point to the cluster whose centroid is the nearest (closest). The most common way to measure distance here is the standard Euclidean distance (the straight-line distance you'd measure with a ruler), but the concept is simply "closest centroid".
// After this step, every data point belongs to one of the  K clusters.

// distance(A,B)= sqrt((xA−xB)2+(yA−yB)2)

double euclidean_distance(double *dot, double *centroïd, int dataset_columns)
{
    int total = 0;
    for (int j = 0; j < dataset_columns; j++)
    {
        double difference = abs(dot[j] - centroïd[j]);
        total += difference * difference;
    }

    return sqrt(total);
}

int compute_closest_centroïd(double *dot, int K, int dataset_columns, double centroids_array[K][dataset_columns]){
    int closest_centroid = 0;
    double min_distance = euclidean_distance(dot, centroids_array[0], dataset_columns);
    for(int i = 0; i < K; i++){
        double distance = euclidean_distance(dot, centroids_array[i], dataset_columns);
        if(distance < min_distance){
            min_distance = distance;
            closest_centroid = i;
        }
    }
    return closest_centroid;
}

void assign_clusters(int K, double **clusters_array, double centroids_array, int dataset_size, int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], int cluster_size)
{
    for (int i = 0; i < dataset_rows; i++)
    {
        for (int j = 0; j < dataset_columns; j++)
        {
            int dot = dataset_array[i][j];
            int closest_centroid = compute_distance(centroids_array, K, dot);
            put_in_cluster(clusters_array, dot, closest_centroid, cluster_size);
            // create cluster
        }
    }
}

void put_in_cluster(int **clusters_array, int *dot, int closest_centroid, int cluster_size)
{
    for (int i = 0; i < cluster_size; i++)
    {
        if (clusters_array[closest_centroid][i] == NULL)
        {
            clusters_array[closest_centroid][i] = dot;
            return;
        }
    }
}

// Update Step: Recalculate the position of each of the  K centroids.
// The new position for a centroid is the mean (average position) of all the data points assigned to its cluster
// in the previous step. If a cluster ends up with no points assigned to it (which can happen, especially with
// poor initialization), the centroid might be removed or randomly reassigned, though typically it stays put until
// points are assigned in a later iteration.

int calculate_average(double *cluster_dots, int cluster_size)
{
    int total = 0;
    for (int i = 0; i < cluster_size; i++)
    {
        total += cluster_dots[i];
    }
    return total / cluster_size;
}

int update_centroids(int K, double **centroids_array, double clusters_array, int cluster_size)
{
    int distance = 0;
    for (int i = 0; i < K; i++)
    {
        int centroid = centroids_array[i];
        int new_centroid_value = calculate_average(clusters_array[i], cluster_size);
        centroids_array[i] = new_centroid_value;
        distance = abs(centroid - new_centroid_value);
    }
    return distance;
}

void RBNF(int dataset_rows, int dataset_columns, double dataset_array[dataset_rows][dataset_columns], int cluster_size)
{
    size_t dataset_size = dataset_rows;
    size_t weight_nb = dataset_columns;

    double weight[weight_nb];
    memset(weight, 0, sizeof(weight)); // on initialise le tableau de poids à 0.

    // LAYER 1

    // LAYER 2

    // a) implementing K-MEANS

    // K is the number of clusters and centroïds we need. To find K, we need to divide the size of the dataset by the number of
    // dots inside each cluster.

    int K = dataset_size / cluster_size;
    double centroids_array = find_centroids(K, dataset_rows, dataset_columns, dataset_array, dataset_size);
    double clusters_array[K][cluster_size];
    assign_clusters(K, clusters_array, centroids_array, dataset_size, dataset_rows, dataset_columns, dataset_array);
    int distance = update_centroids(K, centroids_array);
    if (distance < 2)
    {
        return;
    }
    // When the algorithm stops, the centroids represent the centers of the final clusters, and each data point belongs to the cluster associated with the nearest final centroid.

    //

    // LAYER 3
}