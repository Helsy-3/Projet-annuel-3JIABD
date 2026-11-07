#include <math.h>
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double euclidean_distance(double *dot,double *centroid, int dataset_columns)
{
double total =0.0;

for (int j = 0; j < dataset_columns; j++)
{
double difference = dot[j] - centroid[j];
total += difference * difference;
}

return sqrt(total);
}


int calculer_closest_centroid(
double *dot,
int K,
int dataset_columns,
double centroids_array[K][dataset_columns]
)
{
    int closest_centroid = 0;

double min_distance =
euclidean_distance(
 dot,
 centroids_array[0],
 dataset_columns
 );

 for (int i = 0; i < K; i++)
 {
 double distance =
 euclidean_distance(
dot,
 centroids_array[i],
 dataset_columns
);

 if (distance < min_distance)
 {
min_distance = distance;
 closest_centroid = i;
 } 
}
return closest_centroid;
}

// rbf_distance renvoie une valeur de proximité, pas une valeur géométrique
// Plus la distance est petite, plus la valeur de rbf_distance() est grande.
double rbf_distance(double distance, double sigma){
    return exp(-(distance * distance)/ (2.0 * sigma * sigma));
}

// multiplie chaque valeur de proximité du point au centroide par un poids, puis additionne tous les résultats.
// Les poids indiquent l'importance que le réseau accorde à chaque valeur de proximité pour calculer sa prédiction.
double calculer_dot_output(int K, double dot_proximity_to_centroids[K], double weights[K]){
    double dot_output = 0.0;
    for (int i = 0; i < K; i++){
        dot_output += weights[i] * dot_proximity_to_centroids[i];
    }
    return dot_output;
}

// Perceptron multi couche function
int find_strongest_category_by_neurone_score(double *couche_sortie, int output_length)
{
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

double calculer_loss(double prediction, double cible)
{
    double difference = prediction - cible;
    double loss = 0.5 * difference * difference;
    return loss;
}