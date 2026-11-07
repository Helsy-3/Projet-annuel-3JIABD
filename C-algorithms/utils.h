#ifndef DISTANCE_H
#define DISTANCE_H

void init_pmc_poids(double *weight_array, int weight_array_length);

double euclidean_distance(
double *dot,
double *centroid,
int dataset_columns
);

int calculer_closest_centroid(
double *dot, int K, int dataset_columns,
double centroids_array[K][dataset_columns]
);

double rbf_distance(double distance, double sigma);

double calculer_dot_output(int K, double rbf_closeness[K], double weights[K]);

// Perceptron multi couche function
int find_strongest_category_by_neurone_score(double *couche_sortie, int output_length);

double calculer_loss(double prediction, double cible);

#endif