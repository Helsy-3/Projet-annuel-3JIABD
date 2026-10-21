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

#endif