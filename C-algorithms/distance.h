#ifndef DISTANCE_H
#define DISTANCE_H

double euclidean_distance(
double *dot,
double *centroid,
int dataset_columns
);

int compute_closest_centroid(
double *dot, int K, int dataset_columns,
double centroids_array[K][dataset_columns]
);

#endif