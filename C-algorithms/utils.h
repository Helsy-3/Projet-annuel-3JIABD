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

double rbf(double distance, double sigma);

double compute_output(int K, double rbf_closeness[K], double weights[K]);

#endif