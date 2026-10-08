#include <math.h>
#include "distance.h"

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


int compute_closest_centroid(
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

 for (int i =0; i < K; i++)
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