#include <math.h>
#include "utils.h"

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

double rbf_distance(double distance, double sigma){
    return exp(-(distance * distance)/ (2.0 * sigma * sigma));
}

double calculer_output(int K, double rbf_closeness[K], double weights[K]){
    double output = 0.0;
    for (int i = 0; i < K; i++){
        output += weights[i] * rbf_closeness[i];
    }
    return output;
}