#include <stdio.h>
#include <stdlib.h>


// Déclaration de ta fonction
int train_basic_perceptron(
    double *dataset_one_long_vector,
    int *expected_targets,
    int dataset_length,
    int vector_length,
    double learning_rate,
    int epochs,
    double *weights,
    double *bias
);

int run_basic_perceptron_on_xor(double *X, int *Y, int dataset_length, int vector_length, double learning_rate, int epochs, double *poids, double *biais){
    int erreurs = train_basic_perceptron(
        X,
        Y,
        dataset_length,
        vector_length,
        learning_rate,
        epochs,
        poids,
        biais
    );
    printf("Erreurs XOR brut : %d/4\n", erreurs);
    return erreurs;
}


int from_two_to_three_dimensions(int nb_dot_dimensions){
    return nb_dot_dimensions + nb_dot_dimensions * (nb_dot_dimensions - 1) / 2;
}

void add_3rd_dimension_to_dataset(double *vecteurs_array, int vector_dimensions, int dataset_length, double *new_dimensions_dataset){
    int new_nb_dot_dimensions = from_two_to_three_dimensions(vector_dimensions); 

    for (int i = 0; i < dataset_length; i++){
        // on ajoute une dimension pour chaque point du dataset
        double *current_dot = &vecteurs_array[i * vector_dimensions];   // exemple d'origine (n_features valeurs)
        double *new_dot = &new_dimensions_dataset[i * new_nb_dot_dimensions];     // exemple transforme (n_out valeurs)
        int k = 0;

        // Copier les dimensions originales
        for (int dimension = 0; dimension < vector_dimensions; dimension++){
            new_dot[k++] = current_dot[dimension];                        // on garde d'abord les valeurs d'origine
        }
        // Ajouter le produit croisé (XOR : 2 dimensions)
        new_dot[k++] = current_dot[0] * current_dot[1];        
}
}

int entrainer_perceptron_transforme_dataset(double *dataset, int *Y, int dataset_length, int vecteur_dimensions, double learning_rate, int max_iter, double *weights, double *bias){
    int output_length = from_two_to_three_dimensions(vecteur_dimensions);
    double *new_vecteur_array = malloc(dataset_length * output_length * sizeof(double));
    add_3rd_dimension_to_dataset(dataset, vecteur_dimensions, dataset_length, new_vecteur_array);
    int erreurs = train_basic_perceptron(new_vecteur_array, Y, dataset_length, output_length, learning_rate, max_iter, weights, bias);
    free(new_vecteur_array);
    return erreurs;
}