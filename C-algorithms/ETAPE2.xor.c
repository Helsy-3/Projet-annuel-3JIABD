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

// cette fonction ne prédit pas une droite. 
// elle prédit la classe à laquelle appartient un point
// mais où est la frontière de décision dans tout ça?
// La frontière de décision correspond à tous les points pour lesquels :
// somme : w_1x_1 + w_2x_2 + b = 0
// Concrètement :
// - D'un côté de la droite, la fonction renvoie 1.
// - De l'autre côté, elle renvoie 0.
// - Sur la droite, elle renvoie 1 puisque tu utilises somme >= 0.

int predict_perceptron(double *vecteur, int vector_length, double *weights, double biais){
    double somme = biais;

    for (int i = 0; i < vector_length; i++){
        somme += vecteur[i] * weights[i];
    }
    return somme >= 0 ? 1 : -1;
}

int predict_xor(double *X, int dataset_length, int vector_length, double *weights, double bias, int *predictions) {
    for (int i = 0; i < dataset_length; i++) {
        double *current_dot = &X[i * vector_length];
        predictions[i] = predict_perceptron(
            current_dot,
            vector_length,
            weights,
            bias
        );
    }
    return 0;
}

int predict_xor_transforme(double *X, int dataset_length, int vector_length, double *weights, double bias, int *predictions) {
    int output_length = from_two_to_three_dimensions(vector_length);
    double *new_dataset = malloc(dataset_length * output_length * sizeof(double));
    
    if (new_dataset == NULL) {
        return -1;
    }

    add_3rd_dimension_to_dataset(X, vector_length, dataset_length, new_dataset);
    predict_xor(new_dataset, dataset_length, output_length, weights, bias, predictions);
    free(new_dataset);
    return 0;
}