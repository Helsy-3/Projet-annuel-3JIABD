#include <stdio.h>
#include <string.h>

int train_basic_perceptron(
    double *X, /* double : un nombre décimal (ex. 2.5, 0.75, -1.2). */
    int *Y,
    int dataset_length,
    int vector_length,
    double learning_rate,
    int epochs,
    double *weights,
    double *bias)
{
    for (int epoch = 0; epoch < epochs; epoch++)
    {
        int total_errors = 0;
        for (int i = 0; i < dataset_length; i++){
            double linear_output = *bias;
            for (int j = 0; j < vector_length; j++)
                linear_output += X[i * vector_length + j] * weights[j];
                int prediction = 0;
                if(linear_output >= 0){
                    prediction = 1;
                } else {
                    prediction = -1;
                }

            if (prediction != Y[i]){
                for (int j = 0; j < vector_length; j++)
                    weights[j] += learning_rate * Y[i] * X[i * vector_length + j];
                *bias += learning_rate * Y[i];
                total_errors++;
            }
        }
        if (total_errors == 0)
            break;
    }

    // après l'entrainement, on veut savoir si le modèle est devenu performant
    int post_training_errors = 0;
    for (int i = 0; i < dataset_length; i++){
        double sortie = *bias;
        for (int j = 0; j < vector_length; j++)
            sortie += X[i * vector_length + j] * weights[j];
            int prediction = 0;
            if(sortie >= 0){
                prediction = 1;
            } else {
                prediction = -1;
            }
        if (prediction != Y[i])
            post_training_errors++;
    }
    return post_training_errors;
}

double output_basic_perceptron(double *vector, int vector_length, double *weights_array, double biais){
    double unique_neurone_score = 0.0;
    for(int a = 0; a < vector_length; a++){
            unique_neurone_score += vector[a] * weights_array[a];
    }
    unique_neurone_score += biais;
    return unique_neurone_score;
}


// Sauvegarde les poids et le biais appris dans un fichier texte, pour les reutiliser plus tard
// sans avoir a ré-entrainer (meme principe que svm_sauvegarder() dans ETAPE4.svm.c).
int perceptron_sauvegarder(const char *chemin, double *weights, double bias, int n_features)
{
    FILE *f = fopen(chemin, "w");
    if (f == NULL)
        return -1;

    fprintf(f, "%d %.17g\n", n_features, bias);
    for (int j = 0; j < n_features; j++)
        fprintf(f, "%.17g ", weights[j]);
    fprintf(f, "\n");

    fclose(f);
    return 0;
}

int perceptron_charger(const char *chemin, double *weights, double *bias, int *n_features)
{
    FILE *f = fopen(chemin, "r");
    if (f == NULL)
        return -1;

    if (fscanf(f, "%d %lf", n_features, bias) != 2)
    {
        fclose(f);
        return -1;
    }
    for (int j = 0; j < *n_features; j++)
    {
        if (fscanf(f, "%lf", &weights[j]) != 1)
        {
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}
