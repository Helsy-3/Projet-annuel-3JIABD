#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int X_data[8][2] = {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {6, 6}, {6, 7}, {7, 6}, {7, 7}};
    int Y_pred[8] = {-1, -1, -1, -1, 1, 1, 1, 1};

    double weights[2] = {0.0, 0.0};
    double bias = 0.0;
    double learning_rate = 0.1;
    int repetitions = 1000;

    for (int rep = 0; rep < repetitions; rep++)
    {
        int total_errors = 0;
        for (int i = 0; i < 8; i++)
        {
            // (w1 * x1) + (w2 * x2) + bias
            double linear_output = X_data[i][0] * weights[0] + X_data[i][1] * weights[1];
            linear_output += bias;
            int pred;

            if(linear_output >= 0){
                pred = 1;
            } else {
                pred = -1;
            }

            if(pred != Y_pred[i]){
                weights[0] = learning_rate * Y_pred[i] * X_data[i][0];
                weights[1] = learning_rate * Y_pred[i] * X_data[i][1];
            }
            bias += learning_rate * Y_pred[i];
            total_errors++;
        }
        if (total_errors == 0) {
            printf("Convergence atteinte à l'époque %d !\n", rep + 1);
            break;
        }
    }
    printf("\n--- Modèle entraîné ---\n");
    printf("Poids w1: %.2f\n", weights[0]);
    printf("Poids w2: %.2f\n", weights[1]);
    printf("Biais b : %.2f\n", bias);

     printf("\n--- Vérification des prédictions ---\n");
    for (int i = 0; i < 8; i++) {
        double neurone_sortie = X_data[i][0] * weights[0] + X_data[i][1] * weights[1] + bias;
        int test_pred;
        if(neurone_sortie >= 0){
            test_pred = 1;
        } else {
            test_pred = -1;
        }
        printf("Donnée {%d, %d} -> Prédit: %d (Attendu: %d)\n", X_data[i][0], X_data[i][1], test_pred, Y_pred[i]);
    }
    return 0;
}