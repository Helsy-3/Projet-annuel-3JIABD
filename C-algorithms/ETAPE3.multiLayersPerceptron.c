#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float relu(float x)
{
    if (x > 0)
        return x;

    return 0;
}

int main(void)
{

    int classes[3] = {0, 1, 2};
    int X_pictures_data[3][12] = {{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1}, {0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1}, {1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0}};
    int y_pred[3] = {2, 0, 1}; 

    float neurone_layer_1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float neurone_layer_2[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float output_layer[3] = {0, 0, 0};

    float poids_Entree_C1[12][32]; // 12 pixels en entrée : 32 neurones (384 poids) (un tableau de 12 lignes et 32 colonnes)
    for (int i = 0; i < 12; i++)
    {
        for (int j = 0; j < 32; j++)
        {
            poids_Entree_C1[i][j] = ((float)rand() / RAND_MAX) * 0.1; // petite valeur entre 0 et 0.1
        }
    }

    float poids_C1_C2[32][16]; // 32 neurones -> 16 neurones (512 poids) / Un tableau de 32 lignes avec 16 neurones par ligne
    for (int i = 0; i < 32; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            poids_C1_C2[i][j] = ((float)rand() / RAND_MAX) * 0.1; 
        }
    }

    float poids_C2_Sortie[16][3];
    for (int i = 0; i < 16; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            poids_C2_Sortie[i][j] = ((float)rand() / RAND_MAX) * 0.1; 
        }
    }

    /* ================================================ INITILIZATION =============================================================== */

    float learning_rate = 0.01f;
    int epochs = 10000;

    for (int image = 0; image < 3; image++)
    {

        /* =============================================== COUCHE 1 =================================================================== */
        for (int j = 0; j < 32; j++)
        {
            for (int i = 0; i < 12; i++)
            {
                neurone_layer_1[j] += X_pictures_data[image][i] * poids_Entree_C1[i][j];
            }
            neurone_layer_1[j] = relu(neurone_layer_1[j]);
        }

        /* =============================================== COUCHE 2 =================================================================== */
        for (int i = 0; i < 16; i++)
        {
            for (int y = 0; y < 32; y++)
            {
                neurone_layer_2[i] = neurone_layer_1[y] + poids_C1_C2[y][i];
            }
            neurone_layer_2[i] = relu(neurone_layer_2[i]);
        }

        /* =============================================== COUCHE DE SORTIE =================================================================== */
        float logits[3];
        for (int j = 0; j < 3; j++)
        {
            for (int i = 0; i < 16; i++)
            {
                logits[j] = neurone_layer_2[i] * poids_C2_Sortie[i][j];
            }
        }

        printf("\n");

        printf("Image %d\n", image);

        printf(
            "Probabilite plastique : %.2f%%\n",
            logits[0] * 100.0f);

        printf(
            "Probabilite verre     : %.2f%%\n",
            logits[1] * 100.0f);

        printf(
            "Probabilite metal     : %.2f%%\n",
            logits[2] * 100.0f);

        printf(
            "Classe predite        : %d\n",
            prediction);

        printf(
            "Classe attendue       : %d\n",
            vraie_classe);


        if (prediction == vraie_classe)
        {
            printf("Resultat              : CORRECT\n");
        }
        else
        {
            printf("Resultat              : ERREUR\n");
        }
    }
    return 0;
}
