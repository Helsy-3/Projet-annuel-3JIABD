#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void){

    // verre:0, métal:1, plastique:2
    int predictions[3] = {0, 1, 2};
    int X_pictures_data[3][12] = {{1,0,0,0,1,0,0,0,1,1,1,1}, {0,0,0,0,1,1,0,0,1,1,1,1}, {1,0,0,0,1,0,1,0,1,0,1,0}};
    int y_pred[3] = {2, 0, 1};
    
    // prédictions attendues: la première bouteille est en plastique, la deuxième en verre, la troisième est en métal.

    float neurone_layer_1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float neurone_layer_2[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float output_layer[3] = {0, 0, 0};
    
    // délclaration des matrices (poids)
    float poids_Entree_C1[12][32]; // 12 pixels en entrée : 32 neurones (384 poids) (un tableau de 12 lignes et 32 colonnes)
    float poids_C1_C2[32][16]; // 32 neurones -> 16 neurones (512 poids)
    float poids_C2_Sortie[16][3];

    // initilalisation automatique des poids avec des valeurs aléatoires
    srand(time(NULL));
    
    for(int i = 0; i < 12; i++) {
        for(int j = 0; j < 32; j++) {
            poids_Entree_C1[i][j] = ((float)rand() / RAND_MAX) * 0.1; // petite valeur entre 0 et 0.1
        }
    }

    for(int i = 0; i < 32; i++) {
        for(int j = 0; j < 16; j++) {
            poids_C1_C2[i][j] = ((float)rand() / RAND_MAX) * 0.1; // petite valeur entre 0 et 0.1
        }
    }

    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 3; j++) {
            poids_C2_Sortie[i][j] = ((float)rand() / RAND_MAX) * 0.1; // petite valeur entre 0 et 0.1
        }
    }

    printf("Réseau de neurones initialisé avec succès\n");
    return 0;
}