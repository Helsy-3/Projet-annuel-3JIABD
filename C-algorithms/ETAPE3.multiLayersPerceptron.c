#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IN 12
#define H 8
#define OUT 3
#define N_EXEMPLES 3

// Donnees de test : 3 "images" deja transformees en vecteurs de 12 valeurs
double X[N_EXEMPLES][IN] = {
    {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1},
    {0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1},
    {1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0}};
int vraies_classes[N_EXEMPLES] = {2, 0, 1}; // 0=plastique, 1=verre, 2=metal

double W1[IN][H], W2[H][OUT];
double b1[H] = {0}, b2[OUT] = {0};

double relu(double x) { return x > 0 ? x : 0; }
double relu_deriv(double x) { return x > 0 ? 1 : 0; }

// Remplit un tableau de poids avec de petites valeurs aleatoires
void init_poids_couche(double *poids, int taille)
{
    for (int i = 0; i < taille; i++)
        poids[i] = ((double)rand() / RAND_MAX - 0.5) * 0.2;
}

void init_poids(void)
{
    init_poids_couche(&W1[0][0], IN * H);
    init_poids_couche(&W2[0][0], H * OUT);
}

// Calcule la sortie du reseau pour une entree x (une seule couche cachee)
void forward(double x[IN], double h[H], double out[OUT])
{
    for (int j = 0; j < H; j++)
    {
        h[j] = b1[j];
        for (int i = 0; i < IN; i++)
            h[j] += x[i] * W1[i][j];
        h[j] = relu(h[j]);
    }
    for (int j = 0; j < OUT; j++)
    {
        out[j] = b2[j];
        for (int i = 0; i < H; i++)
            out[j] += h[i] * W2[i][j];
    }
    // softmax : transforme les sorties en probabilites qui totalisent 1
    double somme = 0;
    for (int j = 0; j < OUT; j++)
    {
        out[j] = exp(out[j]);
        somme += out[j];
    }
    for (int j = 0; j < OUT; j++)
        out[j] /= somme;
}

int predire(double out[OUT])
{
    int meilleur = 0;
    for (int j = 1; j < OUT; j++)
        if (out[j] > out[meilleur])
            meilleur = j;
    return meilleur;
}

// Une passe de retropropagation sur un exemple (image, classe attendue)
void retropropager(double x[IN], int classe_attendue, double lr)
{
    double h[H], out[OUT];
    forward(x, h, out);

    double cible[OUT] = {0, 0, 0};
    cible[classe_attendue] = 1.0;

    // gradient de la couche de sortie (softmax + entropie croisee)
    double dz2[OUT];
    for (int j = 0; j < OUT; j++)
        dz2[j] = out[j] - cible[j];

    // gradient de la couche cachee
    double dz1[H];
    for (int i = 0; i < H; i++)
    {
        double somme = 0;
        for (int j = 0; j < OUT; j++)
            somme += dz2[j] * W2[i][j];
        dz1[i] = somme * relu_deriv(h[i]);
    }

    // mise a jour des poids (couche de sortie)
    for (int i = 0; i < H; i++)
        for (int j = 0; j < OUT; j++)
            W2[i][j] -= lr * dz2[j] * h[i];
    for (int j = 0; j < OUT; j++)
        b2[j] -= lr * dz2[j];

    // mise a jour des poids (couche cachee)
    for (int i = 0; i < IN; i++)
        for (int j = 0; j < H; j++)
            W1[i][j] -= lr * dz1[j] * x[i];
    for (int j = 0; j < H; j++)
        b1[j] -= lr * dz1[j];
}

// Entraine le reseau sur des donnees envoyees depuis Python (vraies photos du dataset)
// vecteurs : n_exemples * IN valeurs a la suite (chaque exemple = IN valeurs)
// classes  : n_exemples valeurs (0, 1 ou 2)
int run_pmc_dataset(double *vecteurs, int *classes, int n_exemples)
{
    srand(42);
    init_poids();

    int epochs = 10000;
    double lr = 0.05;

    for (int e = 0; e < epochs; e++)
        for (int n = 0; n < n_exemples; n++)
            retropropager(&vecteurs[n * IN], classes[n], lr);

    int bonnes_predictions = 0;
    for (int n = 0; n < n_exemples; n++)
    {
        double h[H], out[OUT];
        forward(&vecteurs[n * IN], h, out);
        int prediction = predire(out);
        if (prediction == classes[n])
            bonnes_predictions++;
    }
    return bonnes_predictions;
}

// Cas de test : appelle run_pmc_dataset() avec les exemples fixes definis plus haut
int run_pmc(void)
{
    return run_pmc_dataset(&X[0][0], vraies_classes, N_EXEMPLES);
}

int main(void)
{
<<<<<<< HEAD

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
=======
    int bonnes = run_pmc();
    printf("Bonnes predictions : %d/%d\n", bonnes, N_EXEMPLES);
>>>>>>> b927673b30c9732f2802ba1414fe05c146fd1ac6
    return 0;
}
