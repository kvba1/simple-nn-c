#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

float gaussian() {
    float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

struct Value {
    float data;
    float grad;
} typedef Value;

float tanh_(float x) {
    return (expf(x) - expf(-x)) / (expf(x) + expf(-x));
}

float tanh_derivative(float x) {
    float t = tanh_(x);
    return 1 - t * t;
}

float sigmoid(float x){
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

void zero_grads(Value *pWeights, Value *pBias, int num_weights) {
    for (int i = 0; i < num_weights; i++)
    {
        pWeights[i].grad = 0.0;
    }
    pBias->grad = 0.0;
}

int main() {
    srand(time(NULL));
    int num_steps = 5000;

    float x[4][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
    float y[4] =    {0.0, 1.0, 1.0, 0.0};

    float scale = 0.5f;
    Value w1[2] = {{gaussian() * scale, 0.0}, {gaussian() * scale, 0.0}};
    Value w2[2] = {{gaussian() * scale, 0.0}, {gaussian() * scale, 0.0}};
    Value b1 = {0.0, 0.0};
    Value b2 = {0.0, 0.0};
    Value w3[2] = {{gaussian() * scale, 0.0}, {gaussian() * scale, 0.0}};
    Value b3 = {0.0, 0.0};

    float z1[4] = {0.0, 0.0,0.0, 0.0};
    float z1_tanh[4] = {0.0, 0.0,0.0, 0.0};
    float z2[4] = {0.0, 0.0,0.0, 0.0};
    float z2_tanh[4] = {0.0, 0.0,0.0, 0.0};
    float z3[4] = {0.0, 0.0, 0.0, 0.0};
    float z3_sigmoid[4] = {0.0, 0.0, 0.0, 0.0};

    for (int n = 0; n < num_steps; n++)
    {
        float losses[4] = {0.0, 0.0, 0.0, 0.0};
        float loss = 0.0;

        zero_grads(w1, &b1, 2);
        zero_grads(w2, &b2, 2);
        zero_grads(w3, &b3, 2);

        for (int i = 0; i < 4; i++)
        {
            z1[i] = 0.0;
            z1_tanh[i] = 0.0;
            z2[i] = 0.0;
            z2_tanh[i] = 0.0;
            z3[i] = 0.0;
            z3_sigmoid[i] = 0.0;

            for (int j = 0; j < 2; j++){
                z1[i] += x[i][j] * w1[j].data;
                z2[i] += x[i][j] * w2[j].data;
            }

            z1[i] += b1.data;
            z2[i] += b2.data;

            z1_tanh[i] = tanh_(z1[i]);
            z2_tanh[i] = tanh_(z2[i]);

            z3[i] += z1_tanh[i] * w3[0].data;
            z3[i] += z2_tanh[i] * w3[1].data;
            z3[i] += b3.data;

            z3_sigmoid[i] = sigmoid(z3[i]);

            float p = fminf(fmaxf(z3_sigmoid[i], 1e-7f), 1.0f - 1e-7f);
            losses[i] = -(y[i] * logf(p) + (1.0f - y[i]) * logf(1.0f - p));
        }

        for (int i = 0; i < 4; i++)
        {
            loss += losses[i];
        }
        loss /= 4;

        for (int i = 0; i < 4; i++)
        {
            float dLdZ3 = (z3_sigmoid[i] - y[i]) / 4.0f;

            float dLdW31 = dLdZ3 * z1_tanh[i];
            float dLdW32 = dLdZ3 * z2_tanh[i];
            w3[0].grad += dLdW31;
            w3[1].grad += dLdW32;

            b3.grad += dLdZ3;

            float dLdZ1_tanh = dLdZ3 * w3[0].data;
            float dLdZ2_tanh = dLdZ3 * w3[1].data;
            float dLdZ1 = dLdZ1_tanh * tanh_derivative(z1[i]);
            float dLdZ2 = dLdZ2_tanh * tanh_derivative(z2[i]);

            for (int j=0; j<2; j++){
                float dLdW1 = dLdZ1 * x[i][j];
                float dLdW2 = dLdZ2 * x[i][j];

                w1[j].grad += dLdW1;
                w2[j].grad += dLdW2;
            }
            b1.grad += dLdZ1;
            b2.grad += dLdZ2;
        }

        for (int i = 0; i < 2; i++)
        {
            w1[i].data -= .3 * w1[i].grad;
            w2[i].data -= .3 * w2[i].grad;
            w3[i].data -= .3 * w3[i].grad;

        }
        b1.data -= .3 * b1.grad;
        b2.data -= .3 * b2.grad;
        b3.data -= .3 * b3.grad;

        printf("training loss: %f\n", loss);
    }
    return 0;
}