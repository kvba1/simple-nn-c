# simple-nn-c

A minimal feedforward neural network implemented from scratch in C, with manual forward propagation, backpropagation, and gradient descent — no external libraries.

The network is trained to solve the classic **XOR problem**, which is not linearly separable and therefore requires a hidden layer to solve.

## Architecture

- **Input layer**: 2 neurons
- **Hidden layer**: 2 neurons with `tanh` activation
- **Output layer**: 1 neuron with `sigmoid` activation

Trained using binary cross-entropy loss and full-batch gradient descent over the 4 XOR input combinations.

## Why

This project is an exercise in understanding how neural networks work under the hood — implementing weight initialization, forward passes, loss computation, gradient calculation via the chain rule, and parameter updates manually in C, without any ML frameworks.

## Building and running

```sh
gcc -o main main.c -lm
./main
```

This will run 5000 training steps, printing the training loss at each step. The loss should converge toward zero as the network learns the XOR function.

## Notes

- Weights are initialized using a Gaussian distribution (Box-Muller transform).
- Gradients are computed manually via backpropagation for each layer.
- Learning rate is fixed at `0.3`.
