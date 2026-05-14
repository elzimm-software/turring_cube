# Neural Network — Technical Documentation

## Table of Contents

1. [Notation](#notation)
2. [Network Structure](#network-structure)
3. [Weights](#weights)
4. [Biases](#biases)
5. [Forward Pass](#forward-pass)
6. [Activation Functions](#activation-functions)
7. [Loss Function](#loss-function)
8. [Backpropagation](#backpropagation)
9. [Gradient Descent](#gradient-descent)
10. [Training Loop](#training-loop)
11. [Implementation Reference](#implementation-reference)

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $l$ | Layer index (0 = input, 1 = first hidden, ...) |
| $z^{[l]}$ | Pre-activation column vector for layer $l$ |
| $a^{[l]}$ | Post-activation column vector for layer $l$ |
| $W^{[l]}$ | Weight matrix connecting layer $l-1$ to layer $l$ |
| $b^{[l]}$ | Bias vector for layer $l$ |
| $g(z)$ | Activation function applied to $z$ |
| $g'(z)$ | Derivative of the activation function |
| $\delta^{[l]}$ | Delta (error signal) for layer $l$ |
| $\frac{\partial L}{\partial w}$ | Gradient of loss w.r.t. a weight |
| $\alpha$ | Learning rate |
---

## Network Structure

A neural network is a series of **layers**. Each layer contains nodes (neurons). Every node in a layer connects to every node in the next layer; this is called a **fully connected** or **dense** layer.

```
Layer 0       Layer 1       Layer 2
(input)       (hidden)      (output)

  o ———————>  o  ————————>  o
  o ———————>  o
              o
```

Layers are represented as a doubly linked list of `Layer` structs:

```
null ← initial_layer ⇄ hidden_layer ⇄ output_layer → null
```

The `prev` and `next` pointers allow traversal in both directions — forward for the forward pass, backward for backpropagation. A `LayerIterable` is used to walk the list cleanly.

---

## Weights

Each node in a layer must connect to **every** node in the next layer. So the number of weights between two layers is:

```
num_weights = num_inputs × num_outputs
```

Example: a layer with 3 input nodes connecting to a layer with 5 output nodes has `3 × 5 = 15` weights.

Weights are stored as a **flattened matrix** in row-major order:

```
weights[i * num_outputs + j]  =  weight from input node i to output node j
```

### Xavier Initialization

Weights are initialized to random values in the range:

```
±1 / sqrt(num_inputs)
```

This prevents activations from exploding or vanishing as the network gets deeper. Larger input layers get smaller initial weights.

### The Dot Product

For a single output node `j`, its value is the dot product of the input vector and the column of weights for that node:

```
[3]   [4]
[2] · [8]  =  (3×4) + (2×8)  =  28
```

For a full layer this is a matrix multiplication: `W^[l] · a^[l-1]`

---

## Biases

A bias is a number added to a node's value **after** the weight dot product calculation.

- Biases are applied to every node **except** the input layer
- Every non-input node has its own independently learned bias
- Biases start at `0.0` and are updated during training

```
[3]   [4]          bias = 1
[2] · [8]  + 1  =  (3×4 + 2×8) + 1  =  29

[3]   [1]          bias = 4
[2] · [10] + 4  =  (3×1 + 2×10) + 4  =  27
```

---

## Forward Pass

The forward pass propagates inputs through every layer to produce a final output. It has two steps per layer.

### Step 1: Pre-activation (z)

Matrix multiply the weight matrix by the activations from the previous layer, then add the bias vector:

$$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$$

In code, for each output node `j`:

$$z_j = b_j + \sum_{i} a_i \cdot w_{ij}$$

`z^[l]` is the **unprocessed** output of the raw weighted sums before any activation.

### Step 2: Post-activation (a)

Pass each `z` value through the activation function `g`:

$$a^{[l]} = g(z^{[l]})$$

`a^[l]` is the **final** output of the layer, and becomes the input `a^[l-1]` for the next layer.

### Full Chain

```
inputs (a^[0])
   ↓
z^[1] = W^[1] · a^[0] + b^[1]
a^[1] = g(z^[1])
   ↓
z^[2] = W^[2] · a^[1] + b^[2]
a^[2] = g(z^[2])
   ↓
... and so on until the output layer
```

---

## Activation Functions

After computing `z`, we need to **squash** the value into a usable range. This is what activation functions do.

An activation function must be:
1. **Non-linear** — cannot be in the form `g(z) = mz + b`. Without this, the entire network collapses to a simple linear function no matter how many layers it has.
2. **Bounded** — compresses input to a fixed range like `0 to 1`.

### Sigmoid

$$g(z) = \frac{1}{1 + e^{-z}}$$

Output range: `(0, 1)`

Sigmoid models are how a real neuron fires, it is either off (near 0), or on (near 1), or somewhere in the transition window. As `z → +∞`, `g(z) → 1`. As `z → -∞`, `g(z) → 0`.

```
z = -5  →  g(z) ≈ 0.007
z =  0  →  g(z) = 0.500
z =  5  →  g(z) ≈ 0.993
z = 27  →  g(z) ≈ 0.999
```

Implementation note: clamp `z` to `(-500, 500)` before computing `exp` to prevent float overflow.

### Derivative of Sigmoid

The derivative of sigmoid has a convenient form it can be expressed entirely in terms of the sigmoid output itself:

$$g'(z) = g(z) \cdot (1 - g(z))$$

Since `g(z)` is already stored in `outputs` from the forward pass, no recomputation is needed during backpropagation.

### ReLU (for later)

$$g(z) = \max(0, z)$$

More common in modern networks for hidden layers. Not used here yet.

---

## Loss Function

The loss function measures **how wrong the network's prediction is**. The goal of training is to minimize this number.

### Mean Squared Error (MSE)

$$loss = (output - target)^2$$

For multiple outputs:

$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (a_i - \hat{y}_i)^2$$

A loss of `0` means perfect prediction. The loss decreases as the network learns.

---

## Backpropagation

Backpropagation is how the network learns. Given the loss, it computes how much each weight and bias contributed to the error, working **backwards** from the output layer to the input layer using the chain rule.

The chain rule says: if `A` affects `B`, and `B` affects `C`, then how `A` affects `C` is the product of those two relationships. In a neural net the chain is:

```
weights → z → a → loss
```

### Delta

For each layer we compute **delta** `δ^[l]`, the error signal for that layer. It represents how much that layer is responsible for the final error.

**Output layer delta:**

$$\delta^{[L]} = (a^{[L]} - \text{target}) \cdot g'(z^{[L]})$$

Expanding `g'` using the sigmoid derivative property:

$$\delta_j = (a_j - \hat{y}_j) \cdot a_j \cdot (1 - a_j)$$

**Hidden layer delta:**

Hidden layers have no target to compare against. Instead the error signal flows backwards from the next layer:

$$\delta^{[l]} = (W^{[l+1]\top} \cdot \delta^{[l+1]}) \cdot g'(z^{[l]})$$

In code, for each node `i` in the hidden layer:

```
error[i] = sum over j of (next.weights[i * next.num_outputs + j] * next.deltas[j])
delta[i] = error[i] * outputs[i] * (1 - outputs[i])
```

This is why `prev` and `next` pointers are needed. The hidden delta reads from the next layer's weights and deltas while writing to the current layer.

### Weight Gradients

Once deltas are computed, the gradient for each weight is:

$$\frac{\partial L}{\partial w_{ij}} = a^{[l-1]}_i \cdot \delta^{[l]}_j$$

In code:

```
grad_w[i * num_outputs + j] = inputs[i] * deltas[j]
```

The gradient for each bias is simply the delta itself:

$$\frac{\partial L}{\partial b_j} = \delta^{[l]}_j$$

### Backward Pass Order

```
output layer   →  compute_output_delta(target)
hidden layer 2 →  compute_hidden_delta()          (reads from output layer)
hidden layer 1 →  compute_hidden_delta()          (reads from hidden layer 2)
...
```

This is the mirror of the forward pass. The error flows right to left, just as inputs flowed left to right.

---

## Gradient Descent

Once gradients are known, weights and biases are nudged in the direction that reduces the loss:

$$w = w - \alpha \cdot \frac{\partial L}{\partial w}$$
$$b = b - \alpha \cdot \delta$$

Where `α` (learning rate) is a small constant like `0.5` or `1.0` that controls the step size.

- Too large: overshoots the minimum, loss diverges
- Too small: learns very slowly, may get stuck

In code:

```
weights[i * num_outputs + j] = weights[i * num_outputs + j] - learning_rate * grad_w[i * num_outputs + j]
biases[j]                    = biases[j]                    - learning_rate * deltas[j]
```

---

## Training Loop

Training repeats forward pass, backward pass, and weight update over many **epochs**. Each epoch processes all training examples once.

```
for each epoch:
    for each training example (input, target):
        run_forward(head, input)         // compute z and a at every layer
        run_backward(output, target)     // compute deltas at every layer
        run_update(head, input, lr)      // nudge all weights and biases

    compute and log average loss
```

### XOR Results

XOR is the classic first test — it is not linearly separable, meaning a single layer cannot solve it. A network with one hidden layer of 4 nodes, trained for 50,000 epochs at learning rate `0.5`, produces:

```
[0.0, 0.0] → 0.001343   (target: 0)
[0.0, 1.0] → 0.996737   (target: 1)
[1.0, 0.0] → 0.996696   (target: 1)
[1.0, 1.0] → 0.004978   (target: 0)
```

Loss curve:

```
Epoch 0:      0.250035   (random weights)
Epoch 1000:   0.003609   (pattern found)
Epoch 10000:  0.000137   (refining)
Epoch 50000:  0.000025   (converged)
```

---

## Implementation Reference

### Layer Struct

```
pub const Layer -> struct {
    num_inputs:  int,
    num_outputs: int,
    weights:     *double,   // size: num_inputs * num_outputs
    biases:      *double,   // size: num_outputs
    z:           *double,   // pre-activation, size: num_outputs
    outputs:     *double,   // post-activation (a), size: num_outputs
    deltas:      *double,   // error signal, size: num_outputs
    grad_w:      *double,   // weight gradients, size: num_inputs * num_outputs
    prev:        *Layer,
    next:        *Layer,
};
```

### Layer Methods

| Method | Description |
|---|---|
| `init_weights_biases()` | Xavier weight init, biases to 0 |
| `forward(inputs)` | Computes z and a for this layer |
| `compute_output_delta(targets)` | Computes delta for output layer |
| `compute_hidden_delta()` | Computes delta for hidden layer using next layer |
| `update_weights(inputs, lr)` | Updates weights and biases via gradient descent |
| `print_forward()` | Prints z and a vectors |

### Module Functions

| Function | Description |
|---|---|
| `init_layer(in, out)` | Allocates a layer with given dimensions |
| `link_layers(head)` | Infers and sets all prev pointers from next pointers |
| `iter_init(l)` | Creates an iterator over the layer linked list |
| `free_layer(l)` | Frees all heap memory recursively |
| `run_forward(head, inputs)` | Runs forward pass through entire network |
| `run_backward(output, target)` | Runs backward pass through entire network |
| `run_update(head, inputs, lr)` | Updates all weights through entire network |
| `mse_loss(outputs, targets, n)` | Computes mean squared error |

### Memory Layout

All weight/bias/activation arrays are heap-allocated via `alloc` and freed with `free_layer`. The `next` pointer links layers into a singly linked list. `prev` pointers are inferred automatically by `link_layers` — you only need to set `next` manually. Stack-allocated layers must not have `free` called on the struct pointer itself, only on their heap-allocated field arrays.

