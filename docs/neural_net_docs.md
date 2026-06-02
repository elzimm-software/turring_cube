# neural_net

A feedforward neural network built from scratch in [Luma](https://luma-website-mu.vercel.app/), compiled to native code via LLVM. No ML libraries — just weights, matrix math, and sigmoid.

## Architecture

```
inputs → [ hidden layer ] → output
  2    →        4         →   1
```

Each layer is fully connected to the next. Every connection has a weight, every non-input node has a bias.

## Project Structure

```
bin/
  neural            — compiled output
saved_weights/
  and.bin           — trained AND gate weights
  nand.bin          — trained NAND gate weights
  or.bin            — trained OR gate weights
  nor.bin           — trained NOR gate weights
  xor.bin           — trained XOR gate weights
src/
  testing/
    test.lx         — generic test framework (TestSuite, TestCase)
    gate_tests.lx   — logic gate training and truth table tests
    layer_tests.lx  — layer unit tests (init, forward, mse, save/load)
  main.lx           — entry point, runs all test suites
  layer.lx          — Layer struct, forward/backward pass, weight init, save/load
  helper.lx         — sigmoid, random_double utilities
lumix.toml          — build config
```

## Building

```bash
lumix build
ulimit -s unlimited && bin/neural 
```

Or manually:

```bash
luma src/main.lx -l std/libc.lx src/helper.lx std/io.lx src/layer.lx src/testing/test.lx src/testing/gate_test.lx src/testing/layer_test.lx -O3 --no-sanitize -name bin/neural
```

## How It Works

### Layers

A `Layer` holds its weights, biases, and the results of the forward and backward pass:

```
pub const Layer -> struct {
    num_inputs:  int,
    num_outputs: int,
    weights:     *double,  // flattened matrix, size = num_inputs * num_outputs
    biases:      *double,  // size = num_outputs
    z:           *double,  // pre-activation values
    outputs:     *double,  // post-activation values
    deltas:      *double,  // error signals for backprop
    grad_w:      *double,  // weight gradients
    prev:        *Layer,   // linked list to previous layer
    next:        *Layer,   // linked list to next layer
};
```

Layers are linked together as a doubly linked list and traversed with `LayerIterable`.

### Weight Initialization

Weights use **Xavier initialization** which are random values in `±sqrt(6 / (num_inputs + num_outputs))`. This keeps gradients from vanishing or exploding across layers.

Biases start at `0.0`.

### Forward Pass

For each layer, two steps:

**Step 1 — compute z (pre-activation):**
```
z[j] = bias[j] + sum(inputs[i] * weights[i * num_outputs + j])
```

**Step 2 — compute a (post-activation):**
```
a[j] = sigmoid(z[j]) = 1 / (1 + e^(-z[j]))
```

The output `a` of each layer becomes the input to the next layer.

### Backpropagation

**Output layer delta:**
```
delta[j] = (output[j] - target[j]) * sigmoid'(z[j])
```

**Hidden layer delta:**
```
delta[i] = sum(next.weights[i][j] * next.delta[j]) * sigmoid'(z[i])
```

**Weight update (gradient descent):**
```
weights[i][j] -= learning_rate * inputs[i] * delta[j]
biases[j]     -= learning_rate * delta[j]
```

### Save / Load Weights

Trained weights are saved to binary files and can be loaded back without retraining. The format is simple — each layer writes its header (`num_inputs`, `num_outputs`) followed by raw `double` arrays for weights and biases. Layers are written sequentially into a single file.

### Loss

Mean squared error:
```
loss = sum((output[i] - target[i])^2) / n
```

## Example Output

```
Test suite: layer
  ✓ init_layer sets num_inputs
  ✓ init_layer sets num_outputs
  ✓ forward outputs bounded 0..1
  ✓ mse_loss identical is zero
  ✓ mse_loss 1 vs 0 is 1
  ✓ save/load weights round trip (hidden)
  ✓ save/load weights round trip (output)
7 tests, 7 passed, 0 failed

Test suite: logic gates
  ✓ AND
  ✓ NAND
  ✓ OR
  ✓ NOR
  ✓ XOR
5 tests, 5 passed, 0 failed

╔══════╦════════╦════════╦════════╦════════╦════════╗
║ A  B ║  AND   ║  NAND  ║   OR   ║  NOR   ║  XOR   ║
╠══════╬════════╬════════╬════════╬════════╬════════╣
║ 0  0 ║    0   ║    1   ║    0   ║    1   ║    0   ║
║ 0  1 ║    0   ║    1   ║    1   ║    0   ║    1   ║
║ 1  0 ║    0   ║    1   ║    1   ║    0   ║    1   ║
║ 1  1 ║    1   ║    0   ║    1   ║    0   ║    0   ║
╚══════╩════════╩════════╩════════╩════════╩════════╝
```

## Status

- [x] Layer struct with doubly linked list
- [x] Xavier weight initialization
- [x] Forward pass (z and a)
- [x] Sigmoid activation
- [x] MSE loss
- [x] Backpropagation
- [x] Weight updates (gradient descent)
- [x] Training loop
- [x] Save / load weights to binary file
- [x] Logic gate test suite (AND, NAND, OR, NOR, XOR)
- [x] Truth table output