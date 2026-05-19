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

# Turing-Tape Neural Network — Technical Documentation

## Table of Contents

1. [What is a Turing Machine (Plain English)](#what-is-a-turing-machine-plain-english)
2. [Notation](#notation)
3. [Architecture Overview](#architecture-overview)
4. [The 3D Subspace Geometry](#the-3d-subspace-geometry)
5. [Tapes as Layers](#tapes-as-layers)
6. [The Head Mechanism](#the-head-mechanism)
7. [Sparse Vector-Blocked Storage](#sparse-vector-blocked-storage)
8. [State Transitions](#state-transitions)
9. [Forward Pass](#forward-pass)
10. [Bounding the Halting Problem with Epochs](#bounding-the-halting-problem-with-epochs)
11. [Backpropagation Through State](#backpropagation-through-state)
12. [Turing Completeness Argument](#turing-completeness-argument)
13. [Comparison to Standard Neural Net](#comparison-to-standard-neural-net)
14. [Implementation Reference](#implementation-reference)

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

## What is a Turing Machine (Plain English)

Before the math, here is a simple mental model.

Imagine a very long strip of paper, a **tape**, divided into squares. Each square can hold a symbol (a number, a letter, or be blank). You also have a **reading head** a little robot that sits on the tape and can:

1. **Read** the symbol in the square it is currently on
2. **Write** a new symbol to that square
3. **Move** one step left or right
4. **Change its own internal state** (think of this as the robot's "mood" it remembers what mode it is in)

That is the entire machine. Despite being this simple, Alan Turing proved in 1936 that a machine like this can compute *anything that can be computed at all*. The reason is:

- The tape is **unbounded**; it can grow as long as needed (infinite memory)
- The state + transition rules act like a **program**; different states do different things
- The head's ability to move back and forth gives it **random access** to any memory location

A regular neural network is **not** Turing complete because it has fixed width, fixed depth, and no unbounded memory. It is more like a very complicated lookup table. This architecture fixes that.

### The Tape Analogy in This Network

| Turing Machine | This Network |
|---|---|
| The tape | A layer (a 2D subplane in 3D space) |
| A square on the tape | A position `(x, y)` on that layer |
| The symbol in a square | The activation value at `(x, y)` |
| The read/write head | The head vector `H_k` pointing to a position |
| The machine's internal state | The state vector `S_k` carried between layers |
| Moving left/right | Shifting the head position |
| Transition rules | The learned weight matrices |

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $k$ | Tape (layer) index. Tape 0 is input, tape $K$ is output |
| $z = k$ | The Z-coordinate (depth) of tape $k$ in R³ |
| $(x, y)$ | Position on a tape's 2D surface |
| $\mathcal{T}_k$ | The sparse activation map for tape $k$: positions $(x,y)$ → value |
| $H_k \in \mathbb{R}^2$ | Head position on tape $k$. A continuous $(x, y)$ coordinate |
| $S_k \in \mathbb{R}^d$ | State vector at tape $k$. Carries memory forward across tapes |
| $r_k$ | Read vector: the activation values read near head position $H_k$ |
| $W_S, W_H, W_W$ | Learned weight matrices for state, head movement, and write |
| $\epsilon$ | Sparsity threshold. Activations below this are not stored |
| $\mathcal{B}_k$ | Set of active blocks on tape $k$ (sparse storage blocks) |
| $\alpha$ | Learning rate |
| $T$ | Maximum number of epochs (the halting bound) |

---

## Architecture Overview

A standard neural net is a pipeline: input flows forward, each layer transforms it, done. There is no memory between examples.

This architecture adds three things:

1. **Tapes instead of flat layers** — each layer is a 2D surface in 3D space, not just a vector. It can be read from any position, like a memory array.
2. **A state vector** — a small vector that carries information from tape to tape, acting as working memory.
3. **A head** — a differentiable pointer that decides *where* on the tape to read and write each step.

```
        Z-axis (depth)
        │
   k=2  │   [tape 2 — 2D surface at z=2]
        │         ↑ head H_2 reads/writes here
   k=1  │   [tape 1 — 2D surface at z=1]
        │         ↑ head H_1
   k=0  │   [tape 0 — input tape at z=0]
        │
        └─────────────────────── X-axis
       /
      Y-axis
```

Each tape is **sparse**: only positions with activation above threshold $\epsilon$ are stored. This is what makes it theoretically infinitely scalable an empty tape costs nothing.

---

## The 3D Subspace Geometry

**Plain English:** Think of each tape as a sheet of graph paper floating at a fixed height. The height is the tape number. The sheet can be arbitrarily wide and tall — you only pay for the squares you write on.

**Math:** R³ is three-dimensional space with coordinates $(x, y, z)$. A **subspace** at depth $k$ is the set of all points with $z = k$:

$$\text{Tape}_k = \{ (x, y, k) \mid x, y \in \mathbb{R} \}$$

This is a copy of R² embedded in R³. Each tape is isomorphic to R² — it *is* just a 2D plane, but we track its depth so tapes don't overlap and we can reason about connections between them as vectors in 3D.

**Why this matters:** A connection from position $(x_1, y_1, k)$ on tape $k$ to position $(x_2, y_2, k+1)$ on tape $k+1$ is literally a vector in R³:

$$\vec{v} = (x_2 - x_1,\ y_2 - y_1,\ 1)$$

The geometry gives you spatial intuition about information flow. Connections that stay near the same $(x,y)$ are local. Connections that jump far in $x$ or $y$ are long-range.

---

## Tapes as Layers

Each tape $k$ stores a **sparse map** from 2D positions to activation values:

$$\mathcal{T}_k : \mathbb{R}^2 \to \mathbb{R}, \quad \text{defined only where } |\mathcal{T}_k(x,y)| > \epsilon$$

**Plain English:** The tape is like a spreadsheet where most cells are blank. You only store the cells that have a non-trivial value. This is the key to infinite scalability — a tape with 10 active positions costs the same as a tape with 10,000 active positions if the rest are zero.

### Discrete Implementation

In practice, tape positions are discretized to integer grid coordinates $(i, j)$. The tape is stored as a hash map:

```
tape_k: HashMap<(i32, i32), f64>
```

A position $(i, j)$ exists in the map if and only if $|\mathcal{T}_k(i,j)| > \epsilon$. Reading a missing position returns 0.

### Tape Initialization

Tape 0 (input) is written from the input data. All other tapes start empty (all zeros, nothing stored).

---

## The Head Mechanism

**Plain English:** The head is a spotlight. At each tape, it points to a particular $(x, y)$ location. The network reads activations near that spot, does its computation, then decides where to move the spotlight on the next tape.

The head is a **continuous** coordinate — not an integer index — so it is differentiable. We can train it with gradient descent.

### Reading from the Tape

To read from tape $k$ at head position $H_k = (h_x, h_y)$, we use a **soft attention** over nearby positions. For each stored position $(i, j)$:

$$w_{ij} = \exp\!\left(-\frac{(i - h_x)^2 + (j - h_y)^2}{2\sigma^2}\right)$$

**Plain English:** positions close to the head get high weight, positions far away get low weight. $\sigma$ controls the "spread" of the spotlight.

The read vector $r_k$ is the weighted sum of all stored activations:

$$r_k = \frac{\sum_{(i,j) \in \mathcal{B}_k} w_{ij} \cdot \mathcal{T}_k(i,j)}{\sum_{(i,j) \in \mathcal{B}_k} w_{ij}}$$

This is a scalar (or vector if the tape stores vectors at each position). It is differentiable with respect to $H_k$.

### Writing to the Tape

The network also writes a value $v_k$ back to the tape at the head position:

$$\mathcal{T}_{k+1}(i,j) \mathrel{+}= w_{ij} \cdot v_k$$

Positions where the result is below $\epsilon$ are pruned.

### Head Movement

After reading, the head position for the next tape is computed:

$$H_{k+1} = H_k + \Delta H_k$$

where the **movement** $\Delta H_k$ is a 2-vector produced by the network:

$$\Delta H_k = \tanh(W_H \cdot [r_k ; S_k] + b_H)$$

**Plain English:** the network learns how far to move the head left/right/up/down based on what it just read and what it remembers. $\tanh$ bounds the movement to $(-1, 1)$ per step, so the head cannot teleport.

---

## Sparse Vector-Blocked Storage

**Plain English:** Instead of allocating a giant array for every tape, we divide each tape into small fixed-size blocks (e.g. 8×8 squares of positions). A block only exists in memory if at least one of its positions is active. This is like the difference between a dense image and a sparse point cloud.

### Block Structure

Divide the tape into blocks of size $B \times B$ (e.g. $B = 8$). A block at grid-block coordinate $(p, q)$ covers tape positions:

$$\{(i, j) \mid pB \le i < (p+1)B,\ qB \le j < (q+1)B\}$$

A block is stored if any value in it exceeds $\epsilon$. The full tape is:

$$\mathcal{B}_k = \{ \text{block}(p,q) \mid \exists\ (i,j)\ \text{in block}\ (p,q)\ \text{s.t.}\ |\mathcal{T}_k(i,j)| > \epsilon \}$$

### Memory Cost

If $N_k$ is the number of active blocks on tape $k$:

$$\text{memory}(\mathcal{T}_k) = N_k \cdot B^2 \cdot \text{sizeof(f64)}$$

For $B = 8$ and f64 values: each block costs $8 \times 8 \times 8 = 512$ bytes. A tape with 100 active blocks costs 51.2 KB regardless of how far apart those blocks are in coordinate space.

### Pruning

After each write, sweep active blocks and remove any where all values are below $\epsilon$:

```
for block in tape_k.blocks:
    if max(|v| for v in block) < ε:
        tape_k.remove_block(block)
```

---

## State Transitions

The state vector $S_k$ is the network's **working memory** — a dense vector of size $d$ that flows from tape to tape carrying context.

**Plain English:** If the tapes are the notebook, the state vector is what the reader holds in their head while flipping pages. It does not have to be written down anywhere — it just persists.

### State Update

$$S_{k+1} = \tanh\!\left(W_S \cdot [r_k ; S_k] + b_S\right)$$

where $[r_k ; S_k]$ means the read value and state vector concatenated into one input. This is a single dense layer applied at each tape transition.

**Why tanh?** It bounds the state values to $(-1, 1)$, preventing the state from exploding over many tapes. You can substitute an LSTM-style gating mechanism here for better gradient flow (see Backpropagation section).

### Write Value

The value written to the next tape is:

$$v_k = W_W \cdot [r_k ; S_k] + b_W$$

This is what gets distributed across the tape around head position $H_{k+1}$.

### Full Transition Summary

Given tape $k$, head $H_k$, state $S_k$:

1. Read: $r_k \leftarrow \text{soft\_read}(\mathcal{T}_k, H_k)$
2. New state: $S_{k+1} \leftarrow \tanh(W_S [r_k; S_k] + b_S)$
3. Head move: $\Delta H_k \leftarrow \tanh(W_H [r_k; S_k] + b_H)$, then $H_{k+1} \leftarrow H_k + \Delta H_k$
4. Write value: $v_k \leftarrow W_W [r_k; S_k] + b_W$
5. Write to tape: $\mathcal{T}_{k+1}(i,j) \mathrel{+}= w_{ij} \cdot v_k$ for nearby $(i,j)$
6. Prune tape $k+1$

---

## Forward Pass

The full forward pass runs the transition for each tape from $k=0$ to $k=K$:

```
initialize:
    T_0  ← write input data to tape 0 at position H_0 = (0, 0)
    S_0  ← zero vector of size d
    H_0  ← (0, 0)

for k = 0 to K-1:
    r_k    ← soft_read(T_k, H_k)
    S_{k+1} ← tanh(W_S · [r_k; S_k] + b_S)
    ΔH_k   ← tanh(W_H · [r_k; S_k] + b_H)
    H_{k+1} ← H_k + ΔH_k
    v_k    ← W_W · [r_k; S_k] + b_W
    soft_write(T_{k+1}, H_{k+1}, v_k)
    prune(T_{k+1})

output ← soft_read(T_K, H_K)
```

### Comparison to Standard Forward Pass

In your existing network, layer $l$ computes:

$$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}, \qquad a^{[l]} = g(z^{[l]})$$

In this network, tape $k$ computes:

$$r_k = \text{soft\_read}(\mathcal{T}_k, H_k), \qquad S_{k+1} = \tanh(W_S [r_k; S_k] + b_S)$$

The difference: the input to each step is not the full previous layer — it is a **single read vector** at a **learned location**, plus the carried **state**. The tape stores information spatially; the head picks what to look at.

---

## Bounding the Halting Problem with Epochs

**Plain English:** A true Turing machine can loop forever — there is no guarantee it stops. This is the famous Halting Problem: you cannot always know in advance whether a program will finish. For a learning system this is unacceptable. We solve it by simply deciding in advance the maximum amount of work the network is allowed to do.

### Epoch Bound

Define a maximum epoch count $T$. Training runs for exactly $T$ epochs and then stops, regardless of whether the loss has converged. This is not a loss of generality — it is the standard practice in all neural network training. It converts an undecidable problem into a decidable one by construction.

$$\text{Training terminates at } t = T$$

### Tape Depth Bound

Similarly, the number of tapes $K$ is fixed at construction time. The network does not decide dynamically how many tapes to use. This bounds the computation per forward pass:

$$\text{FLOPs per forward pass} \le K \cdot C \cdot N_{\max}$$

where $C$ is the cost per transition and $N_{\max}$ is the maximum active blocks per tape. If tapes are sparse and $N_{\max}$ is small, this is very efficient even for large $K$.

### Why This Still Gives Turing Completeness

Turing completeness is a statement about what can *in principle* be computed, not how long it takes. By making $K$ and $T$ arbitrarily large (but finite), the network can simulate any computation that terminates within $T$ steps on a tape of width $K$. For any specific target computation, there exist values of $K$ and $T$ large enough to solve it.

This is the same argument that makes all real computers Turing-complete despite having finite memory and finite runtime — the bound exists but can be made as large as needed.

---

## Backpropagation Through State

**Plain English:** Training this network is harder than a standard net because the state vector and head position flow from tape to tape — we have to propagate gradients back through all of those connections. This is the same challenge as training a Recurrent Neural Network (RNN), and the same solutions apply.

### Truncated BPTT

The standard approach is **Backpropagation Through Time (BPTT)**: unroll the tape transitions into a computation graph and run standard backprop. The depth of unrolling is $K$ (number of tapes).

For very large $K$, use **truncated BPTT**: detach the gradient graph every $\tau$ tapes. This introduces bias but prevents memory from growing linearly with $K$.

### Gradient Through Soft Read/Write

The soft read operation is differentiable with respect to the head position:

$$\frac{\partial r_k}{\partial H_k} = \frac{\partial}{\partial H_k}\left[\frac{\sum_{ij} w_{ij} \cdot \mathcal{T}_k(i,j)}{\sum_{ij} w_{ij}}\right]$$

Since $w_{ij} = \exp(-(||({i,j}) - H_k||^2 / 2\sigma^2))$, the gradient is a sum of Gaussian-weighted differences — smooth everywhere.

### Vanishing Gradients

The $\tanh$ state update can cause vanishing gradients for large $K$. Solutions in order of preference:

1. Replace the state update with a **GRU** (Gated Recurrent Unit) cell — gates control gradient flow and largely solve this
2. Use **residual connections**: $S_{k+1} = S_k + \tanh(W_S [r_k; S_k] + b_S)$ — the additive skip gives gradients a direct path
3. Use **gradient clipping**: cap $||\nabla|| \le C_{\text{clip}}$ at each update step

---

## Turing Completeness Argument

A system is Turing complete if it can simulate a Universal Turing Machine. We need to show this architecture can implement:

1. **Unbounded memory** — tapes are sparse and expand dynamically. Any position $(i, j)$ can be written. ✓
2. **A state register** — the state vector $S_k$ holds the machine's current state. With $d$ dimensions it can encode any finite number of distinct states. ✓
3. **A transition function** — the weight matrices $W_S, W_H, W_W$ implement a parametric function of (read value, current state) → (new state, head movement, write value). Neural networks are universal function approximators, so any transition table can in principle be approximated. ✓
4. **Read/write access** — the head mechanism provides differentiable read and write at any tape position. ✓
5. **Conditional branching** — the head movement and write value depend on the state and what was read. Different states + inputs lead to different moves and writes, implementing all branches a transition table requires. ✓

All five requirements are met. Therefore the architecture is Turing complete in the limit of $K \to \infty$ and $d \to \infty$.

---

## Comparison to Standard Neural Net

| Property | Standard Net (your doc) | Turing-Tape Net |
|---|---|---|
| Layer representation | Dense vector $a^{[l]} \in \mathbb{R}^n$ | Sparse 2D tape $\mathcal{T}_k : \mathbb{R}^2 \to \mathbb{R}$ |
| Memory between layers | None — each layer is stateless | State vector $S_k$ flows forward |
| Memory between examples | None | State optionally persists (stateful mode) |
| Layer connections | All-to-all dense matrix multiply | Soft attention via head position |
| Storage cost | $O(n^2)$ per layer | $O(N_k)$ active blocks per tape |
| Turing complete | No | Yes (in the limit) |
| Halting bound | N/A | $K$ tapes, $T$ epochs |
| Gradient method | Standard backprop | BPTT through tape transitions |
| Closest existing model | MLP / Feedforward net | Neural Turing Machine + Sparse Transformer |

---

## Implementation Reference

### Core Structs

```
pub const Tape -> struct {
    k:        usize,                        // tape index
    blocks:   HashMap<(i32, i32), Block>,   // active blocks only
    epsilon:  f64,                          // sparsity threshold
};

pub const Block -> struct {
    data:     [B][B]f64,                    // B×B grid of activations
    grad:     [B][B]f64,                    // gradient for backprop
};

pub const TapeNet -> struct {
    tapes:    []Tape,                       // K+1 tapes
    heads:    [][2]f64,                     // head positions H_0..H_K
    states:   [][]f64,                      // state vectors S_0..S_K

    W_S:      [][]f64,                      // state transition weights  [d × (1+d)]
    W_H:      [][]f64,                      // head movement weights     [2 × (1+d)]
    W_W:      [][]f64,                      // write value weights       [1 × (1+d)]

    b_S:      []f64,                        // state bias
    b_H:      []f64,                        // head movement bias
    b_W:      []f64,                        // write bias

    sigma:    f64,                          // Gaussian read/write spread
    d:        usize,                        // state vector dimension
    K:        usize,                        // number of tapes
};
```

### Key Functions

| Function | Description |
|---|---|
| `soft_read(tape, head, sigma)` | Gaussian-weighted read at head position |
| `soft_write(tape, head, value, sigma)` | Gaussian-weighted write to tape |
| `prune_tape(tape, epsilon)` | Remove blocks below threshold |
| `tape_forward(net, input)` | Full forward pass over all K tapes |
| `tape_backward(net, target)` | BPTT through all tape transitions |
| `tape_update(net, lr)` | Update W_S, W_H, W_W via gradient descent |
| `init_tape_net(K, d, sigma, epsilon)` | Allocate and Xavier-init all weight matrices |

### Hyperparameter Guide

| Hyperparameter | Role | Suggested start |
|---|---|---|
| $K$ | Number of tapes (computation depth) | 8–32 |
| $d$ | State vector dimension | 32–128 |
| $B$ | Block size for sparse storage | 8 |
| $\epsilon$ | Sparsity threshold | 1e-4 |
| $\sigma$ | Head read/write spread | 1.0 |
| $\alpha$ | Learning rate | 1e-3 |
| $T$ | Max training epochs (halting bound) | 10,000–100,000 |
| $C_{\text{clip}}$ | Gradient clip norm | 1.0–5.0 |
