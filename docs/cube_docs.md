# Turing-Tape Neural Network — Technical Documentation

## Table of Contents

1. [What is a Turing Machine (Plain English)](#what-is-a-turing-machine-plain-english)
2. [Notation](#notation)
3. [Architecture Overview](#architecture-overview)
4. [The 3D Subspace Geometry](#the-3d-subspace-geometry)
5. [Tapes as Layers](#tapes-as-layers)
6. [Long-Term Memory Store](#long-term-memory-store)
7. [The Head Mechanism](#the-head-mechanism)
8. [Exponential Decay Storage](#exponential-decay-storage)
9. [Sparse Vector-Blocked Storage](#sparse-vector-blocked-storage)
10. [State Transitions](#state-transitions)
11. [Forward Pass](#forward-pass)
12. [Bounding the Halting Problem with Epochs](#bounding-the-halting-problem-with-epochs)
13. [Backpropagation Through State](#backpropagation-through-state)
14. [Turing Completeness Argument](#turing-completeness-argument)
15. [Comparison to Standard Neural Net](#comparison-to-standard-neural-net)
16. [Implementation Reference](#implementation-reference)

---

## What is a Turing Machine (Plain English)

Before the math, here is a simple mental model.

Imagine a very long strip of paper — a **tape** — divided into squares. Each square can hold a symbol (a number, a letter, or be blank). You also have a **reading head**: a little robot that sits on the tape and can:

1. **Read** the symbol in the square it is currently on
2. **Write** a new symbol to that square
3. **Move** one step left or right
4. **Change its own internal state** (think of this as the robot's "mood" — it remembers what mode it is in)

That is the entire machine. Despite being this simple, Alan Turing proved in 1936 that a machine like this can compute *anything that can be computed at all*. The reason is:

- The tape is **unbounded**: it can grow as long as needed (infinite memory)
- The state + transition rules act like a **program**: different states do different things
- The head's ability to move back and forth gives it **random access** to any memory location

A regular neural network is **not** Turing complete because it has fixed width, fixed depth, and no unbounded memory. It is more like a very complicated lookup table. This architecture fixes that.

### The Tape Analogy in This Network

| Turing Machine | This Network |
|---|---|
| The tape | A layer (a 2D subplane in 3D space) |
| A square on the tape | A position `(x, y)` on that layer |
| The symbol in a square | The activation value at `(x, y)` |
| The read/write head | The head vector $H_k$ pointing to a position |
| The machine's internal state | The state vector $S_k$ carried between layers |
| Moving left/right | Shifting the head position |
| Transition rules | The learned weight matrices |
| Long-term paper record | Long-term memory store $M$ |

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $k$ | Tape (layer) index. Tape 0 is input, tape $K$ is output |
| $z = k$ | The Z-coordinate (depth) of tape $k$ in $\mathbb{R}^3$ |
| $(x, y)$ | Position on a tape's 2D surface |
| $\mathcal{T}_k$ | The sparse activation map for tape $k$: positions $(x,y) \to$ value |
| $M$ | The long-term memory store: a persistent sparse map, no decay |
| $H_k \in \mathbb{R}^2$ | Tape head position on tape $k$ |
| $H_M \in \mathbb{R}^2$ | Long-term memory head position (independent of $H_k$) |
| $S_k \in \mathbb{R}^d$ | State vector at tape $k$. Carries memory forward across tapes |
| $r_k$ | Read vector from tape $k$ at head position $H_k$ |
| $m_k$ | Read vector from long-term memory $M$ at head position $H_M$ |
| $g_k$ | Write gate scalar for long-term memory at step $k$ |
| $e_k$ | Erase vector for long-term memory at step $k$ |
| $\gamma$ | Tape activation decay factor. $\gamma \in (0, 1)$ |
| $\gamma_S$ | State carry factor for exponential moving average. $\gamma_S \in (0, 1)$ |
| $\lambda$ | Weight decay coefficient |
| $\theta_\text{commit}$ | Commit threshold for LTM writes (e.g. 0.9) |
| $W_S, W_H, W_W$ | Learned weight matrices for state, head movement, and tape write |
| $W_g, W_e, W_{Wm}, W_{Hm}$ | Learned weight matrices for LTM gate, erase, write value, head move |
| $\epsilon$ | Tape sparsity threshold |
| $\epsilon_M$ | LTM sparsity threshold (typically larger than $\epsilon$) |
| $\mathcal{B}_k$ | Set of active blocks on tape $k$ |
| $\alpha$ | Learning rate |
| $T$ | Maximum number of epochs (the halting bound) |

---

## Architecture Overview

A standard neural net is a pipeline: input flows forward, each layer transforms it, done. There is no memory between examples.

This architecture adds four things:

1. **Tapes instead of flat layers** — each layer is a 2D surface in 3D space. It can be read from any position, like a memory array.
2. **A state vector** — a small vector that carries information from tape to tape, acting as working memory.
3. **A tape head** — a differentiable pointer that decides *where* on the tape to read and write each step.
4. **A long-term memory store** — a persistent sparse map $M$ that does not decay. The network learns what is worth committing to it via a gated write mechanism.

```
        Z-axis (depth)
        │
   k=2  │   [tape 2 — 2D surface at z=2]   ←→  [LTM store M — no decay]
        │         ↑ head H_2                          ↑ head H_M (independent)
   k=1  │   [tape 1 — 2D surface at z=1]
        │         ↑ head H_1
   k=0  │   [tape 0 — input tape at z=0]
        │
        └──────────────────────────────── X-axis
       /
      Y-axis
```

The two memory tiers serve distinct roles:

- **Tapes $\mathcal{T}_k$** hold transient working state. Activations decay with factor $\gamma < 1$ across tapes. Stored in log-domain `LogCell` structs for efficiency.
- **Long-term memory $M$** holds patterns the network has decided are worth persisting. No decay. Written only when a learned gate fires above a commit threshold. Stored in plain `f64` blocks.

Each tape is **sparse**: only positions with activation above threshold $\epsilon$ are stored. $M$ is also sparse, pruned only when values fall below $\epsilon_M$ (not by decay).

---

## The 3D Subspace Geometry

**Plain English:** Think of each tape as a sheet of graph paper floating at a fixed height. The height is the tape number. The sheet can be arbitrarily wide and tall — you only pay for the squares you write on.

**Math:** $\mathbb{R}^3$ is three-dimensional space with coordinates $(x, y, z)$. A subspace at depth $k$ is the set of all points with $z = k$:

$$\text{Tape}_k = \{ (x, y, k) \mid x, y \in \mathbb{R} \}$$

This is a copy of $\mathbb{R}^2$ embedded in $\mathbb{R}^3$. A connection from position $(x_1, y_1, k)$ on tape $k$ to position $(x_2, y_2, k+1)$ on tape $k+1$ is a vector in $\mathbb{R}^3$:

$$\vec{v} = (x_2 - x_1,\ y_2 - y_1,\ 1)$$

The long-term memory store $M$ has no fixed $z$-coordinate — it lives outside the tape stack, accessible from any tape step via $H_M$.

---

## Tapes as Layers

Each tape $k$ stores a **sparse map** from 2D positions to activation values:

$$\mathcal{T}_k : \mathbb{R}^2 \to \mathbb{R}, \quad \text{defined only where } |\mathcal{T}_k(x,y)| > \epsilon$$

Tape positions are discretized to integer grid coordinates $(i, j)$. The tape is stored as a hash map:

```
tape_k: HashMap<(i32, i32), LogCell>
```

A `LogCell` stores the log-magnitude, sign, and write-tape index of the value. Reading a missing position returns 0. See [Exponential Decay Storage](#exponential-decay-storage) for the full struct definition.

### Tape Initialization

Tape 0 (input) is written from the input data. All other tapes start empty. The long-term memory $M$ persists across forward passes in stateful mode, and is reset to empty at the start of a new episode in stateless mode.

---

## Long-Term Memory Store

$M$ is a persistent sparse map sharing the same block storage structure as tapes, but with no decay:

$$M : \mathbb{R}^2 \to \mathbb{R}, \quad \text{defined only where } |M(x,y)| > \epsilon_M$$

It has its own head position $H_M \in \mathbb{R}^2$ that moves independently of the tape heads. This independence is critical: $H_k$ advances with computation, while $H_M$ navigates to wherever similar patterns were previously stored — acting as a content-addressable lookup.

### Why a Separate Head

If $H_M$ were forced to track $H_k$, the LTM would be nothing more than a non-decaying second tape at the same position. The power of LTM comes from the network learning to steer $H_M$ to the right location based on the current state and what it just read. After sufficient training, $H_M$ will diverge significantly from $H_k$ and this is the intended behavior.

### Commit Threshold

Writes to $M$ are gated by $g_k > \theta_\text{commit}$. A high threshold (e.g. 0.9) creates a strong inductive bias toward sparsity: most steps read from $M$ but do not write to it. If $\theta_\text{commit}$ is set too low, $M$ degrades into a second decaying tape and the tier separation is lost.

### Erase-Then-Add Writes

Without an erase step, each commit accumulates at the same position across training examples and the stored values drift toward an uninterpretable average. The NTM-style erase-then-add makes writes *replace* rather than *accumulate*:

$$M(i,j) \leftarrow M(i,j) \cdot \bigl[1 - g_k \cdot e_k \cdot w_{ij}\bigr] \quad \text{(erase)}$$

$$M(i,j) \leftarrow M(i,j) + g_k \cdot w_{ij} \cdot v_M \quad \text{(add)}$$

where $w_{ij}$ is the Gaussian attention weight at position $(i, j)$ relative to $H_M$, and $e_k \in (0,1)$ is the learned erase signal. The combination lets the network partially or fully overwrite a memory location.

### LTM Pruning

$M$ does not decay, but the erase mechanism and sparse commit threshold keep it from growing indefinitely. After each write, prune any block where:

```
max |M(i,j)| over block < ε_M
```

LTM blocks use plain `f64` values — no `LogCell` needed since there is no decay term to apply. This also makes LTM reads slightly cheaper than tape reads.

---

## The Head Mechanism

### Tape Head

The tape head $H_k \in \mathbb{R}^2$ is a continuous coordinate — not an integer index — so it is differentiable and trainable with gradient descent.

**Reading from the tape:** To read from tape $k$ at head position $H_k = (h_x, h_y)$, use a soft attention over nearby positions. For each stored position $(i, j)$:

$$w_{ij} = \exp\!\left(-\frac{(i - h_x)^2 + (j - h_y)^2}{2\sigma^2}\right)$$

The read vector $r_k$ is the normalized weighted sum:

$$r_k = \frac{\sum_{(i,j) \in \mathcal{B}_k} w_{ij} \cdot \mathcal{T}_k(i,j)}{\sum_{(i,j) \in \mathcal{B}_k} w_{ij}}$$

**Writing to the tape:**

$$\mathcal{T}_{k+1}(i,j) \mathrel{+}= w_{ij} \cdot v_k$$

**Tape head movement:**

$$\Delta H_k = \tanh(W_H \cdot [r_k ; m_k ; S_k] + b_H), \quad H_{k+1} \leftarrow H_k + \Delta H_k$$

Note that head movement now conditions on the LTM read $m_k$ as well as the tape read and state, allowing the tape head to factor in long-term context when deciding where to look next.

### LTM Head

The LTM head $H_M \in \mathbb{R}^2$ moves independently:

$$\Delta H_M = \tanh(W_{Hm} \cdot [r_k ; m_k ; S_k] + b_{Hm}), \quad H_M \leftarrow H_M + \Delta H_M$$

**Reading from LTM:**

$$m_k = \frac{\sum_{(i,j) \in M} w^M_{ij} \cdot M(i,j)}{\sum_{(i,j) \in M} w^M_{ij}}, \quad w^M_{ij} = \exp\!\left(-\frac{\|(i,j) - H_M\|^2}{2\sigma^2}\right)$$

Both reads — $r_k$ and $m_k$ — are differentiable with respect to their respective head positions.

---

## Exponential Decay Storage

Exponential decay replaces the standard gradient descent update with a weight decay term and introduces lazy activation decay on tapes. The key insight is that in log-space, multiplicative decay becomes an additive constant, eliminating all per-step multiplications.

### Weight Decay

**Standard update (no decay):**

$$w \leftarrow w - \alpha \cdot \nabla w$$

**With exponential decay:**

$$w \leftarrow w \cdot (1 - \alpha\lambda) - \alpha \cdot \nabla w$$

The factor $(1 - \alpha\lambda)$ shrinks every weight toward zero at each step. Weights that decay below $\ln \epsilon$ are pruned exactly as tape activations are, making the sparse block structure self-cleaning.

**Log-domain form:** Store $\hat{w} = \ln|w|$ plus a sign bit. The decay becomes an additive constant per step:

$$\hat{w} \leftarrow \hat{w} + \ln(1 - \alpha\lambda) - \alpha \cdot \nabla w \cdot e^{-\hat{w}}$$

Since $\ln(1 - \alpha\lambda)$ is a constant computed once per training run, the decay costs a single addition rather than a multiplication.

### Tape Activation Decay

Each tape cell decays between tapes, not just at prune time. With lazy decay, the decay is applied at read time rather than on a periodic sweep:

$$\mathcal{T}_k(i,j) \leftarrow \mathcal{T}_k(i,j) \cdot \gamma^{(k - k_\text{write})}$$

where $k_\text{write}$ is the tape index when the value was last written and $\gamma \in (0,1)$ is the decay factor.

**Log-domain form (no multiply needed):**

$$\log|\mathcal{T}| \leftarrow \log|\mathcal{T}| + (k - k_\text{write}) \cdot \ln\gamma$$

This is a single addition scaled by an integer. The pruning threshold check in log-domain is:

$$\log|\mathcal{T}| < \ln\epsilon \quad \Rightarrow \quad \text{discard}$$

No `exp` or multiply is needed — just a comparison.

### LogCell Struct

Each tape block entry uses a compact log-domain struct instead of a raw `f64`:

```
struct LogCell {
    log_abs: f32,    // ln|value| — decay is an additive constant
    sign:    i8,     // +1 or −1
    k_write: u16,    // tape index of last write (for lazy decay)
}
// 7 bytes vs 8 bytes (f64), plus lazy decay eliminates sweep loops
```

To recover the actual value for Gaussian-weighted reads:

```
actual = sign * exp(log_abs + (k_current - k_write) * ln_gamma)
```

If read performance is a bottleneck, cache `abs_val: f32` alongside `log_abs` and use the log path only for pruning decisions.

**LTM blocks do not use `LogCell`** — they store plain `f64` values. There is no decay term, no `k_write` needed, and avoiding the log/exp round-trip makes LTM reads slightly cheaper.

---

## Sparse Vector-Blocked Storage

Each tape and the LTM store are divided into fixed-size blocks. A block only exists in memory if at least one of its positions is active.

### Block Structure

Divide the tape into blocks of size $B \times B$ (e.g. $B = 8$). A block at grid-block coordinate $(p, q)$ covers tape positions:

$$\{(i, j) \mid pB \le i < (p+1)B,\ qB \le j < (q+1)B\}$$

**Tape block:**

```
pub const Block -> struct {
    data: [B][B]LogCell,    // log-domain storage with lazy decay
    grad: [B][B]f64,        // gradient for backprop (full precision)
};
```

**LTM block:**

```
pub const LTMBlock -> struct {
    data: [B][B]f64,        // full precision, no decay
    grad: [B][B]f64,
};
```

### Memory Cost

If $N_k$ is the number of active blocks on tape $k$:

$$\text{memory}(\mathcal{T}_k) = N_k \cdot B^2 \cdot \text{sizeof(LogCell)} = N_k \cdot 64 \cdot 7 = N_k \cdot 448 \text{ bytes}$$

For LTM with $N_M$ active blocks:

$$\text{memory}(M) = N_M \cdot B^2 \cdot \text{sizeof(f64)} = N_M \cdot 512 \text{ bytes}$$

### Pruning

**Tape:** No active sweep needed. Decay is lazy — applied at read time. A block becomes pruneable when all cells satisfy:

```
log_abs + (k_current - k_write) * ln_gamma < ln(ε)
```

**LTM:** Sweep after each gated write:

```
for block in M.blocks:
    if max(|v| for v in block.data) < ε_M:
        M.remove_block(block)
```

---

## State Transitions

The state vector $S_k$ is the network's working memory — a dense vector of size $d$ that flows from tape to tape carrying context.

### State Update with EMA Decay

The state update uses an exponential moving average (EMA) form, which bounds values to $(-1, 1)$ and provides a residual connection that helps gradient flow:

$$S_{k+1} = \gamma_S \cdot S_k + (1 - \gamma_S) \cdot \tanh\!\bigl(W_S \cdot [r_k ; m_k ; S_k] + b_S\bigr)$$

The carry factor $\gamma_S \in (0,1)$ can be a fixed scalar or a learned per-dimension vector. This is equivalent to an EMA over tape transitions and also acts as a residual connection — the additive path through $\gamma_S \cdot S_k$ gives gradients a direct path backward, largely solving vanishing gradient problems without requiring a full GRU.

### Write Gate and LTM Write Value

The gate $g_k$ decides whether the current step commits to long-term memory:

$$g_k = \sigma\!\bigl(W_g \cdot [r_k ; m_k ; S_k] + b_g\bigr)$$

The write fires only when $g_k > \theta_\text{commit}$. The value written and the erase signal are:

$$v_M = W_{Wm} \cdot [r_k ; m_k ; S_k] + b_{Wm}$$

$$e_k = \sigma\!\bigl(W_e \cdot [r_k ; m_k ; S_k] + b_e\bigr)$$

### Tape Write Value

The value written back to the short-term tape is unchanged from the original formulation:

$$v_k = W_W \cdot [r_k ; m_k ; S_k] + b_W$$

### Full Transition Summary

Given tape $k$, tape head $H_k$, LTM head $H_M$, state $S_k$:

1. **Tape read:** $r_k \leftarrow \text{soft\_read}(\mathcal{T}_k, H_k)$
2. **LTM read:** $m_k \leftarrow \text{soft\_read}(M, H_M)$
3. **State update:** $S_{k+1} \leftarrow \gamma_S \cdot S_k + (1-\gamma_S) \cdot \tanh(W_S [r_k; m_k; S_k] + b_S)$
4. **Tape head move:** $\Delta H_k \leftarrow \tanh(W_H [r_k; m_k; S_k] + b_H)$, then $H_{k+1} \leftarrow H_k + \Delta H_k$
5. **LTM head move:** $\Delta H_M \leftarrow \tanh(W_{Hm} [r_k; m_k; S_k] + b_{Hm})$, then $H_M \leftarrow H_M + \Delta H_M$
6. **Tape write value:** $v_k \leftarrow W_W [r_k; m_k; S_k] + b_W$
7. **Write to tape:** $\mathcal{T}_{k+1}(i,j) \mathrel{+}= w_{ij} \cdot v_k$ for nearby $(i,j)$, then prune $\mathcal{T}_{k+1}$ (lazy — no sweep)
8. **LTM write gate:** $g_k \leftarrow \sigma(W_g [r_k; m_k; S_k] + b_g)$
9. **LTM write (if $g_k > \theta_\text{commit}$):**
   - Erase: $M(i,j) \leftarrow M(i,j) \cdot [1 - g_k \cdot e_k \cdot w^M_{ij}]$
   - Add: $M(i,j) \leftarrow M(i,j) + g_k \cdot w^M_{ij} \cdot v_M$
   - Prune $M$

---

## Forward Pass

```
initialize:
    T_0   ← write input data to tape 0 at H_0 = (0, 0)
    S_0   ← zero vector of size d
    H_0   ← (0, 0)
    H_M   ← (0, 0)   (or persisted from previous example in stateful mode)
    M     ← empty    (or persisted in stateful mode)

for k = 0 to K-1:
    r_k    ← soft_read(T_k, H_k)
    m_k    ← soft_read(M, H_M)

    S_{k+1} ← γ_S · S_k + (1−γ_S) · tanh(W_S · [r_k; m_k; S_k] + b_S)

    ΔH_k   ← tanh(W_H  · [r_k; m_k; S_k] + b_H);   H_{k+1} ← H_k + ΔH_k
    ΔH_M   ← tanh(W_Hm · [r_k; m_k; S_k] + b_Hm);  H_M     ← H_M + ΔH_M

    v_k    ← W_W  · [r_k; m_k; S_k] + b_W
    v_M    ← W_Wm · [r_k; m_k; S_k] + b_Wm
    e_k    ← σ(W_e · [r_k; m_k; S_k] + b_e)
    g_k    ← σ(W_g · [r_k; m_k; S_k] + b_g)

    soft_write(T_{k+1}, H_{k+1}, v_k)
    // lazy decay on T_{k+1} — no sweep, applied at read time

    if g_k > θ_commit:
        M(i,j) ← M(i,j) · [1 − g_k · e_k · w_ij]   // erase
        M(i,j) ← M(i,j) + g_k · w_ij · v_M          // add
        prune(M, ε_M)

output ← soft_read(T_K, H_K)
```

### Comparison to Original Forward Pass

| Step | Original | Updated |
|---|---|---|
| Reads per tape | 1 (tape only) | 2 (tape + LTM) |
| State input | $[r_k; S_k]$ | $[r_k; m_k; S_k]$ |
| State update | $\tanh(\cdot)$ | EMA: $\gamma_S \cdot S_k + (1-\gamma_S) \cdot \tanh(\cdot)$ |
| Tape decay | Prune sweep | Lazy log-domain (no sweep) |
| LTM write | — | Gated erase-then-add |

---

## Bounding the Halting Problem with Epochs

**Plain English:** A true Turing machine can loop forever. For a learning system this is unacceptable. We solve it by deciding in advance the maximum amount of work the network is allowed to do.

### Epoch Bound

Define a maximum epoch count $T$. Training runs for exactly $T$ epochs and then stops, regardless of whether the loss has converged:

$$\text{Training terminates at } t = T$$

### Tape Depth Bound

The number of tapes $K$ is fixed at construction time. The network does not decide dynamically how many tapes to use. This bounds the computation per forward pass:

$$\text{FLOPs per forward pass} \le K \cdot C \cdot N_{\max}$$

where $C$ is the cost per transition and $N_{\max}$ is the maximum active blocks per tape.

### LTM Size Bound

The long-term memory can grow across examples in stateful mode. The sparsity threshold $\epsilon_M$ and the high commit threshold $\theta_\text{commit}$ together bound practical LTM growth, but for hard guarantees an explicit maximum block count $N_M^{\max}$ can be enforced. When $M$ reaches capacity, the lowest-magnitude blocks are evicted first (LRU-magnitude policy).

### Why This Still Gives Turing Completeness

By making $K$, $T$, and $N_M^{\max}$ arbitrarily large (but finite), the network can simulate any computation that terminates within $T$ steps on a tape of width $K$ with LTM of size $N_M^{\max}$. For any specific target computation, there exist values large enough to solve it.

---

## Backpropagation Through State

Training this network is harder than a standard net because the state vector, tape head, and LTM head all flow from tape to tape. This is the same challenge as training an RNN, and the same solutions apply.

### Truncated BPTT

The standard approach is **Backpropagation Through Time (BPTT)**: unroll the tape transitions into a computation graph and run standard backprop. The depth of unrolling is $K$.

For very large $K$, use **truncated BPTT**: detach the gradient graph every $\tau$ tapes. This introduces bias but prevents memory from growing linearly with $K$.

### Gradient Through Soft Read/Write

The soft read is differentiable with respect to both head positions:

$$\frac{\partial r_k}{\partial H_k} = \frac{\partial}{\partial H_k}\left[\frac{\sum_{ij} w_{ij} \cdot \mathcal{T}_k(i,j)}{\sum_{ij} w_{ij}}\right]$$

Since $w_{ij} = \exp(-(||(i,j) - H_k||^2 / 2\sigma^2))$, the gradient is a sum of Gaussian-weighted differences — smooth everywhere. The same holds for $\partial m_k / \partial H_M$.

Gradients flow back through both the tape read path and the LTM read path to their respective weight matrices ($W_S, W_H, W_W$ for tapes; $W_g, W_e, W_{Wm}, W_{Hm}$ for LTM). The gate $g_k$ is differentiable (sigmoid), so the network learns *what is worth remembering* end-to-end.

### Vanishing Gradients

The EMA state update largely addresses vanishing gradients — the additive path $\gamma_S \cdot S_k$ is a residual connection that gives gradients a direct backward path. If vanishing gradients persist at very large $K$:

1. Replace the state update with a full **GRU** cell — gates control gradient flow more precisely
2. Use **gradient clipping**: cap $||\nabla|| \le C_\text{clip}$ at each update step
3. Reduce $K$ and increase $d$ to trade depth for width

---

## Turing Completeness Argument

A system is Turing complete if it can simulate a Universal Turing Machine. Requirements:

1. **Unbounded memory** — tapes are sparse and expand dynamically. LTM also expands dynamically. Any position $(i, j)$ can be written. ✓
2. **A state register** — the state vector $S_k$ holds the machine's current state. With $d$ dimensions it can encode any finite number of distinct states. ✓
3. **A transition function** — the weight matrices implement a parametric function of (tape read, LTM read, current state) → (new state, head movements, write values). Neural networks are universal function approximators. ✓
4. **Read/write access** — the head mechanism provides differentiable read and write at any tape or LTM position. ✓
5. **Persistent memory** — the LTM store $M$ persists across tape steps (and optionally across examples). This gives the system access to unbounded accumulated history. ✓
6. **Conditional branching** — head movements and write values depend on the full context $[r_k; m_k; S_k]$. Different states + inputs lead to different moves and writes, implementing all branches a transition table requires. ✓

All requirements are met. The architecture is Turing complete in the limit of $K \to \infty$, $d \to \infty$, and $N_M^{\max} \to \infty$.

---

## Comparison to Standard Neural Net

| Property | Standard Net | Original Turing-Tape | This Version |
|---|---|---|---|
| Layer representation | Dense vector $a^{[l]}$ | Sparse 2D tape $\mathcal{T}_k$ | Sparse 2D tape $\mathcal{T}_k$ (log-domain) |
| Memory between layers | None | State vector $S_k$ | State vector $S_k$ (EMA decay) |
| Long-term memory | None | None | Persistent LTM store $M$ |
| Memory between examples | None | Optional (stateful) | LTM persists; tapes reset |
| Layer connections | Dense matrix multiply | Soft attention via head | Dual soft attention (tape + LTM) |
| Storage cost | $O(n^2)$ per layer | $O(N_k)$ active blocks | $O(N_k \cdot 7)$ bytes + $O(N_M \cdot 8)$ bytes |
| Activation storage | f64 | f64 | LogCell (7B, lazy decay) |
| Weight decay | None / explicit | None / explicit | Log-domain additive |
| Turing complete | No | Yes (in the limit) | Yes (stronger — with persistent memory) |
| Halting bound | N/A | $K$ tapes, $T$ epochs | $K$ tapes, $T$ epochs, $N_M^{\max}$ LTM blocks |
| Gradient method | Standard backprop | BPTT through tape transitions | BPTT through tape + LTM transitions |
| Closest existing model | MLP | Neural Turing Machine | NTM + Sparse Transformer + EMA state |

---

## Implementation Reference

### Core Structs

```
pub const LogCell -> struct {
    log_abs: f32,    // ln|value| — decay is an additive constant
    sign:    i8,     // +1 or −1
    k_write: u16,    // tape index of last write (for lazy decay)
};

pub const Block -> struct {
    data: [B][B]LogCell,    // log-domain activation storage
    grad: [B][B]f64,        // full-precision gradient for backprop
};

pub const LTMBlock -> struct {
    data: [B][B]f64,        // full precision, no decay
    grad: [B][B]f64,
};

pub const Tape -> struct {
    k:       usize,
    blocks:  HashMap<(i32, i32), Block>,
    epsilon: f64,
    gamma:   f64,    // decay factor — ln_gamma cached as f32
};

pub const LTM -> struct {
    blocks:       HashMap<(i32, i32), LTMBlock>,
    epsilon_M:    f64,
    head:         [2]f64,       // H_M — persists across tapes
    max_blocks:   usize,        // hard cap; 0 = unbounded
};

pub const TapeNet -> struct {
    tapes:   []Tape,            // K+1 tapes
    ltm:     LTM,               // persistent long-term memory

    heads:   [][2]f64,          // tape head positions H_0..H_K
    states:  [][]f64,           // state vectors S_0..S_K

    // Tape transition weights
    W_S:     [][]f64,           // state update       [d × (2+d)]
    W_H:     [][]f64,           // tape head move     [2 × (2+d)]
    W_W:     [][]f64,           // tape write value   [1 × (2+d)]

    // LTM weights
    W_g:     [][]f64,           // write gate         [1 × (2+d)]
    W_e:     [][]f64,           // erase vector       [1 × (2+d)]
    W_Wm:    [][]f64,           // LTM write value    [1 × (2+d)]
    W_Hm:    [][]f64,           // LTM head move      [2 × (2+d)]

    // Biases
    b_S:     []f64,
    b_H:     []f64,
    b_W:     []f64,
    b_g:     []f64,
    b_e:     []f64,
    b_Wm:    []f64,
    b_Hm:    []f64,

    sigma:          f64,        // Gaussian read/write spread
    gamma_S:        f64,        // state EMA carry factor
    lambda:         f64,        // weight decay coefficient
    theta_commit:   f64,        // LTM commit threshold (e.g. 0.9)
    d:              usize,      // state vector dimension
    K:              usize,      // number of tapes
};
```

### Key Functions

| Function | Description |
|---|---|
| `soft_read(tape, head, sigma)` | Gaussian-weighted read; applies lazy log-domain decay |
| `soft_read_ltm(ltm, head, sigma)` | Gaussian-weighted read from LTM (plain f64, no decay) |
| `soft_write(tape, head, value, sigma)` | Gaussian-weighted write to tape; stores as LogCell |
| `ltm_write(ltm, head, gate, erase, value, sigma)` | Erase-then-add gated write to LTM |
| `prune_tape(tape)` | Lazy — checks log_abs + decay offset < ln(ε); no sweep |
| `prune_ltm(ltm)` | Explicit sweep: remove blocks where max abs < ε_M |
| `tape_forward(net, input)` | Full forward pass over all K tapes with dual read |
| `tape_backward(net, target)` | BPTT through tape and LTM transitions |
| `tape_update(net, lr)` | Update all weights with log-domain decay |
| `init_tape_net(K, d, sigma, gamma, gamma_S, lambda, epsilon, epsilon_M, theta_commit)` | Allocate and Xavier-init all weight matrices |
| `reset_tapes(net)` | Clear all tapes; optionally reset LTM (stateless mode) |

### Hyperparameter Guide

| Hyperparameter | Role | Suggested start |
|---|---|---|
| $K$ | Number of tapes (computation depth) | 8–32 |
| $d$ | State vector dimension | 32–128 |
| $B$ | Block size for sparse storage | 8 |
| $\epsilon$ | Tape sparsity threshold | 1e-4 |
| $\epsilon_M$ | LTM sparsity threshold | 1e-3 |
| $\sigma$ | Head read/write spread | 1.0 |
| $\gamma$ | Tape activation decay factor | 0.9 |
| $\gamma_S$ | State EMA carry factor | 0.8 |
| $\lambda$ | Weight decay coefficient | 1e-4 |
| $\theta_\text{commit}$ | LTM write gate threshold | 0.9 |
| $N_M^{\max}$ | Max LTM blocks (0 = unbounded) | 0 or 4096 |
| $\alpha$ | Learning rate | 1e-3 |
| $T$ | Max training epochs (halting bound) | 10,000–100,000 |
| $C_\text{clip}$ | Gradient clip norm | 1.0–5.0 |