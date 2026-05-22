# 📖 Comprehensive Study Manual: Large Language Models (LLMs)

This document is a comprehensive, deep-dive technical reference manual for Decoder-only Transformers and Generative Pre-trained Transformer (GPT) models. It spans foundational history, subword tokenization algorithms, absolute/relative/rotary positional encodings, attention variants (MQA, GQA, KV Caching, FlashAttention), normalization layers, training lifecycles (pre-training, SFT, RLHF, DPO), decoding strategies, and an annotated PyTorch implementation.

---

## 1. Evolution of Sequence Models & The Transformer Paradigm

### 1.1 The Recurrent Bottleneck & Gated Architectures
Before the Transformer, sequential data processing relied heavily on Recurrent Neural Networks (RNNs) and Gated Recurrent Units (LSTMs, GRUs). 

```
RNN Recurrent Step:
h_t = tanh( W_hh * h_{t-1} + W_xh * x_t + b_h )
```

#### The Limitations of RNNs:
1. **Sequential Dependency**: Since the hidden state $h_t$ depends on $h_{t-1}$, computations cannot be parallelized across the time dimension. This makes training on large datasets slow.
2. **Vanishing/Exploding Gradients**: During backpropagation through time (BPTT), gradients are repeatedly multiplied by the weight matrix $W_{hh}$. If eigenvalues of $W_{hh}$ are $< 1$, the gradient vanishes; if $> 1$, the gradient explodes.
3. **Information Bottleneck (Seq2Seq)**: In traditional Encoder-Decoder frameworks, the Encoder compresses the entire input sequence into a single fixed-size context vector $h_N$ representing the sequence bottle-neck.

### 1.2 The Attention Breakthrough
Attention emerged to bypass the compression bottleneck by allowing the decoder to direct a "look back" search over all hidden states of the encoder at each generation step.

```
Encoder States (h_1, ..., h_T) ───┐
                                  v
Query (s_t) ──> Dot-Product Score Calculation ──> Softmax Weights ──> Weighted Context Vector
```

### 1.3 Three Families of Transformers
The 2017 Transformer architecture introduced by Vaswani et al. replaced recurrence entirely with self-attention. This architecture split into three main families:

| Family | Attention Type | Primary Pre-training Objective | Optimal Tasks | Examples |
| :--- | :--- | :--- | :--- | :--- |
| **Encoder-Only** | Bidirectional (Self-attention reads future and past) | Masked Language Modeling (MLM), Next Sentence Prediction | Classification, NER, Embedding Extraction | BERT, RoBERTa |
| **Encoder-Decoder** | Bidirectional (Encoder) + Causal (Decoder) + Cross-attention | Masked span reconstruction, Translation objective | Summarization, Translation, Text-to-text generation | T5, BART |
| **Decoder-Only** | Causal (Masked self-attention, left-to-right only) | Autoregressive Next-Token Prediction | Open-ended text generation, Chat, Code synthesis | GPT series, LLaMA, Mistral |

---

## 2. Data Pipelines: Tokenization & Vocabularies

Tokenization is the process of breaking continuous string data into discrete units (tokens) that can be mapped to integer indices.

```
Raw Text: "unfamiliar"
  ├─ Word-level:    ["unfamiliar"]  (Requires massive vocabulary)
  ├─ Character-level: ["u", "n", "f", "a", "m", "i", "l", "i", "a", "r"] (High sequence length)
  └─ Subword-level:  ["un", "familiar"] (Optimal balance)
```

### 2.1 Subword Tokenization Algorithms

#### A. Byte-Pair Encoding (BPE)
BPE constructs its vocabulary bottom-up, starting with individual characters (and bytes in modern versions) and iteratively merging the most frequent adjacent token pairs.
1. **Initialize**: Split training text into characters. Add a special end-of-word marker (e.g., `</w>`).
2. **Count**: Calculate the frequency of all adjacent character pairs.
3. **Merge**: Merge the most frequent pair $(A, B)$ to form a new token $AB$. Add $AB$ to the vocabulary.
4. **Repeat**: Run for a target number of merge operations.

> [!TIP]
> **Byte-Level BPE (BBPE)**: Original BPE suffered from Out-of-Vocabulary (OOV) tokens when encountering new unicode characters. Modern tokenizers (like GPT-2/3/4 `tiktoken`) represent strings as raw UTF-8 bytes first. Since there are only 256 possible bytes, the base vocabulary is guaranteed to cover any possible string, eliminating OOV tokens.

#### B. WordPiece
Used in BERT, WordPiece is similar to BPE but differs in its merge criterion. Instead of merging the most *frequent* pair, it merges the pair that maximizes the likelihood of the training data according to a unigram language model.
$$\text{Score}(A, B) = \frac{\text{count}(AB)}{\text{count}(A) \times \text{count}(B)}$$
This favors combinations of tokens that occur together significantly more often than expected by their individual rates.

#### C. Unigram
Used in ALBERT and T5, Unigram approaches vocabulary building from the opposite direction. It starts with a very large vocabulary (e.g., all common subwords and words) and iteratively *prunes* tokens. At each step, it calculates the loss penalty (loss decrease) if a token $x$ is removed. It discards the bottom $p\%$ of tokens that contribute the least to maximizing the training corpus likelihood.

---

## 3. Embeddings & Position Encodings

Because self-attention has no inherent recurrence or convolution, it is permutation-invariant: scrambling the order of input tokens yields the exact same attention outputs (only scrambled). We must inject positional information.

```
Token Index ---> Token Embedding Table (Lookup) ───┐
                                                    v
Position Index -> Position Embedding Table (Lookup) -> (Vector Summation) ---> Model Input
```

### 3.1 Mathematical Formulations

#### Sinusoidal Absolute Position Encodings
The original Transformer used fixed, non-learned sinusoidal patterns:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
*Where $pos$ is the token position index, $i$ is the channel index, and $d$ is the embedding dimension.*
* **Rationale**: The geometric progression allows the model to learn to attend by relative positions since $PE_{pos+k}$ can be represented as a linear projection of $PE_{pos}$.

#### Learned Absolute Position Embeddings
GPT-2 and GPT-3 discard sinusoids in favor of a learned embedding table of shape $(\text{Max Context Length}, d)$.
* **Limitation**: The model cannot extrapolate to context lengths larger than those seen during training because positions beyond the max context length have uninitialized embeddings.

#### Rotary Position Embeddings (RoPE)
Modern models (like LLaMA and Mistral) implement RoPE. Instead of *adding* positional vectors, RoPE *rotates* the Query ($Q$) and Key ($K$) vectors in 2D slices of the embedding space by an angle proportional to their sequence position.

For a 2D vector $\mathbf{x} = (x_1, x_2)^T$ at position $m$, we multiply by a rotation matrix:
$$\mathbf{R}_{\theta, m} \mathbf{x} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}$$

For a $d$-dimensional vector, the rotation is applied in independent 2D slices:
$$\mathbf{R}_{\Theta, m}^d = \text{diag}\left( \mathbf{R}_{\theta_1, m}, \mathbf{R}_{\theta_2, m}, \dots, \mathbf{R}_{\theta_{d/2}, m} \right)$$
* **Why it works**: The dot product of a rotated query $\mathbf{q}_m$ and key $\mathbf{k}_n$ preserves relative distance:
  $$\langle \mathbf{R}_{\Theta, m}^d \mathbf{q}, \mathbf{R}_{\Theta, n}^d \mathbf{k} \rangle = \mathbf{q}^T (\mathbf{R}_{\Theta, m}^d)^T \mathbf{R}_{\Theta, n}^d \mathbf{k} = \mathbf{q}^T \mathbf{R}_{\Theta, n-m}^d \mathbf{k}$$
  This makes the attention score dependent strictly on the relative distance $(n-m)$, allowing extrapolation to longer context windows.

#### ALiBi (Attention with Linear Biases)
ALiBi does not add positional embeddings at all. Instead, it adds a static, non-learned negative bias penalty directly to the attention matrix scores, scaled by a head-specific slope $m$:
$$\text{Attention Weights} = \text{Softmax}\left( \frac{Q K^T}{\sqrt{d_k}} - m \cdot d_{\text{distance}} \right)$$
*Where $d_{\text{distance}}$ is a matrix containing relative index offsets.*
* **Benefit**: Extreme length extrapolation capabilities.

---

## 4. The Self-Attention Mechanism

### 4.1 Scaled Dot-Product Attention
For query tensor $Q \in \mathbb{R}^{B \times T \times d_k}$, key tensor $K \in \mathbb{R}^{B \times T \times d_k}$, and value tensor $V \in \mathbb{R}^{B \times T \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V$$

*Where $M$ is the causal mask matrix ($M_{ij} = 0$ for $j \le i$, and $M_{ij} = -\infty$ for $j > i$).*

---

### 4.2 Mathematical Proof: Why Scale by $1/\sqrt{d_k}$?

> [!IMPORTANT]
> **Theorem**: Let query vector $\mathbf{q} \in \mathbb{R}^{d_k}$ and key vector $\mathbf{k} \in \mathbb{R}^{d_k}$ be random vectors whose components are independent and identically distributed (i.i.d.) random variables with mean $0$ and variance $1$. The variance of their dot product $\mathbf{q} \cdot \mathbf{k}$ is $d_k$.

#### Step-by-Step Derivation:
1. Let $\mathbf{q} = [q_1, q_2, \dots, q_{d_k}]$ and $\mathbf{k} = [k_1, k_2, \dots, k_{d_k}]$.
2. The dot product is:
   $$Z = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$
3. Since $q_i$ and $k_i$ are independent:
   $$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i] \mathbb{E}[k_i] = 0 \times 0 = 0$$
4. Compute the variance of an individual term $q_i k_i$:
   $$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - \big(\mathbb{E}[q_i k_i]\big)^2$$
   $$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - 0$$
5. Recall that for any random variable $X$, $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$. Since $\mathbb{E}[q_i] = 0$ and $\text{Var}(q_i) = 1$, we have $\mathbb{E}[q_i^2] = 1$. Similarly, $\mathbb{E}[k_i^2] = 1$.
   $$\text{Var}(q_i k_i) = 1 \times 1 = 1$$
6. Since all components are mutually independent, the variance of the sum is the sum of the variances:
   $$\text{Var}(Z) = \text{Var}\left( \sum_{i=1}^{d_k} q_i k_i \right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = d_k$$

#### The Consequence of Not Scaling:
If we do not scale, then as $d_k$ (the projection head size) grows larger, the variance of the attention scores $Q K^T$ scales linearly with $d_k$. Extremely large values in the score matrix will dominate. 
When passed through the Softmax function:
$$\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
The probability distribution converges toward a one-hot vector (assigning $1.0$ to the maximum value and $0.0$ to all others). In this state, the local gradient of the Softmax function is virtually zero:
$$\frac{\partial \text{Softmax}(z)_i}{\partial z_j} \approx 0$$
This leads to the **vanishing gradient problem**, causing the model to stop learning. Dividing the dot product by $\sqrt{d_k}$ scales the variance of the attention scores back to $1.0$, preserving gradient flow.

---

### 4.3 Attention Projections: MHA, MQA, and GQA

```
Multi-Head Attention (MHA):
Q: [Head 1][Head 2][Head 3][Head 4]
K: [Head 1][Head 2][Head 3][Head 4]
V: [Head 1][Head 2][Head 3][Head 4]

Multi-Query Attention (MQA):
Q: [Head 1][Head 2][Head 3][Head 4]
K: [          Shared Head         ]
V: [          Shared Head         ]

Grouped-Query Attention (GQA):
Q: [Head 1][Head 2] | [Head 3][Head 4]
K: [Shared Group 1] | [Shared Group 2]
V: [Shared Group 1] | [Shared Group 2]
```

* **Multi-Head Attention (MHA)**: Each attention head has unique projection weights for Query, Key, and Value. 
* **Multi-Query Attention (MQA)**: Query projection remains multi-head, but Key and Value projections are shared as a single head across the entire layer. This reduces the memory bandwidth bottleneck during inference generation (KV cache loading).
* **Grouped-Query Attention (GQA)**: A middle ground. Query heads are grouped into clusters, and each cluster shares a single Key and Value head.

---

### 4.4 Inference Optimization: KV Caching

During autoregressive generation, the model predicts one token at a time. To generate token $t+1$, the attention block requires the Queries of the current token, and the Keys and Values of *all* historical tokens $1, \dots, t$.

Without caching, at each generation step, the model must re-run the matrix multiplications to project all past tokens to $Q$, $K$, and $V$. This results in $O(T^2)$ computational complexity.

```
Generation Step t (Without Cache):
Process input sequence: [token_1, token_2, ..., token_t] 
Compute: Q, K, V for all tokens.

Generation Step t (With Cache):
Retrieve Cache: K_prev, V_prev (Shape: B, H, t-1, d_k)
Compute: Q_new, K_new, V_new only for token_t (Shape: B, H, 1, d_k)
Concatenate: K_full = [K_prev, K_new], V_full = [V_prev, V_new]
Update Cache with K_full, V_full
Compute attention using Q_new, K_full, V_full
```

#### Tensor Dimension Operations:
1. At generation step $t$, the input token is a single token of shape $(B, 1, C)$.
2. We project it to get queries, keys, and values:
   $$Q_{\text{new}} \in \mathbb{R}^{B \times H \times 1 \times d_k}, \quad K_{\text{new}} \in \mathbb{R}^{B \times H \times 1 \times d_k}, \quad V_{\text{new}} \in \mathbb{R}^{B \times H \times 1 \times d_k}$$
3. Load the historical Key and Value tensors from RAM (SRAM):
   $$K_{\text{cache}} \in \mathbb{R}^{B \times H \times (t-1) \times d_k}, \quad V_{\text{cache}} \in \mathbb{R}^{B \times H \times (t-1) \times d_k}$$
4. Concatenate along the sequence dimension (dimension 2):
   $$K_{\text{full}} = [K_{\text{cache}} ; K_{\text{new}}] \in \mathbb{R}^{B \times H \times t \times d_k}$$
   $$V_{\text{full}} = [V_{\text{cache}} ; V_{\text{new}}] \in \mathbb{R}^{B \times H \times t \times d_k}$$
5. Compute Attention Output:
   $$\text{Scores} = Q_{\text{new}} K_{\text{full}}^T \in \mathbb{R}^{B \times H \times 1 \times t}$$
   $$\text{Context} = \text{Softmax}(\text{Scores}) V_{\text{full}} \in \mathbb{R}^{B \times H \times 1 \times d_k}$$
6. Write $K_{\text{full}}$ and $V_{\text{full}}$ back to memory for the next step.

---

### 4.5 FlashAttention

Standard self-attention computes the intermediate score matrix $A = \text{Softmax}(Q K^T / \sqrt{d_k})$ and writes it to High Bandwidth Memory (HBM). For sequence length $N$, this matrix requires $O(N^2)$ memory storage. Writing and reading this matrix back and forth to slow GPU HBM creates a memory bandwidth bottleneck.

```
Standard Attention Flow:
HBM (Q, K) ---> SRAM ---> Compute QK^T ---> HBM (O(N^2) Scores) ---> SRAM ---> Softmax ---> HBM (Weights) ---> SRAM ---> Attention Out ---> HBM

FlashAttention Flow:
HBM (Q, K, V) ---> SRAM (Load Block-by-Block Tile) ---> Compute Online Softmax & Local Attn ---> HBM (Attention Out)
```

FlashAttention resolves this by:
1. **Tiling**: Splitting $Q$, $K$, and $V$ matrices into small blocks that fit entirely inside the fast GPU on-chip SRAM memory cache.
2. **Online Softmax**: Computing the Softmax scaling denominator iteratively across blocks using log-sum-exponential scaling, avoiding the need to compute the entire $N \times N$ score matrix before applying Softmax.
3. **Recomputation in Backward Pass**: Instead of storing the $N \times N$ attention matrix for backpropagation, it recomputes the attention values on-the-fly during the backward pass using stored output values, reducing memory consumption from $O(N^2)$ to $O(N)$.

---

## 5. The Transformer Block & Normalization

### 5.1 Pre-LN vs. Post-LN Block Layout

```
Post-LN Layout (Original Transformer):
Input (x) ---> Self-Attention ---> Add ---> LayerNorm ---> FeedForward ---> Add ---> LayerNorm ---> Output
   |                               ^                             |          ^
   └───────────────────────────────┘                             └──────────┘

Pre-LN Layout (Modern LLMs like GPT-2, LLaMA):
Input (x) ---> LayerNorm ---> Self-Attention ---> Add ---> LayerNorm ---> FeedForward ---> Add ---> Output
   |                             |                 ^          |               |             ^
   └─────────────────────────────v─────────────────┘          └───────────────v─────────────┘
```

* **Post-LN**: Normalization is placed after the residual additions. 
  * *Mathematical Problem*: The gradient of early layers scales inversely with the depth of the network. This makes it difficult to train deep models without a precise learning rate warmup phase to prevent gradient explosion.
* **Pre-LN**: Normalization is placed inside the residual branch, directly normalizing the inputs to the attention and feedforward modules.
  * *Benefit*: Ensures stable gradient propagation. The identity mapping pathway ($x + \dots$) remains unobstructed, allowing gradients to flow back directly from output to input. This makes deep networks ($\ge 24$ layers) much easier to train.

---

### 5.2 LayerNorm vs. RMSNorm

#### Layer Normalization (LN)
For a vector $\mathbf{x} \in \mathbb{R}^d$:
$$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$
$$\text{LN}(x)_i = \gamma_i \hat{x}_i + \beta_i$$
*Where $\gamma, \beta$ are learnable scale and shift parameters.*

#### RMSNorm
RMSNorm simplifies LayerNorm by assuming the mean ($\mu$) is zero, normalizing only by the root-mean-square. This provides the same training stability while reducing computational cost:
$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$
$$\text{RMSNorm}(x)_i = \gamma_i \frac{x_i}{\text{RMS}(x)}$$

---

### 5.3 Non-Linear Activation Functions

#### GELU (Gaussian Error Linear Unit)
Standard architectures discard ReLU in favor of GELU. GELU weights inputs by their probability under a Gaussian cumulative distribution:
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad \text{where } X \sim \mathcal{N}(0, 1)$$
* **Approximation**:
  $$\text{GELU}(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right) \right) \right)$$
* **Why it works**: Unlike ReLU which truncates negative inputs to absolute zero (causing dead neurons), GELU has a smooth, non-zero gradient for negative values.

#### SwiGLU (Swish Gated Linear Unit)
SwiGLU replaces the intermediate FeedForward layer with a gated linear architecture:
$$\text{Swish}_\beta(x) = x \cdot \text{Sigmoid}(\beta x)$$
$$\text{SwiGLU}(x) = \left( \text{Swish}_1(xW_1) \otimes xW_2 \right) W_3$$
*Where $\otimes$ is the element-wise Hadamard product.*
* **Benefit**: Gated networks converge faster and achieve lower validation loss than standard MLPs with the same parameter count.

---

## 6. The LLM Training Life Cycle

```
                       [ Massive Web Corpora ]
                                  |
                                  v
                  Stage 1: Pre-training (Autoregressive)
                                  |
                           [ Base Model ]
                                  |
                                  v
                Stage 2: Supervised Fine-Tuning (SFT)
                                  |
                        [ SFT Instruct Model ]
                                  |
            ┌─────────────────────┴─────────────────────┐
            v                                           v
    Stage 3A: RLHF (PPO)                         Stage 3B: DPO
[Reward Model + Actor + Critic]             [Direct Preference Loss]
            │                                           │
            └─────────────────────┬─────────────────────┘
                                  v
                         [ Aligned RLHF Model ]
```

### 6.1 Phase 1: Pre-training
The model is trained on unlabeled text corpora (e.g., Common Crawl, Wikipedia, Github) using the causal next-token prediction task.
* **Loss Function**: Cross-Entropy Loss over sequence tokens:
  $$\mathcal{L} = -\sum_{t=1}^T \log P(x_t \mid x_{1}, \dots, x_{t-1})$$
* **Chinchilla Scaling Laws**: Hoffmann et al. demonstrated that for optimal training compute efficiency, the number of training tokens and the model parameter size should be scaled in equal proportion:
  $$\text{Tokens} \approx 20 \times \text{Parameters}$$

### 6.2 Phase 2: Supervised Fine-Tuning (SFT)
The pre-trained base model is fine-tuned on curated instruct prompt-response pairs to learn the dialogue/conversational format.

```
Instruct Prompt Template:
"Below is an instruction... 
### Instruction: {prompt}
### Response: {response} <|endoftext|>"
```

* **Loss Masking**: To prevent the model from learning to generate prompts rather than answers, target labels for the input prompt tokens are masked (set to `-100` in PyTorch). PyTorch's `nn.CrossEntropyLoss` ignores indices set to `-100`, computing the gradient updates strictly on the response tokens.

### 6.3 Phase 3: Alignment (RLHF & DPO)

#### RLHF (Reinforcement Learning from Human Feedback) via PPO
1. **Train Reward Model**: A separate model is trained on pairwise human comparisons (labeling prompt completions as *chosen* $y_w$ and *rejected* $y_l$).
   $$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\psi(x, y_w) - r_\psi(x, y_l)) \right]$$
2. **Proximal Policy Optimization (PPO)**: The Actor model generates text, the Reward Model evaluates it, and a Critic network guides weight updates. A KL-Divergence penalty compares the active model against the baseline SFT model to prevent the policy from shifting too far and decaying:
   $$\text{Objective} = r_\psi(x, y) - \beta D_{\text{KL}}\big( \pi_\theta(y \mid x) \parallel \pi_{\text{SFT}}(y \mid x) \big)$$

#### DPO (Direct Preference Optimization)
DPO replaces the complex RL framework (Actor, Critic, Reward, and Reference models) with a single loss function optimized directly on preference pairs. DPO mathematically expresses the reward function implicitly in terms of the language model's policy:
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
This simplifies training alignment and reduces resource overhead.

---

## 7. Hugging Face Weight Mapping (Conv1D vs. Linear)

When importing OpenAI's pre-trained GPT-2 checkpoints, developers often encounter a mismatch between Hugging Face's `GPT2Model` and custom scratch implementations:

* **Hugging Face's custom `Conv1D` module**: Despite its name, this is not a standard convolutional layer. It performs a linear projection using a matrix structure transposed relative to standard PyTorch modules:
  $$Y = X W_{\text{HF}} + b$$
  *Where $W_{\text{HF}}$ has shape $(C_{\text{in}}, C_{\text{out}})$.*
* **Standard PyTorch `nn.Linear` module**:
  $$Y = X W_{\text{Linear}}^T + b$$
  *Where $W_{\text{Linear}}$ has shape $(C_{\text{out}}, C_{\text{in}})$.*

To load HF parameters into a standard `nn.Linear` layer, the weight matrix must be transposed:
```python
# Transpose HF weight matrix to match PyTorch expectations
scratch_linear.weight.data = asign_check(scratch_linear.weight.data, hf_conv1d.weight.data.T)
scratch_linear.bias.data = hf_conv1d.bias.data
```

Additionally, in GPT-2, the projections for Query, Key, and Value attention vectors are packed into a single projection matrix. To load these weights into separate layers, split the weights along the output dimension:
```python
# Extract combined weight parameters from state_dict
q_w, k_w, v_w = np.split(hf_weights["h.0.attn.c_attn.weight"], 3, axis=-1)
# Transpose and load into individual target layers
scratch_block.att.w_queries.weight = Parameter(torch.tensor(q_w.T))
scratch_block.att.w_keys.weight = Parameter(torch.tensor(k_w.T))
scratch_block.att.w_values.weight = Parameter(torch.tensor(v_w.T))
```

---

## 8. Decoding & Inference Strategies

During inference generation, the model outputs logit vectors over the vocabulary. Decoding strategies convert these logits into tokens:

```
Logits ---> Temperature Scaling ---> Softmax ---> Probabilities
                                                      │
                                                      ├─> Greedy: Pick Argmax
                                                      ├─> Top-k: Filter top k values
                                                      └─> Top-p: Filter cumulative p threshold
```

1. **Greedy Decoding**: Selects the token with the highest probability:
   $$x_{t+1} = \text{argmax}(P(x \mid x_{1:t}))$$
   * **Drawback**: Often leads to repetitive loops and dry text.
2. **Temperature Scaling**: Divides the logits by a scale factor $T > 0$ before applying Softmax:
   $$P(x_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$
   * $T \rightarrow 0$: Distribution becomes peaky (greedy).
   * $T > 1$: Distribution becomes flat, increasing randomness and creativity.
3. **Top-k Sampling**: Selects the top $k$ tokens with the highest logits, resets the probabilities of all others to $0$, and samples from the subset.
4. **Top-p (Nucleus) Sampling**: Selects the smallest set of tokens whose cumulative probability exceeds a threshold $p \in (0, 1]$. Samples are drawn strictly from this subset.
5. **Beam Search**: Keeps track of the top $B$ (beam width) most likely token sequences (beams) at each step. This is useful for structured tasks like translation or summarization.

---

## 9. Annotated PyTorch Reference Implementation

Here is a modular, heavily annotated reference implementation of a causal Transformer Block, matching the architectural configurations of GPT-2:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention module.
    Projects input vectors to Query, Key, and Value spaces, scales scores,
    applies a causal lower-triangular mask, and computes attention aggregation.
    """
    def __init__(self, d_model, n_heads, context_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        
        # Individual projection matrices for Queries, Keys, and Values
        self.w_queries = nn.Linear(d_model, d_model, bias=True)
        self.w_keys = nn.Linear(d_model, d_model, bias=True)
        self.w_values = nn.Linear(d_model, d_model, bias=True)
        
        # Output projection matrix to project concatenated heads back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        
        # Regularization dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Register the lower-triangular causal mask buffer
        self.register_buffer(
            "tril", 
            torch.tril(torch.ones(context_len, context_len))
        )

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Sequence length, channels (d_model)
        
        # 1. Project to Q, K, V and reshape to (B, n_heads, T, head_size)
        q = self.w_queries(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        k = self.w_keys(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        v = self.w_values(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)
        
        # 2. Compute Scaled Dot-Product scores: (B, H, T, hs) @ (B, H, hs, T) -> (B, H, T, T)
        scores = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        
        # 3. Apply causal mask (zeroing out future values by seting to negative infinity)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # 4. Softmax probability conversion & attention dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 5. Weighted value aggregation: (B, H, T, T) @ (B, H, T, hs) -> (B, H, T, hs)
        out = attn_weights @ v
        
        # 6. Concatenate heads back to shape (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 7. Apply output projection and residual dropout
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """
    Multi-Layer Perceptron (MLP) mapping the representation to a higher-dimensional
    space (4 * d_model) and applying the smooth GELU activation.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single pre-LN Transformer Block containing Layer Normalization,
    Self-Attention, residual shortcut additions, and Feed-Forward mapping.
    """
    def __init__(self, d_model, n_heads, context_len, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_len, dropout)
        self.ffwd = FeedForward(d_model, dropout)

    def forward(self, x):
        # Pre-LN attention block residual connection
        x = x + self.attn(self.ln1(x))
        # Pre-LN Feed-Forward block residual connection
        x = x + self.ffwd(self.ln2(x))
        return x
```
