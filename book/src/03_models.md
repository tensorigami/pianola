# Models

Now that we have reviewed tensor programming in catform, we can study an actual production language model expressed in this formalism. We chose [Qwen3](https://qwen.ai/blog?id=qwen3) as our representative, since it is a popular ungated **open-weight** model with a modern transformer architecture. The Qwen3 family of models comes in several sizes, ranging from `0.6B` to `32B` parameters, where all sizes have the same **architecture** and are thus described by the single `models/qwen3/model.cat` file. The differently sized models only differ in certain **hyperparameters**---the shapes of various components---and **weights**---the actual learned numerical parameters stored as safetensor files.

We will execute the smallest 0.6B model since it runs on many laptops. The model's data is in the folder `models/qwen3/0_6b`---where its hyperparameters are stored in the file `config.toml` and weight safetensors in the folder `models/qwen3/0_6b/weights/`. The `model.cat` file will name weight tensors with a `weights.` prefix and hyperparameters with a `param.` prefix. The runtime will resolve these variable names into their actual values.

Just as in the first chapter, we will study the model computation outside-in like a Matryoshka, beginning with the outer structure, then zooming in on the **transformer**, and finally culminating with the **attention** mechanism.

## The Model Structure

### Type Signature

Recall from [chapter 1](01_inference.md) that we defined the language model as taking a list of integer token indices, along with state, and returning a real-valued logit vector of shape $\mathtt{[V]}$---one entry for each token in the vocabulary---along with an updated state. This gave us the type annotation:

$$
\mathtt{model : (list[int], State) \to (real[V], State)}
$$

We will now make some refinements to this type signature, so that it matches how the model is defined in `model.cat`:
- replace the variable-length $\mathtt{list}$ container with the tensor of static shape $\mathtt{[N]}$ since $\mathtt{N}$ is fixed for a given forward pass
- replace the vocabulary size constant $\mathtt{V}$ with `param.vocab`
- replace the abstract mathematical types $\mathtt{real}$ and $\mathtt{int}$ with the machine datatypes `bf16` and `i32`
- add `weights` as an argument, for the learned weight tensors

Combining these yields the type signature of `main` in `model.cat`:
```catform
main(weights: *, tokens: i32[N], state: *) -> (logits: bf16[param.vocab], state: *)
```

Note that we gave both `weights` and `state` a wildcard type `*`---both of these are trees of tensors of varying shape. This is a notational convenience to keep the pre-flattened `.cat` file free of clutter---many tensors simply pass through as a subtree of arguments, never directly consumed at the top level. Each of these tensors is type annotated within the function that actually consumes them.

Throughout the code, we view the token context as having three phases:

$$
\mathtt{t_0,\dots,t_{M-1}, t_M, \dots, t_{M+N-1}, t_{M+N}, \dots, t_{MAX-1}}
$$

The first $\mathtt{M}$ tokens have been seen in prior forward passes and thus only exist in the cached state. The next $\mathtt{N}$ tokens are precisely those fed into the model. Finally, the remainder of the tokens are possible future tokens within the current completion, allowing us to assign a maximum size of `MAX` to the state buffer.

### Embeddings

Now that we understand the type signature of `main`, let's inspect the computation itself.

```catform
main(weights: *, tokens: i32[N], state: *) -> (logits: bf16[param.vocab], state: *) {
  emb   : bf16[N, param.hidden] = call[embed](tokens, weights.embedding)
  acts  : bf16[N, param.hidden],
  state : *                     = call[transformer](emb, weights.transformer, state)
  logits: bf16[param.vocab]     = call[unembed](acts, weights.lm_head)
}
```

Just as we did in [chapter 1](01_inference.md), we will explore this function outside-in like a Matryoshka. Thus we turn to embedding and unembedding, leaving the main transformer computation to its [own section](#the-transformer).

The transformer acts on **activation** vectors---real vectors of dimension `param.hidden`---for each input token. Thus we must first move from the discrete integers of token indices to real vectors. Embeddings do precisely this: every single token $\mathtt{j}$ in the vocabulary has a learned **embedding** vector $\mathbf{e}_{\mathtt{j}}$. The `embed` function is then implemented as a look-up table of these embedding vectors by index:

```catform
embed(tokens: i32[N], w: bf16[param.vocab, param.hidden]) -> (x: bf16[N, param.hidden]) {
  x: bf16[N, param.hidden] = read["v d -> _ d"](w, tokens)
}
```

Once the transformer has output a final activation vector, we need a way to move back to tokens. Since there are infinitely more possible activation vectors than there are tokens, we cannot actually invert the embedding function. Instead, we dot the activation vector with every single embedding to produce the logit scores per token. Since we only care about the final token's logits, `unembed` first extracts the final activation and then implements this dot product via the obvious contraction: 

```catform
unembed(
  x: bf16[N, param.hidden],
  w: bf16[param.vocab, param.hidden]
) -> (logits: bf16[param.vocab]) {
  last  : bf16[param.hidden] = call[final](x)
  logits: bf16[param.vocab]  = contract["v d, d -> v"](w, last)
}
```

We note that in principle the embedding and unembedding matrices may be distinct---we called `embed` with `weights.embedding` and `unembed` with `weights.lm_head`. In the case of the 0.6B size, these are trained as the same matrix, in a decision called **weight-tying**. Weight-tying is useful for smaller models, since the embedding matrix---given that one of its axes is the rather large vocabulary---takes up a large percentage of its allotted parameters. From the 8B size onward, the Qwen3 family switches to having them be distinct, which gives more degrees of freedom during training.

The embedding/unembedding bookends imply a continuation of the Matryoshka diagrams introduced in [chapter 1](01_inference.md). To write such a diagram, we use the type $\mathtt{Stream}$ for the space where activations live, and refer to it in prose as the **Residual Stream**, and $\mathtt{Tokens}$ for the space where tokens live.

For the latter, however, we are blocked by a seeming asymmetry: we start with a token index integer but end up with a logits vector. Luckily, there is a conceptually illuminating way to fix this. Before explaining it, we note that we avoid doing this in our actual computation since it is needlessly expensive. The conceptual type mismatch can be fixed via **one-hot encoding**---i.e. recasting each token integer $\mathtt{j}$ as a vocabulary (i.e. size $\mathtt{V}$) vector with a $\mathtt{1}$ entry at index $\mathtt{j}$ and $\mathtt{0}$ everywhere else:

$$
\mathtt{j} \mapsto \begin{bmatrix} \mathtt{0} & \cdots & \mathtt{1} & \cdots & \mathtt{0} \end{bmatrix}
$$

We can thus let the $\mathtt{Tokens}$ type be the space of these vocabulary vectors. This gives us the Matryoshka diagram:

$$
\begin{tikzcd}
\mathtt{Tokens}
\arrow[rr, "\mathtt{model}"]
\arrow[d, "\mathtt{embed}"']
&& \mathtt{Tokens}
\\
\mathtt{Stream}
\arrow[rr, "\mathtt{transformer}"']
&& \mathtt{Stream}
\arrow[u, "\mathtt{unembed}"']
\end{tikzcd}
$$

As an extra nicety, with tokens in this form, the `embed` look-up table gets implemented via a tensor contraction!

```catform
embed(tokens: bf16[N, param.vocab], w: bf16[param.vocab, param.hidden]) -> (x: bf16[N, param.hidden]) {
  x: bf16[N, param.hidden] = contract["v d, n v -> n d"](w, tokens)
}
```

The embedding and unembedding functions are thus both tensor contractions---and in the case of weight tying, with the same tensors. It's interesting to consider what would happen if we just unembedded immediately after embedding---it would turn the computation into a contraction with the embedding tensor multiplied by its transpose:

$$
\mathbf{W}\mathbf{W}^{\mathtt{T}} = \begin{bmatrix}
- & \mathbf{e}^{\mathtt{T}}_{\mathtt{1}} & - \\
  & \vdots         &   \\
- & \mathbf{e}^{\mathtt{T}}_{\mathtt{V}} & - \\
\end{bmatrix}\begin{bmatrix}
     |         &         &     |        \\
 \mathbf{e}_{\mathtt{1}}  & \cdots  & \mathbf{e}_{\mathtt{V}} \\
     |         &         &     |
\end{bmatrix} = \begin{bmatrix}
\mathbf{e}_{\mathtt{1}}^{\mathtt{T}}\mathbf{e}_{\mathtt{1}} & \cdots & \mathbf{e}_{\mathtt{1}}^{\mathtt{T}}\mathbf{e}_{\mathtt{V}} \\
\vdots                     & \ddots & \vdots                     \\
\mathbf{e}_{\mathtt{V}}^{\mathtt{T}}\mathbf{e}_{\mathtt{1}} & \cdots & \mathbf{e}_{\mathtt{V}}^{\mathtt{T}}\mathbf{e}_{\mathtt{V}}
\end{bmatrix}
$$

Thus the token $\mathtt{j}$ would get mapped to the $\mathtt{j}^\text{th}$ row, consisting of its dot product with every embedding in the vocabulary:

$$
\begin{bmatrix}
\mathbf{e}_{\mathtt{j}}^{\mathtt{T}}\mathbf{e}_{\mathtt{1}} & \cdots & \mathbf{e}_{\mathtt{j}}^{\mathtt{T}}\mathbf{e}_{\mathtt{V}}
\end{bmatrix}
$$

Given that dot products, at least up to scale, measure similarity between vectors, this would turn the logits into similarity scores. This means that we would sample the next token purely in accord with how similar it is to the most recent token. The transformer is thus the mechanism that turns this static similarity lookup into a context-aware prediction.

### Normalization

Now that we have embedded our tokens into a vector space, we will perform the remainder of our computation in vector spaces. For many of the functions we compute, what we care about is the *direction* of a vector---numerically: the relative ratios of its components---rather than its magnitude. In fact, the magnitude may pose a problem as it grows larger.

Linear functions $\mathtt{L}$ preserve relative ratios between components independent of the magnitude of their inputs since any scaling factor can be factored out via the property:
$$
\mathtt{L(c\mathbf{x})=cL(\mathbf{x})}
$$
In contrast, this does not hold for nonlinear functions, since they might preserve the relationships at some magnitudes but destroy them at others. Let's take an extreme example---so extreme that we can use purposefully imprecise quantities to demonstrate it. Consider a **step function** $\mathtt{f}$ that assigns $\mathtt{0}$ to "small" inputs and $\mathtt{1}$ to "big" ones, based on some threshold $\mathtt{T}$: 

$$
\begin{tikzpicture}[every node/.style={font=\small}]
  % axes
  \draw[->] (-0.3, 0) -- (4.2, 0) node[right]{$\mathtt{x}$};
  \draw[->] (0, -0.3) -- (0, 1.8) node[above]{$\mathtt{f(x)}$};

  % step: y=0 for x < T, jump to y=1 at x=T
  \draw[thick] (0, 0) -- (2, 0);
  \draw[thick, densely dotted] (2, 0) -- (2, 1);
  \draw[thick] (2, 1) -- (4, 1);

  % open / closed endpoints at the jump
  \fill (2, 0) circle (1.5pt);
  \draw[fill=white] (2, 1) circle (1.5pt);

  % axis ticks
  \draw (2, 0.05) -- (2, -0.05) node[below]{$\mathtt{T}$};
  \draw (0.05, 1) -- (-0.05, 1) node[left]{$\mathtt{1}$};
\end{tikzpicture}
$$

Let's say we have two quantities $\mathtt{b}$ and $\mathtt{s}$, where the former big one is meaningfully larger than the latter small one. For our function $\mathtt{f}$ to preserve this fact (despite, of course, annihilating its extent), we would need $\mathtt{s < T < b}$. But if these inputs are scaled too small or too large, then both the big $\mathtt{b}$ and the small $\mathtt{s}$ would get sent to the same output, thus destroying the information of their significant size difference.

As we shall see in the subsequent transformer section, the magnitude of our activations will continue to grow ever larger. Thus, before passing quantities as inputs to a nonlinear function in the model, it is important that they be scaled to a computationally sensitive interval of its domain. We will thus **normalize** our vectors---i.e. scale them to a standardized size while preserving direction---prior to the application of any nonlinear function. We will do so using $\mathtt{RMSNorm}$, which is defined in two steps. 

Recall that the magnitude of a vector is just its length and hence can be calculated, due to the Pythagorean Theorem, by the square root of the sum of the squares of its components:

$$
\mathtt{
\|(x_1,\dots,x_n)\| = \sqrt{x_1^2+\cdots+x_n^2}
}
$$

A classic way to normalize a vector is to divide all of its components by its magnitude, thus yielding a vector of magnitude $\mathtt{1}$. Geometrically, this means that all normalized vectors are a distance of $\mathtt{1}$ from the origin---i.e. they live on the unit (radius $\mathtt{1}$) $\mathtt{n}$-sphere! We can thus visualize such normalization as scaling each vector to the point where it intersects the unit sphere. We depict this scaling in the $\mathtt{n=2}$ case:

$$
\begin{tikzpicture}[every node/.style={font=\small}]
  % axes
  \draw[->] (-2.3, 0) -- (2.3, 0);
  \draw[->] (0, -2.3) -- (0, 2.3);

  % unit circle
  \draw[thick] (0, 0) circle (1.5);

  % at each of 8 angles, one inward and one outward arrow on the same radial
  \foreach \a in {22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5} {
    % inward: long vector retracts onto the circle
    \draw[->, blue] ({2.1*cos(\a)}, {2.1*sin(\a)}) -- ({1.6*cos(\a)}, {1.6*sin(\a)});
    % outward: short vector extends onto the circle
    \draw[->, red]  ({0.6*cos(\a)}, {0.6*sin(\a)}) -- ({1.4*cos(\a)}, {1.4*sin(\a)});
  }
\end{tikzpicture}
$$

From a numerical perspective, there is an issue worth flagging: as the dimension $\mathtt{n}$ grows large, a typical component on the unit sphere gets closer to $\mathtt{0}$, and thus risks escaping the range of numerical precision. If we prefer that the components' sizes remain independent of the dimension, we can instead scale vectors to the sphere containing the unit diagonal vector $\mathtt{\begin{bmatrix}1&\cdots&1\end{bmatrix}}$, which forces a radius of:

$$
\mathtt{
  r = \sqrt{\sum_{i=1}^n1^2} = \sqrt{n}
}
$$

Thus we can retract all vectors to the sphere of radius $\mathtt{\sqrt{n}}$ via dividing by the root mean square $\mathtt{rms}$.

$$
\mathtt{
rms(\mathbf{x}) = \frac{\|\mathbf{x}\|}{\sqrt{n}}
}
$$

This constitutes the first step of $\mathtt{RMSNorm}$, but with two caveats:
- to avoid the risk of dividing by $\mathtt{0}$, we add a constant `param.rms_norm_eps` to the sum of squares
- to avoid the accumulation of rounding error, we upcast to `f32` for this part of the computation

Thus we have described all but the last two lines of the implementation:

```catform
rmsnorm(x: bf16[..., D], w: bf16[D]) -> (out: bf16[..., D]) {
  // upcast
  x_f  : f32[..., D]  = map[f32](x)

  // divide by the perturbed root mean square
  sq   : f32[..., D]  = map[square](x_f)
  var  : f32[..., 1]  = fold["... d -> ... 1", mean](sq)
  eps_t: f32[..., 1]  = tile["-> ..."](param.rms_norm_eps)
  ve   : f32[..., 1]  = map[add](var, eps_t)
  rs   : f32[..., 1]  = map[rsqrt](ve)
  rs_t : f32[..., D]  = tile["... 1 -> ... d"](rs)
  xn   : f32[..., D]  = map[mul](x_f, rs_t)

  // downcast
  xb   : bf16[..., D] = map[bf16](xn)

  // multiply by weight
  w_t  : bf16[..., D] = tile["d -> ... d"](w)
  out  : bf16[..., D] = map[mul](xb, w_t)
}
```

The second step is computationally simple---we multiply each dimension by a learned parameter, restoring a per-dimension degree of freedom that normalization had eliminated.

## The Transformer

The **transformer**, introduced in the seminal paper [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), is the primary architectural component of the contemporary language model. Once the tokens have been embedded as activation vectors in the residual stream, the transformer operates on them via a sequence---of length `param.layers`---of layers. Finally, before unembedding, it applies $\mathtt{RMSNorm}$.

$$
\begin{tikzpicture}[every node/.style={font=\scriptsize}]

\node at (0,1.75) {\large $\mathtt{model}$};
\node at (0,1.25) {\small $\mathtt{transformer}$};

\draw[very thick, rounded corners] (-6.0,-2) rectangle (6.0,2);
\draw[very thick, rounded corners, densely dotted] (-3.9,-1.5) rectangle (3.9,1.5);

\draw[very thick, rounded corners] (-5.4,-0.5) rectangle (-4.2,0.5);
\draw[very thick, rounded corners] ( 5.4,-0.5) rectangle ( 4.2,0.5);
\node at ( 4.8,0) {$\mathtt{unembed}$};
\node at (-4.8,0) {$\mathtt{embed}$};

\draw[very thick] (-7,0) -- (-5.4,0) ;
\draw[very thick] ( 7,0) -- ( 5.4,0) ;

\draw[very thick] (-7, -1) -- (-4.6, -1) to[out=0, in=180, looseness=1.5] (-3.6,-0.15);
\draw[very thick] ( 7, -1) -- ( 2.8, -1) to[out=180, in=0, looseness=1.5] ( 1.8,-0.15);

\draw[very thick, rounded corners] (2.4,-0.5) rectangle (3.6,0.5);
\draw[very thick, rounded corners] (0.6,-0.5) rectangle (1.8,0.5);
\draw[very thick, rounded corners] (-0.6,-0.5) rectangle (-1.8,0.5);
\draw[very thick, rounded corners] (-2.4,-0.5) rectangle (-3.6,0.5);

\node at (3.0,0) {$\mathtt{rmsnorm}$};
\node at (1.2,0) {$\mathtt{layer}$};
\node at (-1.2,0) {$\mathtt{layer}$};
\node at (-3.0,0) {$\mathtt{layer}$};

\draw[very thick] (-3.6,0.15) -- (-4.2,0.15) ;
\draw[very thick] ( 3.6,0) -- ( 4.2,0) ;

\draw[very thick] (-2.4,0.15) -- (-1.8,0.15) ;
\draw[very thick] ( 2.4,0.15) -- ( 1.8,0.15) ;

\draw[very thick] (-2.4,-0.15) -- (-1.8,-0.15) ;

\draw[very thick] (-0.6,0.15) -- (-0.2,0.15) ;
\draw[very thick] ( 0.6,0.15) -- ( 0.2,0.15) ;

\draw[very thick] (-0.6,-0.15) -- (-0.2,-0.15) ;
\draw[very thick] ( 0.6,-0.15) -- ( 0.2,-0.15) ;

\node at (0,0) {$\cdots$};

\end{tikzpicture}
$$

In the case of Qwen3, all of these layers have identical structure, albeit each with its own distinct learned weights. This has been typical until more recently, when models have begun mixing layers, as we shall see in the upcoming architectures chapter. In `qwen3/model.cat` we write the `transformer` function as

```catform
transformer(
  x      : bf16[N, param.hidden],
  weights: *,
  state  : *
) -> (out: bf16[N, param.hidden], state: *) {
  pos       : i32[N]                                    = iota(state.seen)
  rot       : bf16[2, 2, N, param.rope_dim]             = call[build_rot](pos)
  mask      : bf16[N, MAX]                              = call[build_mask](pos)
  x         : bf16[N, param.hidden],
  state.k.* : bf16[param.kv_heads, MAX, param.head_dim],
  state.v.* : bf16[param.kv_heads, MAX, param.head_dim] = loop[layer, param.layers](x, rot, mask, pos, weights.layer.*, state.k.*, state.v.*)
  state.seen: i32[]                                     = map[add](state.seen, N)
  out       : bf16[N, param.hidden]                     = call[rmsnorm](x, weights.norm)
}
```

The `rot` and `mask`---discussed, respectively, in the [RoPE](#rope) and [Attend](#attend) sections---are tables, computed once, that we will reuse in each layer. The `loop[f,n]` syntax is just a [macro](https://en.wikipedia.org/wiki/Macro_(computer_science)), i.e. a notational shorthand that gets replaced before the code ever gets run, via unrolling to `n` lines of calling the function `f`, with appropriate variable renaming. In this case, using `n` for `param.layers`, it ends up looking like:

```catform
x_1, state.k.0,     state.v.0     = call[layer](x      , rot, mask, pos, weights.layer.0,     state.k.0,     state.v.0)
x_2, state.k.1,     state.v.1     = call[layer](x_1    , rot, mask, pos, weights.layer.1,     state.k.1,     state.v.1)
...
x_n, state.k.{n-1}, state.v.{n-1} = call[layer](x_{n-1}, rot, mask, pos, weights.layer.{n-1}, state.k.{n-1}, state.v.{n-1})
```

The layers of Qwen3 follow the now-standard transformer pattern:

$$
\begin{tikzpicture}[every node/.style={font=\scriptsize}]

\node at (0,1.75) {\large $\mathtt{layer}$};
\draw[very thick, rounded corners] (-6.0,-2) rectangle (6.0,2);

\draw[very thick] (-7,1.1) -- (7,1.1) ;
\draw[very thick] (-7,-0.9) -- (-4,-0.9) to[out=0, in=180, looseness=1.5] (-2.5,-0.15);
\draw[very thick] ( 7,-0.9) -- (0.5,-0.9) to[out=180, in=0, looseness=1.5] (  -1,-0.15);

\draw[very thick, rounded corners] (-5,-0.5) rectangle (-3.5,0.5);
\draw[very thick, rounded corners] (-2.5,-0.5) rectangle (-1,0.5);
\node at (-4.25,0) {$\mathtt{rmsnorm}$};
\node at (-1.75,0) {$\mathtt{attention}$};

\draw[very thick] (-5.5,1.1) to[out=-90, in=180, looseness=1] (-5,0) ;
\draw[very thick] (-3.5,0.15) -- (-2.5,0.15) ;
\draw[very thick] (-1,0.15) to[out=0, in=-90, looseness=1] (-0.5,1.1) ;

\draw[very thick, rounded corners] (5,-0.5) rectangle (3.5,0.5);
\draw[very thick, rounded corners] (2.5,-0.5) rectangle (1,0.5);
\node at (4.25,0) {$\mathtt{ffn}$};
\node at (1.75,0) {$\mathtt{rmsnorm}$};

\draw[very thick] (5.5,1.1) to[out=-90, in=0, looseness=1] (5,0) ;
\draw[very thick] (3.5,0) -- (2.5,0) ;
\draw[very thick] (1,0) to[out=180, in=-90, looseness=1] (0.5,1.1) ;

\end{tikzpicture}
$$

Reading the diagram left to right, the activations go through two **residual blocks**, each of which can be conceptualized as the following process:
1. it *reads* from the residual stream by copying the activation tensor
2. it *normalizes* the activation by applying $\mathtt{RMSNorm}$
3. it *computes* a correction by applying a **sub-layer**
4. it *writes* to the residual stream by adding the correction

Or, expressed as an equation:

$$
\mathtt{residual\_block(x) = x + sub\_layer(rmsnorm(x))}
$$

We note that this framing of residual streams and blocks comes from [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html).

At this level of granularity, the architecture is near-identical to the one introduced in the original transformer paper. The main structural departure---standard in modern models---is to normalize *before* applying the sub-layer, rather than after.

In catform, this all amounts to two chained residual blocks---attention followed by feed-forward:

```catform
layer(
  x      : bf16[N, param.hidden],
  rot    : bf16[2, 2, N, param.rope_dim],
  mask   : bf16[N, MAX],
  pos    : i32[N],
  weights: *,
  k_buf  : bf16[param.kv_heads, MAX, param.head_dim],
  v_buf  : bf16[param.kv_heads, MAX, param.head_dim]
) -> (out: bf16[N, param.hidden], k_buf: bf16[param.kv_heads, MAX, param.head_dim], v_buf: bf16[param.kv_heads, MAX, param.head_dim]) {
  xn   : bf16[N, param.hidden]                     = call[rmsnorm](x, weights.ln1)
  attn : bf16[N, param.hidden],
  k_buf: bf16[param.kv_heads, MAX, param.head_dim],
  v_buf: bf16[param.kv_heads, MAX, param.head_dim] = call[attention](xn, mask, rot, pos, k_buf, v_buf, weights.attn)
  x_res: bf16[N, param.hidden]                     = map[add](x, attn)
  xn2  : bf16[N, param.hidden]                     = call[rmsnorm](x_res, weights.ln2)
  ffn  : bf16[N, param.hidden]                     = call[gated_ffn](xn2, weights.ffn)
  out  : bf16[N, param.hidden]                     = map[add](x_res, ffn)
}
```

### Feed-forward Network

The simpler of the two transformer sub-layers is the feed-forward network. This sub-layer processes each token's activations *independently*, with no inter-token communication. The idea is that we wish to apply a nonlinear **threshold** function, such as the one described in the [Normalization subsection](#normalization).

We don't, however, apply this nonlinear function in the residual stream, but in a dedicated space, which we denote $\mathtt{FFN}$. We thus need a projection from $\mathtt{Stream}$ to $\mathtt{FFN}$---we in fact use two: **gate** (G) and **up** (U)---and a projection going the other way: the **down** (D) projection. In this new space, we use the threshold function `swiglu`, described below. This lets us represent the FFN sub-layer as a commutative diagram:

$$
\begin{tikzcd}
\mathtt{Stream}
\arrow[rr, "\mathtt{gated\_ffn}"]
\arrow[d, "\mathtt{G,U}"']
&& \mathtt{Stream}
\\
\mathtt{FFN}
\arrow[rr, "\mathtt{swiglu}"']
&& \mathtt{FFN}
\arrow[u, "\mathtt{D}"']
\end{tikzcd}
$$

This gives us the catform implementation:

```catform
gated_ffn(
  x   : bf16[N, param.hidden],
  gate: bf16[param.hidden, param.ffn_dim],
  up  : bf16[param.hidden, param.ffn_dim],
  down: bf16[param.ffn_dim, param.hidden]
) -> (out: bf16[N, param.hidden]) {
  g  : bf16[N, param.ffn_dim] = contract["... n d, d f -> ... n f"](x, gate)
  u  : bf16[N, param.ffn_dim] = contract["... n d, d f -> ... n f"](x, up)
  h  : bf16[N, param.ffn_dim] = call[swiglu](g, u)
  out: bf16[N, param.hidden]  = contract["... n f, f d -> ... n d"](h, down)
}
```

The goal of a threshold function is to zero out inputs that don't make the threshold. The historically canonical such function was the **ReLU** ("Rectified Linear Unit"):

$$
\mathtt{ReLU}(\mathtt{x}) = \max(\mathtt{0}, \mathtt{x})
$$

$$
\begin{tikzpicture}[every node/.style={font=\small}]
  \draw[->] (-2.5, 0) -- (2.5, 0) node[right]{$\mathtt{x}$};
  \draw[->] (0, -0.3) -- (0, 2.3) node[above]{$\mathtt{ReLU(x)}$};
  \draw[thick, blue] (-2.3, 0) -- (0, 0) -- (2.3, 2.3);
\end{tikzpicture}
$$

ReLU is computationally trivial. By clamping negative inputs to $\mathtt{0}$, it acts as a kind of feature-selection mechanism. Its drawback is the hard zero-clamp: any neuron whose pre-activation lands consistently below zero contributes nothing at all, regardless of how close to zero it sits.

To address this, modern transformers replace ReLU with **SiLU** (also called **Swish**)---a continuous curve that matches ReLU for large $\mathtt{x}$ but has a soft transition through zero, including a small dip below the axis that lets weakly-negative pre-activations still contribute a small signal:

$$
\mathtt{silu}(\mathtt{x}) = \mathtt{x} \cdot \sigma(\mathtt{x}) = \frac{\mathtt{x}}{\mathtt{1} + e^{-\mathtt{x}}}
$$

$$
\begin{tikzpicture}[every node/.style={font=\small}]
  \draw[->] (-3, 0) -- (3, 0) node[right]{$\mathtt{x}$};
  \draw[->] (0, -0.7) -- (0, 2.3) node[above]{$\mathtt{silu(x)}$};
  \draw[thick, blue, smooth, samples=100, domain=-3:2.3] plot ({\x}, {\x / (1 + exp(-\x))});
\end{tikzpicture}
$$

The SwiGLU applies silu to the gate components, then element-wise multiplies the result by the up components.

```catform
swiglu(g: bf16[..., F], u: bf16[..., F]) -> (out: bf16[..., F]) {
  act: bf16[..., F] = map[silu](g)
  out: bf16[..., F] = map[mul](act, u)
}
```

There's no a priori reason why SwiGLU specifically outperforms other activations---the inventor's only public comment was ["we attribute their success, as all else, to divine benevolence."](https://arxiv.org/abs/2002.05202)

What remains is the attention mechanism---the protagonist of the transformer.

## Attention

Thus far, all discussed computations have been per-token, meaning that they transformed the position $\mathtt{j}$ activation $\mathbf{x}_{\mathtt{j}}$ without reference to any other positions. This of course won't suffice to properly understand language, since the interpretation of a token will depend profoundly on prior tokens! 

The **attention mechanism** addresses precisely this. Just as in the feed-forward network, we will not perform the attention computation in the residual stream, but rather in its own dedicated space. The space is often left unnamed, but for conceptual clarity, we will use the type $\mathtt{Attention}$, and refer to it in prose as **attention space**, with dimension given by `param.head_dim`. We thus need projections from $\mathtt{Stream}$ to $\mathtt{Attention}$---we in fact use three: **query** (Q), **key** (K), and **value** (V)---and one going the other way: the **output** (O) projection. We can represent this as a commutative diagram, using `attend` for the core attention computation.

$$
\begin{tikzcd}
\mathtt{Stream}
\arrow[rr, "\mathtt{attention}"]
\arrow[d, "\mathtt{Q,K,V}"']
&& \mathtt{Stream}
\\
\mathtt{Attention}
\arrow[rr, "\mathtt{attend}"']
&& \mathtt{Attention}
\arrow[u, "\mathtt{O}"']
\end{tikzcd}
$$

The rough idea, which we will describe in full detail in its [own subsection](#attend), is that for each suffix token, attention computes a probability distribution across *all* prior positions. This distribution can be thought of as divvying up the given position's "attention". The distribution is extracted from scores that come from each suffix token *querying* all of the other positions' *keys*. It will then use this distribution to compute a weighted sum of information---encoded in the value vectors---from prior tokens.

The `attend` function thus consumes these projected `q`,`k`,`v` vectors. It also consumes a **causal mask** tensor `mask`, which sends the scores produced by querying future tokens to $-\infty$. Thus, a minimal implementation of attention would look as follows.


```catform
attention(
    x   : bf16[N, param.hidden],
    mask: bf16[N, MAX],
    wq  : bf16[param.hidden, param.head_dim],
    wk  : bf16[param.hidden, param.head_dim],
    wv  : bf16[param.hidden, param.head_dim],
    wo  : bf16[param.head_dim, param.hidden]
) -> (out: bf16[N, param.hidden]) {
    // Q K V projections
    q  : bf16[N, param.head_dim] = contract["n d, d h -> n h"](x, wq)
    k  : bf16[N, param.head_dim] = contract["n d, d h -> n h"](x, wk)
    v  : bf16[N, param.head_dim] = contract["n d, d h -> n h"](x, wv)

    // Attend
    val: bf16[N, param.head_dim] = call[attend](q, k, v, mask)

    // O Projection
    out: bf16[N, param.hidden]   = contract["n h, h d -> n d"](val, wo)
}
```

The reader will observe that this function is simpler than the actual `attention` in `qwen3/model.cat`! The first difference to note is the presence of an extra tensor axis---with size `param.heads` for O and Q, and `param.kv_heads` for K and V.

Rather than computing a single attention distribution for each position, the attention mechanism actually computes multiple---`param.heads` to be exact---in parallel. We call each such `attend` computation an **attention head** or just **head**; the corresponding mechanism, **Multi-Head Attention (MHA)**, is the one used in the original transformer. Each head has its own learned projections between the residual stream and attention space. In MHA, the projection tensors bundle these distinct learned projections via an extra attention head tensor axis with size `param.heads`.

Qwen3 uses a newer now-standard variant of MHA called **Grouped Query Attention (GQA)**. In GQA, rather than having entirely separate heads, we have *groups* of heads, where each group shares the same K and V projections, and so is thought of as having a single **KV-head**. The number of groups is then `param.kv_heads`, and hence the group size is given by `param.heads / param.kv_heads`. As we will see in the [KV cache section](#kv-cache), reducing the number of distinct KV heads affords a major reduction in memory overhead.

While adding an extra `param.heads` size axis across the board would suffice to make the above code multi-head, GQA requires an extra tile operation for both key and value vectors, to make the Q, K, and V vectors the same shape so as to parallelize the `attend` operation across every head. This gives us the following updated code block.

```catform
attention(
  x   : bf16[N, param.hidden],
  mask: bf16[N, MAX],
  wq  : bf16[param.hidden, param.heads, param.head_dim],
  wk  : bf16[param.hidden, param.kv_heads, param.head_dim],
  wv  : bf16[param.hidden, param.kv_heads, param.head_dim],
  wo  : bf16[param.head_dim, param.heads, param.hidden]
) -> (out: bf16[N, param.hidden]) {
  // Q K V projections
  q  : bf16[param.heads, N, param.head_dim]    = contract["... n d, d h e -> ... h n e"](x, wq)
  k  : bf16[param.kv_heads, N, param.head_dim] = contract["... n d, d g e -> ... g n e"](x, wk)
  v  : bf16[param.kv_heads, N, param.head_dim] = contract["... n d, d g e -> ... g n e"](x, wv)

  // GQA
  k_e: bf16[param.heads, N, param.head_dim]    = tile["... g n e -> ... (g r) n e"](k)
  v_e: bf16[param.heads, N, param.head_dim]    = tile["... g n e -> ... (g r) n e"](v)

  // Attend
  a  : bf16[param.heads, N, param.head_dim]    = call[attend](q, k_e, v_e, mask)

  // O projection
  out: bf16[N, param.hidden]                   = contract["... h n e, e h d -> ... n d"](a, wo)
}
```

Comparing with the implementation in `qwen3/model.cat`, there are three missing code blocks---each adding arguments to the type signature---between the projections and the GQA; in order:

1. `QK-Norm` with additional arguments `q_norm` and `k_norm`
2. `RoPE` with additional argument `rot`
3. `KV Cache` additional arguments `pos`, `k_buf`, `v_buf`

The latter two of these we defer to dedicated [RoPE](#rope) and [KV cache](#kv-cache) subsections. The QK-Norm---short for **Query-Key Normalization**---however, can be described with little fanfare: Qwen3 makes the popular, albeit non-universal, choice to apply $\mathtt{RMSNorm}$ to the query and key vectors immediately after the projections:

```catform
// QK-Norm
q_n: bf16[param.heads, N, param.head_dim]    = call[rmsnorm](q, q_norm)
k_n: bf16[param.kv_heads, N, param.head_dim] = call[rmsnorm](k, k_norm)
```

The resultant `q_n` and `k_n` then replace `q` and `k` in the subsequent computations en route to being consumed by the `attend` function, which we describe in the [final subsection](#attend).

### RoPE

As discussed, the attention mechanism is the sole component of the transformer that allows tokens to take into account tokens in prior positions. Thus far, however, we have not injected any sort of positional information into the prior tokens. Thus the inputs `2-1=` and `1-2=` would be interpreted in the same way, despite representing distinct meanings.

This issue is addressed geometrically by Rotary Position Embedding, or [RoPE](https://arxiv.org/abs/2104.09864): via rotating vectors in $\mathtt{Attention}$ space! Rotation is easy enough to visualize in two dimensions: simply rotate the entire plane counter-clockwise such that the $\mathtt{x}$-axis shifts by the desired $\theta$ degrees:

$$
\begin{tikzpicture}[every node/.style={font=\small}]
  % axes (all four: dark grey, length 2.5 from origin)
  \draw[->, black!60] (0, 0) -- (2.5, 0)                                 node[right]{$\mathtt{x}$};
  \draw[->, black!60] (0, 0) -- (0, 2.5)                                 node[above]{$\mathtt{y}$};
  \draw[->, black!60] (0, 0) -- ({2.5*cos(35)}, {2.5*sin(35)})           node[right]{$\mathtt{x'}$};
  \draw[->, black!60] (0, 0) -- ({-2.5*sin(35)}, {2.5*cos(35)})          node[above]{$\mathtt{y'}$};

  % original square (off-origin, in Q1)
  \draw[thick, blue] (1.5, 0.3) rectangle (2.0, 0.8);

  % rotated square (rotated around origin by theta)
  \draw[thick, blue, rotate around={35:(0,0)}] (1.5, 0.3) rectangle (2.0, 0.8);

  % rotation arc (no arrowhead) + label
  \draw[black!60] (0.8, 0) arc (0:35:0.8);
  \node at ({1.0*cos(17.5)}, {1.0*sin(17.5)}) {$\theta$};
\end{tikzpicture}
$$

This transformation is equivalent to multiplying by the rotation matrix:

$$
\mathtt{R}_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

What about rotation in arbitrarily many dimensions? There are many possible ways to do this, but the conceptually simplest---in the case of an even number of dimensions---is to group the dimensions into pairs and then apply this two-dimensional rotation to each pair. This is exactly what `rope_rotate` does, with many distinct rotation matrices---stored in the `rot` tensor---in parallel:
1. it splits the `param.head_dim` axis into two axes of size `param.rope_dim` and `2`
2. it applies the rotation matrices stored in `rot` by contracting along the size-`2` axis
3. it merges the `param.rope_dim` and `2` axes back into a single `param.head_dim` axis 

```catform
rope_rotate(
  x  : bf16[..., N, param.head_dim],
  rot: bf16[2, 2, N, param.rope_dim]
) -> (out: bf16[..., N, param.head_dim]) {
  pairs : bf16[..., N, param.rope_dim, 2] = view["... n (two d) -> ... n d two"](x)
  result: bf16[..., N, param.rope_dim, 2] = contract["... n d two, two twop n d -> ... n d twop"](pairs, rot)
  out   : bf16[..., N, param.head_dim]    = view["... n d two -> ... n (two d)"](result)
}
```

The `rot` tensor includes `[N,param.rope_dim]` distinct `[2,2]` rotation matrices, each determined by an angle $\theta$. The formula for the angle in absolute position $\mathtt{j}$ and rope index $\mathtt{k}$ is given by

$$
\theta[\mathtt{j, k}] = \mathtt{j} \phi^{\mathtt{k}}
$$

One can think of each index in the rope axis $\mathtt{k}$ as representing a distinct clock---one whose hand "ticks" by some fixed tick angle. These tick angles vary in a geometric sequence: with `param.rope_step` as the base $\phi$ and `param.rope_dim` as the length $\mathtt{r}$, the sequence is

$$
\mathtt{\phi^0, \phi^1, \phi^2, \dots, \phi^{r-1}}
$$

These tick angles are stored in the `inv_freq` vector:

```catform
compute_inv_freq() -> (inv_freq: f32[param.rope_dim]) {
  // inv_freq[k] = rope_step^k
  k       : f32[param.rope_dim] = iota(0)
  step_t  : f32[param.rope_dim] = tile["-> d"](param.rope_step)
  inv_freq: f32[param.rope_dim] = map[pow](step_t, k)
}
```

Varying along the position axis simply multiplies tick angles by the *absolute* position indices `pos`:

$$
\mathtt{M,\dots,M+N-1}
$$

One can think of incrementing position as moving by a single tick in all of its clocks.

In attention, we apply `rope_rotate` to just queries and keys (after they've been normalized to `q_n` and `k_n`), since only these determine which positions attend to which:

```catform
// RoPE
q_r  : bf16[param.heads, N, param.head_dim]    = call[rope_rotate](q_n, rot)
k_r  : bf16[param.kv_heads, N, param.head_dim] = call[rope_rotate](k_n, rot)
```

The query, key, and value vectors that we pass into `attend` have now all been computed. All that remains is to use them to update our state.

### KV Cache

Throughout this chapter we have continued to reference $\mathtt{State}$, noting that it has allowed us to avoid carrying around prefix tokens. Outside of attention, we clearly didn't need prefix tokens---or any other tokens for that matter---since the computations were per-token. In contrast, we have mentioned that we wish to allow the suffix tokens to "attend to" *all* prior tokens, and not merely the suffix ones! Since only suffix tokens need to do any querying, we only need to store the prefix tokens' key and value vectors! 

We call this the **KV Cache**, and we bundle it into two *buffer* tensors `k_buf` and `v_buf`:

```catform
k_buf : bf16[param.kv_heads, MAX, param.head_dim]
v_buf : bf16[param.kv_heads, MAX, param.head_dim]
```

The `MAX`-size axis of these tensors corresponds to position---$\mathtt{0,\dots,M-1}$ already filled with prefix tokens, and the remainder---up to a fixed `MAX` size---initialized with zeros to be overwritten.

Once we have finished computing our suffix key and value vectors---the former, now `k_r` after applying both QK-Norm and RoPE---we write them to the buffer tensors in the positions $\mathtt{M,\dots,M+N-1}$ in `pos`:

```catform
// KV Cache
k_buf: bf16[param.kv_heads, MAX, param.head_dim] = write["g _ e -> g max e", set](k_r, pos, k_buf)
v_buf: bf16[param.kv_heads, MAX, param.head_dim] = write["g _ e -> g max e", set](v, pos, v_buf)
```

It's an astounding architectural feature of the transformer that there is nothing we need from prior tokens except the computed values of these two tensors!

Now that we have filled in all the missing code blocks in `attention`, we have the query, key, and value vectors to pass into `attend`.

### Attend

Now that we have all of the surrounding pieces in `attention`, we are ready to describe the core `attend` computation. As stated in the section's introduction, the first goal of `attend` is to compute position $\mathtt{j}$'s attention distribution $\mathbf{p}_{\mathtt{j}}$. To do so, we assign a score $\mathtt{s_{j,i}}$ of how much $\mathtt{j}$ should attend to $\mathtt{i}$. Mathematically, we apply the query $\mathbf{q}_{\mathtt{j}}$---which we think of as a covector---to the key $\mathbf{k}_{\mathtt{i}}$

$$
\mathbf{q}_{\mathtt{j}}^{\mathtt{T}}\mathbf{k}_{\mathtt{i}}
$$

Since these dot products parallelize across both $\mathtt{j}$ and $\mathtt{i}$, they can all be assembled into a single contraction

```catform
scores: bf16[N, N] = contract["j h, i h -> j i"](q, k)
```

To turn these scores into valid probabilities, we will apply the softmax function as we did in [the Sampling section](01_inference.md#sampling)---here with temperature $\mathtt{t = 1}$. Prior to doing so, however, we will need to do two things:

1. normalize to maintain unit variance
2. enforce causality

First, the normalization. Because the dot product sums across the contracting dimension, its variance picks up a factor proportional to that dimension's size. To see this, let $\mathtt{x=(x_1,\dots,x_h)}$ and $\mathtt{y=(y_1,\dots,y_h)}$ be vectors where all entries are independent with zero mean and unit variance. The variance of their dot product is then:

$$
\mathtt{
  Var\left(\sum_{i=1}^h x_iy_i\right) = \sum_{i=1}^h Var(x_i)Var(y_i) = \sum_{i=1}^h 1 = h
}
$$

Thus we apply a normalization to remove this factor:

$$
\mathtt{s_{j,i}} = \frac{\mathbf{q}_{\mathtt{j}}^{\mathtt{T}}\mathbf{k}_{\mathtt{i}}}{\sqrt{\mathtt{h}}}
$$

In code, we implement this by calling a helper function `head_scale`:

```catform
scaled: bf16[N, N] = call[head_scale](scores)
```

The scale is the config scalar `param.attn_scale` $= 1/\sqrt{\mathtt{h}} = 1/\sqrt{128} \approx 0.0884$.

Now, the causality. A token can never attend to future tokens---only prior tokens and itself. We thus want to *force* $\mathtt{p_{j,i}}=0$ when $\mathtt{i>j}$. We do so by setting all corresponding scores to negative infinity:

$$
\mathtt{s_{j,i}}=-\infty
$$

This gives the desired outcome since softmax exponentiates every entry, and $\mathtt{e^x\to0}$ as $\mathtt{x\to-\infty}$. We implement this by applying a `causal_mask` helper function, which sets the $\mathtt{i>j}$ entries to $-\infty$ and leaves the others untouched.

```catform
masked: bf16[N, N] = call[causal_mask](scaled)
```

Finally, to extract the probability distribution we apply the softmax function

```catform
attn: bf16[N, N] = call[softmax](masked)
```

Now that we have computed the attention distribution, we compute the expectation of the values across positions.

$$
\mathbf{y}_{\mathtt{j}} = \sum_{\mathtt{i}=0}^{\mathtt{N-1}} \mathtt{p_{j,i}} \, \mathbf{v}_{\mathtt{i}}
$$

This parallelizes over the $\mathtt{j}$ axis and is hence just a contraction over the $\mathtt{i}$ axis:

```catform
val: bf16[N, h] = contract["j i, i h -> j h"](attn, v)
```

Putting this all together, we arrive at the `attend` function in `qwen3/model.cat`:

```catform
attend(
  q   : bf16[..., N, h],
  k   : bf16[..., M, h],
  v   : bf16[..., M, h],
  mask: bf16[N, M]
) -> (val: bf16[..., N, h]) {
  scores: bf16[..., N, M] = contract["... j h, ... i h -> ... j i"](q, k)
  scaled: bf16[..., N, M] = call[head_scale](scores)
  masked: bf16[..., N, M] = call[causal_mask](mask, scaled)
  attn  : bf16[..., N, M] = call[softmax](masked)
  val   : bf16[..., N, h] = contract["... j i, ... i h -> ... j h"](attn, v)
}
```

## The Big Picture

We began this section with the model's type signature, and then slowly deconstructed its internals. Putting it all together, we can represent the main structure of the model in the following schematic.

$$
\begin{tikzpicture}[every node/.style={font=\scriptsize}]

\node at (0,-3.75) {\large $\mathtt{model}$};
\node at (0,-4.25) {\small $\mathtt{transformer}$};

\draw[very thick, rounded corners] (-6.0,-7.5) rectangle (6.0,-3.5);
\draw[very thick, rounded corners, densely dotted] (-3.9,-7) rectangle (3.9,-4);

\draw[very thick, rounded corners] (-5.4,-6) rectangle (-4.2,-5);
\draw[very thick, rounded corners] ( 5.4,-6) rectangle ( 4.2,-5);
\node at ( 4.8,-5.5) {$\mathtt{unembed}$};
\node at (-4.8,-5.5) {$\mathtt{embed}$};

\draw[very thick] (-7,-5.5) -- (-5.4,-5.5) ;
\draw[very thick] ( 7,-5.5) -- ( 5.4,-5.5) ;

\draw[very thick] (-7, -6.5) -- (-4.6, -6.5) to[out=0, in=180, looseness=1.5] (-3.6,-5.65);
\draw[very thick] ( 7, -6.5) -- ( 2.8, -6.5) to[out=180, in=0, looseness=1.5] ( 1.8,-5.65);

\draw[very thick, rounded corners] (2.4,-6) rectangle (3.6,-5);
\draw[very thick, rounded corners] (0.6,-6) rectangle (1.8,-5);
\draw[very thick, rounded corners] (-0.6,-6) rectangle (-1.8,-5);
\draw[very thick, rounded corners] (-2.4,-6) rectangle (-3.6,-5);

\node at (3.0,-5.5) {$\mathtt{rmsnorm}$};
\node at (1.2,-5.5) {$\mathtt{layer}$};
\node at (-1.2,-5.5) {$\mathtt{layer}$};
\node at (-3.0,-5.5) {$\mathtt{layer}$};

\draw[very thick] (-3.6,-5.35) -- (-4.2,-5.35) ;
\draw[very thick] ( 3.6,-5.50) -- ( 4.2,-5.50) ;

\draw[very thick] (-2.4,-5.35) -- (-1.8,-5.35) ;
\draw[very thick] ( 2.4,-5.35) -- ( 1.8,-5.35) ;

\draw[very thick] (-2.4,-5.65) -- (-1.8,-5.65) ;

\draw[very thick] (-0.6,-5.35) -- (-0.2,-5.35) ;
\draw[very thick] ( 0.6,-5.35) -- ( 0.2,-5.35) ;

\draw[very thick] (-0.6,-5.65) -- (-0.2,-5.65) ;
\draw[very thick] ( 0.6,-5.65) -- ( 0.2,-5.65) ;

\node at (0,-5.5) {$\cdots$};


\node at (0,-9.25) {\large $\mathtt{layer}$};
\draw[very thick, rounded corners] (-6.0,-13) rectangle (6.0,-9);

\draw[very thick] (-7,-9.9) -- (7,-9.9) ;
\draw[very thick] (-7,-11.9) -- (-4,-11.9) to[out=0, in=180, looseness=1.5] (-2.5,-11.15);
\draw[very thick] ( 7,-11.9) -- (0.5,-11.9) to[out=180, in=0, looseness=1.5] (  -1,-11.15);

\draw[very thick, rounded corners] (-5,-11.5) rectangle (-3.5,-10.5);
\draw[very thick, rounded corners] (-2.5,-11.5) rectangle (-1,-10.5);
\node at (-4.25,-11) {$\mathtt{rmsnorm}$};
\node at (-1.75,-11) {$\mathtt{attention}$};

\draw[very thick] (-5.5,-9.9) to[out=-90, in=180, looseness=1] (-5,-11) ;
\draw[very thick] (-3.5,-10.85) -- (-2.5,-10.85) ;
\draw[very thick] (-1,-10.85) to[out=0, in=-90, looseness=1] (-0.5,-9.9) ;

\draw[very thick, rounded corners] (5,-11.5) rectangle (3.5,-10.5);
\draw[very thick, rounded corners] (2.5,-11.5) rectangle (1,-10.5);
\node at (4.25,-11) {$\mathtt{ffn}$};
\node at (1.75,-11) {$\mathtt{rmsnorm}$};

\draw[very thick] (5.5,-9.9) to[out=-90, in=0, looseness=1] (5,-11) ;
\draw[very thick] (3.5,-11) -- (2.5,-11) ;
\draw[very thick] (1,-11) to[out=180, in=-90, looseness=1] (0.5,-9.9) ;



\node at (0,-14.75) {\large $\mathtt{attention}$};
\draw[very thick, rounded corners] (-6,-18.5) rectangle (6,-14.5);

\draw[very thick] (-7,-16.225) -- (-5.1,-16.225) ;

\draw[very thick] (-7,-17.625) -- (0.6,-17.625) ;
\draw[very thick] ( 7,-17.625) -- (1.8,-17.625) ;

\draw[very thick, rounded corners] (-5.1,-15.525) -- (-5.6,-15.525) -- (-5.6,-16.925) -- (-5.1,-16.925) ;

\draw[very thick, rounded corners] (-5.1,-15.775) rectangle (-4.1,-15.275);
\draw[very thick, rounded corners] (-5.1,-16.475) rectangle (-4.1,-15.975);
\draw[very thick, rounded corners] (-5.1,-17.175) rectangle (-4.1,-16.675);
\node at (-4.6,-15.525) {$\mathtt{Q}$};
\node at (-4.6,-16.225) {$\mathtt{K}$};
\node at (-4.6,-16.925) {$\mathtt{V}$};

\draw[very thick, rounded corners] (-3.5,-15.775) rectangle (-1.8,-15.275);
\draw[very thick, rounded corners] (-3.5,-16.475) rectangle (-1.8,-15.975);
\node at (-2.65,-15.525) {$\mathtt{rmsnorm}$};
\node at (-2.65,-16.225) {$\mathtt{rmsnorm}$};

\draw[very thick, rounded corners] (-1.2,-15.775) rectangle (0,-15.275);
\draw[very thick, rounded corners] (-1.2,-16.475) rectangle (0,-15.975);
\node at (-0.6,-15.525) {$\mathtt{rope}$};
\node at (-0.6,-16.225) {$\mathtt{rope}$};

\draw[very thick] (-4.1,-15.525) -- (-3.5,-15.525) ;
\draw[very thick] (-1.8,-15.525) -- (-1.2,-15.525) ;
\draw[very thick] (0,-15.525) -- (2.4,-15.525) ;

\draw[very thick] (-4.1,-16.225) -- (-3.5,-16.225) ;
\draw[very thick] (-1.8,-16.225) -- (-1.2,-16.225) ;
\draw[very thick] (0,-16.225) -- (0.6,-16.225) ;
\draw[very thick] (1.8,-16.225) -- (2.4,-16.225) ;

\draw[very thick] (-4.1,-16.925) -- (0.6,-16.925) ;
\draw[very thick] (1.8,-16.925) -- (2.4,-16.925) ;

\draw[very thick, rounded corners] (0.6,-17.875) rectangle (1.8,-15.975);
\node at (1.2,-16.925) {$\mathtt{KV\:cache}$};

\draw[very thick, rounded corners] (2.4,-17.175) rectangle (3.9,-15.275);
\node at (3.15,-16.225) {$\mathtt{attend}$};

\draw[very thick, rounded corners] (4.5,-16.475) rectangle (5.5,-15.975);
\node at (5,-16.225) {$\mathtt{O}$};

\draw[very thick] (3.9,-16.225) -- (4.5,-16.225) ;
\draw[very thick] (5.5,-16.225) -- (7,-16.225) ;

\end{tikzpicture}
$$
