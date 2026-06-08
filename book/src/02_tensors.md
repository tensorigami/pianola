# Tensors

This chapter is a mathematical interlude on **tensors** and their operations, which will be expressed as executable code in our domain-specific tensor programming language ***catform***.

Conceptually a tensor is just a multi-dimensional---or, henceforth, multi-**axis**---array of entries, all of the same **datatype**. Their key feature is that they both capture the mathematics of [linear algebra](https://en.wikipedia.org/wiki/Linear_algebra), and are also a convenient data abstraction for computing with **massively parallel processors**, like [GPUs](https://en.wikipedia.org/wiki/Graphics_processing_unit) and [TPUs](https://en.wikipedia.org/wiki/Tensor_Processing_Unit), that execute many thousands of operations simultaneously.

A **tensor operation** transforms a tensor in a manner that is parallelizable across all but a few---often one or even zero---axes. Due to the pressure of scaling, language model architectures have had to use a constrained set of operations that are maximally parallel. In fact, catform only supports seven such operations, and these suffice to express any modern language model.

We can encode a tensor's **shape** as a tuple of integers. The length of this tuple is the number of axes, while each integer represents the size of a given axis. In catform, we represent the **type** of a tensor with shape `S` and entries in datatype `d` as `d[S]`.

A useful way to think about tensors is as nested vectors. For instance, an `int[2,4]` tensor is just a `2`-vector of a `4`-vector of `int` entries. The order of the numbers in the shape tuple goes from outside-in.

In fact, if we wish to write a concrete tensor in catform, we represent it in exactly that way, wrapped in a `literal` keyword:

```cat
x: int[2,4] = literal([[0, 1, 2, 3], [4, 5, 6, 7]])
```
In fact, *all* data objects in catform are tensors. This includes plain numbers, or **scalars**, which we represent as zero-axis tensors---i.e. those of shape `[]`. For instance, a single real number has the type `real[]`:

```cat
pi: real[] = literal(3.14)
```

The `literal` keyword above is an **introduction**: it brings fresh tensor data into a program---it does not *transform* any tensors, and hence is not an operation. Catform supports two other introductions.

We can introduce a range of data, starting at a specified integer, via the `iota` introduction:

```py
# uv run main.py repl

In [1]: iota(0, 5)
Out[1]: [0, 1, 2, 3, 4]

In [2]: iota(10, 14)
Out[2]: [10, 11, 12, 13]
```

We can also introduce random data:

```py
# uv run main.py repl

In [1]: random[-1.0, 1.0](3)
Out[1]: [-0.007, 0.536, -0.823]

In [2]: random[0.0, 1.0](2, 3)
Out[2]: [[0.132, 0.307, 0.634], [0.490, 0.896, 0.456]]
```

Catform lines all follow the same surface form:

```cat
keyword[*specifiers](*args)
```
where brackets hold compile-time specifiers (constants, patterns, function names fixed when the program is written) and parens hold the inline data or runtime inputs.

## Tensor Operations

Catform has seven native operation types, of which six are primitive. The seventh can be derived from a combination of the others, but it is elevated as native due to its significance and ubiquity.

A catform operation is given by its operation type `op` and one or more **specifiers** `spec`, denoted in square brackets after the operation type. Operations take one or more arguments and output a single value. In catform, we assign the output of an operation as:

```cat
y: d[S] = op[spec_1,...,spec_m](x_1, ..., x_n)
```

In this equation, `y` is the output, `d[S]` its type (with datatype `d` and shape `S`), `op` one of the operation types, `spec_1,...,spec_m` its set of specifiers (in practice, there are at most two) and `x_1, ..., x_n` its arguments.

We will now catalogue the six primitive operation types. A nice way to organize them is:
- a `view` alters the shape but not the values of the entries
- a `map` alters the values but not the shape
- a `fold` compacts an axis, while a `tile` expands it
- a `read` reads from a lookup table, while a `write` writes to it

Each operation is illustrated with interactive examples in the [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop), which allows executing catform operations directly. The REPL's syntax is a bit more lenient than in `.cat` files---type annotations are not required, and literal nested lists are accepted as tensor inputs.

### Views

**View** operations just change the tensor shape without touching the entries---and can therefore be realized without computation. But how do they do so?

Consider the `int[2,4]` tensor

```cat
[
  [0, 1, 2, 3],
  [4, 5, 6, 7],
]
```
From the standpoint of a computer, the entries are originally arranged as a contiguous block of memory. To imbue this block with the structure of a `[2,4]` tensor, we record a metadata tuple called a **stride**. The stride tuple consists of increments used to traverse each axis. We call the position of an entry in the flat memory block its **offset**, and the tuple of per-axis coordinates its **index**.

This tensor's stride is `[4,1]`---incrementing the *inner* (size `4`) vector index advances offset by `1`, while incrementing the index of the *outer* vector---e.g. moving from `2` to `6`---advances `4` offsets. One can relate the offset to the stride and index via a dot product:

$$
\mathtt{offset = index \cdot stride}
$$

For instance, the entry `6` has index `[1,2]`---it is the second entry within the first vector (starting, of course, with "zeroth"). Via our formula, it thus has an offset of `6`:

$$
\mathtt{(4,1)\cdot(1,2) = 4\cdot1 + 1\cdot2 = 6}
$$

View operations thus work by simply changing the shape and stride tuples. For instance, suppose we wish to convert the above `int[2,4]` tensor to its transpose `int[4,2]` tensor:

```cat
[
  [0, 4],
  [1, 5],
  [2, 6],
  [3, 7]
]
```
The reader can check that, to do so, we would change the stride from `[4,1]` to `[1,4]`.

To express such operations in code, we choose as the specifier the evocative syntax of **pattern** strings, popularized by [einops](https://github.com/arogozhnikov/einops). These are best explained through example. For instance, the above transpose is implemented in the following catform function:

```cat
transpose(x: int[2, 4]) -> (y: int[4, 2]) {
    y : int[4, 2] = view["a b -> b a"](x)
}
```

In general, view follows the type rule:

```text
view["... a b -> ... b a"](x: d[..., A, B]) : d[..., B, A]
```

We can execute operations of this sort in the REPL:

```py
# uv run main.py repl

In [1]: view["a b -> b a"]([[1, 2, 3], [4, 5, 6]])
Out[1]: [[1, 4], [2, 5], [3, 6]]
```

Another variant of the view operation comes from merging and splitting axes. For instance, suppose we wished to transform the above to the `int[2,2,2]` tensor

```cat
[
  [
    [0, 1],
    [2, 3],
  ]
  [
    [4, 5], 
    [6, 7]
  ],
]
```

We can express this computation as the following catform function:

```cat
split(x: int[2, 4]) -> (y: int[2, 2, 2]) {
    y : int[2, 2, 2] = view["a (b c) -> a b c"](x)
}
```

We use the pattern `"a (b c) -> a b c"` to convey that the original second axis `(b c)` had the potential to be factored into two new axes `b` and `c`. In this example, the type annotation is critical: it tells us exactly *how* to numerically factor the axis size. Alternatively, pattern strings can contain literal numbers---in this case we could have also written `"a (2 c) -> a 2 c"`, where the size of `c` gets deduced to be `2`.

```py
# uv run main.py repl

In [1]: view["a (2 c) -> a 2 c"]([[1, 2, 3, 4], [5, 6, 7, 8]])
Out[1]: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
```

The interested reader can compute for themselves what happens to the strides in such merges and splits.

In general, if a tensor contains `N` total entries, then it can be viewed as a tensor of any shape `[n_0,...,n_k]` as long as this is a valid factorization of `N`. 

As a consequence of this, views can freely add or remove axes of size `1` from the shape. Visually, this looks like adding brackets around singleton vectors or entries:

```py
# uv run main.py repl

In [1]: view["a -> a 1"]([10,20,30])
Out[1]: [[10], [20], [30]]

In [2]: view["a -> 1 a"]([10,20,30])
Out[2]: [[10, 20, 30]]

In [3]: view["1 a -> a"]([[10, 20, 30]])
Out[3]: [10, 20, 30]
```

In practice, we may work with tensors with many axes, but only operate on the last, i.e. innermost, few, while the leading axes pass through unchanged. Rather than naming each of these pass-through axes, we use `...` to match any number of leading axes that appear identically on both sides of the arrow. For example, `"a b -> b a"` transposes a 2-axis tensor, while `"... a b -> ... b a"` transposes the last two axes of a tensor with any number of axes. The axes matched by `...` are exactly those where independent copies of the operation run in parallel. Catform also uses `...` in type annotations: `int[..., N, M]` denotes a tensor with any number of leading axes as long as the innermost two are `N` and `M`. This convention applies identically to every operation in the rest of the chapter.

### Maps

**Map** operations lift scalar functions to tensors by applying them entry-wise. The first variant lifts a unary (i.e. single argument) function

$$
\mathtt{f:x\to y}
$$

to a function of shape $\mathtt{S}$ tensors

$$
\mathtt{map[f] : x[S]\to y[S]}
$$

We name this operation by using the function $\mathtt{f}$ as the specifier to the op type $\mathtt{map}$. This operation is defined by applying the function to each entry. For instance, the squaring function

$$
\mathtt{square:int\to int}
$$

can be lifted to int-valued tensors as $\mathtt{map[square]}$, applied entry-wise:

```py
In [1]: map[square]([[1, 2], [3, 4]])
Out[1]: [[1, 4], [9, 16]]
```

Maps of unary functions interact nicely with function composition $\mathtt{f \:;\: g}$, via the **functoriality** property:

$$
\mathtt{map[f \:;\: g] = map[f] \:;\: map[g]}
$$

In words, mapping $\mathtt{f}$ then mapping $\mathtt{g}$ is the same as mapping their composition $\mathtt{f \:;\: g}$ in a single pass. Two consecutive unary maps can always be fused into one---a fact that proves useful for optimizing performance.

The other variant of map operations comes from mapping binary (or even $\mathtt{n}$-ary!) operators, i.e. those with two (or more) inputs, such as the classic boolean $\mathtt{and, or}$ and arithmetic $\mathtt{add, mul}$ operators. For instance, adding and multiplying tensors:

```py
In [1]: map[add]([1, 2, 3], [10, 20, 30])
Out[1]: [11, 22, 33]

In [2]: map[mul]([0.5, 1.0, 2.0], [4.0, 4.0, 4.0])
Out[2]: [2.0, 4.0, 8.0]
```

Maps are maximally parallel: every entry coordinate can be computed independently, so mapping tensors of shape `[s_1,...,s_n]` yields `s_1 * ... * s_n` independent computations.

### Folds

**Fold** operations combine all entries along an axis into one, collapsing that axis from size $\mathtt{n}$ to size $\mathtt{1}$. 

To denote a fold operation, we need two specifiers. We first need a pattern to indicate which axis is being collapsed---by replacing the corresponding axis label with `1` on the right-hand side. For example, `"a b c -> a 1 c"` collapses axis `b`. Second, we must specify the **reduction** being used. A reduction is a function that takes any number of inputs of a given type, i.e. a list, and combines them into a single output, hence having a signature

$$
\mathtt{list[x] \to x}
$$

The following reductions are supported:
- $\mathtt{sum,prod}$ for the sum and product of a list of numbers
- $\mathtt{mean}$ for the arithmetic mean of a list of numbers
- $\mathtt{min,max}$ for the minimum and maximum of a list of numbers
- $\mathtt{all,any}$ for checking the truth of all or any of a list of booleans

All of the above reductions are both associative and commutative, so the order in which entries are combined does not matter---the fold is well-defined regardless of how we traverse the axis, giving the hardware more freedom to parallelize. The choice of reduction also constrains the entry type: `mean` requires real-valued entries, while `all` and `any` require booleans.

For example:

```py
# uv run main.py repl

In [1]: fold["a b -> 1 b", sum]([[1, 2, 3], [4, 5, 6]])
Out[1]: [[5, 7, 9]]

In [2]: fold["a b -> a 1", prod]([[1, 2, 3], [4, 5, 6]])
Out[2]: [[6], [120]]
```

In general, fold follows the type rule:

```text
fold["... a b -> ... 1 b", r](x: d[..., A, B]) : d[..., 1, B]
```

where `r: list[d] -> d` is the reduction.

Fold patterns can collapse multiple axes simultaneously, e.g. `"a b c -> 1 1 c"` folds both `a` and `b`---combining entries across two axes is no different from combining across one.

Folds are parallel across all non-folded axes: in `"a b -> 1 b"`, each of the `b` columns is reduced independently.

### Tiles

<!--
TODO: this intro frames tile as 1→n only, but L267 shows the source axis
need not be 1 (e.g. `tile["g -> (g 2)"]`). Either generalize the intro or
flag the 1→n case as the canonical/common one with the general case
introduced later.
-->

**Tile** operations involve repeating an axis, thus expanding that axis from size $\mathtt{1}$ to size $\mathtt{n}$. Like `fold`, a tile operation requires a pattern specifier to indicate which axis is being expanded, and we do so via its flipped pattern `1 b -> a b`, which indicates that the first axis is being expanded from $\mathtt{1}$ to $\mathtt{a}$. Unlike fold, no reduction specifier is needed---the only generic option available is to replicate data.


```py
# uv run main.py repl

In [1]: tile["1 b -> 3 b"]([[1, 2, 3]])
Out[1]: [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
```

In general, tile follows the type rule:

```text
tile["... 1 b -> ... a b"](x: d[..., 1, B]) : d[..., A, B]
```

where `A` is set by the output type annotation.

Note that the output type annotation is essential here---the pattern `"1 b -> a b"` introduces a new axis `a`, and its size is determined by the type annotation. Like views, tile patterns can also contain literal numbers: `"1 b -> 3 b"` replicates to exactly 3 rows, making the type annotation redundant for that axis.

Tiles are parallel across all non-tiled axes: in `"1 b -> a b"`, each of the `b` entries is replicated independently.

Tile patterns can be richer than the `1 -> a` case---the source axis need not have size 1. For instance, `tile["g -> (g 2)"]` replicates each entry twice and merges the new copies back into the same axis:

```py
# uv run main.py repl

In [1]: tile["g -> (g 2)"]([10, 20, 30])
Out[1]: [10, 10, 20, 20, 30, 30]
```

In this sense, tile is strictly more flexible than fold, which always collapses an axis to size `1`.

### Reads

The first four ops are **value-oblivious** in that each output entry can be given by a formula over the input's indices alone---i.e. we never had to look at an actual entry to decide whether or not an index would get wired into a computation output. Read and write break this property: the values stored in the index tensor steer which input entries the output draws from. This is **data-dependent addressing**, the mechanism by which we look up entries by index.

**Read** operations select entries from a tensor using indices from another tensor. Given a data tensor and an index tensor, a read produces an output whose entries are looked up from the data at the positions specified by the indices. Indexing along the `_` axis, a read mirrors the following loop:

```py
for i, index in enumerate(indices):
    out[i] = data[index]
```

Like fold and tile, read uses a pattern specifier to indicate which axis is affected. The marker `_` denotes the axis replaced by the index tensor's shape. For example, the pattern `v d -> _ d` means: axis `v` is replaced by the index shape `_`, while axis `d` passes through unchanged.

Suppose we have a `real[4, 2]` tensor of data and an `int[3]` tensor of indices

```py
# uv run main.py repl

In [1]: data = [[10, 20], [30, 40], [50, 60], [70, 80]]

In [2]: idx = [2, 0, 3]
```

Then the read selects rows `2, 0, 3` from the data:

```py
In [3]: read["v d -> _ d"](data, idx)
Out[3]: [[50, 60], [10, 20], [70, 80]]
```

The `_` in the output shape corresponds to the index tensor's size (`3`), while the non-`_` axes pass through in their same positions.

The indices can take on any shape in fact:

```py
In [4]: idx2 = [[2,3],[0,1]]

In [5]: read["v d -> _ d"](data, idx2)
Out[5]: [[[50, 60], [70, 80]], [[10, 20], [30, 40]]]
```

In general, read follows the type rule:

```text
read["... v d -> ... _ d"](data: d[..., V, D], idx: int[I]) : d[..., I, D]
```

where `I` denotes the (possibly multi-axis) shape of `idx`, which the `_` placeholder expands into in the output.

<!--
TODO: mild redundancy — "lookup table" was already used at L289 ("a read
produces an output whose entries are looked up from the data"). Either
drop this sentence, drop the earlier mention, or link the wiki page only
once.
-->
Read implements a [lookup table](https://en.wikipedia.org/wiki/Lookup_table): the indices select entries along an axis of the data.

Each output entry requires exactly one lookup, so reads are parallel across all output coordinates---each index can be followed independently.

### Writes

**Write** operations are the dual of read. A read uses the index tensor to *pull*: each output entry names the input position it comes from. A write uses the index tensor to know which output index to *push* to. Unlike read, write takes a second specifier: a **reduction** that says how to combine source values that land on the same target slot.

A write can also be conveyed as a loop---the dual one of read---which writes each source entry into the slot named by the index, combining with the existing target value via `*=` (where `*` is the chosen reduction: `sum`, `prod`, `max`, `min`, or `set`, the last one defined by `a * b = b`):

```py
for i, index in enumerate(indices):
    target[index] *= source[i]
```

Slots that no index hits keep their target value; slots that one or more indices hit get folded with the source values via the reduction.

Like read, write uses a pattern specifier to indicate which axis is affected, with the marker `_` denoting the input axis replaced by the index. The write pattern is the reverse of the read pattern: where read uses `v d -> _ d`, write uses `_ d -> v d`---the `_` axis in the input is mapped back into the named axis in the output.

Using the same index tensor `idx = [2, 0, 3]`, we can write a `real[3, 2]` source into a `real[4, 2]` target of zeros, with the `sum` reduction:

```py
# uv run main.py repl

In [1]: source = [[1, 2], [3, 4], [5, 6]]

In [2]: target = [[0, 0], [0, 0], [0, 0], [0, 0]]

In [3]: idx = [2, 0, 3]

In [4]: write["_ d -> v d", sum](source, idx, target)
Out[4]: [[3, 4], [0, 0], [1, 2], [5, 6]]
```

In general, write follows the type rule:

```text
write["... _ d -> ... v d", r](source: d[..., I, D], idx: int[I], target: d[..., V, D]) : d[..., V, D]
```

where `I` denotes the (possibly multi-axis) shape of `idx`, which the `_` placeholder expands into in the source, and `r` is the reduction---as in fold, here applied pairwise: `(d, d) -> d`.

Writes can be parallelized across the non-`_` axes---index collisions can be handled via atomic primitives, so even overlapping writes can proceed in parallel. The first example below demonstrates `sum` accumulation when indices collide; the second shows `set` simply overwriting:

```py
# uv run main.py repl

In [1]: write["_ -> v", sum]([10, 20, 30], [1, 3, 1], [0, 0, 0, 0])
Out[1]: [0, 40, 0, 20]

In [2]: write["_ d -> v d", set]([[1, 2], [3, 4]], [0, 2], [[9, 9], [9, 9], [9, 9]])
Out[2]: [[1, 2], [9, 9], [3, 4]]
```

In Out[1], the index values at source positions `0` and `2` are both `1`, so under `sum` their values (`10 + 30 = 40`) combine. In Out[2], the `set` reduction leaves position `1` at the target's original `9` (unindexed), and replaces positions `0` and `2` with the source rows.

## Functions

<!--
TODO: "function" is used loosely before this section — at L81/L119
("the above transpose is implemented in the following catform function")
and throughout the Maps section in the mathematical sense ("scalar
functions", "function composition", "unary function"). The catform
*function abstraction* (signature + body + curly brackets) is only
formally introduced here. Either: (a) rename earlier "catform function"
uses to avoid the implicit forward reference, (b) explicitly note here
that we're now formalizing what was used informally, or (c) decide it's
fine because earlier uses are clear from context.
-->

A `.cat` file represents a single computation, taking one or more input tensors and transforming them via a sequence of tensor operations, until the desired output tensors have been computed. While it is possible to express this computation as a single list of operations, catform offers [function abstraction](https://en.wikipedia.org/wiki/Function_(computer_programming)) for the sake of code organization and readability. Every `.cat` file has an [entry point](https://en.wikipedia.org/wiki/Entry_point) function named `main`, from which the entire computation expands---as we will see shortly via a process called *flattening*.

The function declaration syntax is given by the function's name, its tuple of typed arguments, an arrow, its tuple of typed return values, and a body of assignment lines in curly brackets:

```cat
func(x_1: d_1[S_1], ..., x_n: d_n[S_n]) -> (y_1: d'_1[S'_1], ..., y_m: d'_m[S'_m]) {
    ...
}
```

The function's output values are whichever assigned variables in the body match the return names declared after the arrow.

The assignment lines in a function body are either operations:

```cat
y: d[S] = op[spec_1,...,spec_m](x_1, ..., x_n)
```

or calls to other functions defined in the file:

```cat
y: d[S] = call[f](x_1, ..., x_n)
```

Since functions may return multiple outputs, we can assign them to multiple variables at once via the following multi-line syntax:

```cat
u: U,
v: V  = call[f](x)
```

It is also convenient to have notation for repeated calls of a given function---we do so via `loop`, which repeats a function a fixed number of times, e.g.

```cat
y : T = loop[f, 3](y, s, w.*)
```

expands to the three lines

```cat
y_1 : T = call[f](y,   s, w.0)
y_2 : T = call[f](y_1, s, w.1)
y_3 : T = call[f](y_2, s, w.2)
```
The `loop` syntax allows for three distinct variable behaviors:
- an argument like `y`, which appears on both sides, is **threaded**---we enforce variables to be immutable, so each iteration produces a fresh name (`y_1`, `y_2`, `y_3`) that feeds into the next
- an argument like `w.*` marked with `.*` is **indexed**---the `.*` is replaced with `.0`, `.1`, `.2`.
- all other arguments, like `s`, are merely repeated unchanged.
Neither `call` nor `loop` is an operation---they are abstractions that organize the program for readability. Prior to execution, they get removed via **flattening**: we recursively [unroll](https://en.wikipedia.org/wiki/Loop_unrolling) every `loop` and [inline](https://en.wikipedia.org/wiki/Inline_expansion) every `call` in `main` until it consists solely of operations. The result is the "single list of operations" mentioned at the start of this section---one flat `main` that represents the entire module's computation.

The flattened `main` follows [**static single assignment**](https://en.wikipedia.org/wiki/Static_single-assignment_form)---meaning that each variable is only assigned once. In fact, it is a [**straight-line program**](https://en.wikipedia.org/wiki/Straight-line_program)---no branches, no loops, just a sequence of operations from top to bottom.

## Tensor Contractions

We elevate one family of functions---**tensor contraction**---to a derived operation type, because of its centrality to linear algebra. We first review the mathematics, perhaps shedding new light on it, and then show how contractions decompose into the more primitive operation types.

Let's review some linear algebra. For the time being, forget about tensors---we will connect back to them soon. For this section, we will deviate from some of our notational conventions to show their correspondence with the conventional mathematics notation. Recall that a vector in $\mathbb{R}^{\mathtt{n}}$ is by convention denoted as a **column** vector, e.g. the following $\mathbb{R}^{\mathtt{3}}$ vector

$$
\begin{bmatrix}
\mathtt{1} \\ \mathtt{2} \\ \mathtt{3}
\end{bmatrix}
$$

Why was there such an emphasis on these vectors being written as columns rather than rows? What semantic meaning did that distinction carry? Recall that a matrix was depicted as a two dimensional array, e.g. the following $\mathtt{2\times 3}$ matrix

$$
\begin{bmatrix}
\mathtt{4} & \mathtt{5} & \mathtt{6} \\
\mathtt{7} & \mathtt{8} & \mathtt{9}
\end{bmatrix}
$$

A matrix carried further meaning than merely being a two dimensional array of numbers---in particular an $\mathtt{n\times m}$ matrix---i.e. one with $\mathtt{m}$ columns and $\mathtt{n}$ rows---corresponded to a **linear map** of type

$$
\mathbb{R}^{\mathtt{m}}\to\mathbb{R}^{\mathtt{n}}
$$

A particularly simple type of matrix is the one with a single row, i.e. a $\mathtt{1\times m}$ matrix, which we also call an $\mathbb{R}^{\mathtt{m}}$ **row vector** or **covector**, which is a function of type

$$
\mathbb{R}^{\mathtt{m}}\to\mathbb{R}
$$

This means that if we have an $\mathbb{R}^{\mathtt{m}}$ covector $\mathbf{w}$ and an $\mathbb{R}^{\mathtt{m}}$ vector $\mathbf{v}$, we have a function of type $\mathbb{R}^{\mathtt{m}}\to\mathbb{R}$ along with an element of its domain $\mathbb{R}^{\mathtt{m}}$---which means we can *apply* the function to the vector:

$$
\mathbf{w}(\mathbf{v}):\mathbb{R}
$$

In practice, we obtain a covector by transposing a column vector $\mathbf{w}$, and denote the result $\mathbf{w}^{\mathtt{T}}$. With this notation, we drop the parentheses and write the application as:

$$
\mathbf{w}^{\mathtt{T}}\mathbf{v}:\mathbb{R}
$$

We compute this value in terms of the entries of the covector and vector using the **dot product**:

$$
\begin{bmatrix}\mathtt{c_1} & \cdots & \mathtt{c_m} \end{bmatrix}\begin{bmatrix} \mathtt{x_1} \\ \vdots \\ \mathtt{x_m} \end{bmatrix} = \mathtt{c_1 x_1 + \cdots + c_m x_m}
$$

Multiply entry-by-entry, then sum---this is exactly a map followed by a fold. We can write this as a catform program using our six primitive tensor operations. If we view both vector and covector as $\mathtt{real[m]}$ tensors:

```cat
dot_product(w: real[m], v: real[m]) -> (val: real[]) {
    wv  : real[m] = map[mul](w, v)
    val_: real[1] = fold["m -> 1", sum](wv)
    val : real[]  = view["1 -> "](val_)
}
```

The dot product recurs in the context of applying an $\mathtt{n\times m}$ matrix function $\mathbf{W}:\mathbb{R}^{\mathtt{m}}\to\mathbb{R}^{\mathtt{n}}$ to an $\mathbb{R}^{\mathtt{m}}$ vector $\mathbf{v}$. In fact, it can be used to give an interpretation of what an $\mathtt{n\times m}$ matrix does, by interpreting it as $\mathtt{n}$ parallel $\mathbb{R}^{\mathtt{m}}$ covectors, each applied independently to the argument $\mathbf{v}$ to arrive at an $\mathbb{R}^{\mathtt{n}}$ vector:

$$
\begin{bmatrix}
- & \mathbf{w}^{\mathtt{T}}_{\mathtt{1}} & - \\
  & \vdots         &   \\
- & \mathbf{w}^{\mathtt{T}}_{\mathtt{n}} & - \\
\end{bmatrix}\mathbf{v} = \begin{bmatrix} \mathbf{w}_{\mathtt{1}}^{\mathtt{T}}\mathbf{v} \\ \vdots \\ \mathbf{w}_{\mathtt{n}}^{\mathtt{T}}\mathbf{v} \end{bmatrix}
$$

To write this in catform, we do so as above, except must first tile the vector to be the same shape as the matrix:

```cat
matrix_apply(w: real[n,m], v: real[m]) -> (vect: real[n]) {
    vv   : real[n,m] = tile["m -> n m"](v)
    wvv  : real[n,m] = map[mul](w, vv)
    vect_: real[n,1] = fold["n m -> n 1", sum](wvv)
    vect : real[n]   = view["n 1 -> n"](vect_)
}
```

We can extend this idea to the matrix multiplication of an $\mathtt{n\times m}$ matrix by an $\mathtt{m\times k}$ matrix---by viewing, as before, the former as $\mathtt{n}$ parallel $\mathbb{R}^{\mathtt{m}}$ covectors, and the latter as $\mathtt{k}$ parallel $\mathbb{R}^{\mathtt{m}}$ vectors. This allows us to think of matrix multiplication as $\mathtt{n\times k}$ worth of independent dot products.

$$
\begin{bmatrix}
- & \mathbf{w}^{\mathtt{T}}_{\mathtt{1}} & - \\
  & \vdots         &   \\
- & \mathbf{w}^{\mathtt{T}}_{\mathtt{n}} & - \\
\end{bmatrix}\begin{bmatrix}
     |         &         &     |        \\
 \mathbf{v}_{\mathtt{1}}  & \cdots  & \mathbf{v}_{\mathtt{k}} \\
     |         &         &     |
\end{bmatrix} = \begin{bmatrix}
\mathbf{w}_{\mathtt{1}}^{\mathtt{T}}\mathbf{v}_{\mathtt{1}} & \cdots & \mathbf{w}_{\mathtt{1}}^{\mathtt{T}}\mathbf{v}_{\mathtt{k}} \\
\vdots                     & \ddots & \vdots                     \\
\mathbf{w}_{\mathtt{n}}^{\mathtt{T}}\mathbf{v}_{\mathtt{1}} & \cdots & \mathbf{w}_{\mathtt{n}}^{\mathtt{T}}\mathbf{v}_{\mathtt{k}}
\end{bmatrix}
$$

To write this in catform, we do so as above, except must now tile *both* of the matrices to be the same shape:

```cat
matrix_multiply(w: real[n,m], v: real[m,k]) -> (mat: real[n,k]) {
    ww   : real[n,m,k] = tile["n m -> n m k"](w)
    vv   : real[n,m,k] = tile["m k -> n m k"](v)
    wwvv : real[n,m,k] = map[mul](ww, vv)
    mat_ : real[n,1,k] = fold["n m k -> n 1 k", sum](wwvv)
    mat  : real[n,k]   = view["n 1 k -> n k"](mat_)
}
```

Higher dimensional variants of such computations become harder to visualize, but the operation remains well defined. For instance, we may have an $\mathtt{o\times n}$ rectangular grid of parallel $\mathbb{R}^{\mathtt{m}}$ covectors and a rectangular prism grid of $\mathtt{k\times j \times i}$ parallel $\mathbb{R}^{\mathtt{m}}$ vectors, and calculate $\mathtt{o\times n\times k\times j\times i}$ dot products. Although the visuals become more challenging to imagine, the catform does not---this example is nearly identical to the above:

```cat
tensor_contraction(w: real[o,n,m], v: real[m,k,j,i]) -> (tens: real[o,n,k,j,i]) {
    ww    : real[o,n,m,k,j,i] = tile["o n m -> o n m k j i"](w)
    vv    : real[o,n,m,k,j,i] = tile["m k j i -> o n m k j i"](v)
    wwvv  : real[o,n,m,k,j,i] = map[mul](ww, vv)
    tens_ : real[o,n,1,k,j,i] = fold["o n m k j i -> o n 1 k j i", sum](wwvv)
    tens  : real[o,n,k,j,i]   = view["o n 1 k j i -> o n k j i"](tens_)
}
```

Just as we easily expanded the number of axes of the set of dot product calculations, we could also have expanded the number of axes that we summed across---there is nothing special about the covector, vector pair that enforces them to be single axis! The degenerate case has zero contracting axes---no shared indices are summed over. The result is the **outer product**: every entry of one tensor paired with every entry of the other, as in `"n, d -> n d"`. We call the summed-over axes **contracting axes** (often called contracting dimensions elsewhere). This general operation is so common---and carries so much mathematical semantics that we have to restrain ourselves from explicating further (the curious reader can explore [Penrose graphical notation](https://en.wikipedia.org/wiki/Penrose_graphical_notation))---that we elevate it to a derived form with native status alongside the six primitive operations above. As with many other operations, we specify such contractions with a pattern as follows:

$$
\mathtt{contract["as\:\:bs, bs\:\:cs \to as\:\:cs"] : (d[as, bs], d[bs, cs]) \to d[as, cs]}
$$

where $\mathtt{as, bs, cs}$ can all be any list of values. In catform this looks like:

```cat
z: d[as, cs] = contract["as bs, bs cs -> as cs"](x, y)
```

The contracting axes are those that appear in both inputs but not in the output. Contractions are parallel across all non-contracting axes: in `"n m, m k -> n k"`, the $\mathtt{n \times k}$ output entries are independent dot products, each computed in parallel. Likewise, `"... n d, o d -> ... n o"` contracts over `d` regardless of leading axes. When `...` appears on both inputs---as in `"... n h e, ... s h e -> ... h n s"`---it matches the same set of leading axes in each.

The four contraction specializations demonstrated above---dot product, matrix-vector, matrix multiply, and outer product---each reduce to a single `contract` op:

```py
# uv run main.py repl

In [1]: contract["m, m ->"]([1, 2, 3, 4], [5, 6, 7, 8])
Out[1]: 70

In [2]: contract["n m, m -> n"]([[1, 2, 3], [4, 5, 6]], [7, 8, 9])
Out[2]: [50, 122]

In [3]: contract["n m, m k -> n k"]([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
Out[3]: [[58, 64], [139, 154]]

In [4]: contract["n, m -> n m"]([1, 2, 3], [4, 5, 6, 7])
Out[4]: [[4, 5, 6, 7], [8, 10, 12, 14], [12, 15, 18, 21]]
```

## Numerics

Everything above used `real` and `int` as mathematical types. A machine has a finite number of bits, so every numeric value it stores must fit in a fixed-width binary word.

[Integers](https://en.wikipedia.org/wiki/Integer) are exact, so the only sacrifice is range. A $\mathtt{b}$-bit word can represent $\mathtt{2^b}$ distinct values. The standard convention, [**two's complement**](https://en.wikipedia.org/wiki/Two%27s_complement), splits them between negative and non-negative: a $\mathtt{b}$-bit signed integer represents values in $[-2^{\mathtt{b}-1},\: 2^{\mathtt{b}-1}-1]$.

[Real numbers](https://en.wikipedia.org/wiki/Real_number) are infinitely precise, so approximation is unavoidable. To represent them, we must choose a **numeric representation**. The standard format is given by the [**IEEE 754 floating-point**](https://en.wikipedia.org/wiki/IEEE_754) specification. We can derive this representation as follows. Given a real number $\mathtt{x}$, we can first extract its $\mathtt{sign}$---$\mathtt{0}$ if non-negative and $\mathtt{1}$ if negative---and rewrite it in terms of a non-negative real $\mathtt{r}$:

$$
\mathtt{x=(-1)^{sign}r}
$$

Then, let $\mathtt{exponent}$ denote the unique integer such that

$$
\mathtt{2^{exponent} \leq  r < 2^{exponent+1}}
$$

This lets us further deconstruct $\mathtt{x}$ as

$$
\mathtt{x=(-1)^{sign}2^{exponent}t}
$$

Note that by the choice of $\mathtt{exponent}$, $\mathtt{t}$ is always in the interval $\mathtt{[1,2)}$. We make the substitution $\mathtt{t=1+mantissa}$, where $\mathtt{mantissa}$ is a non-negative real number in the interval $\mathtt{[0,1)}$ and hence can be expressed in binary as $\mathtt{0.b_1b_2\dots}$. This gives us the unique decomposition used by floating point numbers:

$$
\mathtt{x = (-1)^{sign}2^{exponent}(1+mantissa)}
$$

The triple (sign, exponent, mantissa) uniquely characterizes a floating-point value. A floating-point *type* is determined by choosing how many bits to allocate to each component.

A given floating point representation has a fixed number $\mathtt{b}$---in practice a power of $\mathtt{2}$---of bits. It always allocates $\mathtt{1}$ bit to the sign, and then assigns $\mathtt{e}$ for the exponent and $\mathtt{m}$ for the mantissa. Assigning more bits to the exponent allows the representation to capture a wider *range* of numbers---much larger or much smaller---while assigning more to the mantissa gives higher *precision*---more binary digits before rounding occurs. Many numerical formats---specified by $\mathtt{b}, \mathtt{e}$, and $\mathtt{m}$---have been used in numerical computing.

If we wish to convert a tensor from numeric datatype `S` to `T`, we can do so by applying `map[T]`, where we overload `T` to denote the recast function of type $\mathtt{S\to T}$.

The two float types we use---[`f32`](https://en.wikipedia.org/wiki/Single-precision_floating-point_format) and [`bf16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)---both have `8` exponent bits and thus the same numerical range. They differ dramatically in precision, however, as `f32` has `23` mantissa bits while `bf16` has just `7`.

## Tensor Programming

<!--
TODO: the word "introductions" is used here (and at L651 "showcasing
introductions") as a category name for `literal` and `random`, but the
term was never defined earlier in the chapter as a category distinct
from ops. Also, `random` is first encountered in the `ball_vol` body
without any prior introduction — its semantics get explained only in
the line-by-line walkthrough afterward. Either: (a) introduce
"introductions" as a category in the Tensor Operations section
alongside the six primitive ops, (b) introduce `random` with a brief
example before `ball_vol`, or (c) reframe this sentence to not lean on
the undefined term.
-->

We now have all the ingredients: tensors and their types, seven operation types including contraction, introductions for constants and random data, and concrete numeric dtypes. Before we turn to language models, we put all of these components together in a single computation---with no machine learning in sight. Inspired by an excellent [3Blue1Brown lecture](https://youtu.be/fsLh-NYhOoU), we compute the $\mathtt{n}$-volume of the **unit $\mathtt{n}$-ball**, defined as the set of points in $\mathbb{R}^{\mathtt{n}}$ with distance at most $\mathtt{1}$ from the origin. Its volume is a classical quantity that at first grows and then shrinks as $\mathtt{n}$ increases---the unit ball becomes vanishingly small in high dimensions:

| $\mathtt{n}$ | Name | Volume | $\approx$ |
|---|---|---|---|
| $\mathtt{1}$ | segment | $2$ | `2.00` |
| $\mathtt{2}$ | disk | $\pi$ | `3.14` |
| $\mathtt{3}$ | ball | $\frac{4}{3}\pi$ | `4.19` |
| $\mathtt{4}$ | $\mathtt{4}$-ball | $\frac{1}{2}\pi^2$ | `4.93` |
| $\mathtt{5}$ | $\mathtt{5}$-ball | $\frac{8}{15}\pi^2$ | `5.26` |
| $\mathtt{10}$ | $\mathtt{10}$-ball | $\frac{1}{120}\pi^5$ | `2.55` |
| $\mathtt{100}$ | $\mathtt{100}$-ball | $\frac{1}{50!}\pi^{50}$ | $2.37 \times 10^{-40}$ |

The volume peaks at $\mathtt{n = 5}$ and then declines. The closed forms for even and odd dimensions are

$$
\mathtt{V_{2k}} = \frac{\pi^\mathtt{k}}{\mathtt{k!}} \qquad\qquad \mathtt{V_{2k+1}} = \frac{\mathtt{2}(\mathtt{4}\pi)^\mathtt{k} \, \mathtt{k!}}{(\mathtt{2k+1})!}
$$

Since both formulae have factorials in the denominator that eventually dominate the powers of $\pi$, the volume converges to zero. We can estimate it empirically via [Monte Carlo sampling](https://en.wikipedia.org/wiki/Monte_Carlo_method). The function takes the dimension $\mathtt{n}$ and sample count $\mathtt{S}$ as parameters, then generates random data inside---showcasing introductions. For each of $\mathtt{S}$ random points, we sample $\mathtt{n}$ coordinates uniformly from the interval $[\mathtt{-1}, \mathtt{1}]$.

A point lies inside the ball when

$$
\|\mathbf{x}\|^2 = \mathtt{x_1^2 + \cdots + x_n^2 \leq 1}
$$

The volume can be calculated entirely via catform operations:

```cat
// uv run main.py run book/book.cat ball_vol 5 10000
ball_vol(n: i32[], S: i32[]) -> (vol: f32[]) {
    // draw random sample
    x     : f32[n, S] = random[-1.0, 1.0]
    
    // compute the square norm for every sample
    sq    : f32[n, S] = map[mul](x, x)
    norm  : f32[1, S] = fold["n S -> 1 S", sum](sq)

    // compute the ratio of points in the ball
    inside: f32[1, S] = map[leq1](norm)
    frac  : f32[1, 1] = fold["1 S -> 1 1", mean](inside)
    ratio : f32[]     = view["1 1 ->"](frac)

    // scale the ratio by the volume of the hypercube
    side  : f32[]     = map[exp2](n)
    vol   : f32[]     = map[mul](ratio, side)
}
```

Line by line: `random` introduces an $\mathtt{f32[n, S]}$ tensor of uniform samples. `map[mul]` squares every coordinate. `fold[sum]` collapses the $\mathtt{n}$ axis, yielding a squared norm $\|\mathbf{x}\|^2$ for each sample. `map[leq1]` returns `1.0` for each sample whose norm is at most `1.0` and `0.0` otherwise---the indicator of being inside the ball. `fold[mean]` averages this indicator across all $\mathtt{S}$ samples, and `view` strips the unit axes to a scalar. The final two lines compute $\mathtt{2^n}$ via `exp2`---the volume of the $\mathtt{[-1,1]^n}$ hypercube---and multiply to convert the fraction into a volume.

With $\mathtt{n = 2}$ and large $\mathtt{S}$, `ball_vol` returns approximately `3.14` ($\approx \pi$). The operations that will specify a language model are not specific to language models---they are general-purpose tensor operations. A forward pass through a transformer and a Monte Carlo integration are the same kind of thing: a straight-line composition of maps and folds.
