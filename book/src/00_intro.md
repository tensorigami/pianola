# Structure and Execution of Language Models

This textbook gives a complete mathematical description of language models, starting with AI agents and then moving outside-in to the individual operations. If you can't wait to see how everything fits together, check out the diagram at the end of this page.

Beyond the text itself, this project has two software components:

1. [***Catform***](https://github.com/tensorigami/catform), a [domain-specific language](https://en.wikipedia.org/wiki/Domain-specific_language) for expressing tensor computations
2. [***Pianola***](https://github.com/tensorigami/pianola), an engine that executes language models written in catform

This software makes the book interactive---all logic introduced in the text is runnable via an interactive [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop) environment.

## Organization

Contemporary AI programs broadly have two layers---an outer scaffolding layer and an inner layer, which is the **language model** itself. Contrary to popular misconception, the language model is a purely deterministic mathematical function. The mathematics of such models is expressed as **tensor** computations. The book is thus organized as follows.

1. [Chapter 1](01_inference.md) covers the core logic of **inference**---converting structured context into tokens (atomic units of language) and then **autoregressively**---one token at a time---sampling from the language model outputs to generate a response. This chapter references Pianola's implementation in `src/pianola/inference.py` line by line.
2. [Chapter 2](02_tensors.md) covers tensors and their computations in catform---short for categorical form, inspired by [category theory](https://en.wikipedia.org/wiki/Category_theory)---a notation designed from first principles to capture the algebra of tensor operations. Catform is minimal in that it uses just six primitive tensor operations to express the full computation of any modern transformer-based language model, and does so no more verbosely than the typical Python implementation.
3. [Chapter 3](03_models.md) covers the complete mathematical description of [Qwen3](https://qwen.ai/blog?id=qwen3), a popular and representative **open-weight** language model, by following its complete specification in the catform file `models/qwen3/model.cat`.

## Philosophy

Our title is a direct homage to the seminal programming textbook [*Structure and Interpretation of Computer Programs*](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html). The choice to abstract away tensor programming into its own language was inspired by one of the book's core ideas.

> *The general technique of isolating the parts of a program that deal with how data objects are represented from the parts of a program that deal with how data objects are used is a powerful design methodology called data abstraction.*
>
> — SICP, Section 2.1

A catform file specifies the **structure** of a computation, while Pianola performs its **execution**. More precisely, a `.cat` file is a stand-alone [declarative](https://en.wikipedia.org/wiki/Declarative_programming) artifact that describes *what* to compute, without prescribing *how*. Pianola---named for the [player piano](https://en.wikipedia.org/wiki/Player_piano)---runs this computation using a chosen framework; currently [PyTorch](https://pytorch.org/) or [JAX](https://en.wikipedia.org/wiki/Google_JAX), and extensible in principle to any hardware backend. This makes catform [portable](https://en.wikipedia.org/wiki/Software_portability) across frameworks without changing a single line of code.


## Audience

This book is written for two kinds of reader. The first kind is the programmer---especially one working in or around AI---who wants a mathematically complete understanding of language models. The second kind has a background in mathematics and desires an entry point into AI that speaks in their language and meets their standard of precision.

The primary programming prerequisite for this book is a basic understanding of Python---specifically functions, types---including enums, dataclasses, and containers like tuples and dicts---and basic control flow. Python is universally used in AI engineering, and we follow suit. The primary mathematical background is a basic understanding of linear algebra---not a full course, just familiarity with vectors, linear maps, matrices, and the notion of dimension. Later on, in the chapter on gradients, the reader will want familiarity with differential calculus---and single-variable calculus suffices.

We will use the programming convention of calling collections *types* and their constituents *terms*. Those with a traditional mathematics background can safely substitute these words, in their mind, with *sets* and *elements*, respectively, since the theoretical distinction between these foundations is not relevant in our context. Throughout this text, most concepts are stated twice: once as a mathematical expression and once as a snippet of working code. This is not mere redundancy, but the point: the two notations say the same thing to different readers---including the one we call a computer---and seeing them side by side reveals that the distance between the blackboard and the terminal is shorter than it appears.

## Notation

In light of this, we briefly set some notational conventions. We declare that a term $\mathtt{x}$ is of type $\mathtt{X}$ by writing:

$$
\mathtt{x} : \mathtt{X}
$$

In programs---both Python and catform alike---a type annotation looks like:

```catform
x: X
```

Given types $\mathtt{X}$ and $\mathtt{Y}$, a function $\mathtt{f}$ takes an input $\mathtt{x}:\mathtt{X}$ and returns an output $\mathtt{f(x)}:\mathtt{Y}$. In mathematical notation:

$$
\mathtt{f} : \mathtt{X} \to \mathtt{Y}
$$

In Python:

```python
def f(x: X) -> Y: ...
```

In catform:

```catform
f(x: X) -> (y: Y) {...}
```

Given two functions $\mathtt{f} : \mathtt{X} \to \mathtt{Y}$ and $\mathtt{g} : \mathtt{Y} \to \mathtt{Z}$, their *composition* is the function that applies $\mathtt{f}$ first and then $\mathtt{g}$. In mathematical notation, we represent this either as an *arrow diagram*:

$$
\begin{tikzcd}
\mathtt{X} \arrow[r, "\mathtt{f}"] & \mathtt{Y} \arrow[r, "\mathtt{g}"] & \mathtt{Z}
\end{tikzcd}
$$

or as a binary operator. The classical notation writes composition as $\mathtt{g} \circ \mathtt{f}$---read aloud as "$\mathtt{g}$ of $\mathtt{f}$", applied right to left. We often prefer to match the diagrammatic order and use *forward* composition, written $\mathtt{f} \:;\: \mathtt{g}$ and read aloud as "$\mathtt{f}$ then $\mathtt{g}$".

In programs, our convention for readability is to express composition line by line:

```python
y = f(x)
z = g(y)
```

## The Big Picture

$$
\begin{tikzpicture}[every node/.style={font=\scriptsize}]

\node at (0,1.5) {\large $\mathtt{infer}$};

\draw[thick, rounded corners,               ] (-6.0,-2) rectangle (6.0,2);
\draw[thick, rounded corners, densely dotted] (-4.8,-1) rectangle (4.8,1);
\draw[thick, rounded corners, densely dotted] (-3.0,-0.75) rectangle (3.0,0.75);

\draw[thick, rounded corners] (-5.4,-0.5) rectangle (-4.2,0.5);
\draw[thick, rounded corners] (5.4,-0.5) rectangle (4.2,0.5);
\node at (4.8,0) {$\mathtt{parse}$};
\node at (-4.8,0) {$\mathtt{template}$};

\draw[thick] (-7,0) -- (-5.4,0) ;
\draw[thick] (7,0) -- (5.4,0) ;

\draw[thick, rounded corners] (-3.6,-0.5) rectangle (-2.4,0.5);
\draw[thick, rounded corners] (3.6,-0.5) rectangle (2.4,0.5);
\node at (3,0) {$\mathtt{decode}$};
\node at (-3,0) {$\mathtt{encode}$};

\draw[thick] (-4.2,0) -- (-3.6,0) ;
\draw[thick] (4.2,0) -- (3.6,0) ;

\draw[thick, rounded corners] (-2.00,-0.33) rectangle (-1.2,0.33);
\draw[thick, rounded corners] (-0.8,-0.33) rectangle (-0.00,0.33);
\draw[thick, rounded corners] (1.2,-0.33) rectangle (2.00,0.33);

\draw[thick] (-2.4,0) -- (-2,0) ;
\draw[thick] (2.4,0) -- (2,0) ;

\draw[thick] (-1.2,0.1) -- (-.8,0.1) ;
\draw[thick] (-1.2,-0.1) -- (-.8,-0.1) ;

\draw[thick] (0,0.1) -- (.4,0.1) ;
\draw[thick] (0,-0.1) -- (.4,-0.1) ;

\draw[thick] (0.8,0.1) -- (1.2,0.1) ;
\draw[thick] (0.8,-0.1) -- (1.2,-0.1) ;

\node at (.6,0) {$\cdots$};

\node at (-1.6,0) {$\mathtt{cache}$};
\node at (-0.4,0) {$\mathtt{fwd}$};
\node at (1.6,0) {$\mathtt{fwd}$};

\node at (0,-4) {\large $\mathtt{forward}$};
\node at (0,-6.25) {\small $\mathtt{transformer}$};

\draw[thick, rounded corners] (-6.0,-7.5) rectangle (6.0,-3.5);
\draw[thick, rounded corners, densely dotted] (-4.8,-6.5) rectangle (4.8,-4.5);

\draw[thick, rounded corners] (-5.4,-6) rectangle (-4.2,-5);
\draw[thick, rounded corners] (5.4,-6) rectangle (4.2,-5);
\node at (4.8,-5.5) {$\mathtt{embed}$};
\node at (-4.8,-5.5) {$\mathtt{unembed}$};

\draw[thick] (-7,-5.5) -- (-5.4,-5.5) ;
\draw[thick] (7,-5.5) -- (5.4,-5.5) ;

\draw[thick, rounded corners] (2.4,-6) rectangle (3.6,-5);
\draw[thick, rounded corners] (0.6,-6) rectangle (1.8,-5);
\draw[thick, rounded corners] (-0.6,-6) rectangle (-1.8,-5);
\draw[thick, rounded corners] (-2.4,-6) rectangle (-3.6,-5);

\node at (3.0,-5.5) {$\mathtt{rmsnorm}$};
\node at (1.2,-5.5) {$\mathtt{layer}$};
\node at (-1.2,-5.5) {$\mathtt{layer}$};
\node at (-3.0,-5.5) {$\mathtt{layer}$};

\draw[thick] (-3.6,-5.5) -- (-4.2,-5.5) ;
\draw[thick] (3.6,-5.5) -- (4.2,-5.5) ;

\draw[thick] (-2.4,-5.5) -- (-1.8,-5.5) ;
\draw[thick] (2.4,-5.5) -- (1.8,-5.5) ;

\draw[thick] (-0.6,-5.5) -- (-0.2,-5.5) ;
\draw[thick] (0.6,-5.5) -- (0.2,-5.5) ;

\node at (0,-5.5) {$\cdots$};

\node at (0,-9.5) {\large $\mathtt{layer}$};
\draw[thick, rounded corners] (-6.0,-13) rectangle (6.0,-9);

\draw[thick] (-7,-10.4) -- (7,-10.4) ;

\draw[thick, rounded corners] (-5.1,-12.1) rectangle (-3.4,-11.1);
\draw[thick, rounded corners] (-2.7,-12.1) rectangle (-0.8,-11.1);
\node at (-4.25,-11.6) {$\mathtt{rmsnorm}$};
\node at (-1.75,-11.6) {$\mathtt{attention}$};

\draw[thick, rounded corners] (-5.6,-10.4) -- (-5.6,-11.6) -- (-5.1,-11.6) ;
\draw[thick] (-3.4,-11.6) -- (-2.7,-11.6) ;
\draw[thick, rounded corners] (-0.8,-11.6) -- (-0.2,-11.6) -- (-0.2,-10.4) ;

\draw[thick, rounded corners] (0.8,-12.1) rectangle (2.5,-11.1);
\draw[thick, rounded corners] (3.2,-12.1) rectangle (5.1,-11.1);
\node at (1.65,-11.6) {$\mathtt{rmsnorm}$};
\node at (4.15,-11.6) {$\mathtt{ffn}$};

\draw[thick, rounded corners] (0.2,-10.4) -- (0.2,-11.6) -- (0.8,-11.6) ;
\draw[thick] (2.5,-11.6) -- (3.2,-11.6) ;
\draw[thick, rounded corners] (5.1,-11.6) -- (5.6,-11.6) -- (5.6,-10.4) ;

\node at (0,-15) {\large $\mathtt{attention}$};
\draw[thick, rounded corners] (-6,-18.5) rectangle (6,-14.5);
\draw[thick, rounded corners, densely dotted] (-4.5,-17.75) rectangle (4.5,-15.25);

\draw[thick] (-7,-16.5) -- (-5,-16.5) ;
\draw[thick, rounded corners] (-5,-15.8) -- (-5.5,-15.8) -- (-5.5,-17.2) -- (-5,-17.2) ;

\draw[thick, rounded corners] (-5,-16.05) rectangle (-4,-15.55);
\draw[thick, rounded corners] (-5,-16.75) rectangle (-4,-16.25);
\draw[thick, rounded corners] (-5,-17.45) rectangle (-4,-16.95);
\node at (-4.5,-15.8) {$\mathtt{Q}$};
\node at (-4.5,-16.5) {$\mathtt{K}$};
\node at (-4.5,-17.2) {$\mathtt{V}$};

\draw[thick, rounded corners] (-3,-16.05) rectangle (-1.3,-15.55);
\draw[thick, rounded corners] (-3,-16.75) rectangle (-1.3,-16.25);
\node at (-2.15,-15.8) {$\mathtt{rmsnorm}$};
\node at (-2.15,-16.5) {$\mathtt{rmsnorm}$};

\draw[thick, rounded corners] (-0.75,-16.05) rectangle (0.45,-15.55);
\draw[thick, rounded corners] (-0.75,-16.75) rectangle (0.45,-16.25);
\node at (-0.15,-15.8) {$\mathtt{rope}$};
\node at (-0.15,-16.5) {$\mathtt{rope}$};

\draw[thick] (-4,-15.8) -- (-3,-15.8) ;
\draw[thick] (-1.3,-15.8) -- (-0.75,-15.8) ;
\draw[thick] (0.45,-15.8) -- (1.5,-15.8) ;

\draw[thick] (-4,-16.5) -- (-3,-16.5) ;
\draw[thick] (-1.3,-16.5) -- (-0.75,-16.5) ;
\draw[thick] (0.45,-16.5) -- (1.5,-16.5) ;

\draw[thick] (-4,-17.2) -- (1.5,-17.2) ;

\draw[thick, rounded corners] (1.5,-17.45) rectangle (3,-15.55);
\node at (2.25,-16.5) {$\mathtt{attend}$};

\draw[thick, rounded corners] (4,-16.75) rectangle (5,-16.25);
\node at (4.5,-16.5) {$\mathtt{O}$};

\draw[thick] (3,-16.5) -- (4,-16.5) ;
\draw[thick] (5,-16.5) -- (7,-16.5) ;

\end{tikzpicture}
$$


## Interactivity

This book is more illuminating---and more fun---when you can see the math actually running on your computer. Setup instructions are in the [README](https://github.com/tensorigami/pianola#setup).
