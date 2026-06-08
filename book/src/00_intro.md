# Structure and Execution of Language Models

This is an interactive textbook on the mathematics of language models. We start with AI agents and then work outside-in, peeling away layers of abstraction until we have accounted for every single arithmetic operation. If you can't wait to see how everything fits together, check out the diagram at the end of this page. We separate out the core, the **language model** itself, from the outer shell, everything else. These two levels are of a different nature---the language model, which is the primary entity that AI labs create, is a deterministic tensor computation, while the outer shell is a more typical program. In other words:

> *the language model is a mathematical function*, and one that you can write down.

We make the nonstandard choice to express the language model in [***Catform***](https://github.com/tensorigami/catform), our own [domain-specific language](https://en.wikipedia.org/wiki/Domain-specific_language) for [**tensor**](https://en.wikipedia.org/wiki/Tensor) programs---i.e. those that transform multi-dimensional arrays of data. Right now, models and their components are either described informally using math and diagram notation, or written to be executed in some tensor framework like [PyTorch](https://pytorch.org/) or [JAX](https://en.wikipedia.org/wiki/Google_JAX). The vision of catform is to close this gap by providing a mathematically inspired language for both *writing* and *executing* the model as a mathematical function. It was created with the following design principles in mind:
- *minimalism*: it supports just seven primitive tensor transformations.
- *expressivity*: it can express any modern open-weight language model.
- *soundness*: every term is type-annotated, and every operation type-checked.
- *explicitness*: every decision is stated in the program, not presumed by the framework.

The design of catform owes a debt to the [einops](https://github.com/arogozhnikov/einops) library, the [HLO](https://openxla.org/xla/operation_semantics) intermediate representation, and the [Haskell](https://www.haskell.org/) programming language. Its name, short for "Categorical Form", is an homage to [category theory](https://en.wikipedia.org/wiki/Category_theory).

Language models written in catform can be run on different kinds of devices by [***Pianola***](https://github.com/tensorigami/pianola), which interprets them in a chosen framework. Pianola also exposes an interactive [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop) environment, which lets you run and experiment with all of the code snippets in the book.

## Organization

The core of the book consists of the following chapters.

1. [Chapter 1](01_inference.md) introduces the concept of an AI agent and then covers the core logic of **inference**, which transforms structured context into a form that can be fed to a language model to produce meaningful outputs. This chapter mirrors the code in Pianola's inference implementation in `src/pianola/inference.py`.
2. [Chapter 2](02_tensors.md) is a mathematical interlude on tensors and their computations, expressed in catform.
3. [Chapter 3](03_models.md) covers the complete mathematical description of [Qwen3](https://qwen.ai/blog?id=qwen3), a popular and representative **open-weight** language model, by following its specification in the catform file `models/qwen3/model.cat`.

Future chapters will cover:

4. *Architectures* - on transformer variants such as mixture-of-experts (MoE) and linear attention
5. *Performance* - on writing and reasoning about efficient implementations of language models
6. *Gradients* - on programmatically differentiating tensor programs
7. *Training* - on algorithms for training models

## Philosophy

Our title is a direct homage to the seminal programming textbook [*Structure and Interpretation of Computer Programs*](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/index.html). The choice to abstract away tensor programming into its own language was inspired by one of the book's core ideas.

> *The general technique of isolating the parts of a program that deal with how data objects are represented from the parts of a program that deal with how data objects are used is a powerful design methodology called data abstraction.*
>
> — SICP, Section 2.1

A catform file specifies the **structure** of a language model, while Pianola performs its **execution**. A catform program is a [declarative](https://en.wikipedia.org/wiki/Declarative_programming) artifact that describes *what* the model computes without prescribing *how*. Such [separation of concerns](https://en.wikipedia.org/wiki/Separation_of_concerns) allows the language model, as a mathematical function, to exist as a single stand-alone file, rather than be entangled in a codebase. Pianola---named for the [player piano](https://en.wikipedia.org/wiki/Player_piano)---runs the catform program using a chosen framework. This makes catform [portable](https://en.wikipedia.org/wiki/Software_portability) across frameworks without changing a single line of code.

## Audience

This book is written for two kinds of readers---those with a background in programming, and those with one in math. We want the programmer to feel closer to the math and the mathematician closer to the programming, so that both can come away understanding what is actually being computed when language models are run.

The primary programming prerequisite for this book is a working familiarity---functions, types, and control flow---with Python, the standard language for AI engineering. The primary mathematical background is a basic understanding of linear algebra---specifically vectors, linear maps, matrices, and the notion of dimension.

Throughout this text, we will state many concepts using both mathematical expressions and working code. This is not mere redundancy: the two notations say the same thing to different readers---including the one we call a computer---and seeing them side by side reveals that the distance between the blackboard and the terminal is shorter than it appears.

## Notation

In light of this, we briefly set some notational choices. We use the programming convention of calling collections *types* and their constituents *terms*. Those with a traditional mathematics background can safely substitute these words, in their mind, with *sets* and *elements*, respectively, since the theoretical distinction between these foundations is not relevant in our context. We declare that a term $\mathtt{x}$ is of type $\mathtt{X}$ by writing:

$$
\mathtt{x} : \mathtt{X}
$$

In programs---both Python and catform alike---a type annotation looks like:

```py
x: X
```

We can also give type annotations to functions, e.g.

$$
\mathtt{f} : \mathtt{X} \to \mathtt{Y}
$$

means that $\mathtt{f}$ takes input in $\mathtt{X}$ and returns outputs $\mathtt{Y}$. In Python this is written as:

```py
def f(x: X) -> Y: ...
```

Mathematically, many computations can be seen as elaborate function compositions, so we introduce notation for expressing them. Given two functions $\mathtt{f} : \mathtt{X} \to \mathtt{Y}$ and $\mathtt{g} : \mathtt{Y} \to \mathtt{Z}$, their *composition* is the function that applies $\mathtt{f}$ first and then $\mathtt{g}$. In mathematical notation, we can represent this as an *arrow diagram*:

$$
\begin{tikzcd}
\mathtt{X} \arrow[r, "\mathtt{f}"] & \mathtt{Y} \arrow[r, "\mathtt{g}"] & \mathtt{Z}
\end{tikzcd}
$$

We can also represent this in-line as a binary operation. The classical notation writes composition as $\mathtt{g} \circ \mathtt{f}$---read aloud as "$\mathtt{g}$ of $\mathtt{f}$", applied right to left. We often prefer to match the diagrammatic order and use *forward* composition, written $\mathtt{f} \:;\: \mathtt{g}$ and read aloud as "$\mathtt{f}$ then $\mathtt{g}$".

To express that two compositions are equal, we use [**commutative diagrams**](https://en.wikipedia.org/wiki/Commutative_diagram), e.g. to show that $\mathtt{f = u \:;\: g \:;\: v}$ we can write

$$
\begin{tikzcd}
\mathtt{A}
\arrow[rr, "\mathtt{f}"]
\arrow[d, "\mathtt{u}"']
&& \mathtt{D}
\\
\mathtt{B}
\arrow[rr, "\mathtt{g}"']
&& \mathtt{C}
\arrow[u, "\mathtt{v}"']
\end{tikzcd}
$$

Most compositions, however, are more complex than sequential application as they involve routing distinct components of a function's output to different functions' inputs. In this case we use [**string diagrams**](https://en.wikipedia.org/wiki/String_diagram), which draw functions as boxes and data as wires. Splitting wires represents copying data, while merging them represents combining data, usually via addition.

$$
\begin{tikzpicture}[every node/.style={font=\scriptsize}]

\draw[very thick, rounded corners] (-2, -0.4) rectangle (-1, 0.4);
\node at (-1.5, 0) {$\mathtt{f}$};

\draw[very thick, rounded corners] (1, -0.4) rectangle (2, 0.4);
\node at (1.5, 0) {$\mathtt{g}$};

\draw[very thick, rounded corners] (1, -1.2) rectangle (2, -0.6);
\node at (1.5, -0.9) {$\mathtt{h}$};

\draw[very thick] (-3, 0) -- (-2, 0);

\draw[very thick] (-1, 0.15) -- (1, 0.15);
\draw[very thick] (-1, -0.15) -- (1, -0.15);

\draw[very thick, rounded corners] (0, -0.15) -- (0, -0.9) -- (1, -0.9);

\draw[very thick] (2, 0) -- (3, 0);
\draw[very thick, rounded corners] (2, -0.9) -- (3, -0.9) -- (3, 0);
\draw[very thick] (3, 0) -- (4, 0);

\end{tikzpicture}
$$

The fact that functions are boxes allows string diagrams to be nested, in what are sometimes called **wiring diagrams**. For instance, to express that $\mathtt{f = g \:;\: h}$ and additionally that $\mathtt{g = a \:;\: b}$, we can depict this nested composition:

$$
\begin{tikzpicture}[every node/.style={font=\scriptsize}]

\node at (-1, 1.5) {\large $\mathtt{f}$};

\draw[very thick, rounded corners] (-5, -1.75) rectangle (3, 1.75);
\draw[very thick, rounded corners, densely dotted] (-4, -1) rectangle (0, 1);

\node at (-2, 0.75) {\normalsize $\mathtt{g}$};

\draw[very thick, rounded corners] (-3.5, -0.4) rectangle (-2.5, 0.4);
\draw[very thick, rounded corners] (-1.5, -0.4) rectangle (-.5, 0.4);
\node at (-3, 0) {$\mathtt{a}$};
\node at (-1, 0) {$\mathtt{b}$};

\draw[very thick, rounded corners] (.5, -0.4) rectangle (1.5, 0.4);
\node at (1, 0) {$\mathtt{h}$};

\draw[very thick] (-6.0, 0) -- (-3.5, 0);
\draw[very thick] (-2.5, 0) -- (-1.5, 0);
\draw[very thick] (-0.5, 0) -- (  .5, 0);
\draw[very thick] ( 1.5, 0) -- (   4, 0);

\end{tikzpicture}
$$

## The Big Picture

Wiring diagrams are powerfully expressive---with them, we can more or less depict the entire structure that the book will reveal in detail. The following Matryoshka of wiring diagrams describes the whole inference process at different levels of zoom.

$$
\begin{tikzpicture}[every node/.style={font=\scriptsize}]

\node at (0,1.75) {\large $\mathtt{infer}$};

\draw[very thick, rounded corners,               ] (-6.0,-2) rectangle (6.0,2);
\draw[very thick, rounded corners, densely dotted] (-4.5,-1.25) rectangle (4.5,1.25);
\draw[very thick, rounded corners, densely dotted] (-3.1,-.825) rectangle (3.1,.825);

\draw[very thick, rounded corners] (-5.6,-0.33) rectangle (-4.8,0.33);
\draw[very thick, rounded corners] ( 5.6,-0.33) rectangle ( 4.8,0.33);
\node at (-5.2,0) {$\mathtt{templ}$};
\node at ( 5.2,0) {$\mathtt{parse}$};

\draw[very thick] (-7,0) -- (-5.6,0);
\draw[very thick] ( 7,0) -- ( 5.6,0);

\draw[very thick] (-7, -1) -- (-4, -1) to[out=0, in=180, looseness=1] (-2.8,-.15);
\draw[very thick] ( 7, -1) -- ( 4, -1) to[out=180, in=0, looseness=1] ( 2.4,-.15);

\draw[very thick, rounded corners] (-4.2,-0.33) rectangle (-3.4,0.33);
\draw[very thick, rounded corners] ( 4.2,-0.33) rectangle ( 3.4,0.33);
\node at (-3.8,0) {$\mathtt{enc}$};
\node at ( 3.8,0) {$\mathtt{dec}$};

\draw[very thick] (-4.8,0) -- (-4.2,0);
\draw[very thick] ( 4.8,0) -- ( 4.2,0);

\draw[very thick, rounded corners] (-2.8,-0.33) rectangle (-2.0,0.33);
\draw[very thick, rounded corners] (-1.6,-0.33) rectangle (-0.8,0.33);
\draw[very thick, rounded corners] (-0.2,-0.33) rectangle ( 0.6,0.33);
\draw[very thick, rounded corners] ( 2.4,-0.33) rectangle ( 1.6,0.33);

\draw[very thick] (-2.8,.15) -- (-3.4,.15) ;
\draw[very thick] ( 2.4,.15) -- ( 3.4,.15) ;

\draw[very thick] (-2.0, 0.15) -- (-1.6, 0.15) ;
\draw[very thick] (-2.0,-0.15) -- (-1.6,-0.15) ;

\fill (-0.6, 0.05) -- (-0.6, 0.25) -- (-0.4, 0.15) -- cycle;
\fill ( 2.6, 0.05) -- ( 2.6, 0.25) -- ( 2.8, 0.15) -- cycle;

\draw[very thick] (-.8, 0.15) -- (-.2, 0.15) ;
\draw[very thick] (-.8,-0.15) -- (-.2,-0.15) ;

\draw[very thick] (0.6, 0.15) -- (1, 0.15) ;
\draw[very thick] (0.6,-0.15) -- (1,-0.15) ;

\draw[very thick] (1.2, 0.15) -- (1.6, 0.15) ;
\draw[very thick] (1.2,-0.15) -- (1.6,-0.15) ;

\node at (1.1,0) {$\cdots$};

\node at (-2.4,0) {$\mathtt{cache}$};
\node at (-1.2,0) {$\mathtt{model}$};
\node at ( 0.2,0) {$\mathtt{model}$};
\node at ( 2,0) {$\mathtt{model}$};


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

\draw[very thick] (-7, -10.4) -- (7,-10.4) ;
\draw[very thick] (-7, -12.4) -- (-4, -12.4) to[out=0, in=180, looseness=1.5] (-2.5,-11.65);
\draw[very thick] ( 7, -12.4) -- (0.5, -12.4) to[out=180, in=0, looseness=1.5] (  -1,-11.65);

\draw[very thick, rounded corners] (-5,-12) rectangle (-3.5,-11);
\draw[very thick, rounded corners] (-2.5,-12) rectangle (-1,-11);
\node at (-4.25,-11.5) {$\mathtt{rmsnorm}$};
\node at (-1.75,-11.5) {$\mathtt{attention}$};

\draw[very thick] (-5.5,-10.4) to[out=-90, in=180, looseness=1] (-5,-11.5) ;
\draw[very thick] (-3.5,-11.35) -- (-2.5,-11.35) ;
\draw[very thick] (-1,-11.35) to[out=0, in=-90, looseness=1] (-0.5,-10.4) ;

\draw[very thick, rounded corners] (5,-12) rectangle (3.5,-11);
\draw[very thick, rounded corners] (2.5,-12) rectangle (1,-11);
\node at (4.25,-11.5) {$\mathtt{ffn}$};
\node at (1.75,-11.5) {$\mathtt{rmsnorm}$};

\draw[very thick] (5.5,-10.4) to[out=-90, in=0, looseness=1] (5,-11.5) ;
\draw[very thick] (3.5,-11.5) -- (2.5,-11.5) ;
\draw[very thick] (1,-11.5) to[out=180, in=-90, looseness=1] (0.5,-10.4) ;



\node at (0,-14.75) {\large $\mathtt{attention}$};
\draw[very thick, rounded corners] (-6,-18.5) rectangle (6,-14.5);

\draw[very thick] (-7,-16.5) -- (-5.1,-16.5) ;

\draw[very thick] (-7,-17.75) -- (0.6,-17.75) ;
\draw[very thick] ( 7,-17.75) -- (1.8,-17.75) ;

\draw[very thick, rounded corners] (-5.1,-15.8) -- (-5.6,-15.8) -- (-5.6,-17.2) -- (-5.1,-17.2) ;

\draw[very thick, rounded corners] (-5.1,-16.05) rectangle (-4.1,-15.55);
\draw[very thick, rounded corners] (-5.1,-16.75) rectangle (-4.1,-16.25);
\draw[very thick, rounded corners] (-5.1,-17.45) rectangle (-4.1,-16.95);
\node at (-4.6,-15.8) {$\mathtt{Q}$};
\node at (-4.6,-16.5) {$\mathtt{K}$};
\node at (-4.6,-17.2) {$\mathtt{V}$};

\draw[very thick, rounded corners] (-3.5,-16.05) rectangle (-1.8,-15.55);
\draw[very thick, rounded corners] (-3.5,-16.75) rectangle (-1.8,-16.25);
\node at (-2.65,-15.8) {$\mathtt{rmsnorm}$};
\node at (-2.65,-16.5) {$\mathtt{rmsnorm}$};

\draw[very thick, rounded corners] (-1.2,-16.05) rectangle (0,-15.55);
\draw[very thick, rounded corners] (-1.2,-16.75) rectangle (0,-16.25);
\node at (-0.6,-15.8) {$\mathtt{rope}$};
\node at (-0.6,-16.5) {$\mathtt{rope}$};

\draw[very thick] (-4.1,-15.8) -- (-3.5,-15.8) ;
\draw[very thick] (-1.8,-15.8) -- (-1.2,-15.8) ;
\draw[very thick] (0,-15.8) -- (2.4,-15.8) ;

\draw[very thick] (-4.1,-16.5) -- (-3.5,-16.5) ;
\draw[very thick] (-1.8,-16.5) -- (-1.2,-16.5) ;
\draw[very thick] (0,-16.5) -- (0.6,-16.5) ;
\draw[very thick] (1.8,-16.5) -- (2.4,-16.5) ;

\draw[very thick] (-4.1,-17.2) -- (0.6,-17.2) ;
\draw[very thick] (1.8,-17.2) -- (2.4,-17.2) ;

\draw[very thick, rounded corners] (0.6,-18.25) rectangle (1.8,-16.25);
\node at (1.2,-17.25) {$\mathtt{KV\:cache}$};

\draw[very thick, rounded corners] (2.4,-17.45) rectangle (3.9,-15.55);
\node at (3.15,-16.5) {$\mathtt{attend}$};

\draw[very thick, rounded corners] (4.5,-16.75) rectangle (5.5,-16.25);
\node at (5,-16.5) {$\mathtt{O}$};

\draw[very thick] (3.9,-16.5) -- (4.5,-16.5) ;
\draw[very thick] (5.5,-16.5) -- (7,-16.5) ;

\end{tikzpicture}
$$

If your curiosity is piqued, you can read the next chapter to deconstruct the first wiring diagrams, and then continue onward for the remaining ones.

## Interactivity

This book is more illuminating---and more fun---when you can see the math actually running on your computer. To follow along, set up the codebase locally:

**1. Clone the repository.**

```bash
git clone https://github.com/tensorigami/pianola.git
cd pianola
```

**2. Install [`uv`](https://docs.astral.sh/uv/)** which manages Python (no separate Python install needed):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**3. Install dependencies.**

```bash
uv sync
```

**4. Fetch the model artifacts** for Qwen3 0.6B (no login required):

```bash
uv run main.py populate qwen3/0_6b
```
