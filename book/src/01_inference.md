# Inference

What exactly is an AI **agent**? Such systems typically consist of a loop in which a **scaffolding** or **harness** sends requests to a language model **inference server** and then processes their responses, potentially inducing another request, and so on until some stopping condition.

$$
\begin{tikzpicture}[every node/.style={font=\scriptsize}]

\node at (0,1.75) {\large $\mathtt{agent}$};

\draw[very thick, rounded corners] (-6.0,-2) rectangle (6.0,2);

\draw[very thick, rounded corners] (-5.5,-1.0) rectangle (-1.0,1.0);
\draw[very thick, rounded corners] ( 1.0,-1.0) rectangle ( 5.5,1.0);

\node at (-3.25,0) {\normalsize$\mathtt{harness}$};
\node at ( 3.25,0) {\normalsize$\mathtt{inference\:server}$};

\draw[very thick] (-1,0.5) to[out=30, in=150] node[above] {$\mathtt{request}$} (1,0.5);
\draw[very thick] (-0.1,0.9) -- (0.1,0.8) -- (-0.1,0.7);
\draw[very thick] ( 1,-0.5) to[out=-150, in=-30] node[below] {$\mathtt{response}$} (-1,-0.5);
\draw[very thick] ( 0.1,-0.7) -- (-0.1,-0.8) -- ( 0.1,-0.9);

\end{tikzpicture}
$$

The harness can include whatever data and computations are needed to give the application its form: be it a chatbot, structured workflow, or fully autonomous agent. While agent engineering is an interesting subject, this book will just study the inference component.

An inference server is typically accessed via an [API endpoint](https://en.wikipedia.org/wiki/API), and our codebase comes equipped with one. Start it locally with the following terminal command

```sh
uv run main.py serve qwen3/0_6b
```

Don't expect too much---on either speed or output quality---as this is a small open model running unoptimized on your laptop! In another terminal window, you can make a request to this local `qwen3/0_6b` via `curl`, e.g.

```bash
curl http://localhost:8000/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{
   "model": "qwen3/0_6b",
   "messages": [{"role": "user", "content": "answer directly. 1+2="}]
 }'
```
gives the following response:

```json
{
  "id": "chatcmpl-1779927539503",
  "object": "chat.completion",
  "created": 1779927539,
  "model": "qwen3/0_6b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<think>\nOkay, so the question is 1+2=, and I need to answer it directly. Let me think. Well, when do you add numbers? Dirac's box, right? Like when they need to find the total from two values. 1 and 2. Exchanging that, adding them together. So 1+2 is 3, right? That makes sense. No need to complicate it. Maybe they just want the sum, so 3. I don't see any tricks here, this is a simple addition problem.\n</think>\n\n3"
      },
      "finish_reason": "stop"
    }
  ],
  "cached_tokens": 0
}
```

In the request, `model` is just a string identifier, while the `messages` field contains the **context**, i.e. the actual input we wish to feed to the language model. In this case, the context is just a list of **messages**, each containing both a `role` and some textual `content`. In the response, the primary field is `choices`, which lists a number---specified in the request, defaulting to one---of distinct outputs. The `message` field of a choice contains the actual **inference**. 

A production inference server like [vLLM](https://github.com/vllm-project/vllm) and [SGLang](https://github.com/sgl-project/sglang) needs to efficiently orchestrate many such requests. But from the perspective of the consumer, the server is exposing a function that takes in context---in the form of a list of messages---and returns a subsequent message:

$$
\mathtt{infer : Context \to Message}
$$

Or, a bit more precisely---for the time being black boxing what constitutes the **state**---a **stateful** version of this function:

$$
\mathtt{infer : (Context, State) \to (Message, State)}
$$

The core of this book---its first three chapters---will successively deconstruct `infer` until we have a complete description, down to every mathematical operation. As mentioned in the [introduction](00_intro.md), the book will parallel the code in [***Pianola***](https://tensorigami.github.io/pianola/). This chapter will follow `src/pianola/inference.py`---for instance, here is the type signature of `infer`:

```python
def infer(
    template: Template,
    tokenizer: Tokenizer,
    sampler: Sampler,
    model: Catform,
    context: Context,
    state: State,
) -> tuple[Message, State]:
```

The first thing you will notice is all of the arguments beyond the actual data of `context` and `state`---we will get to each in due course. The most important of these is `model`---the actual **language model** itself, which, as discussed in the introduction, is a deterministic tensor program. We will refer to computing this program on inputs as a **forward pass**. In this chapter, we will deconstruct `infer` until we reduce it to the model forward pass. Then, after a mathematical interlude on tensor programming in the [next chapter](02_tensors.md), we will spend the [third chapter](03_models.md) deconstructing the forward pass itself.

## Templating

To proceed further, we'll want to inspect `messages` a bit more deeply. Every contemporary model family supports at least three roles, which reflect that conversational chatbots were the first dominant way of interacting with language models.

- `system` for persistent instructions---the so-called **system prompt**---appearing just once at the beginning of the context
- `user` for external input, historically the human user's conversation turn
- `assistant` for the model's output

We will cover a fourth role---`tool`---in the next subsection. Here's an example context:

```json
[
  {"role": "system",    "content": "You are legendary mathematician Alexander Grothendieck."},
  {"role": "user",      "content": "What is your favorite prime?"},
  {"role": "assistant", "content": "<think>\nIt should look prime\nwithout being prime.\n</think>\n\n57"},
  {"role": "user",      "content": "Why that one?"}
]
```

The language model doesn't consume structured dictionaries directly---a **template** first converts them into raw text. The template is codified as a [Jinja2](https://jinja.palletsprojects.com/) source string. Qwen3's template---the `chat_template` field of `models/qwen3/tokenizer_config.json`---converts the above structured context to the following raw text, using special tokens to delimit each message:

```
<|im_start|>system
You are legendary mathematician Alexander Grothendieck.<|im_end|>
<|im_start|>user
What is your favorite prime?<|im_end|>
<|im_start|>assistant
57<|im_end|>
<|im_start|>user
Why that one?<|im_end|>
<|im_start|>assistant
```

The rendered text ends with an *open* `<|im_start|>assistant` tag---the **generation prompt**---which cues the model to produce the next assistant turn.

These delimiters fit a convention called [**ChatML**](https://github.com/openai/openai-python/blob/release-v0.28.1/chatml.md). Other model families may use different delimiters but the same principle applies. The template is model-specific: it is distributed with the model. Crucially, the model is *trained* on data formatted with its template, so the template is not an independent formatting choice, but rather is baked into the model and cannot be swapped freely. One can, however, also feed text without any delimiters to the model.

Once the model is given raw text, it can then `generate` new raw text! This generated text may carry its own tagged structure. For instance, Qwen3's outputs, such as the one shown earlier, begin with **chain of thought** reasoning, demarcated between the tags `<think>` and `</think>`. Notice that the assistant message's `<think>` reasoning from the structured context was absent from the rendered text above: the template *strips* chain of thought from prior assistant turns, so the model never re-reads its own past reasoning. Once generation is finished, the generated text gets **parsed** back to the `Message` format---for now, just by setting the `role` to `"assistant"` and the `content` to the output text. We have thus defined the function body of `infer`:

```python
def infer(
    template: Template,
    tokenizer: Tokenizer,
    sampler: Sampler,
    model: Catform,
    context: Context,
    state: State,
) -> tuple[Message, State]:
    """infer = template ; generate ; parse."""
    text             = template(context)                                # Context -> str
    inference, state = generate(tokenizer, sampler, model, text, state) # str     -> str
    return template.parse(context, inference), state                    # str     -> Message
```

The commented type signatures record how these functions transform the context across formats, ignoring state and auxiliary arguments. While we focus purely on context processing, we will use these simplified type signatures. With this in mind, we can depict the relationship between the above functions in a commutative diagram:

$$
\begin{tikzcd}
\mathtt{Context}
\arrow[rr, "\mathtt{infer}"]
\arrow[d, "\mathtt{template}"']
&& \mathtt{Message}
\\
\mathtt{Text}
\arrow[rr, "\mathtt{generate}"']
&& \mathtt{Text}
\arrow[u, "\mathtt{parse}"']
\end{tikzcd}
$$

Before deconstructing generation, we investigate tool-use capability.

### Tools

Just as the inference server exposes `infer` to the harness, the harness can expose its own functions---**tools**---to the language model! Suppressing the harness's and inference server's other internals, this lets us enrich our above agent diagram:

$$
\begin{tikzpicture}[every node/.style={font=\scriptsize}]

\node at (0,1.75) {\large $\mathtt{agent}$};

\draw[very thick, rounded corners] (-6.0,-2) rectangle (6.0,2);

\draw[very thick, rounded corners] (-5.5,-1.0) rectangle (-1.0,1.0);
\draw[very thick, rounded corners] ( 1.0,-1.0) rectangle ( 5.5,1.0);

\node at (-3.25,0.75) {\normalsize$\mathtt{harness}$};
\node at ( 3.25,0.75) {\normalsize$\mathtt{inference\:server}$};

\draw[very thick] (-1,0.5) to[out=30, in=150] node[above] {$\mathtt{request}$} (1,0.5);
\draw[very thick] (-0.1,0.9) -- (0.1,0.8) -- (-0.1,0.7);
\draw[very thick] ( 1,-0.5) to[out=-150, in=-30] node[below] {$\mathtt{response}$} (-1,-0.5);
\draw[very thick] ( 0.1,-0.7) -- (-0.1,-0.8) -- ( 0.1,-0.9);

\draw[very thick, rounded corners] (2.5,-0.5) rectangle (4,0.25);
\node[] at (3.25,-0.125) {$\mathtt{infer}$};
\draw[very thick] (2,0) -- (2.5,0);
\draw[very thick] (2,-0.25) -- (2.5,-0.25);
\draw[very thick] (4,0) -- (4.5,0);
\draw[very thick] (4,-0.25) -- (4.5,-0.25);

\draw[very thick, rounded corners] (-4,0.1) rectangle (-2.5,0.45);
\node[] at (-3.25,0.275) {$\mathtt{tool_0}$};
\draw[very thick] (-4.4,0.275) -- (-4,0.275);
\draw[very thick] (-2.5,0.275) -- (-2.1,0.275);

\node[] at (-3.25,-0.15) {$\vdots$};

\draw[very thick, rounded corners] (-4,-0.75) rectangle (-2.5,-0.4);
\node[] at (-3.25,-0.575) {$\mathtt{tool_n}$};
\draw[very thick] (-4.4,-0.575) -- (-4,-0.575);
\draw[very thick] (-2.5,-0.575) -- (-2.1,-0.575);

\end{tikzpicture}
$$

The harness lets the model know of these exposed functions by including tool definitions in the request body, as such:

```json
{
  "model": "qwen3/0_6b",
  "messages": [{"role": "user", "content": "What is 19 times 3? Use the calculator tool."}],
  "tools": [{
    "type": "function",
    "function": {
      "name": "calculator",
      "description": "Perform arithmetic on two numbers.",
      "parameters": {
        "type": "object",
        "properties": {
          "operation": {"type": "string", "enum": ["add", "multiply", "subtract", "divide"]},
          "left":  {"type": "number"},
          "right": {"type": "number"}
        },
        "required": ["operation", "left", "right"]
      }
    }
  }]
}
```
The `function` payload inside each `tools` entry is a [JSON](https://en.wikipedia.org/wiki/JSON) (the ubiquitous structured-data format) serialization of the `Tool` type:

```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
```

The inclusion of tools gives us the full `Context` type.

```python
@dataclass(frozen=True)
class Context:
    messages: list[Message]
    tools: tuple[Tool, ...] = ()
```

Some templates, including the one used by Qwen3, have more substantive behavior when it comes to producing raw text from `tools`. When the `tools` field is non-empty, the template *appends* their explanation, as in the following text, to the *system prompt*:

```json
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"function": {"description": "Perform arithmetic on two numbers.", "name": "calculator", "parameters": {"properties": {"left": {"type": "number"}, "operation": {"enum": ["add", "multiply", "subtract", "divide"], "type": "string"}, "right": {"type": "number"}}, "required": ["operation", "left", "right"], "type": "object"}}, "type": "function"}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```
This system prompt amendment consists of:
1. a standard prompt that informs the model of the existence of tools
2. the `tools` field copied verbatim as JSON inside the `<tools></tools>` delimiters
3. a standard instruction on how to output text that calls these tools

As is the case with templates in general, the model is trained with text in this format, and thus it is ill-advised to feed the model a hand-crafted raw text context for tools.

During inference, the model can then decide to make a **tool call** by generating text in the template format:

```
<tool_call>
{"name": "calculator", "arguments": {"left": 19, "operation": "multiply", "right": 3}}
</tool_call>
```

Up to this point, `parse` has simply wrapped the model output inside an `assistant` message. Tool calls, however, are parsed more surgically. The parser scans the generated text for `<tool_call></tool_call>` blocks and lifts their contents into a structured `tool_calls` field. One can see precisely such an output in the response to the above request:

```json
{
  "id": "chatcmpl-1780079379210",
  "object": "chat.completion",
  "created": 1780079379,
  "model": "qwen3/0_6b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "<think>\nOkay, the user is asking, \"What is 19 times 3?\" and they want me to use the calculator tool. Let me check the available functions. The tool provided is called calculator and has parameters left, operation, and right. The operation can be add, multiply, subtract, or divide. Since the user is asking for multiplication, the operation should be \"multiply\". The left and right numbers are 19 and 3 respectively. I need to make sure all required parameters are included. Let me structure the tool call with those values.\n</think>",
        "tool_calls": [
          {
            "id": "call_0",
            "type": "function",
            "function": {
              "name": "calculator",
              "arguments": "{\"left\": 19, \"operation\": \"multiply\", \"right\": 3}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "cached_tokens": 0
}
```
Each entry in the `tool_calls` field is a serialized `ToolCall` type:

```python
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
```

With this appreciation of how tool calling works, we can now understand the full `Message` type:

```python
@dataclass(frozen=True)
class Message:
    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()    # assistant only: tool invocations from the model
    tool_call_id: str = ""                   # tool only: which call this answers
```

We emphasize that all the model did was produce text---it did not actually execute any code to compute the function. Rather, the harness receives the `tool_call` JSON, and treats it as a kind of request that the model made to the harness. It is entirely up to the harness as to what to do with this! A straightforward way for the harness to fulfill this request would be to execute the function locally, and then append its result to the context as a `tool` role message, e.g.

```json
{"role": "tool", "tool_call_id": "call_0", "content": "57"}
```

Observe that the `tool_call_id` matches the `id`---assigned by `parse` when lifting `<tool_call>` blocks---from the `tool_calls` entry above, telling the model which call this result answers. When the harness sends this updated context back, the template renders it into the following raw text---which is what the model reads when producing its final answer:

```
<|im_start|>system
[...system prompt with tool definitions, as shown earlier...]<|im_end|>
<|im_start|>user
What is 19 times 3? Use the calculator tool.<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "calculator", "arguments": {"left": 19, "operation": "multiply", "right": 3}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
57
</tool_response><|im_end|>
<|im_start|>assistant
```

Note that Qwen3's template renders `tool`-role messages as `user` turns, using the `<tool_response>` tags to mark the content as tool output. Other model families may use a dedicated `<|im_start|>tool` turn instead. Upon reading this rendered context, the model can then resume inference.

Tool calls allow language models to achieve tasks that they can *specify* but may struggle to do directly, as is the case with numerical computations. Perhaps more importantly, they are essential to agentic capability, since without them, the language model couldn't participate in a nontrivial feedback loop with the harness depicted earlier. With more general tools like a computer terminal or an internet browser, a language model can in principle do anything on a computer that a human can.

## Tokenization

We have reduced the problem of inference to that of consuming text and using it to generate more text. But what exactly *is* text? It is natural to think of raw text as a list of items of some given unit---think characters or words---which we call a **token**. To make this conversion, we will need a **tokenizer**, which consists of several components. The first is a **vocabulary**, i.e. an enumerated list of tokens, along with an **encoding** function that decomposes text into a list of tokens, often encoded via their integer indices:

$$
\mathtt{encode} : \mathtt{Text} \to \mathtt{list}[\mathtt{int}]
$$

We call the number of tokens the **vocabulary size** and for now denote it with the symbol $\mathtt{V}$. In principle, one could make many different choices of tokenization scheme. The main tradeoff to consider is that between vocabulary size and the size of the tokenized output. Two naive choices for the vocabulary are characters and words. Using characters yields a small vocabulary, but produces very long token sequences. Using words shortens sequences, but leads to an impractically large (effectively unbounded) vocabulary.

Nearly all modern language models---including Qwen3---use a middle-ground approach called **byte pair encoding** (BPE). The procedure for **training** a BPE tokenizer is as follows. The vocabulary is initialized with individual bytes. Bytes are even more granular than characters, since a single character may be encoded as multiple bytes under [UTF-8](https://en.wikipedia.org/wiki/UTF-8). Then, using a reference text corpus, the most frequently occurring adjacent pair of tokens is iteratively merged into a single new token until the vocabulary reaches a target size. We save both the vocabulary and the ordered sequence of merge rules. This core loop is simple enough to describe in a few lines of pseudocode:

```python
vocabulary = {all single bytes}
merge_rules = []
while len(vocabulary) < target_size:
    (a, b) = most_frequent_adjacent_pair(corpus)
    vocabulary.add(a + b)
    merge_rules.append((a, b))
```

Given a trained tokenizer, the $\mathtt{encode}$ function is implemented as the following sequence:

1. the text is first [normalized](https://en.wikipedia.org/wiki/Unicode_equivalence#Normal_forms) to a canonical unicode form
2. the text is then split into semantic units with a [regex](https://en.wikipedia.org/wiki/Regular_expression)---this is called **pre-tokenization**
3. finally, within each chunk, adjacent tokens are iteratively merged using the merge rules

The pre-tokenization regex is load-bearing: BPE merges only ever happen *within* a chunk, so whatever pre-tokenization splits apart, the merge step can never glue back together. Qwen3's regex enforces several deliberate design choices:

- **digits split one at a time** (`\p{N}`, not `\p{N}+`)---multi-digit numbers always decompose into individual digits, giving the model a clean per-digit representation for arithmetic
- **English contractions split off** (the alternation `(?i:'s|'t|'re|'ve|'m|'ll|'d)`)---`"don't"` becomes `'don'` + `'t'`, so the model sees the contraction suffix as its own grammatical unit
- **leading space attaches to the following word** (the `[^\r\n\p{L}\p{N}]?\p{L}+` clause)---`" world"` becomes a single `Ġworld` token, so the vocabulary doesn't need separate entries for `"world"` and `" world"`

To make the merge process concrete, the REPL is preloaded with a small hand-crafted `merge_rules` list and two helpers, `merge_step` (apply one rule once) and `bpe_encode` (apply all rules in order, starting from individual characters):

```py
# uv run main.py repl

In [1]: merge_rules
Out[1]: [('t', 'h'), ('th', 'e'), ('h', 'e'), ('e', 'r'), ('o', 'r')]

In [2]: unmerged = list("theory")

In [3]: unmerged
Out[3]: ['t', 'h', 'e', 'o', 'r', 'y']

In [4]: merge_step(unmerged, merge_rules[0])
Out[4]: ['th', 'e', 'o', 'r', 'y']

In [5]: merge_step(_, merge_rules[1])
Out[5]: ['the', 'o', 'r', 'y']

In [6]: merge_step(_, merge_rules[2])
Out[6]: ['the', 'o', 'r', 'y']

In [7]: merge_step(_, merge_rules[3])
Out[7]: ['the', 'o', 'r', 'y']

In [8]: merge_step(_, merge_rules[4])
Out[8]: ['the', 'or', 'y']

In [9]: bpe_encode(unmerged)
Out[9]: ['the', 'or', 'y']
```

You can edit `merge_rules` or call `bpe_encode` with your own list to see how the encoding changes.

Different model families train their tokenizers on different reference corpora and to different vocabulary sizes, but the underlying algorithm is the same. Qwen3's tokenizer has a vocabulary with `151,669` tokens.

You can play with Qwen3's tokenizer using our built-in [REPL](https://en.wikipedia.org/wiki/Read%E2%80%93eval%E2%80%93print_loop), based on [IPython](https://ipython.org/), pre-loaded with the helper function `tokenize`. You can run it via the terminal command

```bash
uv run main.py repl
```

It will open an interactive coding environment directly in the terminal:

```python
In [1]: tokenize("Hello, world!")
Out[1]: ['Hello', ',', 'Ġworld', '!']

In [2]: tokenize("The quick brown fox")
Out[2]: ['The', 'Ġquick', 'Ġbrown', 'Ġfox']

In [3]: tokenize("tokenization")
Out[3]: ['token', 'ization']

In [4]: tokenize("transformer")
Out[4]: ['transform', 'er']

In [5]: tokenize("x = 3.14")
Out[5]: ['x', 'Ġ=', 'Ġ', '3', '.', '1', '4']

In [6]: tokenize("the year 2025")
Out[6]: ['the', 'Ġyear', 'Ġ', '2', '0', '2', '5']
```

Try your own strings to build intuition for how BPE decomposes text. Some patterns you will observe:
- leading spaces get tokenized as `Ġ`, the encoding of the space byte
- common words survive as single tokens: `Hello`, `world`, `quick`, `transform`
- rarer compounds are split into recognizable subwords: `token`+`ization`, `transform`+`er`
- every digit is its own token regardless of magnitude (a consequence of the pre-tokenization regex)
- punctuation is handled byte by byte

Now that we have an encoding function, we can also define a **decoding** function

$$
\mathtt{decode} : \mathtt{list}[\mathtt{int}] \to \mathtt{Text}
$$

by simply reversing the above process: 

1. each index is mapped back to its vocabulary entry
2. the entries are concatenated
3. the resulting bytes are decoded as text

The round-trip recovers the original text up to unicode normalization:

$$
\begin{tikzcd}
\mathtt{Text}
\arrow[rr, "\mathtt{normalize}"]
\arrow[dr, "\mathtt{encode}"']
&& \mathtt{Text}
\\
& \mathtt{list[int]}
\arrow[ur, "\mathtt{decode}"']
&
\end{tikzcd}
$$

The rules for both encoding and decoding are packaged in a single JSON file---`tokenizer.json`---distributed alongside each model. The curious reader can open `models/qwen3/tokenizer.json`. The three most load-bearing fields are:

- `model`---which contains the BPE `vocab` and `merges` trained above
- `added_tokens`---special tokens beyond the BPE vocabulary; includes the ChatML delimiters `<|im_start|>` / `<|im_end|>` (the latter doubles as the chat **end-of-sequence**) and `<|endoftext|>` (the base end-of-document / pad token)
- `pre_tokenizer`---the [regex](https://en.wikipedia.org/wiki/Regular_expression) pattern that splits text into chunks before merge rules are applied

Instead of implementing encoding and decoding by hand, we use the industry-standard `tokenizers` library from Hugging Face, which provides a compiled [Rust](https://www.rust-lang.org/) binary that reads a `tokenizer.json` file and exposes `encode` and `decode` functions. Once a language model consumes a sequence of tokens---its *actual* input type---it can be used to generate a sequence of output tokens, called a **completion**. This gives us the following implementation: 

```python
def generate(
    tokenizer: Tokenizer,
    sampler: Sampler,
    model: Catform,
    context: str,
    state: State,
) -> tuple[str, State]:
    """generate = encode ; complete ; decode."""
    tokens            = tokenizer.encode(context).ids           # str       -> list[int]
    completion, state = complete(sampler, model, tokens, state) # list[int] -> list[int]
    return tokenizer.decode(completion), state                  # list[int] -> str
```

Thus we have reduced the problem of defining our text-to-text function to that of defining one that is tokens-to-tokens:

$$
\mathtt{complete} : \mathtt{list}[\mathtt{int}] \to \mathtt{list}[\mathtt{int}]
$$

We thus have another commutative diagram:

$$
\begin{tikzcd}
\mathtt{Text}
\arrow[rr, "\mathtt{generate}"]
\arrow[d, "\mathtt{encode}"']
&& \mathtt{Text}
\\
\mathtt{list[int]}
\arrow[rr, "\mathtt{complete}"']
&& \mathtt{list[int]}
\arrow[u, "\mathtt{decode}"']
\end{tikzcd}
$$

We now turn to the production of tokens.

## Autoregressive Completion

While there are other sorts of language models---e.g. [diffusion language models](https://spacehunterinf.github.io/blog/2025/diffusion-language-models/)---the vast majority of those in circulation at the time of writing (2026) are **autoregressive**, meaning that they complete their output sequence one token at a time. Thus the $\mathtt{complete}$ function might be thought of as the iterative application of a more elemental function which computes the next token

$$
\mathtt{next\_token} : \mathtt{list}[\mathtt{int}] \to \mathtt{int}
$$

With such a function, we could then define the completion $[\mathtt{y_0}, \dots, \mathtt{y_{N-1}}]$ of a token sequence $[\mathtt{x_0}, \dots, \mathtt{x_{M-1}}]$ as the successive applications of $\mathtt{next\_token}$, performed while some "keep generating" condition holds:

$$\begin{align*}
\mathtt{y_0} &= \mathtt{next\_token}([\mathtt{x_0}, \dots, \mathtt{x_{M-1}}])
\\
\mathtt{y_1} &= \mathtt{next\_token}([\mathtt{x_0}, \dots, \mathtt{x_{M-1}}, \mathtt{y_0}])
\\
\mathtt{y_2} &= \mathtt{next\_token}([\mathtt{x_0}, \dots, \mathtt{x_{M-1}}, \mathtt{y_0},\mathtt{y_1} ])\\
\vdots
\\
\mathtt{y_{N-1}} &= \mathtt{next\_token}([\mathtt{x_0}, \dots, \mathtt{x_{M-1}}, \mathtt{y_0},\dots,\mathtt{y_{N-2}}])
\end{align*}
$$

In pseudocode one may write this as the loop:

```py
completion = []
while keep_generating:
    token = next_token(context + completion)
    completion.append(token)
return completion
```

Each call to `next_token` above re-runs the model on the entire prior context---even though we already did that work in the previous step. We can avoid the redundant recomputation by keeping a **cache** of intermediate values from prior steps, so each subsequent call only needs to consume one new token:

$$\begin{align*}
\mathtt{y_0} &= \mathtt{next\_token}([\mathtt{x_0}, \dots, \mathtt{x_{M-1}}])
\\
\mathtt{y_1} &= \mathtt{next\_token}([\mathtt{y_0}],     \mathtt{cache})
\\
\mathtt{y_2} &= \mathtt{next\_token}([\mathtt{y_1}],     \mathtt{cache})\\
\vdots
\\
\mathtt{y_{N-1}} &= \mathtt{next\_token}([\mathtt{y_{N-2}}], \mathtt{cache})
\end{align*}
$$

This also means that the next token computation must be stateful

$$
\mathtt{next\_token : (list[int], State) \to (int, State)}
$$

Note that, since we received $\mathtt{x_0}, \dots, \mathtt{x_{M-1}}$ all at once, we could not reduce that first computation to a single token input. We call this first step the **prefill**, and each subsequent one a **decode**.

In fact, this cache is precisely what our opaque $\mathtt{State}$ type was carrying:

```py
@dataclass(frozen=True)
class State:
    cached: list[int]              # token ids the cache reflects
    cache: dict[str, Any]          # cached values from prior forward passes
```

Furthermore, most inference servers maintain this state across requests. Recall from the discussion on tool use that agentic flows often chain multiple requests as inference interacts with the harness. So before the autoregressive loop, we compute the longest common **prefix** between the stored context and the incoming context---the part already cached---and feed only the remaining **suffix** to the prefill stage.

```py
def prefix(context: list[int], state: State) -> tuple[list[int], State]:
    """Truncate state.cached to the longest common prefix with context;
    return the uncached suffix and the truncated state."""
    matched = takewhile(lambda ab: ab[0] == ab[1], zip(state.cached, context))
    cached = [a for a, _ in matched]
    suffix = context[len(cached):]
    return suffix, State(cached=cached, cache={**state.cache, "state.seen": len(cached)})
```

Thus our above pseudocode loop gets refined to:

```py
context, state = prefix(context, state)
completion = []
while sampler.keep_generating(completion):
    token, state = next_token(context, state)
    completion.append(token)
    context      = [token]
return completion, state
```

The above however is not the full story. You may have heard that language models, in contrast to classic computation, are "probabilistic" rather than "deterministic". Technically this is not exactly correct: there is a clean separation of the deterministic component---which, somewhat ironically, is the language model itself---and the probabilistic component, which **samples** tokens from a distribution derived from the model's output. More precisely, a model's forward pass takes tokens as input and outputs a numerical score---called a **logit**---for each token in the vocabulary. We can represent these output **logits** as having the type $\mathtt{real[V]}$---a real-valued vector of size $\mathtt{V}$, where the value at index $\mathtt{k}$ is the logit associated to the $\mathtt{k}^\text{th}$ token. Thus, ignoring the state, we can think of the model as having the following type:

$$
\mathtt{model} : \mathtt{list[int]} \to \mathtt{real[V]}
$$

The sampling process then converts the logits vector $\mathtt{real[V]}$ to a probability distribution and then randomly samples a token index. Thus, we have that the $\mathtt{next\_token}$ function is really a composition of the model forward pass, followed by sampling:

$$
\mathtt{next\_token = model ; sample}
$$

We use a `Sampler` object to package both the sampling process and the `keep_generating` condition---which ends completion as soon as either an **end-of-sequence** token has been generated or `max_tokens` tokens have been generated. This gives us the actual loop of the `complete` function:

```py
def complete(
    sampler: Sampler,
    model: Catform,
    context: list[int],
    state: State,
) -> tuple[list[int], State]:
    """Autoregressive completion."""
    context, state = prefix(context, state)
    completion = []
    while sampler.keep_generating(completion):
        logits, state = model(context, state)
        token          = sampler.sample(logits)
        completion.append(token)
        context        = [token]
    return completion, state
```

We defer the deconstruction of $\mathtt{model}$ and $\mathtt{State}$ to the [third chapter](03_models.md). In the meantime, we turn to the sampling process.

### Sampling

How do we sample a token from the logits array $\mathtt{x}$? The first matter is to turn the logits $\mathtt{x_1,\dots,x_V}$ into an actual probability distribution. Recall that a probability distribution $\mathtt{p}$ assigns a non-negative number $\mathtt{p_k}$ to outcome $\mathtt{k}$ with the requirement that these numbers sum to $\mathtt{1}$:

$$
\sum_{\mathtt{k}}\mathtt{p_k} = 1
$$

The relevant distribution for us is the probability $\mathtt{p_k}$ of the next token being $\mathtt{k}$. The logits are not at all guaranteed to sum to $\mathtt{1}$, so we cannot just set $\mathtt{p_k=x_k}$. Naively, we could enforce this by simply dividing each logit $\mathtt{x_k}$ by the total sum $\mathtt{sum(x)}$ of all of the logits:

$$
\mathtt{p_k} = \frac{\mathtt{x_k}}{\mathtt{sum(x)}}
$$

This, however, only produces a valid probability distribution when all logits $\mathtt{x_k}$ are non-negative, which is also not guaranteed. To fix this, we can observe that we can generalize the above by first applying a function $\mathtt{f : real \to real}$ to all the logits and then normalizing by the sum:

$$
\mathtt{p_k} = \frac{\mathtt{f(x_k)}}{\mathtt{sum(f(x))}}
$$

The question now is how to choose the $\mathtt{f}$. We want any choice to satisfy two properties:

- **non-negativity**: $\mathtt{f(x)\geq 0}$ for all $\mathtt{x}$---so that the output is a valid probability distribution
- **monotonicity**: if $\mathtt{x<y}$ then $\mathtt{f(x) < f(y)}$---so that higher logits mean higher probabilities

While many functions satisfy both criteria, the exponential family, parameterized by $\mathtt{t}$, is a natural choice:

$$
\mathtt{f(x) = e^{x/t}}
$$

Taken together, we have just defined the **softmax** function, which takes an array of real numbers and outputs a probability distribution on them:

$$
\mathtt{[x_1,\dots,x_V]\mapsto \left[\frac{e^{x_1/t}}{sum(e^{x/t})},\dots, \frac{e^{x_V/t}}{sum(e^{x/t})}\right]}
$$

Choosing the exponential carries one extra advantage: the logits become **shift-invariant**---adding the same quantity to all logits yields the same output distribution. This allows us to avoid numerical overflow by subtracting the maximum logit value $\mathtt{x_{max}}$ from all logits:

$$
\mathtt{[x_1,\dots,x_V]\mapsto \left[\frac{e^{(x_1-x_{max})/t}}{sum(e^{(x-x_{max})/t})},\dots, \frac{e^{(x_V-x_{max})/t}}{sum(e^{(x-x_{max})/t})}\right]}
$$

We call the parameter $\mathtt{t}$ the **temperature**, and use it to modulate the **entropy**---that is, the level of randomness---of the resulting distribution. In the limit $\mathtt{t \to 0}$, this yields a deterministic output where we simply select the token with the highest logit. In the limit $\mathtt{t \to \infty}$, this yields the uniform distribution across tokens. In practice, temperatures typically range from $\mathtt{0}$ to $\mathtt{2}$, with $\mathtt{1}$ as a common default.

Beyond temperature, there are two common filtering strategies to restrict the candidate set of tokens to sample. **Top-$\mathtt{k}$** keeps only the $\mathtt{k}$ highest-probability tokens, while **Top-$\mathtt{p}$**, or **nucleus sampling**, keeps the smallest set of tokens whose cumulative probability exceeds a threshold $\mathtt{p}$. Both are methods to zero out low-probability tokens and can be used independently or together.

To play around with the sampler, the REPL preloads Qwen3 0.6B, along with two helpers `tokenize_ids` and `forward_pass`, so we can interact with the full pipeline:

```python
# uv run main.py repl

In [1]: prompt = "To be or not to be, that is the"

In [2]: tokens = tokenize_ids(prompt)

In [3]: logits = forward_pass(tokens)

In [4]: s = Sampler(logits)

In [5]: s
Out[5]:
  0.8355  ' question'
  0.0185  ' essence'
  0.0119  ' same'
  0.0099  ' matter'
  0.0053  ' price'
  0.0041  ' classic'
  0.0030  ' call'
  0.0030  ' thing'
  0.0030  ' spirit'
  0.0023  ' first'

In [6]: s.topk(5)

In [7]: s
Out[7]:
  0.9483  ' question'
  0.0210  ' essence'
  0.0135  ' same'
  0.0112  ' matter'
  0.0060  ' price'

In [8]: s.pick()
Out[8]: ' question'
```

`Sampler(logits)` applies softmax to the logits, producing a probability distribution over the full vocabulary. The model is `83.55%` confident that the next token is `' question'`. Calling `.topk(5)` filters to the five most likely tokens and thus renormalizes the probability of ` question` to `0.9483`.

We emphasize that the sampler configuration parameters---`temperature`, `top_k`, `top_p`, and `max_tokens`---are independent of the model. They are optional fields in the API request alongside the `model` and `messages`, with sensible defaults when they are not specified.

## The Full Picture

We began this chapter with agents interacting with an inference API and then we peeled away layers until we got to the model forward pass, the subject of the [third chapter](03_models.md). At the code level, this nesting is visible in the function signatures themselves. Each layer is the one below with one more configuration argument:

```py
infer(template, tokenizer, sampler, model, context, state)
generate(       tokenizer, sampler, model, context, state)
complete(                  sampler, model, context, state)
                                    model( context, state)
```

The full composite is best represented in a wiring diagram. All data flows along wires from left to right, in and out of function boxes. The sampling is represented with an $\mathtt{S}$-labelled triangle since technically the act of drawing a random sample is not a function.

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
\draw[very thick] ( 2.4,.15) -- ( 2.6,.15) ;
\draw[very thick] ( 2.9,.15) -- ( 3.4,.15) ;

\draw[very thick] (-2.0, 0.15) -- (-1.6, 0.15) ;
\draw[very thick] (-2.0,-0.15) -- (-1.6,-0.15) ;

\draw[very thick] (-0.6, 0.0) -- (-0.6, 0.3) -- (-0.3, 0.15) -- cycle;
\node at (-0.5, 0.15) {$\mathtt{S}$};
\draw[very thick] ( 2.6, 0.0) -- ( 2.6, 0.3) -- ( 2.9, 0.15) -- cycle;
\node at ( 2.7, 0.15) {$\mathtt{S}$};

\draw[very thick] (-.8, 0.15) -- (-.6, 0.15) ;
\draw[very thick] (-.3, 0.15) -- (-.2, 0.15) ;
\draw[very thick] (-.8,-0.15) -- (-.2,-0.15) ;

\draw[very thick] (0.6, 0.15) -- (1, 0.15) ;
\draw[very thick] (0.6,-0.15) -- (1,-0.15) ;

\draw[very thick] (1.2, 0.15) -- (1.6, 0.15) ;
\draw[very thick] (1.2,-0.15) -- (1.6,-0.15) ;

\node at (1.1,0) {$\cdots$};

\node at (-2.4,0) {$\mathtt{prefix}$};
\node at (-1.2,0) {$\mathtt{model}$};
\node at ( 0.2,0) {$\mathtt{model}$};
\node at ( 2,0) {$\mathtt{model}$};

\end{tikzpicture}
$$

Now that we have unraveled all of the scaffolding, what remains is the $\mathtt{model}$ itself. Before we can open it up, we need the mathematical language in which it is written: tensors and their operations.
