// highlight.js 10.x language definition for catform (.cat)
// Color roles defined in SYNTAX.md — highlight by syntactic position.
function catform(hljs) {
  var COMMENT = hljs.COMMENT("//", "$");

  var STRING = {
    className: "string",
    begin: '"', end: '"',
  };

  var NUMBER = {
    className: "number",
    begin: /-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/,
  };

  // ── Content-based (match anywhere) ────────────────────────

  // §5: Dotted names — prefix @attr, path segments @literal
  var DOTTED_NAME = {
    className: "attr",
    begin: /\b[a-zA-Z_]\w*\./,
    end: /(?=[^.\w]|$)/,
    contains: [
      { className: "literal", begin: /[a-zA-Z_]\w*/ }
    ],
  };

  // ── Type context (after colon) ────────────────────────────

  // Shape bracket inside type: [N, param.hidden, 2, (two d)]
  var SHAPE_BRACKET = {
    begin: /\[/, end: /\]/,
    contains: [
      COMMENT,
      NUMBER,
      DOTTED_NAME,
      { begin: /\(/, end: /\)/, contains: [
        { className: "meta", begin: /\b[a-zA-Z_]\w*/ },
      ]},
      { className: "meta", begin: /\b[a-zA-Z_]\w*/ },
    ],
  };

  // Tuple type: (dtype[shape], dtype[shape])
  var TUPLE_TYPE = {
    begin: /\(/, end: /\)/,
    contains: [
      COMMENT,
      DOTTED_NAME,
      { className: "type", begin: /\b[a-zA-Z_']\w*/ },
      SHAPE_BRACKET,
    ],
  };

  // Type annotation (§2): ": dtype[shape]" or ": (tuple)"
  // Begins at colon, ends before = , ) { }
  var TYPE_ANNOTATION = {
    begin: /:\s*/,
    excludeBegin: true,
    end: /(?=\s*[=,){}])/,
    contains: [
      COMMENT,
      TUPLE_TYPE,
      { className: "type", begin: /\b[a-zA-Z_']\w*/ },
      SHAPE_BRACKET,
    ],
  };

  // ── Specifier context (inside brackets after op) ──────────

  var SPECIFIER_BLOCK = {
    begin: /\[/, end: /\]/,
    contains: [
      STRING,
      NUMBER,
      DOTTED_NAME,
      // key=value keywords like over=, count= (§6)
      { className: "attr", begin: /\b\w+(?==)/ },
      // specifier function — everything else (§5)
      { className: "built_in", begin: /\b[a-zA-Z_]\w*/ },
    ],
  };

  // ── Argument context (inside parens after op) ─────────────

  var ARG_LIST = {
    begin: /\(/, end: /\)/,
    contains: [
      COMMENT,
      DOTTED_NAME,
      NUMBER,
      STRING,
      // bare identifiers: no class → default text (§1)
    ],
  };

  // ── Op call (after =) ─────────────────────────────────────

  // = op_name[specifiers](args)
  var OP_CALL = {
    begin: /=\s*/,
    excludeBegin: true,
    end: /$/,
    contains: [
      COMMENT,
      // op name: first word after = (§3)
      { className: "keyword", begin: /\b[a-zA-Z_]\w*/ },
      SPECIFIER_BLOCK,
      ARG_LIST,
    ],
  };

  // ── Function definition ───────────────────────────────────

  // Function name at column 0 — same color as op names (§2)
  var FUNC_NAME = {
    className: "keyword",
    begin: /^[a-zA-Z_]\w*(?=\s*[\(])/,
  };

  // Arrow -> (§13)
  var ARROW = {
    className: "keyword",
    begin: /->/,
  };

  return {
    name: "Catform",
    aliases: ["cat"],
    contains: [
      COMMENT,
      STRING,
      FUNC_NAME,
      TYPE_ANNOTATION,
      OP_CALL,
      ARROW,
      DOTTED_NAME,
      NUMBER,
    ],
  };
}

if (typeof hljs !== "undefined") {
  hljs.registerLanguage("catform", catform);
  document.querySelectorAll("code.language-catform").forEach(function(block) {
    block.className = "language-catform";
    block.textContent = block.textContent;
    hljs.highlightBlock(block);
  });
}
