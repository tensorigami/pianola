// Scroll-spy: highlight the current section in the sidebar as the reader scrolls.

(function () {
    "use strict";

    var headings = document.querySelectorAll(".content main h2, .content main h3");
    if (!headings.length) return;

    // Build a map from heading id → sidebar link
    var links = {};
    headings.forEach(function (h) {
        if (!h.id) return;
        // Sidebar links end with #<id>
        var a = document.querySelector('.sidebar a[href$="#' + CSS.escape(h.id) + '"]');
        if (a) links[h.id] = a;
    });

    if (!Object.keys(links).length) return;

    var observer = new IntersectionObserver(
        function (entries) {
            entries.forEach(function (entry) {
                var a = links[entry.target.id];
                if (!a) return;
                if (entry.isIntersecting) {
                    // Clear previous
                    Object.values(links).forEach(function (el) {
                        el.classList.remove("scroll-active");
                    });
                    a.classList.add("scroll-active");
                }
            });
        },
        {
            // Trigger when heading crosses the top 20% of the viewport
            rootMargin: "0px 0px -80% 0px",
            threshold: 0,
        }
    );

    headings.forEach(function (h) {
        if (h.id && links[h.id]) observer.observe(h);
    });
})();

// Fix TikZ SVG ID collisions: each diagram defines glyphs like "glyph-0-0"
// but when multiple SVGs share a page, the browser resolves IDs globally.
// Prefix every ID (and its references) per SVG so they don't collide.

(function () {
    "use strict";

    var svgs = document.querySelectorAll("div > svg");
    svgs.forEach(function (svg, i) {
        var prefix = "tikz" + i + "-";

        // Rewrite id attributes in <defs>
        svg.querySelectorAll("[id]").forEach(function (el) {
            el.id = prefix + el.id;
        });

        // Rewrite xlink:href and href references
        svg.querySelectorAll("[*|href]").forEach(function (el) {
            var attr = el.getAttributeNS("http://www.w3.org/1999/xlink", "href")
                     || el.getAttribute("href");
            if (attr && attr.startsWith("#")) {
                var newRef = "#" + prefix + attr.slice(1);
                if (el.hasAttributeNS("http://www.w3.org/1999/xlink", "href")) {
                    el.setAttributeNS("http://www.w3.org/1999/xlink", "href", newRef);
                }
                if (el.hasAttribute("href")) {
                    el.setAttribute("href", newRef);
                }
            }
        });

        // Rewrite clip-path="url(#...)" and similar url() references
        svg.querySelectorAll("[clip-path], [fill], [stroke], [mask], [filter]").forEach(function (el) {
            ["clip-path", "fill", "stroke", "mask", "filter"].forEach(function (attr) {
                var val = el.getAttribute(attr);
                if (val && val.indexOf("url(#") !== -1) {
                    el.setAttribute(attr, val.replace(/url\(#/g, "url(#" + prefix));
                }
            });
        });
    });
})();
