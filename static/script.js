/* ===========================================================================
   BiteRate
   User-supplied text is only ever inserted with textContent / text nodes.
   =========================================================================== */

const CLASSES = ["Positive", "Neutral", "Negative"];
const MAX_CHARS = 5000;
const BATCH_CHUNK = 100; // server limit per /predict_all request

const EXAMPLES = [
    "Hands down the best tacos in the city. The salsa verde is addictive and the staff were lovely. Five stars!",
    "The ambiance was nice and the dessert was excellent, but the mains were bland and we waited forty minutes for a table.",
    "I wouldn't say the food was bad, but it definitely wasn't worth the price. Not coming back.",
];

const $ = (id) => document.getElementById(id);
const state = { items: [], nextId: 1, filter: "all", query: "", last: null };

document.addEventListener("DOMContentLoaded", () => {
    initAnalyzer();
    initBatch();
});

/* ---------- helpers ---------- */
function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

async function postJson(url, body) {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
}

async function withLoading(btn, fn) {
    btn.disabled = true;
    btn.classList.add("is-loading");
    try {
        return await fn();
    } finally {
        btn.disabled = false;
        btn.classList.remove("is-loading");
    }
}

let toastTimer;
function toast(message) {
    const node = $("toast");
    node.textContent = message;
    node.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (node.hidden = true), 2800);
}

const fmt = (n) => n.toLocaleString("en-US");
const plural = (n, word) => `${fmt(n)} ${word}${n === 1 ? "" : "s"}`;

function sentimentTag(sentiment) {
    const tag = el("span", `sent sent-${sentiment.toLowerCase()}`);
    tag.append(el("i", "dot"), document.createTextNode(sentiment));
    return tag;
}

function renderStack(container, values) {
    container.replaceChildren();
    const total = CLASSES.reduce((sum, c) => sum + (values[c] || 0), 0);
    container.setAttribute("aria-label",
        CLASSES.map((c) => `${c} ${total ? Math.round(((values[c] || 0) / total) * 100) : 0}%`).join(", "));
    const segs = CLASSES.filter((c) => values[c]).map((c) => {
        const seg = el("div", `stack-seg ${c.toLowerCase()}`);
        seg.style.flexGrow = "0";
        container.append(seg);
        return [seg, values[c]];
    });
    void container.offsetWidth; // commit the zero widths so the grow transition runs
    segs.forEach(([seg, v]) => (seg.style.flexGrow = v));
}

/* ---------- analyzer ---------- */
function initAnalyzer() {
    const input = $("analyzerInput");
    // Only surface the counter when the limit is close; otherwise it is noise.
    const count = () => {
        const n = input.value.length;
        $("charCount").textContent = n > MAX_CHARS * 0.9 ? `${fmt(n)} / ${fmt(MAX_CHARS)}` : "";
    };
    input.addEventListener("input", count);
    input.addEventListener("keydown", (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
    });
    document.querySelectorAll("[data-example]").forEach((btn) =>
        btn.addEventListener("click", () => {
            input.value = EXAMPLES[Number(btn.dataset.example)];
            count();
            analyze();
        })
    );
    $("analyzeBtn").addEventListener("click", analyze);
    $("addToBatchBtn").addEventListener("click", () => {
        if (!state.last) return;
        addItems([state.last]);
        $("addToBatchBtn").disabled = true;
        toast("Added to batch");
    });
}

async function analyze() {
    const input = $("analyzerInput");
    const text = input.value.trim();
    if (!text) {
        input.focus();
        return toast("Type or paste a review first");
    }
    try {
        const result = await withLoading($("analyzeBtn"), () => postJson("/predict", { review: text }));
        state.last = { text, result };
        renderResult(text, result);
        $("addToBatchBtn").disabled = false;
    } catch (err) {
        toast(err.message);
    }
}

function renderResult(text, result) {
    $("analyzerResult").hidden = false;

    const label = $("verdictLabel");
    label.textContent = result.sentiment;
    label.className = `verdict-label is-${result.sentiment.toLowerCase()}`;
    $("verdictConf").textContent = `${Math.round(result.confidence)}%`;

    renderStack($("probBar"), result.probabilities);
    const legend = $("probLegend");
    legend.replaceChildren();
    CLASSES.forEach((c) => {
        const li = el("li");
        li.append(el("i", `dot ${c.toLowerCase()}`), document.createTextNode(c), el("span", "num", `${Math.round(result.probabilities[c])}%`));
        legend.append(li);
    });

    renderHighlighted($("highlighted"), text, result.highlights || []);
}

function renderHighlighted(container, text, highlights) {
    container.replaceChildren();
    let cursor = 0;
    highlights.forEach((h) => {
        if (h.start < cursor) return;
        container.append(document.createTextNode(text.slice(cursor, h.start)));
        const mark = el("mark", `${h.polarity}${h.strength >= 0.5 ? " strong" : ""}`, text.slice(h.start, h.end));
        mark.title = `${h.polarity === "pos" ? "Positive" : "Negative"} signal`;
        container.append(mark);
        cursor = h.end;
    });
    container.append(document.createTextNode(text.slice(cursor)));
}

/* ---------- batch ---------- */
function initBatch() {
    const input = $("batchInput");
    const lines = () => input.value.split("\n").map((l) => l.trim()).filter(Boolean);
    const count = () => ($("lineCount").textContent = plural(lines().length, "review"));
    input.addEventListener("input", count);

    $("loadSamplesBtn").addEventListener("click", () => {
        input.value = JSON.parse($("sampleReviews").textContent).join("\n");
        count();
    });

    $("batchBtn").addEventListener("click", async () => {
        const texts = lines();
        if (!texts.length) {
            input.focus();
            return toast("Add at least one review, one per line");
        }
        const tooLong = texts.findIndex((t) => t.length > MAX_CHARS);
        if (tooLong !== -1) return toast(`Line ${tooLong + 1} is over ${fmt(MAX_CHARS)} characters`);
        try {
            const results = await withLoading($("batchBtn"), async () => {
                const out = [];
                for (let i = 0; i < texts.length; i += BATCH_CHUNK) {
                    const chunk = texts.slice(i, i + BATCH_CHUNK);
                    out.push(...(await postJson("/predict_all", { reviews: chunk })).results);
                }
                return out;
            });
            addItems(texts.map((text, i) => ({ text, result: results[i] })));
            input.value = "";
            count();
            $("batchResults").scrollIntoView({ behavior: "smooth", block: "start" });
        } catch (err) {
            toast(err.message);
        }
    });

    document.querySelectorAll(".tab").forEach((tab) =>
        tab.addEventListener("click", () => {
            state.filter = tab.dataset.filter;
            document.querySelectorAll(".tab").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
            renderList();
        })
    );
    $("batchSearch").addEventListener("input", (e) => {
        state.query = e.target.value.trim().toLowerCase();
        renderList();
    });
    $("exportBtn").addEventListener("click", exportCsv);
    $("clearBtn").addEventListener("click", () => {
        state.items = [];
        renderBatch();
    });
}

function addItems(entries) {
    entries.forEach(({ text, result }) => state.items.push({ id: state.nextId++, text, result }));
    renderBatch();
}

function renderBatch() {
    const has = state.items.length > 0;
    $("batchResults").hidden = !has;
    $("batchEmpty").hidden = has;
    if (!has) return;

    const counts = { Positive: 0, Neutral: 0, Negative: 0 };
    state.items.forEach((item) => counts[item.result.sentiment]++);
    const total = state.items.length;

    const summary = $("summaryText");
    summary.replaceChildren(document.createTextNode(`${plural(total, "review")}: `));
    CLASSES.forEach((c, i) => {
        summary.append(el("span", c.toLowerCase(), `${Math.round((counts[c] / total) * 100)}% ${c.toLowerCase()}`));
        summary.append(document.createTextNode(i < CLASSES.length - 1 ? ", " : "."));
    });
    renderStack($("summaryBar"), counts);

    document.querySelector('[data-count="all"]').textContent = fmt(total);
    CLASSES.forEach((c) => (document.querySelector(`[data-count="${c}"]`).textContent = fmt(counts[c])));
    renderList();
}

function renderList() {
    const list = $("batchList");
    list.replaceChildren();
    const visible = state.items.filter((item) =>
        (state.filter === "all" || item.result.sentiment === state.filter) &&
        (!state.query || item.text.toLowerCase().includes(state.query))
    );
    visible.forEach((item) => {
        const li = el("li", "review-item");
        li.tabIndex = 0;
        li.append(sentimentTag(item.result.sentiment), el("p", "review-text", item.text),
                  el("span", "review-conf num", `${Math.round(item.result.confidence)}%`));
        const open = () => openInAnalyzer(item.text);
        li.addEventListener("click", open);
        li.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                open();
            }
        });
        list.append(li);
    });
    $("noMatches").hidden = visible.length > 0;
}

function openInAnalyzer(text) {
    const input = $("analyzerInput");
    input.value = text;
    input.dispatchEvent(new Event("input"));
    $("analyze").scrollIntoView({ behavior: "smooth", block: "start" });
    analyze();
}

function exportCsv() {
    const esc = (v) => `"${String(v).replace(/"/g, '""')}"`;
    const lines = ["review,sentiment,confidence,positive,neutral,negative"];
    state.items.forEach(({ text, result: r }) =>
        lines.push([esc(text), r.sentiment, r.confidence, r.probabilities.Positive, r.probabilities.Neutral, r.probabilities.Negative].join(","))
    );
    const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    const a = el("a");
    a.href = url;
    a.download = "biterate-results.csv";
    a.click();
    URL.revokeObjectURL(url);
}
