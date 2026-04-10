/* ===========================================================================
   BiteRate — Dashboard JavaScript
   =========================================================================== */

// Track the next review ID for dynamically added rows
let nextReviewId = 13;

document.addEventListener("DOMContentLoaded", () => {
    initTooltips();
    initClock();
    initSearch();
    initFilter();
    initPredictAll();
    initAddReview();
    updateSummaryCards(false); // initial count (no animation)

    // Set nextReviewId based on existing rows
    const existingRows = document.querySelectorAll("#reviewTableBody .review-row");
    if (existingRows.length > 0) {
        const ids = Array.from(existingRows).map(r => parseInt(r.getAttribute("data-id")) || 0);
        nextReviewId = Math.max(...ids) + 1;
    }
});

/* -----------------------------------------------------------------------
   Bootstrap Tooltips
   ----------------------------------------------------------------------- */
function initTooltips() {
    const tooltipEls = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipEls.forEach(el => new bootstrap.Tooltip(el));
}

/* -----------------------------------------------------------------------
   Live Clock
   ----------------------------------------------------------------------- */
function initClock() {
    const el = document.getElementById("liveClock");
    function tick() {
        const now = new Date();
        el.textContent = now.toLocaleTimeString("en-GB", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
        });
    }
    tick();
    setInterval(tick, 1000);
}

/* -----------------------------------------------------------------------
   Summary Cards — count-up animation
   ----------------------------------------------------------------------- */
function updateSummaryCards(animate = true) {
    const rows = document.querySelectorAll("#reviewTableBody .review-row");
    let total = rows.length;
    let pos = 0, neg = 0, neu = 0;

    rows.forEach(row => {
        const s = row.getAttribute("data-sentiment");
        if (s === "Positive") pos++;
        else if (s === "Negative") neg++;
        else if (s === "Neutral") neu++;
    });

    if (animate) {
        animateCount("totalCount", total);
        animateCount("positiveCount", pos);
        animateCount("negativeCount", neg);
        animateCount("neutralCount", neu);
    } else {
        document.getElementById("totalCount").textContent = total;
        document.getElementById("positiveCount").textContent = pos;
        document.getElementById("negativeCount").textContent = neg;
        document.getElementById("neutralCount").textContent = neu;
    }
}

function animateCount(elementId, target) {
    const el = document.getElementById(elementId);
    const current = parseInt(el.textContent) || 0;
    if (current === target) return;

    const duration = 400; // ms
    const startTime = performance.now();

    function step(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = Math.round(current + (target - current) * eased);
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

/* -----------------------------------------------------------------------
   Per-row Predict
   ----------------------------------------------------------------------- */
async function predictSingle(btnEl, reviewId) {
    const row = btnEl.closest(".review-row");
    const reviewText = row.querySelector(".review-full-text").value;

    // Show spinner
    btnEl.disabled = true;
    btnEl.innerHTML = '<span class="spinner-border spinner-border-sm"></span>';

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ review: reviewText }),
        });
        const data = await res.json();
        applyPrediction(row, data.sentiment, data.confidence);
    } catch (err) {
        console.error("Prediction error:", err);
        btnEl.innerHTML = "Retry";
        btnEl.disabled = false;
    }
}

function applyPrediction(row, sentiment, confidence) {
    // Update data attribute
    row.setAttribute("data-sentiment", sentiment);

    // Sentiment badge
    const sentCell = row.querySelector(".col-sentiment");
    const badgeClass = sentiment === "Positive" ? "badge-positive"
                     : sentiment === "Negative" ? "badge-negative"
                     : "badge-neutral";
    const label = sentiment === "Positive" ? "🟢 Positive"
                : sentiment === "Negative" ? "🔴 Negative"
                : "⚪ Neutral";
    sentCell.innerHTML = `<span class="badge ${badgeClass}">${label}</span>`;

    // Confidence bar
    const confCell = row.querySelector(".col-confidence");
    const fillClass = sentiment === "Positive" ? "fill-positive"
                    : sentiment === "Negative" ? "fill-negative"
                    : "fill-neutral";
    confCell.innerHTML = `
        <div class="confidence-bar-wrapper">
            <div class="confidence-track">
                <div class="confidence-fill ${fillClass}" style="width: 0"></div>
            </div>
            <span class="confidence-pct">${confidence}%</span>
        </div>`;
    // Animate bar after a frame
    requestAnimationFrame(() => {
        const fill = confCell.querySelector(".confidence-fill");
        fill.style.width = confidence + "%";
    });

    // Button → done
    const btn = row.querySelector(".col-action .btn-predict, .col-action button");
    btn.innerHTML = "✅";
    btn.classList.add("btn-done");
    btn.disabled = true;

    // Update cards
    updateSummaryCards(true);
}

/* -----------------------------------------------------------------------
   Predict All
   ----------------------------------------------------------------------- */
function initPredictAll() {
    const btn = document.getElementById("predictAllBtn");
    btn.addEventListener("click", predictAll);
}

async function predictAll() {
    const btn = document.getElementById("predictAllBtn");
    const labelSpan = btn.querySelector(".btn-label");
    const spinnerSpan = btn.querySelector(".spinner-border");

    // Collect all review texts
    const rows = Array.from(document.querySelectorAll("#reviewTableBody .review-row"));
    const texts = rows.map(r => r.querySelector(".review-full-text").value);

    // Disable button, show spinner
    btn.disabled = true;
    labelSpan.classList.add("d-none");
    spinnerSpan.classList.remove("d-none");

    // Disable all per-row buttons
    rows.forEach(r => {
        const b = r.querySelector(".col-action button");
        if (b && !b.classList.contains("btn-done")) b.disabled = true;
    });

    try {
        const res = await fetch("/predict_all", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ reviews: texts }),
        });
        const data = await res.json();
        const results = data.results;

        // Apply results one by one with stagger
        for (let i = 0; i < rows.length; i++) {
            await delay(100);
            applyPrediction(rows[i], results[i].sentiment, results[i].confidence);
        }

        showToast("✅ All reviews analysed!");
    } catch (err) {
        console.error("Predict all error:", err);
        showToast("❌ Something went wrong. Please try again.");
    } finally {
        btn.disabled = false;
        labelSpan.classList.remove("d-none");
        spinnerSpan.classList.add("d-none");
    }
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/* -----------------------------------------------------------------------
   Search
   ----------------------------------------------------------------------- */
function initSearch() {
    const input = document.getElementById("searchInput");
    input.addEventListener("input", applyFilters);
}

/* -----------------------------------------------------------------------
   Filter Dropdown
   ----------------------------------------------------------------------- */
function initFilter() {
    const select = document.getElementById("filterSelect");
    select.addEventListener("change", applyFilters);
}

/* -----------------------------------------------------------------------
   Combined filter logic
   ----------------------------------------------------------------------- */
function applyFilters() {
    const query = document.getElementById("searchInput").value.toLowerCase().trim();
    const filterValue = document.getElementById("filterSelect").value;
    const rows = document.querySelectorAll("#reviewTableBody .review-row");

    rows.forEach(row => {
        const text = row.querySelector(".review-full-text").value.toLowerCase();
        const sentiment = row.getAttribute("data-sentiment");

        const matchesSearch = !query || text.includes(query);
        const matchesFilter = filterValue === "all" || sentiment === filterValue;

        row.style.display = (matchesSearch && matchesFilter) ? "" : "none";
    });
}

/* -----------------------------------------------------------------------
   Toast Notification
   ----------------------------------------------------------------------- */
function showToast(message) {
    const toastEl = document.getElementById("liveToast");
    const bodyEl = document.getElementById("toastBody");
    bodyEl.textContent = message;
    const toast = bootstrap.Toast.getOrCreateInstance(toastEl, { delay: 3000 });
    toast.show();
}

/* -----------------------------------------------------------------------
   Add New Review
   ----------------------------------------------------------------------- */
function initAddReview() {
    const btn = document.getElementById("predictNewReviewBtn");
    btn.addEventListener("click", addNewReview);

    // Reset modal state when it closes
    const modal = document.getElementById("addReviewModal");
    modal.addEventListener("hidden.bs.modal", () => {
        document.getElementById("newReviewText").value = "";
        document.getElementById("newReviewError").classList.add("d-none");
    });

    // Clear error when user starts typing
    document.getElementById("newReviewText").addEventListener("input", () => {
        document.getElementById("newReviewError").classList.add("d-none");
    });
}

async function addNewReview() {
    const textarea = document.getElementById("newReviewText");
    const errorEl = document.getElementById("newReviewError");
    const btn = document.getElementById("predictNewReviewBtn");
    const labelSpan = btn.querySelector(".btn-label");
    const spinnerSpan = btn.querySelector(".spinner-border");
    const reviewText = textarea.value.trim();

    // Validate
    if (!reviewText) {
        errorEl.classList.remove("d-none");
        textarea.focus();
        return;
    }

    // Show spinner
    btn.disabled = true;
    labelSpan.classList.add("d-none");
    spinnerSpan.classList.remove("d-none");

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ review: reviewText }),
        });
        const data = await res.json();

        // Build new row
        const id = nextReviewId++;
        const truncated = reviewText.length > 80 ? reviewText.substring(0, 80) + "…" : reviewText;
        const sentiment = data.sentiment;
        const confidence = data.confidence;

        const badgeClass = sentiment === "Positive" ? "badge-positive"
                         : sentiment === "Negative" ? "badge-negative"
                         : "badge-neutral";
        const badgeLabel = sentiment === "Positive" ? "🟢 Positive"
                         : sentiment === "Negative" ? "🔴 Negative"
                         : "⚪ Neutral";
        const fillClass = sentiment === "Positive" ? "fill-positive"
                        : sentiment === "Negative" ? "fill-negative"
                        : "fill-neutral";

        const tr = document.createElement("tr");
        tr.className = "review-row";
        tr.setAttribute("data-id", id);
        tr.setAttribute("data-sentiment", sentiment);
        tr.style.animationDelay = "0ms";

        tr.innerHTML = `
            <td class="col-num">${id}</td>
            <td class="col-text">
                <span class="review-text-truncated" data-bs-toggle="tooltip" data-bs-placement="top" title="${reviewText.replace(/"/g, '&quot;')}">${truncated}</span>
                <input type="hidden" class="review-full-text" value="${reviewText.replace(/"/g, '&quot;')}">
            </td>
            <td class="col-sentiment">
                <span class="badge ${badgeClass}">${badgeLabel}</span>
            </td>
            <td class="col-confidence">
                <div class="confidence-bar-wrapper">
                    <div class="confidence-track">
                        <div class="confidence-fill ${fillClass}" style="width: 0"></div>
                    </div>
                    <span class="confidence-pct">${confidence}%</span>
                </div>
            </td>
            <td class="col-action">
                <button class="btn btn-sm btn-predict btn-done" disabled>✅</button>
            </td>`;

        // Append to table
        document.getElementById("reviewTableBody").appendChild(tr);

        // Init tooltip on the new row
        const tooltipEl = tr.querySelector('[data-bs-toggle="tooltip"]');
        if (tooltipEl) new bootstrap.Tooltip(tooltipEl);

        // Animate confidence bar
        requestAnimationFrame(() => {
            const fill = tr.querySelector(".confidence-fill");
            if (fill) fill.style.width = confidence + "%";
        });

        // Update summary cards
        updateSummaryCards(true);

        // Close modal
        const modal = bootstrap.Modal.getInstance(document.getElementById("addReviewModal"));
        modal.hide();

        // Scroll to new row
        tr.scrollIntoView({ behavior: "smooth", block: "center" });

        // Toast
        showToast(`✅ Review #${id} added — ${sentiment} (${confidence}%)`);

    } catch (err) {
        console.error("Add review error:", err);
        showToast("❌ Something went wrong. Please try again.");
    } finally {
        btn.disabled = false;
        labelSpan.classList.remove("d-none");
        spinnerSpan.classList.add("d-none");
    }
}
