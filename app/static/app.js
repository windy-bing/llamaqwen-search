const cardEl = document.querySelector("#daily-card");
const drawButton = document.querySelector("#draw-card");
const form = document.querySelector("#ask-form");
const questionEl = document.querySelector("#question");
const answerEl = document.querySelector("#answer");

const clientId = getClientId();
let currentCardId = null;

function getClientId() {
  const key = "llamaqwen-search-client-id";
  let value = localStorage.getItem(key);
  if (!value) {
    value = crypto.randomUUID();
    localStorage.setItem(key, value);
  }
  return value;
}

function renderCard(card) {
  currentCardId = card.id;
  cardEl.classList.remove("is-changing");
  cardEl.innerHTML = `
    <div class="card-title">${escapeHtml(card.title)}</div>
    <div class="card-text">${escapeHtml(card.text)}</div>
    <div class="card-source">
      来源 · ${escapeHtml(card.source)}<br />
      ${escapeHtml(card.reference)}
    </div>
    <div class="card-action">${escapeHtml(card.action)}</div>
  `;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function loadTodayCard() {
  const response = await fetch("/api/card/today", {
    headers: { "X-Client-Id": clientId },
  });
  if (!response.ok) throw new Error("今日卡片加载失败");
  const data = await response.json();
  renderCard(data.card);
}

drawButton.addEventListener("click", async () => {
  drawButton.disabled = true;
  cardEl.classList.add("is-changing");
  try {
    const url = new URL("/api/card/draw", window.location.origin);
    if (currentCardId) url.searchParams.set("exclude_id", currentCardId);

    const response = await fetch(url, { method: "POST" });
    if (!response.ok) throw new Error("抽卡失败");
    const data = await response.json();
    window.setTimeout(() => renderCard(data.card), 120);
  } catch (error) {
    cardEl.classList.remove("is-changing");
    cardEl.innerHTML = `<span class="muted">${escapeHtml(error.message || "抽卡失败")}</span>`;
  } finally {
    window.setTimeout(() => {
      drawButton.disabled = false;
    }, 140);
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionEl.value.trim();
  if (!question) return;

  answerEl.textContent = "检索中...";
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await response.json();

  if (!response.ok) {
    answerEl.textContent = data.detail || "请求失败";
    return;
  }

  const sources = (data.sources || [])
    .map((source) => {
      const name = [source.file, source.page ? `第 ${source.page} 页` : ""].filter(Boolean).join(" · ");
      return `<li>${escapeHtml(name || "未知来源")}<br />${escapeHtml(source.excerpt || "")}</li>`;
    })
    .join("");

  answerEl.innerHTML = `
    <div>${escapeHtml(data.answer).replaceAll("\n", "<br />")}</div>
    ${sources ? `<ol class="source-list">${sources}</ol>` : ""}
  `;
});

loadTodayCard().catch(() => {
  cardEl.innerHTML = '<span class="muted">卡片加载失败</span>';
});
