// sol_chat 웹 화면. 빌드 도구 없이 브라우저에서 바로 동작한다.
"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  login: $("login"), loginForm: $("login-form"), loginCode: $("login-code"), loginError: $("login-error"),
  app: $("app"), sidebar: $("sidebar"), backdrop: $("backdrop"), menu: $("menu"),
  newChat: $("new-chat"), sessionList: $("session-list"), logout: $("logout"),
  title: $("chat-title"), modelSelect: $("model-select"),
  messages: $("messages"), empty: $("empty"),
  composer: $("composer"), input: $("input"), send: $("send"),
};

// ---- 브라우저 저장소 (비공개 창 등에서 막혀도 동작하도록) ----
const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch {} },
  remove(key) { try { localStorage.removeItem(key); } catch {} },
};

function makeClientId() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, "0")).join("");
}

const state = {
  accessCode: storage.get("sol_chat.access_code") || "",
  clientId: storage.get("sol_chat.client_id") || makeClientId(),
  sessionId: null,
  models: [],
  defaultModel: null,
  busy: false,
};
storage.set("sol_chat.client_id", state.clientId);

// ---- API ----
class ApiError extends Error {
  constructor(status, detail) { super(detail); this.status = status; }
}

function headers() {
  return { "Content-Type": "application/json", "X-Access-Code": state.accessCode, "X-Client-Id": state.clientId };
}

async function api(path, options = {}) {
  const res = await fetch(path, { ...options, headers: headers() });
  if (!res.ok) throw new ApiError(res.status, await errorDetail(res));
  return res.json();
}

async function errorDetail(res) {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return "요청 형식이 올바르지 않아요.";
  } catch {}
  return `오류가 났어요 (HTTP ${res.status}).`;
}

// SSE 스트림 읽기: "event: x\ndata: {...}\n\n" 블록 단위.
// 서버는 항상 done 또는 error 로 끝낸다. 둘 다 없이 연결이 닫히면 답변이 중간에 끊긴 것.
async function streamChat(body, onEvent) {
  const res = await fetch("/api/chat/stream", { method: "POST", headers: headers(), body: JSON.stringify(body) });
  if (!res.ok) throw new ApiError(res.status, await errorDetail(res));
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;
      if (event === "done" || event === "error") finished = true;
      onEvent(event, JSON.parse(data));
    }
  }
  if (!finished) throw new IncompleteStream();
}

class IncompleteStream extends Error {
  constructor() { super("답변이 중간에 끊겼어요. 서버에 저장된 대화를 다시 불러왔어요."); }
}

// ---- 화면: 로그인 ----
function showLogin(message) {
  els.app.hidden = true;
  els.login.hidden = false;
  els.loginError.hidden = !message;
  els.loginError.textContent = message || "";
  els.loginCode.value = "";
  els.loginCode.focus();
}

async function tryEnter() {
  try {
    await api("/api/auth/check");
  } catch (e) {
    if (e.status === 401) {
      showLogin(state.accessCode ? "비밀번호가 올바르지 않아요." : "");
      return;
    }
    showLogin(e.message);
    return;
  }
  storage.set("sol_chat.access_code", state.accessCode);
  els.login.hidden = true;
  els.app.hidden = false;
  await loadModels();
  newChat(); // 대화 목록도 함께 불러옴
}

els.loginForm.addEventListener("submit", (ev) => {
  ev.preventDefault();
  state.accessCode = els.loginCode.value.trim();
  tryEnter();
});

els.logout.addEventListener("click", () => {
  storage.remove("sol_chat.access_code");
  state.accessCode = "";
  showLogin("");
});

// ---- 화면: 모델 / 대화 목록 ----
async function loadModels() {
  const data = await api("/api/models");
  state.models = data.models;
  state.defaultModel = data.default_model;
  els.modelSelect.innerHTML = "";
  for (const m of data.models) {
    const opt = document.createElement("option");
    opt.value = m.name;
    opt.textContent = m.label;
    els.modelSelect.append(opt);
  }
  els.modelSelect.value = storage.get("sol_chat.model") || data.default_model;
  if (!els.modelSelect.value) els.modelSelect.value = data.default_model;
}

els.modelSelect.addEventListener("change", () => storage.set("sol_chat.model", els.modelSelect.value));

async function refreshSessions() {
  let sessions = [];
  try { sessions = await api("/api/sessions"); } catch (e) { return handleError(e); }
  els.sessionList.innerHTML = "";
  for (const s of sessions) {
    const li = document.createElement("li");
    li.className = "session-item" + (s.id === state.sessionId ? " active" : "");
    const title = document.createElement("span");
    title.className = "title";
    title.textContent = s.title;
    const del = document.createElement("button");
    del.className = "btn-ghost delete";
    del.type = "button";
    del.textContent = "✕";
    del.title = "대화 삭제";
    del.addEventListener("click", (ev) => { ev.stopPropagation(); deleteSession(s.id); });
    li.append(title, del);
    li.addEventListener("click", () => openSession(s.id));
    els.sessionList.append(li);
  }
}

async function openSession(id) {
  if (state.busy) return;
  if (await renderSession(id)) {
    closeSidebar();
    refreshSessions();
  }
}

// 서버에 저장된 대화 내용으로 화면을 다시 그린다
async function renderSession(id) {
  let detail;
  try { detail = await api(`/api/sessions/${id}`); } catch (e) { handleError(e); return false; }
  state.sessionId = id;
  els.title.textContent = detail.title;
  if (state.models.some((m) => m.name === detail.model)) els.modelSelect.value = detail.model;
  clearMessages();
  for (const m of detail.messages) addMessage(m.role, m.content);
  return true;
}

async function deleteSession(id) {
  if (!confirm("이 대화를 삭제할까요?")) return;
  try { await api(`/api/sessions/${id}`, { method: "DELETE" }); } catch (e) { return handleError(e); }
  if (id === state.sessionId) newChat();
  refreshSessions();
}

function newChat() {
  if (state.busy) return;
  state.sessionId = null;
  els.title.textContent = "새 대화";
  clearMessages();
  closeSidebar();
  refreshSessions();
  els.input.focus();
}

els.newChat.addEventListener("click", newChat);

// ---- 화면: 메시지 ----
function clearMessages() {
  els.messages.replaceChildren(els.empty);
  els.empty.hidden = false;
}

function addMessage(role, text) {
  els.empty.hidden = true;
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text; // textContent 라서 HTML 이 실행되지 않음
  els.messages.append(div);
  scrollToBottom();
  return div;
}

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function handleError(e) {
  if (e instanceof ApiError && e.status === 401) {
    showLogin("비밀번호를 다시 입력해주세요.");
    return;
  }
  addMessage("error", e.message || "알 수 없는 오류가 났어요.");
}

function setBusy(busy) {
  state.busy = busy;
  els.send.disabled = busy;
  els.modelSelect.disabled = busy;
}

async function sendMessage() {
  const text = els.input.value.trim();
  if (!text || state.busy) return;
  els.input.value = "";
  autoResize();
  addMessage("user", text);
  const bubble = addMessage("assistant", "");
  bubble.classList.add("pending");
  setBusy(true);

  let failed = false;
  try {
    await streamChat({ session_id: state.sessionId, message: text, model: els.modelSelect.value }, (event, data) => {
      if (event === "session") {
        state.sessionId = data.session_id;
      } else if (event === "token") {
        bubble.textContent += data.text;
        scrollToBottom();
      } else if (event === "done") {
        els.title.textContent = data.title;
      } else if (event === "error") {
        failed = true;
        bubble.remove();
        addMessage("error", data.detail);
      }
    });
  } catch (e) {
    failed = true;
    if (e instanceof IncompleteStream && state.sessionId) {
      // 화면에 보이는 내용과 서버 저장 내용이 다를 수 있으므로 서버 기준으로 맞춘다
      await renderSession(state.sessionId);
      addMessage("error", e.message);
    } else {
      bubble.remove();
      handleError(e);
      if (e.status === 404) state.sessionId = null; // 서버 재시작 등으로 대화가 사라진 경우
    }
  } finally {
    bubble.classList.remove("pending");
    setBusy(false);
    if (!failed && !bubble.textContent) bubble.textContent = "(빈 응답)";
    refreshSessions();
    els.input.focus();
  }
}

els.composer.addEventListener("submit", (ev) => { ev.preventDefault(); sendMessage(); });
els.input.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); sendMessage(); }
});

function autoResize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 180) + "px";
}
els.input.addEventListener("input", autoResize);

// ---- 휴대폰 사이드바 ----
function closeSidebar() { els.sidebar.classList.remove("open"); els.backdrop.hidden = true; }
els.menu.addEventListener("click", () => { els.sidebar.classList.add("open"); els.backdrop.hidden = false; });
els.backdrop.addEventListener("click", closeSidebar);

tryEnter();
