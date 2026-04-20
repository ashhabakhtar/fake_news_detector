/**
 * app.js — AI News Verifier
 * Gemini-style chat interface with dark/light mode, history, and animated results.
 */

/* ══════════════════════════════════════════════════════════════════════════════
   STATE
══════════════════════════════════════════════════════════════════════════════ */
const state = {
  theme:       localStorage.getItem('nv-theme') || 'light',
  sidebarOpen: false,
  isAnalyzing: false,
  history:     JSON.parse(localStorage.getItem('nv-history') || '[]'),
};

/* ══════════════════════════════════════════════════════════════════════════════
   ELEMENT CACHE
══════════════════════════════════════════════════════════════════════════════ */
const $ = (id) => document.getElementById(id);

const el = {
  html:            document.documentElement,
  sidebar:         $('sidebar'),
  overlay:         $('sidebarOverlay'),
  menuBtn:         $('menuBtn'),
  closeSidebarBtn:  $('closeSidebarBtn'),
  themeToggle:     $('themeToggle'),
  welcomeScreen:   $('welcomeScreen'),
  chatArea:        $('chatArea'),
  newsInput:       $('newsInput'),
  sendBtn:         $('sendBtn'),
  charCount:       $('charCount'),
  historyList:     $('historyList'),
  newChatBtn:      $('newChatBtn'),
  sampleRealBtn:   $('sampleRealBtn'),
  sampleFakeBtn:   $('sampleFakeBtn'),
};

/* ══════════════════════════════════════════════════════════════════════════════
   THEME
══════════════════════════════════════════════════════════════════════════════ */
function setTheme(t) {
  state.theme = t;
  el.html.setAttribute('data-theme', t);
  localStorage.setItem('nv-theme', t);
}
function toggleTheme() {
  setTheme(state.theme === 'light' ? 'dark' : 'light');
}

/* ══════════════════════════════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════════════════════════════ */
function openSidebar() {
  state.sidebarOpen = true;
  el.sidebar.classList.add('open');
  el.overlay.classList.add('active');
}
function closeSidebar() {
  state.sidebarOpen = false;
  el.sidebar.classList.remove('open');
  el.overlay.classList.remove('active');
}

/* ══════════════════════════════════════════════════════════════════════════════
   TEXTAREA HELPERS
══════════════════════════════════════════════════════════════════════════════ */
function autoResize(ta) {
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
}

function updateInputState() {
  const len = el.newsInput.value.length;
  el.charCount.textContent = `${len.toLocaleString()} / 10,000`;
  const ready = el.newsInput.value.trim().length > 5 && !state.isAnalyzing;
  el.sendBtn.disabled = !ready;
}

/* ══════════════════════════════════════════════════════════════════════════════
   HISTORY
══════════════════════════════════════════════════════════════════════════════ */
function saveHistory(text, isReal) {
  const item = {
    id:      Date.now(),
    snippet: text.slice(0, 55) + (text.length > 55 ? '…' : ''),
    isReal,
    ts:      new Date().toISOString(),
  };
  state.history.unshift(item);
  state.history = state.history.slice(0, 25);
  localStorage.setItem('nv-history', JSON.stringify(state.history));
  renderHistory();
}

function renderHistory() {
  if (!state.history.length) {
    el.historyList.innerHTML = '<p class="no-history">No analyses yet</p>';
    return;
  }
  el.historyList.innerHTML = state.history.map((h) =>
    `<div class="history-item">
       <span class="history-icon">${h.isReal ? '✅' : '⚠️'}</span>
       <span class="history-text">${escHtml(h.snippet)}</span>
     </div>`
  ).join('');
}

/* ══════════════════════════════════════════════════════════════════════════════
   WELCOME / CHAT TOGGLE
══════════════════════════════════════════════════════════════════════════════ */
function showWelcome(show) {
  el.welcomeScreen.classList.toggle('hidden', !show);
}

/* ══════════════════════════════════════════════════════════════════════════════
   MESSAGE RENDERERS
══════════════════════════════════════════════════════════════════════════════ */
function addUserMessage(text) {
  showWelcome(false);
  const snippet = text.length > 220 ? text.slice(0, 220) + '…' : text;
  const div = createElement(
    `<div class="message msg-user">
       <div class="user-bubble">${escHtml(snippet)}</div>
     </div>`
  );
  el.chatArea.appendChild(div);
  scrollToBottom(div);
}

function addTypingIndicator() {
  const div = createElement(
    `<div class="message msg-ai" id="typingMsg">
       <div class="ai-avatar" aria-hidden="true">🕵️</div>
       <div class="typing-indicator">
         <div class="typing-dots">
           <div class="dot"></div>
           <div class="dot"></div>
           <div class="dot"></div>
         </div>
         <span class="typing-text">Analyzing article…</span>
       </div>
     </div>`
  );
  el.chatArea.appendChild(div);
  scrollToBottom(div);
}

function removeTypingIndicator() {
  const el2 = $('typingMsg');
  if (el2) el2.remove();
}

function addResultCard(data) {
  const R = 22;
  const C = 2 * Math.PI * R;          // ≈ 138.23
  const offset = C * (1 - data.confidence / 100);

  const isReal = data.is_real;
  const conf   = data.confidence.toFixed(1);
  const sub    = data.subjectivity;
  const sent   = data.sentiment;

  // Subjectivity
  const subWidth = Math.round(sub * 100);
  const subHigh  = sub > 0.5;
  const subClass = subHigh ? 'danger' : 'success';
  const subNote  = subHigh
    ? '🚩 Sounds more like opinion than a factual report'
    : '✅ Writing appears neutral and objective';

  // Sentiment
  const sentAbs   = Math.abs(sent);
  const sentWidth  = Math.round(sentAbs * 100);
  const sentNote   = sent >  0.25 ? '🟢 Noticeably positive tone'
                   : sent < -0.25 ? '🔴 Noticeably negative tone'
                   : '⚪ Relatively neutral tone';

  const cardHtml =
    `<div class="message msg-ai">
       <div class="ai-avatar" aria-hidden="true">🕵️</div>
       <div class="result-card">

         <!-- HEADER -->
         <div class="card-header">
           <div class="verdict-badge">
             <div class="verdict-icon ${isReal ? 'real' : 'fake'}">
               ${isReal ? '✅' : '⚠️'}
             </div>
             <div>
               <div class="verdict-label">AI Verdict</div>
               <div class="verdict-value ${isReal ? 'real' : 'fake'}">
                 ${isReal ? 'Likely Real News' : 'Likely Fake / Misleading'}
               </div>
             </div>
           </div>
           <div class="confidence-group">
             <div class="confidence-meta">
               <div class="conf-title">${isReal ? 'Trust' : 'Risk'} Score</div>
               <div class="conf-score">${conf}%</div>
             </div>
             <div class="ring-wrap">
               <svg viewBox="0 0 58 58" aria-hidden="true">
                 <circle class="ring-bg" cx="29" cy="29" r="${R}"/>
                 <circle class="ring-fill ${isReal ? 'real' : 'fake'}"
                         cx="29" cy="29" r="${R}"
                         stroke-dasharray="${C.toFixed(2)}"
                         stroke-dashoffset="${C.toFixed(2)}"/>
               </svg>
               <div class="ring-label">${Math.round(data.confidence)}%</div>
             </div>
           </div>
         </div>

         <!-- BODY -->
         <div class="card-body">
           <div class="metric-tile">
             <div class="tile-label">Subjectivity</div>
             <div class="tile-value">${sub.toFixed(2)}</div>
             <div class="tile-bar">
               <div class="tile-bar-fill ${subClass}" data-w="${subWidth}"></div>
             </div>
             <div class="tile-note">${subNote}</div>
           </div>
           <div class="metric-tile">
             <div class="tile-label">Sentiment Tone</div>
             <div class="tile-value">${sent >= 0 ? '+' : ''}${sent.toFixed(2)}</div>
             <div class="tile-bar">
               <div class="tile-bar-fill accent" data-w="${sentWidth}"></div>
             </div>
             <div class="tile-note">${sentNote}</div>
             <div class="tile-hint">Scale: −1 (Negative) → 0 (Neutral) → +1 (Positive)</div>
           </div>
         </div>

         <!-- FOOTER -->
         <div class="card-footer">
           <span class="word-tally">📝 ${data.word_count} words analyzed</span>
           <button class="copy-btn" data-real="${isReal}" data-conf="${conf}">Copy result</button>
         </div>

       </div>
     </div>`;

  const div = createElement(cardHtml);
  el.chatArea.appendChild(div);

  // Animate ring + bars after two rAF ticks (ensures transition fires)
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      const ring = div.querySelector('.ring-fill');
      if (ring) ring.style.strokeDashoffset = offset.toFixed(2);

      div.querySelectorAll('.tile-bar-fill[data-w]').forEach((bar) => {
        bar.style.width = bar.dataset.w + '%';
      });
    });
  });

  scrollToBottom(div);
}

function addErrorCard(msg) {
  const div = createElement(
    `<div class="message msg-ai">
       <div class="ai-avatar" aria-hidden="true">🕵️</div>
       <div class="error-card">
         <span class="error-icon">⚠️</span>
         <div>
           <div class="error-title">Analysis failed</div>
           <div class="error-body">${escHtml(msg)}</div>
         </div>
       </div>
     </div>`
  );
  el.chatArea.appendChild(div);
  scrollToBottom(div);
}

/* ══════════════════════════════════════════════════════════════════════════════
   ANALYZE
══════════════════════════════════════════════════════════════════════════════ */
async function analyze() {
  const text = el.newsInput.value.trim();
  if (!text || state.isAnalyzing) return;

  state.isAnalyzing = true;
  el.sendBtn.disabled = true;

  // Show user bubble and clear input
  addUserMessage(text);
  el.newsInput.value = '';
  autoResize(el.newsInput);
  updateInputState();

  // Show typing
  addTypingIndicator();

  try {
    const res  = await fetch('/api/analyze', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
    });
    const data = await res.json();

    await delay(200); // brief pause for visual smoothness
    removeTypingIndicator();

    if (res.ok) {
      addResultCard(data);
      saveHistory(text, data.is_real);
    } else {
      addErrorCard(data.error || 'Something went wrong. Please try again.');
    }
  } catch {
    removeTypingIndicator();
    addErrorCard('Could not reach the analysis server. Make sure backend.py is running on port 5000.');
  } finally {
    state.isAnalyzing = false;
    updateInputState();
  }
}

/* ══════════════════════════════════════════════════════════════════════════════
   COPY RESULT (event delegation)
══════════════════════════════════════════════════════════════════════════════ */
el.chatArea.addEventListener('click', (e) => {
  const btn = e.target.closest('.copy-btn');
  if (!btn) return;

  const isReal = btn.dataset.real === 'true';
  const conf   = btn.dataset.conf;
  const text   =
    `AI News Verifier Result\n` +
    `Verdict: ${isReal ? 'Likely Real News' : 'Likely Fake / Misleading'}\n` +
    `Confidence: ${conf}%`;

  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = '✓ Copied!';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.textContent = 'Copy result';
      btn.classList.remove('copied');
    }, 2200);
  });
});

/* ══════════════════════════════════════════════════════════════════════════════
   SAMPLE CONTENT
══════════════════════════════════════════════════════════════════════════════ */
const SAMPLES = {
  real: `The Federal Reserve announced today that it will hold interest rates steady at their current level, citing stable economic indicators and a continued gradual decline in inflation. Fed Chair Jerome Powell reiterated the central bank's commitment to its 2% inflation target while noting that the labor market remains resilient. The decision was in line with expectations from most economists and financial analysts following recent employment and consumer price data.`,
  fake: `BREAKING!!! Scientists have CONFIRMED that drinking lemon water CURES cancer overnight!! Big Pharma is DESPERATELY trying to hide this SHOCKING discovery from the public!!! Share this before they DELETE it!!! Over 1 MILLION people have already been CURED using this incredible method doctors DON'T want you to know about!! Click NOW to learn the SECRET!!!`,
};

function loadSample(type) {
  el.newsInput.value = SAMPLES[type];
  autoResize(el.newsInput);
  updateInputState();
  el.newsInput.focus();
}

/* ══════════════════════════════════════════════════════════════════════════════
   NEW CHAT
══════════════════════════════════════════════════════════════════════════════ */
function newChat() {
  el.chatArea.innerHTML = '';
  showWelcome(true);
  closeSidebar();
  el.newsInput.focus();
}

/* ══════════════════════════════════════════════════════════════════════════════
   UTILS
══════════════════════════════════════════════════════════════════════════════ */
function escHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function createElement(html) {
  const tmp = document.createElement('div');
  tmp.innerHTML = html.trim();
  return tmp.firstElementChild;
}

function scrollToBottom(el2) {
  el2.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

/* ══════════════════════════════════════════════════════════════════════════════
   INIT
══════════════════════════════════════════════════════════════════════════════ */
function init() {
  // Apply saved theme
  setTheme(state.theme);

  // Render history sidebar
  renderHistory();

  // Theme
  el.themeToggle.addEventListener('click', toggleTheme);

  // Sidebar
  el.menuBtn.addEventListener('click', openSidebar);
  el.closeSidebarBtn.addEventListener('click', closeSidebar);
  el.overlay.addEventListener('click', closeSidebar);
  el.newChatBtn.addEventListener('click', newChat);

  // Samples
  el.sampleRealBtn.addEventListener('click', () => loadSample('real'));
  el.sampleFakeBtn.addEventListener('click', () => loadSample('fake'));

  // Input
  el.newsInput.addEventListener('input', () => {
    autoResize(el.newsInput);
    updateInputState();
  });

  el.newsInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (!el.sendBtn.disabled) analyze();
    }
  });

  el.sendBtn.addEventListener('click', analyze);

  // Initial state
  updateInputState();
}

document.addEventListener('DOMContentLoaded', init);
