# app.py
import asyncio, os
from typing import List, Dict, Any
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8002"))

BOT_HOST = os.getenv("BOT_HOST", "127.0.0.1")
BOT_PORT = int(os.getenv("BOT_PORT", "9000"))
BOT_UPDATE_URL = f"http://{BOT_HOST}:{BOT_PORT}/update"

PUSH_INTERVAL_SEC = float(os.getenv("PUSH_INTERVAL_SEC", "0.2"))  # 5x/Sek

app = FastAPI(title="KI-NPC Mini-Map")

# NPC fix auf (10,10); nur Spieler bewegbar
state = {
    "npcs": [
        {"id": "npc1", "name": "Moderator", "x": 10.0, "y": 10.0},
        {"id": "npc2", "name": "NPC 1", "x": 14.0, "y": 10.0},
        {"id": "npc3", "name": "NPC 2", "x": 18.0, "y": 10.0},
    ],
    "players": [{"id": "p1", "name": "Spieler1", "x": 18.0, "y": 14.0}],
}
_last_push_ok = True
_last_push_msg = "noch nix"
_push_enabled = True

if not os.path.exists("static"): os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health") 
def health(): 
    return {"ok": True, "app": "minimap"}

@app.get("/config")
def cfg():
    return {"bot_update_url": BOT_UPDATE_URL,
            "push_interval_sec": PUSH_INTERVAL_SEC,
            "push_enabled": _push_enabled}

@app.get("/state_mini")
def state_mini():
    return {"npcs": state["npcs"], "players": state["players"],
            "push_enabled": _push_enabled,
            "last_push_ok": _last_push_ok, "last_push_msg": _last_push_msg}

@app.post("/move")
async def move(req: Request):
    body = await req.json()
    who = body.get("who"); x = float(body.get("x",0)); y=float(body.get("y",0))
    if who and who.startswith("npc"):
        for npc in state["npcs"]:
            if npc.get("id") == who:
                npc.update({"x": x, "y": y})
                break
    else:
        # Spieler anhand der ID suchen, fallback: erster Spieler
        pid = body.get("player_id") or (state["players"][0]["id"] if state["players"] else None)
        if pid:
            for player in state["players"]:
                if player.get("id") == pid:
                    player.update({"x": x, "y": y})
                    break
    return {"ok": True}

@app.post("/push_once")
async def push_once():
    ok, msg = await _push_positions()
    return {"ok": ok, "msg": msg}

@app.post("/toggle_push")
def toggle_push():
    global _push_enabled; _push_enabled = not _push_enabled
    return {"ok": True, "push_enabled": _push_enabled}

@app.get("/", response_class=HTMLResponse)
def index():
    html = """<!doctype html><html><head><meta charset="utf-8"/><title>Mini-Map</title>
<style>
body { font-family: ui-sans-serif, system-ui; margin:0; display:flex; height:100vh; color:#e5e7eb; }
#left { flex:1; display:flex; align-items:center; justify-content:center; background:#0b1220; padding:16px; box-sizing:border-box; }
#right { width:420px; background:#111827; padding:16px; overflow:auto; box-sizing:border-box; }
canvas { background:#0f172a; border:1px solid #374151; border-radius:8px; box-shadow:0 10px 30px rgba(0,0,0,0.35); }
button { margin-right:8px; margin-bottom:8px; background:#2563eb; border:none; color:#e5e7eb; padding:6px 10px; border-radius:6px; cursor:pointer; transition:background 0.2s ease; }
button.secondary { background:#374151; }
button:disabled { opacity:0.55; cursor:not-allowed; }
.row { margin:12px 0; }
.row h3 { margin:0 0 6px 0; font-size:1rem; color:#93c5fd; }
code { background:#1f2937; padding:2px 6px; border-radius:4px; display:inline-block; margin-top:4px; }
label { display:block; font-size:0.85rem; margin-bottom:4px; color:#9ca3af; }
select, textarea { width:100%; padding:6px; border-radius:6px; background:#1f2937; color:#e5e7eb; border:1px solid #374151; margin-bottom:6px; font-family:inherit; box-sizing:border-box; }
textarea { resize:vertical; }
small { color:#9ca3af; }
ul { list-style:none; padding-left:16px; margin:0; }
li { margin:2px 0; font-size:0.9rem; }
.status { font-size:0.85rem; color:#9ca3af; }
.button-row { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:6px; }
.npc-prompts { display:flex; flex-direction:column; gap:10px; }
.prompt-block { background:#151c2f; border:1px solid #1f2937; border-radius:8px; padding:8px; }
.prompt-block label { font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em; color:#c7d2fe; margin-bottom:4px; }
#conversationLog { background:#0f172a; border:1px solid #1f2937; border-radius:8px; padding:10px; max-height:240px; overflow-y:auto; font-size:0.9rem; }
.log-entry { margin-bottom:6px; line-height:1.4; }
.log-entry:last-child { margin-bottom:0; }
.log-entry strong { color:#bfdbfe; }
#finalSolution { margin-top:10px; padding:10px; border-radius:8px; border:1px solid rgba(37,99,235,0.5); background:#1d283a; color:#facc15; font-weight:600; display:none; }
#finalSolution.visible { display:block; }
.small-note { font-size:0.8rem; color:#6b7280; margin-bottom:6px; }
</style></head><body>
<div id="left"><canvas id="map" width="600" height="420"></canvas></div>
<div id="right">
  <h2>Mini-Map</h2>
  <div class="row">Bot Update URL:<br><code id="cfg"></code></div>
  <div class="row"><button id="toggle">Push an/aus</button><button id="pushonce" class="secondary">Push einmal</button></div>
  <div class="row"><div>Letzter Push: <span id="last"></span></div><div>Push aktiv: <span id="active"></span></div></div>
  <div class="row">
    <h3>Gesprächssteuerung</h3>
    <div class="button-row">
      <button id="startConversation">Start</button>
      <button id="stopConversation" class="secondary" disabled>Stop</button>
      <button id="resetConversation" class="secondary">Refresh</button>
    </div>
    <label for="taskDescription">Aufgabenstellung</label>
    <textarea id="taskDescription" rows="3" placeholder="Beschreibe die Aufgabe, die moderiert werden soll..."></textarea>
    <div class="small-note">Trage hier die Anforderungen und die Persönlichkeit der NPCs als System-Prompt ein.</div>
    <div class="npc-prompts">
      <div class="prompt-block">
        <label for="prompt-moderator">Moderator (NPC 1)</label>
        <textarea id="prompt-moderator" data-npc-prompt="npc1" rows="3">Ruhig, strukturiert und wertschätzend. Fasse Ergebnisse klar zusammen und achte auf einen respektvollen Ton.</textarea>
      </div>
      <div class="prompt-block">
        <label for="prompt-npc2">NPC 1</label>
        <textarea id="prompt-npc2" data-npc-prompt="npc2" rows="3">Analytisch, detailverliebt und lösungsorientiert. Denkt logisch und mag Schritt-für-Schritt-Pläne.</textarea>
      </div>
      <div class="prompt-block">
        <label for="prompt-npc3">NPC 2</label>
        <textarea id="prompt-npc3" data-npc-prompt="npc3" rows="3">Kreativ, intuitiv und empathisch. Bringt frische Ideen ein und achtet auf zwischenmenschliche Aspekte.</textarea>
      </div>
    </div>
  </div>
  <div class="row">
    <h3>Gesprächsverlauf</h3>
    <div id="conversationLog"></div>
    <div id="finalSolution"></div>
  </div>
  <div class="row">
    <h3>Entfernungen (Meter)</h3>
    <ul id="distances"></ul>
    <div class="status">Aktueller Radius für Gespräche: <span id="radius">4.0</span> m</div>
  </div>
  <div class="row">
    <h3>Audio-Routing</h3>
    <label for="micSelect">Mikrofon (Input)</label>
    <select id="micSelect"></select>
    <label for="speakerSelect">Lautsprecher (Output)</label>
    <select id="speakerSelect"></select>
    <button id="refreshDevices" class="secondary">Geräteliste aktualisieren</button>
    <div class="status" id="deviceStatus"></div>
  </div>
  <div class="row">
    <h3>Debug</h3>
    <a href="http://__BOT_HOST__:__BOT_PORT__/state" target="_blank">Bot /state</a><br/>
    <a href="http://__BOT_HOST__:__BOT_PORT__/log" target="_blank">Bot /log</a><br/>
    <a href="http://__BOT_HOST__:__BOT_PORT__/devices" target="_blank">Bot /devices</a>
  </div>
  <div class="row"><small>Ziehe die Marker für Moderator (cyan), NPC&nbsp;1 (gelb) und NPC&nbsp;2 (grün) sowie Spieler (lila), um Positionen festzulegen.</small></div>
</div>
<script>
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const SCALE = 12;
const BOT_BASE = `http://__BOT_HOST__:__BOT_PORT__`;
const NPC_COLORS = ['#38bdf8', '#fbbf24', '#34d399'];
const PLAYER_COLOR = '#a78bfa';
const DEFAULT_NPCS = [
  { id: 'npc1', name: 'Moderator', x: 10.0, y: 10.0 },
  { id: 'npc2', name: 'NPC 1', x: 14.0, y: 10.0 },
  { id: 'npc3', name: 'NPC 2', x: 18.0, y: 10.0 }
];
const DEFAULT_PLAYERS = [
  { id: 'p1', name: 'Spieler1', x: 18.0, y: 14.0 }
];

let npcs = DEFAULT_NPCS.map((npc) => ({ ...npc }));
let players = DEFAULT_PLAYERS.map((player) => ({ ...player }));
let greetRadius = 4.0;
let dragging = null;
let suspendRefresh = false;

const conversationState = {
  running: false,
  stopRequested: false,
  clearOnStop: false,
  solutions: {},
  compromiseRound: 0,
  finalSolution: ''
};

class ConversationAbort extends Error {
  constructor() {
    super('conversation aborted');
  }
}

function canvasPosToWorld(e) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / SCALE;
  const y = (e.clientY - rect.top) / SCALE;
  return { x, y };
}

function drawGrid() {
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += SCALE * 2) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += SCALE * 2) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }
}

function drawCircle(worldX, worldY, radius, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.arc(worldX * SCALE, worldY * SCALE, radius * SCALE, 0, Math.PI * 2);
  ctx.stroke();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  ctx.font = '13px sans-serif';
  ctx.fillStyle = '#e5e7eb';
  ctx.fillText('1 Kästchen = 2 m', 10, 20);

  npcs.forEach((npc, idx) => {
    drawCircle(npc.x, npc.y, greetRadius, 'rgba(56,189,248,0.08)');
    ctx.fillStyle = NPC_COLORS[idx % NPC_COLORS.length];
    ctx.beginPath();
    ctx.arc(npc.x * SCALE, npc.y * SCALE, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#e5e7eb';
    ctx.fillText(`${npc.name || npc.id}`, npc.x * SCALE + 12, npc.y * SCALE - 10);
  });

  players.forEach((pl) => {
    ctx.fillStyle = PLAYER_COLOR;
    ctx.beginPath();
    ctx.arc(pl.x * SCALE, pl.y * SCALE, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#e5e7eb';
    ctx.fillText(`${pl.name || pl.id}`, pl.x * SCALE + 12, pl.y * SCALE - 10);
  });
}

function updateDistances() {
  const list = document.getElementById('distances');
  list.innerHTML = '';
  for (let i = 0; i < npcs.length; i++) {
    for (let j = i + 1; j < npcs.length; j++) {
      const npcA = npcs[i];
      const npcB = npcs[j];
      const d = Math.hypot(npcA.x - npcB.x, npcA.y - npcB.y).toFixed(2);
      const li = document.createElement('li');
      li.textContent = `${npcA.name || npcA.id} ↔ ${npcB.name || npcB.id}: ${d} m`;
      list.appendChild(li);
    }
  }
  players.forEach((pl) => {
    npcs.forEach((npc) => {
      const d = Math.hypot(pl.x - npc.x, pl.y - npc.y).toFixed(2);
      const li = document.createElement('li');
      li.textContent = `${pl.name || pl.id} ↔ ${npc.name || npc.id}: ${d} m`;
      list.appendChild(li);
    });
  });
}

function pickEntity(worldX, worldY) {
  const threshold = 1.2;
  for (let i = 0; i < npcs.length; i++) {
    const npc = npcs[i];
    if (Math.hypot(npc.x - worldX, npc.y - worldY) <= threshold) {
      return { type: 'npc', index: i };
    }
  }
  for (let i = 0; i < players.length; i++) {
    const pl = players[i];
    if (Math.hypot(pl.x - worldX, pl.y - worldY) <= threshold) {
      return { type: 'player', index: i };
    }
  }
  return null;
}

canvas.addEventListener('mousedown', (e) => {
  const pos = canvasPosToWorld(e);
  dragging = pickEntity(pos.x, pos.y);
});

canvas.addEventListener('mousemove', async (e) => {
  if (!dragging) return;
  const pos = canvasPosToWorld(e);
  if (dragging.type === 'npc') {
    const npc = npcs[dragging.index];
    npc.x = pos.x; npc.y = pos.y;
    await fetch('/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ who: npc.id, x: npc.x, y: npc.y })
    });
  } else if (dragging.type === 'player') {
    const pl = players[dragging.index];
    pl.x = pos.x; pl.y = pos.y;
    await fetch('/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ who: 'player', player_id: pl.id, x: pl.x, y: pl.y })
    });
  }
  updateDistances();
  draw();
});

['mouseup','mouseleave'].forEach((ev) => canvas.addEventListener(ev, () => { dragging = null; }));

function ensureDefaults(list, defaults) {
  const result = list.map((item) => ({ ...item }));
  defaults.forEach((def) => {
    if (!result.some((item) => item.id === def.id)) {
      result.push({ ...def });
    }
  });
  return result;
}

async function refreshLocal(force=false) {
  if (suspendRefresh && !force) return null;
  try {
    const r = await fetch('/state_mini');
    const s = await r.json();
    const npcList = (s.npcs && s.npcs.length ? s.npcs : DEFAULT_NPCS).map((npc) => ({ ...npc }));
    const playerList = (s.players && s.players.length ? s.players : DEFAULT_PLAYERS).map((player) => ({ ...player }));
    npcs = ensureDefaults(npcList, DEFAULT_NPCS);
    players = ensureDefaults(playerList, DEFAULT_PLAYERS);
    document.getElementById('last').textContent = s.last_push_ok ? (`OK: ${s.last_push_msg}`) : (`Fehler: ${s.last_push_msg}`);
    document.getElementById('active').textContent = s.push_enabled ? 'AN' : 'AUS';
    updateDistances();
    draw();
    return s;
  } catch (err) {
    console.error('state_mini fehlgeschlagen', err);
    if (!npcs.length) {
      npcs = DEFAULT_NPCS.map((npc) => ({ ...npc }));
    }
    if (!players.length) {
      players = DEFAULT_PLAYERS.map((player) => ({ ...player }));
    }
    updateDistances();
    draw();
    return null;
  }
}

async function loadCfg() {
  try {
    const r = await fetch('/config');
    const s = await r.json();
    document.getElementById('cfg').textContent = `${s.bot_update_url}   (alle ${s.push_interval_sec}s)`;
  } catch (err) {
    document.getElementById('cfg').textContent = 'Konfiguration nicht abrufbar';
  }
}

let currentDevices = { devices: [], defaults: {} };
let resolvedIn = null;
let resolvedOut = null;

function populateDeviceSelects() {
  const micSel = document.getElementById('micSelect');
  const speakerSel = document.getElementById('speakerSelect');
  micSel.innerHTML = '';
  speakerSel.innerHTML = '';
  const devices = currentDevices.devices || [];
  devices.forEach((dev, idx) => {
    if (dev.max_input_channels > 0) {
      const opt = document.createElement('option');
      opt.value = idx;
      opt.textContent = `[${idx}] ${dev.name} (${dev.max_input_channels} In)`;
      if (resolvedIn === idx) opt.selected = true;
      micSel.appendChild(opt);
    }
    if (dev.max_output_channels > 0) {
      const opt = document.createElement('option');
      opt.value = idx;
      opt.textContent = `[${idx}] ${dev.name} (${dev.max_output_channels} Out)`;
      if (resolvedOut === idx) opt.selected = true;
      speakerSel.appendChild(opt);
    }
  });
  if (!micSel.value && currentDevices.default_input) {
    micSel.value = currentDevices.default_input.index;
  }
  if (!speakerSel.value && currentDevices.default_output) {
    speakerSel.value = currentDevices.default_output.index;
  }
}

async function refreshDevices(showStatus=true) {
  const status = document.getElementById('deviceStatus');
  status.textContent = showStatus ? 'Lade Geräteliste ...' : '';
  try {
    const r = await fetch(`${BOT_BASE}/devices`);
    const data = await r.json();
    if (data.error) {
      status.textContent = `Fehler: ${data.error}`;
    } else {
      currentDevices = data;
      populateDeviceSelects();
      status.textContent = 'Geräte aktualisiert.';
    }
  } catch (err) {
    status.textContent = 'Geräte konnten nicht geladen werden.';
  }
}

async function refreshBotState() {
  try {
    const r = await fetch(`${BOT_BASE}/state`);
    const s = await r.json();
    if (s.radius) {
      greetRadius = s.radius;
      document.getElementById('radius').textContent = greetRadius.toFixed(2);
    }
    if (typeof s.resolved_in === 'number') resolvedIn = s.resolved_in;
    if (typeof s.resolved_out === 'number') resolvedOut = s.resolved_out;
    populateDeviceSelects();
  } catch (err) {
    console.warn('Bot-State nicht abrufbar', err);
  }
}

function getNpcById(id) {
  return npcs.find((npc) => npc.id === id) || null;
}

function getNpcName(id) {
  const npc = getNpcById(id);
  return npc ? (npc.name || npc.id) : id;
}

function describePrompt(prompt) {
  if (!prompt) return 'einen neutralen Ansatz verfolgt';
  const sentences = prompt.split(/\\r?\\n|[.!?]/).map((s) => s.trim()).filter(Boolean);
  const first = sentences.length ? sentences[0] : prompt.trim();
  return first.length > 140 ? `${first.slice(0, 137)}...` : first;
}

function getPromptForNpc(id) {
  const el = document.querySelector(`[data-npc-prompt="${id}"]`);
  return el ? el.value.trim() : '';
}

function appendLog(message, speaker=null) {
  const log = document.getElementById('conversationLog');
  if (!log) return;
  const entry = document.createElement('div');
  entry.className = 'log-entry';
  if (speaker) {
    const strong = document.createElement('strong');
    strong.textContent = `${speaker}: `;
    entry.appendChild(strong);
    entry.appendChild(document.createTextNode(message));
  } else {
    entry.textContent = message;
  }
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

function updateFinalSolution(text) {
  const finalEl = document.getElementById('finalSolution');
  if (!finalEl) return;
  if (text) {
    finalEl.textContent = text;
    finalEl.classList.add('visible');
  } else {
    finalEl.textContent = '';
    finalEl.classList.remove('visible');
  }
}

function clearConversationUI() {
  const log = document.getElementById('conversationLog');
  if (log) log.innerHTML = '';
  updateFinalSolution('');
  conversationState.solutions = {};
  conversationState.compromiseRound = 0;
  conversationState.finalSolution = '';
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function checkForAbort() {
  if (conversationState.stopRequested) {
    throw new ConversationAbort();
  }
}

async function sendMove(id, x, y) {
  try {
    await fetch('/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ who: id, x, y })
    });
  } catch (err) {
    console.warn('Konnte Bewegung nicht senden', err);
  }
}

function computeMeetingPoint(moderator, target) {
  if (!moderator || !target) return null;
  const dx = target.x - moderator.x;
  const offsetX = dx >= 0 ? -0.9 : 0.9;
  const offsetY = 0.8;
  return { x: target.x + offsetX, y: target.y + offsetY };
}

function computeCelebrationPoint() {
  return { x: 12, y: 8 };
}

async function animateMoveNpc(id, destination, duration = 2600) {
  const npc = getNpcById(id);
  if (!npc || !destination) return;
  const steps = Math.max(10, Math.floor(duration / 90));
  const startX = npc.x;
  const startY = npc.y;
  const targetX = destination.x;
  const targetY = destination.y;
  const stepDuration = duration / steps;
  suspendRefresh = true;
  try {
    for (let step = 1; step <= steps; step++) {
      checkForAbort();
      const t = step / steps;
      npc.x = startX + (targetX - startX) * t;
      npc.y = startY + (targetY - startY) * t;
      draw();
      updateDistances();
      await wait(stepDuration);
    }
  } finally {
    suspendRefresh = false;
  }
  checkForAbort();
  await sendMove(id, npc.x, npc.y);
  await refreshLocal(true);
}

function createProposal(id, task) {
  const name = getNpcName(id);
  const promptSummary = describePrompt(getPromptForNpc(id));
  const baseTask = task ? `für "${task}"` : 'für die aktuelle Aufgabe';
  return `${name} schlägt ${baseTask} eine Lösung vor, die ${promptSummary.toLowerCase()}.`;
}

function createOpinion(id, otherName, otherSolution) {
  const name = getNpcName(id);
  const promptSummary = describePrompt(getPromptForNpc(id));
  const stance = promptSummary.toLowerCase().includes('kritisch') ? 'kritisch' : 'aufgeschlossen';
  return `${name} bewertet den Vorschlag von ${otherName} ${stance} und möchte dabei sicherstellen, dass ${promptSummary.toLowerCase()}.`;
}

function createSharedSolution(task) {
  const base = task ? `Gemeinsame Lösung für "${task}":` : 'Gemeinsame Lösung:';
  const styleA = describePrompt(getPromptForNpc('npc2')).toLowerCase();
  const styleB = describePrompt(getPromptForNpc('npc3')).toLowerCase();
  return `${base} Wir kombinieren ${styleA} mit ${styleB}, damit sich beide Ansätze ergänzen.`;
}

function createCompromise(id, otherName, otherSolution, task) {
  if (conversationState.compromiseRound >= 2 || (otherSolution && otherSolution.startsWith('Gemeinsame Lösung'))) {
    const shared = createSharedSolution(task);
    conversationState.finalSolution = shared;
    return shared;
  }
  const name = getNpcName(id);
  const promptSummary = describePrompt(getPromptForNpc(id));
  return `${name} schlägt vor, wichtige Elemente von ${otherName} zu übernehmen und sie mit dem eigenen Stil (${promptSummary.toLowerCase()}) zu verbinden.`;
}

function solutionsAligned(a, b) {
  if (!a || !b) return false;
  return a.trim().toLowerCase() === b.trim().toLowerCase();
}

function tokenizeSolution(text) {
  if (!text) return [];
  return text
    .toLowerCase()
    .replace(/[^a-zäöüß0-9 \\t\\n\\r]/gi, ' ')
    .split(/[ \\t\\n\\r]+/)
    .filter((word) => word.length > 1);
}

function compareSolutions(a, b) {
  const wordsA = new Set(tokenizeSolution(a));
  const wordsB = new Set(tokenizeSolution(b));
  if (!wordsA.size || !wordsB.size) {
    return {
      score: 0,
      summary: 'Ich kann die Vorschläge kaum vergleichen, da sie sehr unterschiedlich beschrieben wurden.',
    };
  }
  let intersection = 0;
  wordsA.forEach((word) => {
    if (wordsB.has(word)) intersection += 1;
  });
  const union = new Set([...wordsA, ...wordsB]);
  const ratio = union.size ? intersection / union.size : 0;
  let verdict;
  if (ratio >= 0.75) {
    verdict = 'Die beiden Vorschläge sind nahezu identisch in ihren Kernaussagen.';
  } else if (ratio >= 0.5) {
    verdict = 'Es gibt viele Überschneidungen – ihr seid euch in weiten Teilen einig.';
  } else if (ratio >= 0.3) {
    verdict = 'Einige Aspekte ähneln sich, aber die Schwerpunkte unterscheiden sich noch spürbar.';
  } else {
    verdict = 'Eure Sichtweisen gehen deutlich auseinander und betonen ganz andere Punkte.';
  }
  return {
    score: ratio,
    summary: `${verdict} (Ähnlichkeit: ${(ratio * 100).toFixed(0)}%)`,
  };
}

async function approachNpc(targetId, message) {
  const moderator = getNpcById('npc1');
  const target = getNpcById(targetId);
  if (!moderator || !target) return;
  const targetName = getNpcName(targetId);
  appendLog(`Ich komme zu dir, ${targetName}.`, getNpcName('npc1'));
  const meetPoint = computeMeetingPoint(moderator, target);
  await animateMoveNpc('npc1', meetPoint);
  checkForAbort();
  if (message) {
    appendLog(message, getNpcName('npc1'));
  }
}

function setControlsRunning(running) {
  if (startBtn) startBtn.disabled = running;
  if (stopBtn) stopBtn.disabled = !running;
}

async function runConversation() {
  if (conversationState.running) return;
  if (!getNpcById('npc1') || !getNpcById('npc2') || !getNpcById('npc3')) {
    npcs = DEFAULT_NPCS.map((npc) => ({ ...npc }));
    players = DEFAULT_PLAYERS.map((player) => ({ ...player }));
    draw();
  }
  conversationState.running = true;
  conversationState.stopRequested = false;
  conversationState.clearOnStop = false;
  conversationState.solutions = {};
  conversationState.compromiseRound = 0;
  conversationState.finalSolution = '';
  setControlsRunning(true);
  const task = document.getElementById('taskDescription').value.trim();
  const moderatorName = getNpcName('npc1');
  const npcA = 'npc2';
  const npcB = 'npc3';
  const npcAName = getNpcName(npcA);
  const npcBName = getNpcName(npcB);
  try {
    await refreshLocal(true);
    appendLog(`Ich moderiere jetzt die Aufgabe${task ? ` "${task}"` : ''}.`, moderatorName);
    await approachNpc(npcA, `Bitte erklär mir deine beste Lösung${task ? ` für "${task}"` : ''} und welches Ergebnis du erwartest.`);
    checkForAbort();
    conversationState.solutions[npcA] = createProposal(npcA, task);
    appendLog(conversationState.solutions[npcA], npcAName);
    await wait(900);
    checkForAbort();

    await approachNpc(npcB, `Mit Blick auf dein Ergebnis: Wie würdest du die Aufgabe${task ? ` "${task}"` : ''} lösen?`);
    checkForAbort();
    conversationState.solutions[npcB] = createProposal(npcB, task);
    appendLog(conversationState.solutions[npcB], npcBName);
    await wait(900);
    checkForAbort();

    await animateMoveNpc('npc1', computeCelebrationPoint(), 1600);
    checkForAbort();
    const comparison = compareSolutions(conversationState.solutions[npcA], conversationState.solutions[npcB]);
    const summary = `Ich habe beide Lösungen verglichen. ${comparison.summary}`;
    appendLog(summary, moderatorName);
    conversationState.finalSolution = summary;
    updateFinalSolution(summary);
  } catch (err) {
    if (err instanceof ConversationAbort) {
      appendLog('Das Gespräch wurde angehalten.', moderatorName);
    } else {
      console.error('Fehler im Gespräch', err);
      appendLog('Es ist ein Fehler im Gespräch aufgetreten.', 'System');
    }
  } finally {
    setControlsRunning(false);
    const shouldClear = conversationState.clearOnStop;
    conversationState.running = false;
    conversationState.stopRequested = false;
    if (shouldClear) {
      conversationState.clearOnStop = false;
      clearConversationUI();
    }
  }
}

function requestStop(clear=false, message=null) {
  if (!conversationState.running) {
    if (clear) {
      clearConversationUI();
    }
    return;
  }
  conversationState.stopRequested = true;
  conversationState.clearOnStop = clear;
  if (message) {
    appendLog(message, getNpcName('npc1'));
  }
}

const startBtn = document.getElementById('startConversation');
const stopBtn = document.getElementById('stopConversation');
const resetBtn = document.getElementById('resetConversation');

startBtn.addEventListener('click', () => {
  runConversation();
});

stopBtn.addEventListener('click', () => {
  if (!conversationState.running) return;
  requestStop(false, 'Ich stoppe das Gespräch für den Moment.');
});

resetBtn.addEventListener('click', () => {
  if (conversationState.running) {
    requestStop(true, 'Gespräch wird zurückgesetzt.');
  } else {
    clearConversationUI();
  }
});

setControlsRunning(false);

const promptFields = document.querySelectorAll('[data-npc-prompt]');
promptFields.forEach((field) => {
  const key = `npc_prompt_${field.dataset.npcPrompt}`;
  const stored = window.localStorage ? window.localStorage.getItem(key) : null;
  if (stored) field.value = stored;
  field.addEventListener('input', () => {
    try {
      if (window.localStorage) {
        window.localStorage.setItem(key, field.value);
      }
    } catch (err) {
      console.warn('Konnte Prompt nicht speichern', err);
    }
  });
});

document.getElementById('toggle').onclick = async () => { await fetch('/toggle_push', { method: 'POST' }); refreshLocal(); };
document.getElementById('pushonce').onclick = async () => { await fetch('/push_once', { method: 'POST' }); refreshLocal(); };
document.getElementById('refreshDevices').onclick = () => refreshDevices();

document.getElementById('micSelect').addEventListener('change', async (e) => {
  const idx = parseInt(e.target.value, 10);
  if (Number.isInteger(idx)) {
    await fetch(`${BOT_BASE}/set_in/${idx}`, { method: 'POST' });
    resolvedIn = idx;
    document.getElementById('deviceStatus').textContent = `Input gesetzt auf Gerät ${idx}.`;
  }
});

document.getElementById('speakerSelect').addEventListener('change', async (e) => {
  const idx = parseInt(e.target.value, 10);
  if (Number.isInteger(idx)) {
    await fetch(`${BOT_BASE}/set_out/${idx}`, { method: 'POST' });
    resolvedOut = idx;
    document.getElementById('deviceStatus').textContent = `Output gesetzt auf Gerät ${idx}.`;
  }
});

setInterval(() => refreshLocal(), 800);
setInterval(refreshBotState, 4000);
loadCfg();
refreshLocal();
refreshBotState();
refreshDevices(false);
draw();
</script></body></html>
"""
    return HTMLResponse(html.replace("__BOT_HOST__", BOT_HOST).replace("__BOT_PORT__", str(BOT_PORT)))

async def _push_positions():
    global _last_push_ok, _last_push_msg
    payload = {"npcs": state["npcs"], "players": state["players"]}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.post(BOT_UPDATE_URL, json=payload)
        if r.status_code == 200:
            _last_push_ok = True; _last_push_msg = "gesendet"; return True, "OK"
        else:
            _last_push_ok = False; _last_push_msg = f"HTTP {r.status_code}"; return False, _last_push_msg
    except Exception as e:
        _last_push_ok = False; _last_push_msg = f"Fehler: {e}"; return False, _last_push_msg

@app.on_event("startup")
async def start_pusher():
    async def loop():
        await asyncio.sleep(0.5)
        while True:
            if _push_enabled: await _push_positions()
            await asyncio.sleep(PUSH_INTERVAL_SEC)
    asyncio.create_task(loop())
