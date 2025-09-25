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
        {"id": "npc1", "name": "NPC 1", "x": 10.0, "y": 10.0},
        {"id": "npc2", "name": "NPC 2", "x": 14.0, "y": 10.0},
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
    html = """
<!doctype html><html><head><meta charset="utf-8"/><title>Mini-Map</title>
<style>
body { font-family: ui-sans-serif, system-ui; margin:0; display:flex; height:100vh; color:#e5e7eb; }
#left { flex:1; display:flex; align-items:center; justify-content:center; background:#0b1220; padding:16px; box-sizing:border-box; }
#right { width:380px; background:#111827; padding:16px; overflow:auto; box-sizing:border-box; }
canvas { background:#0f172a; border:1px solid #374151; border-radius:8px; box-shadow:0 10px 30px rgba(0,0,0,0.35); }
button { margin-right:8px; margin-bottom:8px; background:#2563eb; border:none; color:#e5e7eb; padding:6px 10px; border-radius:6px; cursor:pointer; }
button.secondary { background:#374151; }
.row { margin:12px 0; }
.row h3 { margin:0 0 6px 0; font-size:1rem; color:#93c5fd; }
code { background:#1f2937; padding:2px 6px; border-radius:4px; display:inline-block; margin-top:4px; }
label { display:block; font-size:0.85rem; margin-bottom:2px; color:#9ca3af; }
select { width:100%; padding:6px; border-radius:6px; background:#1f2937; color:#e5e7eb; border:1px solid #374151; margin-bottom:6px; }
small { color:#9ca3af; }
ul { list-style:none; padding-left:16px; margin:0; }
li { margin:2px 0; font-size:0.9rem; }
.status { font-size:0.85rem; color:#9ca3af; }
</style></head><body>
<div id="left"><canvas id="map" width="600" height="420"></canvas></div>
<div id="right">
  <h2>Mini-Map</h2>
  <div class="row">Bot Update URL:<br><code id="cfg"></code></div>
  <div class="row"><button id="toggle">Push an/aus</button><button id="pushonce" class="secondary">Push einmal</button></div>
  <div class="row"><div>Letzter Push: <span id="last"></span></div><div>Push aktiv: <span id="active"></span></div></div>
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
  <div class="row"><small>Ziehe die Marker für NPC&nbsp;1 (cyan), NPC&nbsp;2 (gelb) und Spieler (lila), um Positionen festzulegen.</small></div>
</div>
<script>
const canvas = document.getElementById('map');
const ctx = canvas.getContext('2d');
const SCALE = 12;
const BOT_BASE = `http://__BOT_HOST__:__BOT_PORT__`;
const NPC_COLORS = ['#22d3ee', '#fbbf24'];
const PLAYER_COLOR = '#a78bfa';
let npcs = [];
let players = [];
let greetRadius = 4.0;
let dragging = null;

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
    drawCircle(npc.x, npc.y, greetRadius, 'rgba(34,211,238,0.1)');
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
  if (npcs.length >= 2) {
    const d = Math.hypot(npcs[0].x - npcs[1].x, npcs[0].y - npcs[1].y).toFixed(2);
    const li = document.createElement('li');
    li.textContent = `${npcs[0].name || 'NPC1'} ↔ ${npcs[1].name || 'NPC2'}: ${d} m`;
    list.appendChild(li);
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

async function refreshLocal() {
  try {
    const r = await fetch('/state_mini');
    const s = await r.json();
    npcs = s.npcs || [];
    players = s.players || [];
    document.getElementById('last').textContent = s.last_push_ok ? (`OK: ${s.last_push_msg}`) : (`Fehler: ${s.last_push_msg}`);
    document.getElementById('active').textContent = s.push_enabled ? 'AN' : 'AUS';
    updateDistances();
    draw();
  } catch (err) {
    console.error('state_mini fehlgeschlagen', err);
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

setInterval(refreshLocal, 800);
setInterval(refreshBotState, 4000);
loadCfg();
refreshLocal();
refreshBotState();
refreshDevices(false);
draw();
</script></body></html>"""
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
