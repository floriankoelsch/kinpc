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
    "npc": {"x": 10.0, "y": 10.0},
    "players": [{"id":"p1","name":"Spieler1","x":18.0,"y":14.0}],
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
    return {"npc": state["npc"], "players": state["players"],
            "push_enabled": _push_enabled,
            "last_push_ok": _last_push_ok, "last_push_msg": _last_push_msg}

@app.post("/move")
async def move(req: Request):
    body = await req.json()
    who = body.get("who"); x = float(body.get("x",0)); y=float(body.get("y",0))
    if who == "npc":
        # NPC bleibt fix – ignoriere
        return {"ok": True, "ignored": "npc fixed"}
    else:
        if state["players"]:
            state["players"][0].update({"x":x,"y":y})
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
    return HTMLResponse(f"""
<!doctype html><html><head><meta charset="utf-8"/><title>Mini-Map</title>
<style>
body {{ font-family: ui-sans-serif, system-ui; margin:0; display:flex; height:100vh; }}
#left {{ flex:1; display:flex; align-items:center; justify-content:center; background:#0b1220; }}
#right {{ width:360px; background:#111827; color:#e5e7eb; padding:16px; overflow:auto; }}
canvas {{ background:#0f172a; border:1px solid #374151; border-radius:8px; }}
button {{ margin-right:8px; margin-bottom:8px; }}
.row {{ margin:8px 0; }} code {{ background:#1f2937; padding:2px 6px; border-radius:4px; }}
</style></head><body>
<div id="left"><canvas id="map" width="480" height="360"></canvas></div>
<div id="right">
  <h2>Mini-Map</h2>
  <div class="row">Bot Update URL:<br><code id="cfg"></code></div>
  <div class="row"><button id="toggle">Push an/aus</button><button id="pushonce">Push einmal</button></div>
  <div class="row"><div>Letzter Push: <span id="last"></span></div><div>Push aktiv: <span id="active"></span></div></div>
  <div class="row">
    <a href="http://{BOT_HOST}:{BOT_PORT}/state" target="_blank">Bot /state</a><br/>
    <a href="http://{BOT_HOST}:{BOT_PORT}/log" target="_blank">Bot /log</a><br/>
    <a href="http://{BOT_HOST}:{BOT_PORT}/devices" target="_blank">Bot /devices</a>
  </div>
  <div class="row"><small>NPC ist fix; bewege nur den Spieler (lila).</small></div>
</div>
<script>
const canvas=document.getElementById('map'); const ctx=canvas.getContext('2d');
let npc={{x:10,y:10}}, player={{x:18,y:14}}; let dragging=null;
function draw(){{ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle="#1f2937"; for(let x=0;x<canvas.width;x+=24){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();}}
  for(let y=0;y<canvas.height;y+=24){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();}}
  ctx.fillStyle="#22d3ee"; ctx.beginPath(); ctx.arc(npc.x*12,npc.y*12,8,0,Math.PI*2); ctx.fill(); ctx.fillText("NPC (fix)", npc.x*12+10, npc.y*12-10);
  ctx.fillStyle="#a78bfa"; ctx.beginPath(); ctx.arc(player.x*12,player.y*12,8,0,Math.PI*2); ctx.fill(); ctx.fillText("Spieler", player.x*12+10, player.y*12-10);
  const dx=npc.x-player.x, dy=npc.y-player.y; ctx.fillStyle="#e5e7eb"; ctx.fillText("Dist: "+Math.hypot(dx,dy).toFixed(2), 10, 20);
}}
canvas.addEventListener('mousedown',(e)=>{{ // immer den Spieler packen
  dragging='player';
}});
canvas.addEventListener('mousemove',async(e)=>{{if(dragging!=='player')return; const r=canvas.getBoundingClientRect();
  const x=(e.clientX-r.left)/12; const y=(e.clientY-r.top)/12;
  player={{x,y}}; await fetch('/move',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{who:'p1',x,y}})}}); draw();}});
canvas.addEventListener('mouseup',()=>dragging=null); canvas.addEventListener('mouseleave',()=>dragging=null);
async function refreshLocal(){{const r=await fetch('/state_mini'); const s=await r.json(); npc=s.npc; player=s.players[0];
  document.getElementById('last').textContent=s.last_push_ok?("OK: "+s.last_push_msg):("Fehler: "+s.last_push_msg);
  document.getElementById('active').textContent=s.push_enabled?"AN":"AUS"; draw();}}
async function loadCfg(){{const r=await fetch('/config'); const s=await r.json(); document.getElementById('cfg').textContent=s.bot_update_url+"   (alle "+s.push_interval_sec+"s)";}}
document.getElementById('toggle').onclick=async()=>{{await fetch('/toggle_push',{{method:'POST'}}); refreshLocal();}};
document.getElementById('pushonce').onclick=async()=>{{await fetch('/push_once',{{method:'POST'}}); refreshLocal();}};
setInterval(refreshLocal,500); loadCfg(); refreshLocal(); draw();
</script></body></html>""")

async def _push_positions():
    global _last_push_ok, _last_push_msg
    payload = {"npc": state["npc"], "players": state["players"]}
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
