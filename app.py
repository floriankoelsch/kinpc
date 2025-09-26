# app.py
import asyncio, os, threading, uuid
from copy import deepcopy
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import httpx
from pydantic import BaseModel, Field

APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "8002"))

BOT_HOST = os.getenv("BOT_HOST", "127.0.0.1")
BOT_PORT = int(os.getenv("BOT_PORT", "9000"))
BOT_UPDATE_URL = f"http://{BOT_HOST}:{BOT_PORT}/update"

PUSH_INTERVAL_SEC = float(os.getenv("PUSH_INTERVAL_SEC", "0.2"))  # 5x/Sek

def _gen_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class AgentCreate(BaseModel):
    id: Optional[str] = None
    name: str
    role: str = "npc"
    prompt: str = ""
    voice_id: Optional[str] = None
    x: float = 0.0
    y: float = 0.0


class EnvironmentCreate(BaseModel):
    name: str
    owner_id: str
    description: Optional[str] = None
    width_m: float = Field(80.0, gt=10.0)
    height_m: float = Field(56.0, gt=10.0)
    grid_m: float = Field(2.0, gt=0.5)
    pixels_per_meter: float = Field(12.0, ge=4.0)
    radius_m: float = Field(4.0, gt=0.0)
    agents: Optional[List[AgentCreate]] = None


class CustomWeightPayload(BaseModel):
    id: Optional[str] = None
    label: str
    value: float = Field(0.25, ge=0.0, le=1.0)


class WeightingUpdate(BaseModel):
    solution_focus: Optional[float] = Field(None, ge=0.0, le=1.0)
    speed: Optional[float] = Field(None, ge=0.0, le=1.0)
    thoroughness: Optional[float] = Field(None, ge=0.0, le=1.0)
    custom: Optional[List[CustomWeightPayload]] = None


class CustomWeightCreate(BaseModel):
    label: str
    value: Optional[float] = Field(0.2, ge=0.0, le=1.0)


class AgentUpdate(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    voice_id: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    role: Optional[str] = None


class MapUpdate(BaseModel):
    width_m: Optional[float] = Field(None, gt=10.0)
    height_m: Optional[float] = Field(None, gt=10.0)
    grid_m: Optional[float] = Field(None, gt=0.5)
    pixels_per_meter: Optional[float] = Field(None, ge=4.0)
    radius_m: Optional[float] = Field(None, gt=0.0)


class CustomerCreate(BaseModel):
    name: str
    contact_email: Optional[str] = None


class UserCreate(BaseModel):
    name: str
    email: Optional[str] = None
    role: str = "user"


class SetEnvironmentRequest(BaseModel):
    environment_id: str


class TokenLimitSet(BaseModel):
    customer_id: str
    service: str
    limit: int = Field(..., ge=0)


class TokenUsageReport(BaseModel):
    customer_id: str
    service: str
    used: int = Field(..., ge=0)


class TaskUpdate(BaseModel):
    task: str = ""


class SaasState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.customers: Dict[str, Dict[str, Any]] = {}
        self.users: Dict[str, Dict[str, Any]] = {}
        self.environments: Dict[str, Dict[str, Any]] = {}
        self.environment_states: Dict[str, Dict[str, Any]] = {}
        self.token_limits: Dict[str, Dict[str, int]] = {}
        self.token_usage: Dict[str, Dict[str, int]] = {}
        self.current_environment_id: Optional[str] = None
        self._bootstrap_defaults()

    # ---------- Bootstrap ----------
    def _bootstrap_defaults(self) -> None:
        customer = self.create_customer(CustomerCreate(name="Demo GmbH", contact_email="demo@example.com"))
        self.create_user(customer["id"], UserCreate(name="Admin", email="admin@demo.example", role="admin"))
        map_defaults = {
            "width_m": 84.0,
            "height_m": 60.0,
            "grid_m": 2.0,
            "pixels_per_meter": 14.0,
            "radius_m": 4.0,
        }
        default_env = EnvironmentCreate(
            name="Innovation Lab",
            owner_id=customer["id"],
            description="Virtuelle Projektfläche für Workshops",
            width_m=map_defaults["width_m"],
            height_m=map_defaults["height_m"],
            grid_m=map_defaults["grid_m"],
            pixels_per_meter=map_defaults["pixels_per_meter"],
            radius_m=map_defaults["radius_m"],
        )
        env = self.create_environment(default_env)
        self.current_environment_id = env["id"]

    # ---------- Utilities ----------
    def _default_agents(self, map_conf: Dict[str, float]) -> List[Dict[str, Any]]:
        width = map_conf.get("width_m", 80.0)
        height = map_conf.get("height_m", 56.0)
        base_y = height * 0.45
        return [
            {
                "id": "npc1",
                "name": "Moderator",
                "role": "moderator",
                "prompt": (
                    "Ruhig, strukturiert und wertschätzend. Fasse Ergebnisse klar zusammen, stelle gezielte Fragen "
                    "und behalte die Zielsetzung im Blick."
                ),
                "voice_id": None,
                "x": width * 0.35,
                "y": base_y,
            },
            {
                "id": "npc2",
                "name": "NPC 1",
                "role": "npc",
                "prompt": (
                    "Analytisch, detailverliebt und lösungsorientiert. Denkt logisch, liebt strukturierte Pläne und "
                    "argumentiert anhand von Daten."
                ),
                "voice_id": None,
                "x": width * 0.55,
                "y": base_y - 6,
            },
            {
                "id": "npc3",
                "name": "NPC 2",
                "role": "npc",
                "prompt": (
                    "Kreativ, intuitiv und empathisch. Bringt ungewöhnliche Ideen ein und achtet auf Teamdynamik und "
                    "Stimmung."
                ),
                "voice_id": None,
                "x": width * 0.58,
                "y": base_y + 6,
            },
        ]

    def _default_players(self, map_conf: Dict[str, float]) -> List[Dict[str, Any]]:
        return [
            {
                "id": "p1",
                "name": "Spieler1",
                "x": map_conf.get("width_m", 80.0) * 0.52,
                "y": map_conf.get("height_m", 56.0) * 0.72,
            }
        ]

    def _ensure_customer(self, customer_id: str) -> None:
        if customer_id not in self.customers:
            raise HTTPException(status_code=404, detail="Kunde nicht gefunden")

    # ---------- Customer / User ----------
    def create_customer(self, data: CustomerCreate) -> Dict[str, Any]:
        with self.lock:
            customer_id = _gen_id("cust")
            payload = {
                "id": customer_id,
                "name": data.name.strip(),
                "contact_email": (data.contact_email or "").strip() or None,
                "users": [],
            }
            self.customers[customer_id] = payload
            return deepcopy(payload)

    def list_customers(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [deepcopy(cust) for cust in self.customers.values()]

    def create_user(self, customer_id: str, data: UserCreate) -> Dict[str, Any]:
        self._ensure_customer(customer_id)
        with self.lock:
            user_id = _gen_id("user")
            payload = {
                "id": user_id,
                "customer_id": customer_id,
                "name": data.name.strip(),
                "email": (data.email or "").strip() or None,
                "role": data.role.strip() or "user",
            }
            self.users[user_id] = payload
            self.customers[customer_id]["users"].append(payload)
            return deepcopy(payload)

    def list_users(self, customer_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.lock:
            result = [deepcopy(u) for u in self.users.values()]
        if customer_id:
            result = [u for u in result if u["customer_id"] == customer_id]
        return result

    # ---------- Environment ----------
    def create_environment(self, data: EnvironmentCreate) -> Dict[str, Any]:
        self._ensure_customer(data.owner_id)
        map_conf = {
            "width_m": data.width_m,
            "height_m": data.height_m,
            "grid_m": data.grid_m,
            "pixels_per_meter": data.pixels_per_meter,
            "radius_m": min(data.radius_m, 4.0),
        }
        agent_defs = data.agents or [AgentCreate(**a) if isinstance(a, dict) else a for a in self._default_agents(map_conf)]
        agents: Dict[str, Dict[str, Any]] = {}
        moderator_id: Optional[str] = None
        for idx, raw in enumerate(agent_defs):
            if isinstance(raw, AgentCreate):
                agent = raw
            else:
                agent = AgentCreate(**raw)
            agent_id = agent.id or ("npc" + str(idx + 1))
            agent_payload = {
                "id": agent_id,
                "name": agent.name.strip() or agent_id,
                "role": agent.role or ("npc" if idx else "moderator"),
                "prompt": agent.prompt.strip(),
                "voice_id": (agent.voice_id or "").strip() or None,
                "x": float(agent.x),
                "y": float(agent.y),
            }
            if agent_payload["role"] == "moderator" and not moderator_id:
                moderator_id = agent_id
            agents[agent_id] = agent_payload
        if not moderator_id:
            # Fallback: erster Agent wird Moderator
            first_id = next(iter(agents))
            agents[first_id]["role"] = "moderator"
            moderator_id = first_id

        env_id = _gen_id("env")
        env_payload = {
            "id": env_id,
            "name": data.name.strip(),
            "description": data.description or "",
            "owner_id": data.owner_id,
            "map": map_conf,
            "moderator_id": moderator_id,
            "agents": agents,
            "weights": {
                "solution_focus": 0.45,
                "speed": 0.25,
                "thoroughness": 0.30,
                "custom": [],
            },
            "task": "",
            "status": "idle",
        }

        npc_positions = [
            {
                "id": aid,
                "name": agent_data["name"],
                "x": agent_data.get("x", 0.0),
                "y": agent_data.get("y", 0.0),
            }
            for aid, agent_data in agents.items()
            if agent_data.get("role") in {"npc", "moderator"}
        ]
        players = self._default_players(map_conf)
        env_state = {
            "npcs": npc_positions,
            "players": players,
            "push": {"enabled": True, "last_ok": True, "last_msg": "noch nix"},
        }

        with self.lock:
            self.environments[env_id] = env_payload
            self.environment_states[env_id] = env_state
        return deepcopy(env_payload)

    def list_environment_ids(self) -> List[str]:
        with self.lock:
            return list(self.environments.keys())

    def list_environments(self, owner_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.lock:
            envs = [deepcopy(env) for env in self.environments.values()]
        if owner_id:
            envs = [env for env in envs if env["owner_id"] == owner_id]
        return envs

    def get_environment(self, env_id: str) -> Dict[str, Any]:
        with self.lock:
            env = self.environments.get(env_id)
            if not env:
                raise HTTPException(status_code=404, detail="Environment nicht gefunden")
            return deepcopy(env)

    def get_environment_state(self, env_id: str) -> Dict[str, Any]:
        with self.lock:
            state = self.environment_states.get(env_id)
            if not state:
                raise HTTPException(status_code=404, detail="Environment-State nicht gefunden")
            return deepcopy(state)

    def set_current_environment(self, env_id: str) -> Dict[str, Any]:
        self.get_environment(env_id)  # Validierung
        with self.lock:
            self.current_environment_id = env_id
            return {"environment_id": env_id}

    def get_current_environment(self) -> Optional[str]:
        with self.lock:
            return self.current_environment_id

    def update_agent(self, env_id: str, agent_id: str, payload: AgentUpdate) -> Dict[str, Any]:
        with self.lock:
            env = self.environments.get(env_id)
            if not env:
                raise HTTPException(status_code=404, detail="Environment nicht gefunden")
            agent = env["agents"].get(agent_id)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent nicht gefunden")
            if payload.name is not None:
                agent["name"] = payload.name.strip() or agent["name"]
            if payload.prompt is not None:
                agent["prompt"] = payload.prompt.strip()
            if payload.voice_id is not None:
                agent["voice_id"] = payload.voice_id.strip() or None
            if payload.role is not None:
                agent["role"] = payload.role
            if payload.x is not None:
                agent["x"] = float(payload.x)
            if payload.y is not None:
                agent["y"] = float(payload.y)
            # Update state names/positions synchronisiert halten
            state = self.environment_states.get(env_id)
            if state:
                for npc in state["npcs"]:
                    if npc["id"] == agent_id:
                        npc["name"] = agent["name"]
                        if payload.x is not None:
                            npc["x"] = float(payload.x)
                        if payload.y is not None:
                            npc["y"] = float(payload.y)
                        break
            return deepcopy(agent)

    def move_entity(self, env_id: str, who: Optional[str], x: float, y: float, player_id: Optional[str] = None) -> None:
        with self.lock:
            state = self.environment_states.get(env_id)
            if not state:
                raise HTTPException(status_code=404, detail="Environment-State nicht gefunden")
            if who and who.startswith("npc"):
                for npc in state["npcs"]:
                    if npc["id"] == who:
                        npc["x"] = float(x)
                        npc["y"] = float(y)
                        break
                agent = self.environments[env_id]["agents"].get(who)
                if agent:
                    agent["x"] = float(x)
                    agent["y"] = float(y)
            else:
                target_id = player_id
                if not target_id and state["players"]:
                    target_id = state["players"][0]["id"]
                if target_id:
                    found = False
                    for player in state["players"]:
                        if player["id"] == target_id:
                            player["x"] = float(x)
                            player["y"] = float(y)
                            found = True
                            break
                    if not found:
                        state["players"].append({"id": target_id, "name": target_id, "x": float(x), "y": float(y)})

    def update_weights(self, env_id: str, payload: WeightingUpdate) -> Dict[str, Any]:
        with self.lock:
            env = self.environments.get(env_id)
            if not env:
                raise HTTPException(status_code=404, detail="Environment nicht gefunden")
            weights = env["weights"]
            if payload.solution_focus is not None:
                weights["solution_focus"] = float(payload.solution_focus)
            if payload.speed is not None:
                weights["speed"] = float(payload.speed)
            if payload.thoroughness is not None:
                weights["thoroughness"] = float(payload.thoroughness)
            if payload.custom is not None:
                custom_items = []
                for item in payload.custom:
                    cid = item.id or _gen_id("crit")
                    custom_items.append({"id": cid, "label": item.label, "value": float(item.value)})
                weights["custom"] = custom_items
            return deepcopy(weights)

    def add_custom_weight(self, env_id: str, payload: CustomWeightCreate) -> Dict[str, Any]:
        with self.lock:
            env = self.environments.get(env_id)
            if not env:
                raise HTTPException(status_code=404, detail="Environment nicht gefunden")
            cid = _gen_id("crit")
            entry = {"id": cid, "label": payload.label.strip(), "value": float(payload.value or 0.2)}
            env["weights"].setdefault("custom", []).append(entry)
            return deepcopy(entry)

    def remove_custom_weight(self, env_id: str, custom_id: str) -> None:
        with self.lock:
            env = self.environments.get(env_id)
            if not env:
                raise HTTPException(status_code=404, detail="Environment nicht gefunden")
            custom = env["weights"].get("custom", [])
            env["weights"]["custom"] = [item for item in custom if item.get("id") != custom_id]

    def update_map(self, env_id: str, payload: MapUpdate) -> Dict[str, Any]:
        with self.lock:
            env = self.environments.get(env_id)
            if not env:
                raise HTTPException(status_code=404, detail="Environment nicht gefunden")
            map_conf = env["map"]
            if payload.width_m is not None:
                map_conf["width_m"] = float(payload.width_m)
            if payload.height_m is not None:
                map_conf["height_m"] = float(payload.height_m)
            if payload.grid_m is not None:
                map_conf["grid_m"] = float(payload.grid_m)
            if payload.pixels_per_meter is not None:
                map_conf["pixels_per_meter"] = float(payload.pixels_per_meter)
            if payload.radius_m is not None:
                map_conf["radius_m"] = min(float(payload.radius_m), 4.0)
            return deepcopy(map_conf)

    def update_task(self, env_id: str, task: str) -> Dict[str, Any]:
        with self.lock:
            env = self.environments.get(env_id)
            if not env:
                raise HTTPException(status_code=404, detail="Environment nicht gefunden")
            env["task"] = task.strip()
            return {"task": env["task"]}

    # ---------- Push handling ----------
    def get_push_state(self, env_id: str) -> Dict[str, Any]:
        with self.lock:
            state = self.environment_states.get(env_id)
            if not state:
                raise HTTPException(status_code=404, detail="Environment-State nicht gefunden")
            return deepcopy(state["push"])

    def toggle_push(self, env_id: str) -> Dict[str, Any]:
        with self.lock:
            state = self.environment_states.get(env_id)
            if not state:
                raise HTTPException(status_code=404, detail="Environment-State nicht gefunden")
            state["push"]["enabled"] = not state["push"].get("enabled", True)
            return deepcopy(state["push"])

    def record_push(self, env_id: str, ok: bool, msg: str) -> None:
        with self.lock:
            state = self.environment_states.get(env_id)
            if not state:
                return
            state["push"].update({"last_ok": ok, "last_msg": msg})

    def describe_environment_for_frontend(self, env_id: str) -> Dict[str, Any]:
        env = self.get_environment(env_id)
        return {
            "id": env["id"],
            "name": env["name"],
            "description": env.get("description", ""),
            "map": env["map"],
            "weights": env["weights"],
            "task": env.get("task", ""),
            "moderator_id": env.get("moderator_id"),
            "agents": list(env["agents"].values()),
        }

    def describe_environment_for_bot(self, env_id: str) -> Dict[str, Any]:
        env = self.get_environment(env_id)
        return {
            "id": env["id"],
            "name": env["name"],
            "radius_m": env["map"].get("radius_m", 4.0),
            "task": env.get("task", ""),
            "weights": env.get("weights", {}),
            "agents": [
                {
                    "id": agent["id"],
                    "name": agent["name"],
                    "prompt": agent.get("prompt", ""),
                    "voice_id": agent.get("voice_id"),
                    "role": agent.get("role", "npc"),
                }
                for agent in env["agents"].values()
            ],
        }

    # ---------- Token usage ----------
    def set_token_limit(self, payload: TokenLimitSet) -> Dict[str, Any]:
        self._ensure_customer(payload.customer_id)
        with self.lock:
            limits = self.token_limits.setdefault(payload.customer_id, {})
            limits[payload.service] = payload.limit
            return {"customer_id": payload.customer_id, "service": payload.service, "limit": payload.limit}

    def record_token_usage(self, payload: TokenUsageReport) -> Dict[str, Any]:
        self._ensure_customer(payload.customer_id)
        with self.lock:
            usage = self.token_usage.setdefault(payload.customer_id, {})
            usage[payload.service] = usage.get(payload.service, 0) + payload.used
            return {"customer_id": payload.customer_id, "service": payload.service, "used": usage[payload.service]}

    def get_token_overview(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "limits": deepcopy(self.token_limits),
                "usage": deepcopy(self.token_usage),
            }


saas = SaasState()

app = FastAPI(title="KI-NPC Mini-Map")

if not os.path.exists("static"): os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/health") 
def health(): 
    return {"ok": True, "app": "minimap"}

@app.get("/config")
def cfg():
    env_id = saas.get_current_environment()
    push_state = {"enabled": False}
    if env_id:
        push_state = saas.get_push_state(env_id)
    return {
        "bot_update_url": BOT_UPDATE_URL,
        "push_interval_sec": PUSH_INTERVAL_SEC,
        "push_enabled": push_state.get("enabled", False),
        "current_environment_id": env_id,
    }

@app.get("/api/customers")
def api_list_customers():
    return saas.list_customers()


@app.post("/api/customers")
def api_create_customer(payload: CustomerCreate):
    return saas.create_customer(payload)


@app.get("/api/customers/{customer_id}/users")
def api_list_users(customer_id: str):
    return saas.list_users(customer_id)


@app.post("/api/customers/{customer_id}/users")
def api_create_user(customer_id: str, payload: UserCreate):
    return saas.create_user(customer_id, payload)


@app.get("/api/environments")
def api_list_environments(customer_id: Optional[str] = None):
    return saas.list_environments(owner_id=customer_id)


@app.post("/api/environments")
def api_create_environment(payload: EnvironmentCreate):
    env = saas.create_environment(payload)
    if saas.get_current_environment() is None:
        saas.set_current_environment(env["id"])
    return env


@app.get("/api/environments/{env_id}")
def api_get_environment(env_id: str):
    return saas.describe_environment_for_frontend(env_id)


@app.patch("/api/environments/{env_id}/map")
def api_update_map(env_id: str, payload: MapUpdate):
    return saas.update_map(env_id, payload)


@app.patch("/api/environments/{env_id}/weights")
def api_update_weights(env_id: str, payload: WeightingUpdate):
    return saas.update_weights(env_id, payload)


@app.post("/api/environments/{env_id}/weights/custom")
def api_add_weight(env_id: str, payload: CustomWeightCreate):
    return saas.add_custom_weight(env_id, payload)


@app.delete("/api/environments/{env_id}/weights/custom/{custom_id}")
def api_remove_weight(env_id: str, custom_id: str):
    saas.remove_custom_weight(env_id, custom_id)
    return {"ok": True}


@app.patch("/api/environments/{env_id}/agents/{agent_id}")
def api_update_agent(env_id: str, agent_id: str, payload: AgentUpdate):
    return saas.update_agent(env_id, agent_id, payload)


@app.post("/api/environments/{env_id}/task")
def api_update_task(env_id: str, payload: TaskUpdate):
    return saas.update_task(env_id, payload.task)


@app.get("/api/session/environment")
def api_get_session_environment():
    env_id = saas.get_current_environment()
    if not env_id:
        return {"environment_id": None}
    return saas.describe_environment_for_frontend(env_id)


@app.post("/api/session/environment")
def api_set_session_environment(payload: SetEnvironmentRequest):
    return saas.set_current_environment(payload.environment_id)


@app.get("/api/admin/token-usage")
def api_token_overview():
    return saas.get_token_overview()


@app.post("/api/admin/token-limits")
def api_set_token_limit(payload: TokenLimitSet):
    return saas.set_token_limit(payload)


@app.post("/api/admin/token-usage")
def api_report_token_usage(payload: TokenUsageReport):
    return saas.record_token_usage(payload)

@app.get("/state_mini")
def state_mini(environment_id: Optional[str] = None):
    env_id = environment_id or saas.get_current_environment()
    if not env_id:
        raise HTTPException(status_code=404, detail="Kein Environment ausgewählt")
    env_descriptor = saas.describe_environment_for_frontend(env_id)
    env_state = saas.get_environment_state(env_id)
    push = saas.get_push_state(env_id)
    return {
        "environment_id": env_id,
        "environment": env_descriptor,
        "npcs": env_state["npcs"],
        "players": env_state["players"],
        "push_enabled": push.get("enabled", False),
        "last_push_ok": push.get("last_ok", True),
        "last_push_msg": push.get("last_msg", ""),
    }

@app.post("/move")
async def move(req: Request):
    body = await req.json()
    env_id = body.get("environment_id") or saas.get_current_environment()
    if not env_id:
        raise HTTPException(status_code=400, detail="Kein Environment aktiv")
    who = body.get("who")
    x = float(body.get("x", 0.0))
    y = float(body.get("y", 0.0))
    saas.move_entity(env_id, who, x, y, player_id=body.get("player_id"))
    return {"ok": True}

@app.post("/push_once")
async def push_once():
    env_id = saas.get_current_environment()
    if not env_id:
        raise HTTPException(status_code=400, detail="Kein Environment aktiv")
    ok, msg = await _push_positions(env_id)
    return {"ok": ok, "msg": msg}

@app.post("/toggle_push")
async def toggle_push(req: Request):
    data: Dict[str, Any] = {}
    try:
        if int(req.headers.get("content-length", "0")) > 0:
            data = await req.json()
    except Exception:
        data = {}
    env_id = (
        data.get("environment_id")
        or req.query_params.get("environment_id")
        or saas.get_current_environment()
    )
    if not env_id:
        raise HTTPException(status_code=400, detail="Kein Environment aktiv")
    push_state = saas.toggle_push(env_id)
    return {"ok": True, "push_enabled": push_state.get("enabled", False)}

@app.get("/", response_class=HTMLResponse)
def index():
    html = """<!doctype html><html><head><meta charset="utf-8"/><title>Mini-Map</title>
<style>
body { font-family: ui-sans-serif, system-ui; margin:0; display:flex; height:100vh; color:#e5e7eb; }
#left { flex:1; display:flex; align-items:center; justify-content:center; background:#0b1220; padding:16px; box-sizing:border-box; }
#right { width:480px; background:#111827; padding:16px; overflow:auto; box-sizing:border-box; }
canvas { background:#0f172a; border:1px solid #374151; border-radius:8px; box-shadow:0 12px 36px rgba(0,0,0,0.35); max-width:100%; height:auto; }
button { margin-right:8px; margin-bottom:8px; background:#2563eb; border:none; color:#e5e7eb; padding:6px 10px; border-radius:6px; cursor:pointer; transition:background 0.2s ease; }
button.secondary { background:#374151; }
button:disabled { opacity:0.55; cursor:not-allowed; }
.row { margin:12px 0; }
.row-sub { font-size:0.9rem; color:#cbd5f5; margin:6px 0; }
.row h3 { margin:0 0 6px 0; font-size:1rem; color:#93c5fd; }
code { background:#1f2937; padding:2px 6px; border-radius:4px; display:inline-block; margin-top:4px; }
label { display:block; font-size:0.85rem; margin-bottom:4px; color:#9ca3af; }
select, textarea { width:100%; padding:6px; border-radius:6px; background:#1f2937; color:#e5e7eb; border:1px solid #374151; margin-bottom:6px; font-family:inherit; box-sizing:border-box; }
textarea { resize:vertical; }
small { color:#9ca3af; }
.map-info { font-size:0.85rem; color:#9ca3af; background:#0f172a; border:1px solid #1f2937; padding:8px 10px; border-radius:8px; }
ul { list-style:none; padding-left:16px; margin:0; }
li { margin:2px 0; font-size:0.9rem; }
.status { font-size:0.85rem; color:#9ca3af; }
.button-row { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:6px; }
.npc-prompts { display:flex; flex-direction:column; gap:12px; }
.prompt-block { background:#151c2f; border:1px solid #1f2937; border-radius:8px; padding:10px; }
.prompt-block label { font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em; color:#c7d2fe; margin-bottom:4px; }
.prompt-block textarea { min-height:78px; }
#promptContainer .prompt-actions { display:flex; justify-content:space-between; align-items:center; margin-top:6px; font-size:0.75rem; color:#9ca3af; }
#conversationLog { background:#0f172a; border:1px solid #1f2937; border-radius:8px; padding:10px; max-height:240px; overflow-y:auto; font-size:0.9rem; }
.log-entry { margin-bottom:6px; line-height:1.4; }
.log-entry:last-child { margin-bottom:0; }
.log-entry strong { color:#bfdbfe; }
#finalSolution { margin-top:10px; padding:10px; border-radius:8px; border:1px solid rgba(37,99,235,0.5); background:#1d283a; color:#facc15; font-weight:600; display:none; }
#finalSolution.visible { display:block; }
.small-note { font-size:0.8rem; color:#6b7280; margin-bottom:6px; }
.weights { display:flex; flex-direction:column; gap:10px; }
.slider-row { display:flex; align-items:center; gap:12px; background:#151c2f; border:1px solid #1f2937; border-radius:8px; padding:8px 10px; }
.slider-row label { flex:1; font-size:0.85rem; color:#c7d2fe; text-transform:uppercase; letter-spacing:0.03em; }
.slider-row input[type=range] { flex:1.4; }
.slider-row span.value { width:52px; text-align:right; font-variant-numeric:tabular-nums; color:#facc15; font-size:0.85rem; }
.slider-row button.remove { background:#b91c1c; margin-bottom:0; }
.slider-row button.remove:hover { background:#ef4444; }
</style></head><body>
<div id="left"><canvas id="map" width="980" height="840"></canvas></div>
<div id="right">
  <h2>KI-NPC Control Center</h2>
  <div class="row map-info"><div>Kartengröße: <span id="mapMeta">lädt...</span></div></div>
  <div class="row">
    <h3>Environment</h3>
    <label for="environmentSelect">Environment auswählen</label>
    <select id="environmentSelect"></select>
    <div class="status" id="environmentSummary"></div>
  </div>
  <div class="row">
    <h3>Bot &amp; Push</h3>
    <div class="row-sub">Bot Update URL:<br><code id="cfg"></code></div>
    <div class="button-row">
      <button id="toggle">Push an/aus</button>
      <button id="pushonce" class="secondary">Push einmal</button>
      <button id="refreshState" class="secondary">State aktualisieren</button>
    </div>
    <div class="row-sub">
      <div>Letzter Push: <span id="last"></span></div>
      <div>Push aktiv: <span id="active"></span></div>
    </div>
  </div>
  <div class="row">
    <h3>Gewichtung der Lösung</h3>
    <div id="weightingContainer" class="weights"></div>
    <div class="button-row">
      <button id="addCriterion" class="secondary">Kriterium hinzufügen</button>
    </div>
    <small class="small-note">Gewichte bestimmen, wie der Moderator Lösungen bewertet. Werte werden automatisch gespeichert.</small>
  </div>
  <div class="row">
    <h3>Gesprächssteuerung</h3>
    <div class="button-row">
      <button id="startConversation">Start</button>
      <button id="stopConversation" class="secondary" disabled>Stop</button>
      <button id="resetConversation" class="secondary">Refresh</button>
    </div>
    <label for="taskDescription">Aufgabenstellung</label>
    <textarea id="taskDescription" rows="3" placeholder="Beschreibe die Aufgabe, die moderiert werden soll..."></textarea>
    <div class="small-note">Die Aufgabenstellung wird als Kontext an Moderator und NPCs übergeben.</div>
    <div id="promptContainer" class="npc-prompts"></div>
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
const BOT_BASE = `http://__BOT_HOST__:__BOT_PORT__`;
const NPC_COLORS = ['#38bdf8', '#fbbf24', '#34d399', '#f472b6', '#60a5fa'];
const PLAYER_COLOR = '#a78bfa';
const DEFAULT_MAP = { width_m: 84, height_m: 60, grid_m: 2, pixels_per_meter: 14 };
const DEFAULT_NPCS = [
  { id: 'npc1', name: 'Moderator', x: 28.0, y: 24.0 },
  { id: 'npc2', name: 'NPC 1', x: 42.0, y: 20.0 },
  { id: 'npc3', name: 'NPC 2', x: 46.0, y: 28.0 }
];
const DEFAULT_PLAYERS = [
  { id: 'p1', name: 'Spieler1', x: 44.0, y: 40.0 }
];

let mapConfig = { ...DEFAULT_MAP };
let pixelsPerMeter = mapConfig.pixels_per_meter;
let npcs = DEFAULT_NPCS.map((npc) => ({ ...npc }));
let players = DEFAULT_PLAYERS.map((player) => ({ ...player }));
let greetRadius = 4.0;
let currentEnvironmentId = null;
let environmentAgents = {};
let weightingsState = { solution_focus: 0.45, speed: 0.25, thoroughness: 0.3, custom: [] };
let dragging = null;
let suspendRefresh = false;
let promptUpdateTimers = {};
let weightUpdateTimer = null;
let taskUpdateTimer = null;
let environmentOptions = [];

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

function applyMapConfig(cfg = {}) {
  mapConfig = { ...mapConfig, ...cfg };
  pixelsPerMeter = mapConfig.pixels_per_meter || 14;
  const widthPx = Math.round((mapConfig.width_m || 84) * pixelsPerMeter);
  const heightPx = Math.round((mapConfig.height_m || 60) * pixelsPerMeter);
  canvas.width = widthPx;
  canvas.height = heightPx;
  const mapMeta = document.getElementById('mapMeta');
  if (mapMeta) {
    mapMeta.textContent = `${mapConfig.width_m.toFixed(0)} m × ${mapConfig.height_m.toFixed(0)} m | Raster ${mapConfig.grid_m} m`;
  }
  draw();
}

function canvasPosToWorld(e) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) / pixelsPerMeter;
  const y = (e.clientY - rect.top) / pixelsPerMeter;
  return { x, y };
}

function drawGrid() {
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 1;
  const step = Math.max(1, mapConfig.grid_m || 2) * pixelsPerMeter;
  for (let x = 0; x < canvas.width; x += step) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }
  for (let y = 0; y < canvas.height; y += step) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }
}

function drawCircle(worldX, worldY, radius, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  ctx.arc(worldX * pixelsPerMeter, worldY * pixelsPerMeter, radius * pixelsPerMeter, 0, Math.PI * 2);
  ctx.stroke();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGrid();
  ctx.font = '13px sans-serif';
  ctx.fillStyle = '#e5e7eb';
  ctx.fillText(`1 Kästchen = ${mapConfig.grid_m} m`, 10, 20);

  npcs.forEach((npc, idx) => {
    drawCircle(npc.x, npc.y, greetRadius, 'rgba(56,189,248,0.08)');
    ctx.fillStyle = NPC_COLORS[idx % NPC_COLORS.length];
    ctx.beginPath();
    ctx.arc(npc.x * pixelsPerMeter, npc.y * pixelsPerMeter, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#e5e7eb';
    ctx.fillText(`${npc.name || npc.id}`, npc.x * pixelsPerMeter + 12, npc.y * pixelsPerMeter - 10);
  });

  players.forEach((pl) => {
    ctx.fillStyle = PLAYER_COLOR;
    ctx.beginPath();
    ctx.arc(pl.x * pixelsPerMeter, pl.y * pixelsPerMeter, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#e5e7eb';
    ctx.fillText(`${pl.name || pl.id}`, pl.x * pixelsPerMeter + 12, pl.y * pixelsPerMeter - 10);
  });
}

function updateEnvironmentSummary(env) {
  const summaryEl = document.getElementById('environmentSummary');
  if (!summaryEl) return;
  if (!env) {
    summaryEl.textContent = 'Kein Environment ausgewählt.';
    return;
  }
  const baseWeights = env.weights || {};
  const toPercent = (val, fallback) => Math.round(((typeof val === 'number' ? val : fallback) || 0) * 100);
  const parts = [
    `Lösungsorientiert ${toPercent(baseWeights.solution_focus, 0.45)}%`,
    `Schnelligkeit ${toPercent(baseWeights.speed, 0.25)}%`,
    `Gründlichkeit ${toPercent(baseWeights.thoroughness, 0.3)}%`,
  ];
  const custom = Array.isArray(baseWeights.custom) ? baseWeights.custom.length : 0;
  if (custom) {
    parts.push(`+${custom} individuelle Kriterien`);
  }
  const desc = env.description ? ` – ${env.description}` : '';
  summaryEl.textContent = `${env.name}${desc}. ${parts.join(' | ')}`;
}

function renderPromptFields(agentList) {
  const container = document.getElementById('promptContainer');
  if (!container) return;
  container.innerHTML = '';
  environmentAgents = {};
  const agents = Array.isArray(agentList) ? agentList : [];
  const relevant = agents.filter((agent) => ['moderator', 'npc'].includes(agent.role || 'npc'));
  relevant.forEach((agent) => {
    environmentAgents[agent.id] = { ...agent };
    const block = document.createElement('div');
    block.className = 'prompt-block';
    const label = document.createElement('label');
    label.textContent = `${agent.name || agent.id} (${agent.role === 'moderator' ? 'Moderator' : 'NPC'})`;
    block.appendChild(label);
    const textarea = document.createElement('textarea');
    textarea.rows = 3;
    textarea.value = agent.prompt || '';
    textarea.dataset.npcPrompt = agent.id;
    textarea.addEventListener('input', (e) => {
      const value = e.target.value;
      environmentAgents[agent.id].prompt = value;
      schedulePromptUpdate(agent.id, value);
    });
    block.appendChild(textarea);
    const info = document.createElement('div');
    info.className = 'prompt-actions';
    info.innerHTML = `<span>Voice: ${agent.voice_id || 'Standard'}</span><span>ID: ${agent.id}</span>`;
    block.appendChild(info);
    container.appendChild(block);
  });
  if (!relevant.length) {
    const empty = document.createElement('div');
    empty.className = 'status';
    empty.textContent = 'Keine NPC-Prompts hinterlegt.';
    container.appendChild(empty);
  }
}

function createSliderRow({ key, label, value, isCustom = false, id }) {
  const row = document.createElement('div');
  row.className = 'slider-row';
  const lab = document.createElement('label');
  lab.textContent = label;
  row.appendChild(lab);
  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = '0';
  slider.max = '100';
  slider.step = '5';
  slider.value = Math.round(((typeof value === 'number' ? value : 0) || 0) * 100);
  const valueEl = document.createElement('span');
  valueEl.className = 'value';
  valueEl.textContent = `${slider.value}%`;
  slider.addEventListener('input', (e) => {
    valueEl.textContent = `${e.target.value}%`;
  });
  slider.addEventListener('change', (e) => {
    const fraction = Math.max(0, Math.min(100, parseFloat(e.target.value))) / 100;
    if (isCustom) {
      const idx = weightingsState.custom.findIndex((item) => item.id === id);
      if (idx >= 0) {
        weightingsState.custom[idx].value = fraction;
      }
    } else {
      weightingsState[key] = fraction;
    }
    scheduleWeightUpdate();
  });
  row.appendChild(slider);
  row.appendChild(valueEl);
  if (isCustom) {
    const removeBtn = document.createElement('button');
    removeBtn.className = 'secondary remove';
    removeBtn.textContent = 'Entfernen';
    removeBtn.addEventListener('click', () => removeCustomCriterion(id));
    row.appendChild(removeBtn);
  }
  return row;
}

function renderWeightings(weights) {
  const container = document.getElementById('weightingContainer');
  if (!container) return;
  const base = weights || {};
  weightingsState = {
    solution_focus: typeof base.solution_focus === 'number' ? base.solution_focus : 0.45,
    speed: typeof base.speed === 'number' ? base.speed : 0.25,
    thoroughness: typeof base.thoroughness === 'number' ? base.thoroughness : 0.3,
    custom: Array.isArray(base.custom) ? base.custom.map((item) => ({ ...item })) : [],
  };
  container.innerHTML = '';
  [
    { key: 'solution_focus', label: 'Lösungsorientiert', value: weightingsState.solution_focus },
    { key: 'speed', label: 'Schnelligkeit', value: weightingsState.speed },
    { key: 'thoroughness', label: 'Gründlichkeit', value: weightingsState.thoroughness },
  ].forEach((cfg) => container.appendChild(createSliderRow(cfg)));
  weightingsState.custom.forEach((item) => {
    container.appendChild(
      createSliderRow({ key: item.id, label: item.label || item.id, value: item.value, isCustom: true, id: item.id })
    );
  });
}

function scheduleWeightUpdate() {
  if (!currentEnvironmentId) return;
  if (weightUpdateTimer) window.clearTimeout(weightUpdateTimer);
  weightUpdateTimer = window.setTimeout(async () => {
    weightUpdateTimer = null;
    try {
      await fetch(`/api/environments/${currentEnvironmentId}/weights`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          solution_focus: weightingsState.solution_focus,
          speed: weightingsState.speed,
          thoroughness: weightingsState.thoroughness,
          custom: weightingsState.custom.map((item) => ({ id: item.id, label: item.label, value: item.value })),
        }),
      });
    } catch (err) {
      console.warn('Konnte Gewichtungen nicht speichern', err);
    }
  }, 600);
}

async function addCustomCriterion() {
  if (!currentEnvironmentId) return;
  const label = window.prompt('Name des neuen Kriteriums');
  if (!label) return;
  try {
    const res = await fetch(`/api/environments/${currentEnvironmentId}/weights/custom`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, value: 0.2 }),
    });
    if (res.ok) {
      await refreshLocal(true);
    }
  } catch (err) {
    console.warn('Konnte Kriterium nicht anlegen', err);
  }
}

async function removeCustomCriterion(customId) {
  if (!currentEnvironmentId || !customId) return;
  try {
    await fetch(`/api/environments/${currentEnvironmentId}/weights/custom/${customId}`, { method: 'DELETE' });
    await refreshLocal(true);
  } catch (err) {
    console.warn('Konnte Kriterium nicht entfernen', err);
  }
}

async function loadEnvironmentOptions() {
  try {
    const res = await fetch('/api/environments');
    const data = await res.json();
    environmentOptions = Array.isArray(data) ? data : [];
    populateEnvironmentSelect();
  } catch (err) {
    console.warn('Konnte Environment-Liste nicht laden', err);
  }
}

function populateEnvironmentSelect() {
  const select = document.getElementById('environmentSelect');
  if (!select) return;
  select.innerHTML = '';
  if (!Array.isArray(environmentOptions) || environmentOptions.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = 'Keine Environments vorhanden';
    opt.disabled = true;
    opt.selected = true;
    select.appendChild(opt);
    return;
  }
  environmentOptions.forEach((env) => {
    const opt = document.createElement('option');
    opt.value = env.id;
    opt.textContent = env.name;
    if (env.id === currentEnvironmentId) opt.selected = true;
    select.appendChild(opt);
  });
}

function schedulePromptUpdate(npcId, value) {
  if (!currentEnvironmentId || !npcId) return;
  if (promptUpdateTimers[npcId]) window.clearTimeout(promptUpdateTimers[npcId]);
  promptUpdateTimers[npcId] = window.setTimeout(async () => {
    promptUpdateTimers[npcId] = null;
    try {
      await fetch(`/api/environments/${currentEnvironmentId}/agents/${npcId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: value }),
      });
    } catch (err) {
      console.warn('Konnte Prompt nicht speichern', err);
    }
  }, 600);
}

function scheduleTaskUpdate(value) {
  if (!currentEnvironmentId) return;
  if (taskUpdateTimer) window.clearTimeout(taskUpdateTimer);
  taskUpdateTimer = window.setTimeout(async () => {
    taskUpdateTimer = null;
    try {
      await fetch(`/api/environments/${currentEnvironmentId}/task`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task: value || '' }),
      });
    } catch (err) {
      console.warn('Konnte Aufgabe nicht speichern', err);
    }
  }, 600);
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
    await sendMove(npc.id, npc.x, npc.y);
  } else if (dragging.type === 'player') {
    const pl = players[dragging.index];
    pl.x = pos.x; pl.y = pos.y;
    await sendMove('player', pl.x, pl.y, pl.id);
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
  const url = currentEnvironmentId ? `/state_mini?environment_id=${encodeURIComponent(currentEnvironmentId)}` : '/state_mini';
  try {
    const r = await fetch(url);
    if (r.status === 404) {
      console.warn('Kein Environment verfügbar');
      return null;
    }
    const s = await r.json();
    currentEnvironmentId = s.environment_id || currentEnvironmentId;
    if (s.environment) {
      applyMapConfig(s.environment.map || DEFAULT_MAP);
      renderPromptFields(s.environment.agents || []);
      renderWeightings(s.environment.weights || {});
      updateEnvironmentSummary(s.environment);
      const taskField = document.getElementById('taskDescription');
      if (taskField && document.activeElement !== taskField) {
        taskField.value = s.environment.task || '';
      }
    }
    greetRadius = Math.min(4, (s.environment && s.environment.map && s.environment.map.radius_m) || greetRadius);
    document.getElementById('radius').textContent = greetRadius.toFixed(1);
    const npcList = (s.npcs && s.npcs.length ? s.npcs : DEFAULT_NPCS).map((npc) => ({ ...npc }));
    const playerList = (s.players && s.players.length ? s.players : DEFAULT_PLAYERS).map((player) => ({ ...player }));
    npcs = ensureDefaults(npcList, DEFAULT_NPCS);
    players = ensureDefaults(playerList, DEFAULT_PLAYERS);
    document.getElementById('last').textContent = s.last_push_ok ? (`OK: ${s.last_push_msg}`) : (`Fehler: ${s.last_push_msg}`);
    document.getElementById('active').textContent = s.push_enabled ? 'AN' : 'AUS';
    if (!environmentOptions.length) {
      loadEnvironmentOptions();
    } else {
      populateEnvironmentSelect();
    }
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
    if (s.current_environment_id) {
      currentEnvironmentId = s.current_environment_id;
    }
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
  if (environmentAgents[id] && environmentAgents[id].prompt) {
    return environmentAgents[id].prompt.trim();
  }
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

async function sendMove(id, x, y, playerId=null) {
  try {
    const payload = { who: id, x, y, environment_id: currentEnvironmentId || undefined };
    if (playerId) payload.player_id = playerId;
    await fetch('/move', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
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

document.getElementById('toggle').onclick = async () => {
  await fetch(`/toggle_push?environment_id=${encodeURIComponent(currentEnvironmentId || '')}`, { method: 'POST' });
  refreshLocal(true);
};
document.getElementById('pushonce').onclick = async () => {
  await fetch('/push_once', { method: 'POST' });
  refreshLocal(true);
};
document.getElementById('refreshState').onclick = () => refreshLocal(true);
document.getElementById('addCriterion').onclick = () => addCustomCriterion();
const envSelect = document.getElementById('environmentSelect');
if (envSelect) {
  envSelect.addEventListener('change', async (e) => {
    const envId = e.target.value;
    if (!envId || envId === currentEnvironmentId) return;
    try {
      await fetch('/api/session/environment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ environment_id: envId })
      });
      currentEnvironmentId = envId;
      await refreshLocal(true);
    } catch (err) {
      console.warn('Konnte Environment nicht setzen', err);
    }
  });
}
const taskField = document.getElementById('taskDescription');
if (taskField) {
  taskField.addEventListener('input', (e) => {
    scheduleTaskUpdate(e.target.value);
  });
}
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

setInterval(() => refreshLocal(), 1200);
setInterval(refreshBotState, 4000);

applyMapConfig(DEFAULT_MAP);

async function boot() {
  await loadCfg();
  await loadEnvironmentOptions();
  await refreshLocal(true);
  refreshBotState();
  refreshDevices(false);
}

boot();
</script></body></html>
"""
    return HTMLResponse(html.replace("__BOT_HOST__", BOT_HOST).replace("__BOT_PORT__", str(BOT_PORT)))

async def _push_positions(env_id: str):
    env_state = saas.get_environment_state(env_id)
    env_descriptor = saas.describe_environment_for_bot(env_id)
    payload = {
        "environment_id": env_id,
        "environment": env_descriptor,
        "npcs": env_state["npcs"],
        "players": env_state["players"],
    }
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.post(BOT_UPDATE_URL, json=payload)
        if r.status_code == 200:
            saas.record_push(env_id, True, "gesendet")
            return True, "OK"
        else:
            msg = f"HTTP {r.status_code}"
            saas.record_push(env_id, False, msg)
            return False, msg
    except Exception as e:
        msg = f"Fehler: {e}"
        saas.record_push(env_id, False, msg)
        return False, msg

@app.on_event("startup")
async def start_pusher():
    async def loop():
        await asyncio.sleep(0.5)
        while True:
            for env_id in saas.list_environment_ids():
                push_state = saas.get_push_state(env_id)
                if push_state.get("enabled", False):
                    await _push_positions(env_id)
            await asyncio.sleep(PUSH_INTERVAL_SEC)
    asyncio.create_task(loop())
