#!/usr/bin/env python3
"""Bootstrap-Skript fuer den KI-NPC-Prototyp.

Dieses Skript sorgt dafuer, dass eine virtuelle Umgebung vorhanden ist,
installiert (oder aktualisiert) alle benoetigten Abhaengigkeiten und startet
anschliessend Mini-Map (FastAPI/Uvicorn) sowie den Bot. Es funktioniert sowohl
unter Windows als auch unter Linux/macOS.

Verwendung:
    python bootstrap.py            # installiert und startet alles
    python bootstrap.py --install-only   # nur Pakete installieren
    python bootstrap.py --reset-venv     # virtuelle Umgebung neu erstellen
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS_FILE = ROOT / "requirements.txt"
BOT_PRIMARY = ROOT / "npc_bot_proximity.py"
BOT_FALLBACK = ROOT / "npc_bot.py"


class BootstrapError(RuntimeError):
    """Sammel-Exception fuer bootstrapbezogene Fehler."""


def log(msg: str) -> None:
    print(f"[bootstrap] {msg}")


def python_executable() -> Path:
    """Pfad zur Python-Exe innerhalb der venv ermitteln."""
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def pip_executable() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"


def ensure_venv(reset: bool = False) -> None:
    """Stellt sicher, dass eine virtuelle Umgebung existiert."""
    if reset and VENV_DIR.exists():
        log("Entferne bestehende virtuelle Umgebung…")
        shutil.rmtree(VENV_DIR)
    if not VENV_DIR.exists():
        log("Erstelle virtuelle Umgebung (.venv)…")
        import venv

        builder = venv.EnvBuilder(with_pip=True, upgrade_deps=False)
        builder.create(VENV_DIR)
    py = python_executable()
    if not py.exists():
        raise BootstrapError("Python-Interpreter in der virtuellen Umgebung wurde nicht gefunden.")


def read_requirements() -> list[str]:
    """Liest requirements.txt ein und filtert Kommentare/Leerzeilen."""
    if not REQUIREMENTS_FILE.exists():
        log("requirements.txt nicht gefunden – verwende Minimalpakete.")
        return ["fastapi", "uvicorn"]
    packages: list[str] = []
    for raw in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        packages.append(line)
    return packages


def install_dependencies() -> None:
    ensure_venv()
    pip = pip_executable()
    if not pip.exists():
        raise BootstrapError("pip wurde in der virtuellen Umgebung nicht gefunden.")

    log("Aktualisiere pip…")
    subprocess.check_call([str(pip), "install", "--upgrade", "pip"])

    packages = read_requirements()
    if packages:
        log("Installiere benötigte Pakete…")
        cmd = [str(pip), "install", "--upgrade", *packages]
        subprocess.check_call(cmd)
    else:
        log("Keine Pakete in requirements.txt gefunden.")


def start_processes() -> None:
    """Startet App (uvicorn) und Bot parallel."""
    ensure_venv()
    python = python_executable()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPATH", str(ROOT))

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = os.getenv("APP_PORT", "8002")
    app_module = os.getenv("APP_MODULE", "app:app")

    app_cmd = [str(python), "-m", "uvicorn", app_module, "--host", host, "--port", str(port)]
    log("Starte Mini-Map (uvicorn)…")
    app_proc = subprocess.Popen(app_cmd, cwd=str(ROOT), env=env)

    bot_script: Path | None = None
    if BOT_PRIMARY.exists():
        bot_script = BOT_PRIMARY
    elif BOT_FALLBACK.exists():
        bot_script = BOT_FALLBACK

    bot_proc = None
    if bot_script is not None:
        log(f"Starte Bot-Skript {bot_script.name}…")
        bot_proc = subprocess.Popen([str(python), str(bot_script)], cwd=str(ROOT), env=env)
    else:
        log("Hinweis: Kein Bot-Skript gefunden – starte nur die Mini-Map.")

    log("Beide Prozesse laufen. STRG+C beendet sie.")
    try:
        app_proc.wait()
    except KeyboardInterrupt:
        log("Stoppsignal empfangen – beende Prozesse…")
    finally:
        for proc in (app_proc, bot_proc):
            if proc and proc.poll() is None:
                proc.terminate()
        for proc in (app_proc, bot_proc):
            if proc:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap fuer KI-NPC")
    parser.add_argument("--install-only", action="store_true", help="nur Abhaengigkeiten installieren")
    parser.add_argument("--reset-venv", action="store_true", help="virtuelle Umgebung neu aufbauen")
    parser.add_argument("--no-run", action="store_true", help="nach Installation nicht starten")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        ensure_venv(reset=args.reset_venv)
        install_dependencies()
        if not (args.install_only or args.no_run):
            start_processes()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Fehler beim Ausfuehren von {exc.cmd}: Exit-Code {exc.returncode}")
    except BootstrapError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
