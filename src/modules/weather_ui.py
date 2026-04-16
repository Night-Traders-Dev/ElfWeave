from datetime import date as Date
from typing import Any, Tuple, Optional
import shutil
from rich.console import Console
from rich.text import Text
from .weather_logic import WeatherReport, _coord, _uv_label, _obs_to_local

# ══════════════════════════════════════════════════════════════════════
#  Weather card helpers (Rich, printed after live UI closes)
# ══════════════════════════════════════════════════════════════════════

_COND_MAP = [
    ("thunder","⛈","bold yellow"), ("blizzard","❄","bold cyan"), ("snow","❄","bold cyan"),
    ("sleet","🌨","cyan"), ("drizzle","🌦","sky_blue1"), ("rain","🌧","sky_blue1"),
    ("shower","🌦","sky_blue1"), ("fog","🌫","grey70"), ("mist","🌫","grey70"),
    ("haze","🌫","grey70"), ("overcast","☁","grey70"), ("cloudy","🌥","grey74"),
    ("partly","⛅","yellow3"), ("clear","☀","bright_yellow"), ("sunny","☀","bright_yellow"),
    ("wind","💨","cyan"),
]

def _cstyle(desc: str) -> Tuple[str, str]:
    d = (desc or "").lower()
    for kw, em, col in _COND_MAP:
        if kw in d:
            return em, col
    return "🌡", "white"

def _uv_col(uv: str) -> str:
    try:
        v = int(uv)
        return ("green","yellow3","orange3","red","bright_red")[
            0 if v<=2 else 1 if v<=5 else 2 if v<=7 else 3 if v<=10 else 4]
    except Exception:
        return "white"

def _bar(val: str, step: int = 10, width: int = 10) -> str:
    try:
        v = int(val)
        filled = min(width, max(0, v // step))
        return "█" * filled + "░" * (width - filled)
    except Exception:
        return "?" * width

def _moon_em(phase: str) -> str:
    p = (phase or "").lower()
    m = {"new":"🌑","waxing crescent":"🌒","first quarter":"🌓","waxing gibbous":"🌔",
         "full":"🌕","waning gibbous":"🌖","last quarter":"🌗","waning crescent":"🌘"}
    for k, v in m.items():
        if k in p:
            return v
    return "🌙"

def _day_label(date_str: str, i: int) -> str:
    try:
        d = Date.fromisoformat(date_str)
        s = d.strftime("%a %b %d").replace(" 0", " ")
        return ("Today", "Tomorrow")[i] + f" ({s})" if i < 2 else s
    except Exception:
        return date_str

def _delta_text(now: Any, prev: Any, unit: str = "") -> Text:
    t = Text()
    try:
        d = int(str(now)) - int(str(prev))
        if d > 0:   t.append(f"+{d}{unit}", "green")
        elif d < 0: t.append(f"{d}{unit}", "red")
        else:       t.append(f"±0{unit}", "grey50")
    except Exception:
        pass
    return t

def _term_width(console: Console) -> int:
    try:
        return max(40, int(console.size.width))
    except Exception:
        return shutil.get_terminal_size((100, 30)).columns

def _compact_mode(console: Console) -> bool:
    w = _term_width(console)
    return w < 110

def _tiny_mode(console: Console) -> bool:
    w = _term_width(console)
    return w < 78

def _safe_text(value: Any, fallback: str = "—") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    return s if s else fallback

def _limit(s: str, width: int) -> str:
    s = str(s)
    if width <= 3:
        return s[:width]
    return s if len(s) <= width else s[: width - 1] + "…"

def _weather_ascii(desc: str, is_day: bool = True) -> str:
    d = (desc or "").lower()
    if "thunder" in d:
        return "   .-.   \n  (⚡ )  \n (___) ) \n  ʻ ʻ ʻ  "
    if "snow" in d or "blizzard" in d or "sleet" in d or "ice" in d:
        return "   .-.   \n  (   ). \n (___(__)\n  * * *  "
    if "rain" in d or "drizzle" in d or "shower" in d:
        return "   .-.   \n  (   ). \n (___(__)\n  ʻ ʻ ʻ  "
    if "fog" in d or "mist" in d or "haze" in d:
        return " _ - _ - \n  _ - _  \n _ - _ - \n  - _ -  "
    if "cloud" in d or "overcast" in d:
        return "    .--.  \n .-(    ).\n(___.__)__)"
    if is_day:
        return "   \\ | /  \n '-.☼.-' \n   / | \\  "
    return "   _..._  \n .:::::::. \n ::::::::: \n `:::::::' "
