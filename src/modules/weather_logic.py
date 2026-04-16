import json
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, List, Tuple, Dict
from urllib.parse import quote
from urllib.request import Request, urlopen

# ══════════════════════════════════════════════════════════════════════
#  Data models
# ══════════════════════════════════════════════════════════════════════

@dataclass
class HourlySlot:
    time_label:  str          # "06:00"
    temp_f:      str
    feels_f:     str
    desc:        str
    wind_mph:    str
    wind_dir:    str
    rain_chance: int
    snow_chance: int
    uv_index:    str
    humidity:    str
    cloud_cover: str
    is_current:  bool = False


@dataclass
class DayForecast:
    date:              str
    desc:              str
    high_f:            str
    low_f:             str
    uv_index:          str
    rain_chance:       int
    snow_chance:       int
    sunrise:           str
    sunset:            str
    moon_phase:        str
    moon_illumination: str
    hourly:            List[HourlySlot] = field(default_factory=list)


@dataclass
class WeatherReport:
    location_label:   str
    lat:              str
    lon:              str
    population:       str
    queried_location: str
    desc:             str
    temp_f:           str
    feels_f:          str
    humidity:         str
    wind_mph:         str
    wind_dir:         str
    wind_deg:         str
    uv_index:         str
    visibility_mi:    str
    pressure_mb:      str
    precip_in:        str
    cloud_cover:      str
    obs_time:         str
    tz_name:         str = ""
    forecast:         List[DayForecast] = field(default_factory=list)


@dataclass
class MemorySnapshot:
    date:             str
    location_key:     str
    location_label:   str
    queried_location: str
    desc:             str
    temp_f:           str
    feels_f:          str
    humidity:         str
    wind_mph:         str
    wind_dir:         str
    uv_index:         str
    precip_in:        str
    cloud_cover:      str
    obs_time:         str


@dataclass
class WeatherComparison:
    found:         bool
    previous_date: str       = ""
    summary:       str       = ""
    bullets:       List[str] = field(default_factory=list)

# ══════════════════════════════════════════════════════════════════════
#  Geocoding + cache
# ══════════════════════════════════════════════════════════════════════

_geocache: Dict[str, Tuple[float, float]] = {}
_geocache_lock = threading.Lock()
GEOCACHE_PATH = Path.home() / ".weather_geocache.json"

def _load_geocache() -> None:
    global _geocache
    if GEOCACHE_PATH.exists():
        try:
            raw = json.loads(GEOCACHE_PATH.read_text())
            _geocache = {k: tuple(v) for k, v in raw.items()}  # type: ignore
        except Exception:
            _geocache = {}


def _save_geocache() -> None:
    GEOCACHE_PATH.write_text(
        json.dumps({k: list(v) for k, v in _geocache.items()}, indent=2)
    )


def geocode(location: str, timeout: int = 20) -> Optional[Tuple[float, float]]:
    key = location.lower().strip()
    with _geocache_lock:
        if key in _geocache:
            return _geocache[key]

    url = (
        "https://nominatim.openstreetmap.org/search"
        f"?q={quote(location)}&format=json&limit=1&addressdetails=0"
    )
    req = Request(url, headers={"User-Agent": "weather-agent/2.1 (local)"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            results = json.loads(resp.read().decode())
        if results:
            coords: Tuple[float, float] = (float(results[0]["lat"]), float(results[0]["lon"]))
            with _geocache_lock:
                _geocache[key] = coords
                _save_geocache()
            return coords
    except Exception:
        pass
    return None

# ══════════════════════════════════════════════════════════════════════
#  Weather fetch + parse
# ══════════════════════════════════════════════════════════════════════

def _fv(lst: list, key: str, fb: str = "N/A") -> str:
    if isinstance(lst, list) and lst:
        return str(lst[0].get(key) or fb).strip()
    return fb

def _parse_time(raw: str) -> str:
    try:
        h = int(raw) // 100
        return f"{h:02d}:00"
    except Exception:
        return raw

def _parse_hourly(day: dict, today: bool) -> List[HourlySlot]:
    slots: List[HourlySlot] = []
    now_h = datetime.now().hour if today else -1
    for h in day.get("hourly", []):
        label  = _parse_time(str(h.get("time", "0")))
        slot_h = int(h.get("time", 0)) // 100
        is_now = today and (slot_h <= now_h < slot_h + 3)
        slots.append(HourlySlot(
            time_label  = label,
            temp_f      = h.get("tempF",          "N/A"),
            feels_f     = h.get("FeelsLikeF",     "N/A"),
            desc        = _fv(h.get("weatherDesc", [{"value": ""}]), "value", ""),
            wind_mph    = h.get("windspeedMiles",  "N/A"),
            wind_dir    = h.get("winddir16Point",  ""),
            rain_chance = int(h.get("chanceofrain",  0)),
            snow_chance = int(h.get("chanceofsnow",  0)),
            uv_index    = str(h.get("uvIndex",     "")),
            humidity    = h.get("humidity",        "N/A"),
            cloud_cover = h.get("cloudcover",      "N/A"),
            is_current  = is_now,
        ))
    return slots

def _get_tz(lat: str, lon: str) -> str:
    try:
        from timezonefinder import TimezoneFinder
        tf = TimezoneFinder()
        return tf.timezone_at(lat=float(lat), lng=float(lon)) or 'UTC'
    except ImportError:
        return 'UTC'

def _obs_to_local(obs_str: str, tz_name: str) -> str:
    if not obs_str: return ""
    try:
        from zoneinfo import ZoneInfo
        t = datetime.strptime(obs_str.strip(), "%I:%M %p")
        utc_dt = datetime.now(tz=timezone.utc).replace(hour=t.hour, minute=t.minute)
        local = utc_dt.astimezone(ZoneInfo(tz_name))
        return local.strftime("%I:%M %p %Z").lstrip("0")
    except Exception:
        return obs_str.strip() + " UTC"

def parse_weather(data: dict, queried: str) -> WeatherReport:
    cur = (data.get("current_condition") or [{}])[0]
    near = (data.get("nearest_area") or [{}])[0]
    days = data.get("weather") or []
    area = _fv(near.get("areaName", [{"value": queried}]), "value", queried)
    region = _fv(near.get("region", []), "value", "")
    country = _fv(near.get("country", []), "value", "")
    label = ", ".join(dict.fromkeys([v for v in (area, region, country) if v and v != "N/A"]))
    forecast: List[DayForecast] = []
    today_date = Date.today().isoformat()
    for i, day in enumerate(days[:3]):
        astro = (day.get("astronomy") or [{}])[0]
        hourly = day.get("hourly") or []
        mid = hourly[len(hourly)//2] if hourly else {}
        is_today = (day.get("date", "") == today_date) or (i == 0)
        forecast.append(DayForecast(
            date=day.get("date", ""),
            desc=_fv(mid.get("weatherDesc", [{"value": ""}]), "value", ""),
            high_f=day.get("maxtempF", "N/A"),
            low_f=day.get("mintempF", "N/A"),
            uv_index=str(day.get("uvIndex", "")),
            rain_chance=max((int(h.get("chanceofrain", 0)) for h in hourly), default=0),
            snow_chance=max((int(h.get("chanceofsnow", 0)) for h in hourly), default=0),
            sunrise=astro.get("sunrise", ""), sunset=astro.get("sunset", ""),
            moon_phase=astro.get("moon_phase", ""), moon_illumination=astro.get("moon_illumination", ""),
            hourly=_parse_hourly(day, is_today),
        ))
    return WeatherReport(
        location_label=label, lat=near.get("latitude", ""), lon=near.get("longitude", ""),
        population=near.get("population", ""), queried_location=queried,
        desc=_fv(cur.get("weatherDesc", [{"value": "Unknown"}]), "value", "Unknown"),
        temp_f=cur.get("temp_F", "N/A"), feels_f=cur.get("FeelsLikeF", "N/A"),
        humidity=cur.get("humidity", "N/A"), wind_mph=cur.get("windspeedMiles", "N/A"),
        wind_dir=cur.get("winddir16Point", ""), wind_deg=cur.get("winddirDegree", ""),
        uv_index=cur.get("uvIndex", ""), visibility_mi=cur.get("visibilityMiles", "N/A"),
        pressure_mb=cur.get("pressure", "N/A"), precip_in=cur.get("precipInches", "0.0"),
        cloud_cover=cur.get("cloudcover", "N/A"), obs_time=cur.get("observation_time", ""),
        tz_name=_get_tz(near.get("latitude", ""), near.get("longitude", "")),
        forecast=forecast,
    )

def fetch_weather(location: str, coords: Optional[Tuple[float, float]], timeout: int = 20) -> WeatherReport:
    query_str = f"{coords[0]:.4f},{coords[1]:.4f}" if coords else location.strip()
    url = f"https://wttr.in/{quote(query_str, safe=',')}?format=j1"
    req = Request(url, headers={"User-Agent": "curl/8.0", "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return parse_weather(data, location.strip())

# ══════════════════════════════════════════════════════════════════════
#  Memory
# ══════════════════════════════════════════════════════════════════════

MEMORY_PATH = Path.home() / ".weather_agent_memory.json"

def _loc_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")

def load_memory() -> List[MemorySnapshot]:
    if not MEMORY_PATH.exists(): return []
    try:
        raw = json.loads(MEMORY_PATH.read_text())
        return [MemorySnapshot(**item) for item in raw.get("snapshots", [])]
    except Exception: return []

def save_memory(snapshots: List[MemorySnapshot]) -> None:
    MEMORY_PATH.write_text(json.dumps({"snapshots": [asdict(s) for s in snapshots]}, indent=2))

def find_prev_snapshot(r: WeatherReport, snaps: List[MemorySnapshot]) -> Optional[MemorySnapshot]:
    key = _loc_key(r.queried_location or r.location_label or "unknown")
    today = Date.today()
    cands: List[Tuple[Date, MemorySnapshot]] = []
    for s in snaps:
        if s.location_key != key: continue
        try: d = Date.fromisoformat(s.date)
        except Exception: continue
        if d < today: cands.append((d, s))
    if not cands: return None
    exact = [s for d, s in cands if d == today - timedelta(days=1)]
    if exact: return exact[-1]
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

def upsert_snapshot(snaps: List[MemorySnapshot], cur: MemorySnapshot) -> List[MemorySnapshot]:
    for i, s in enumerate(snaps):
        if s.location_key == cur.location_key and s.date == cur.date:
            snaps[i] = cur
            return snaps[-500:]
    snaps.append(cur)
    snaps.sort(key=lambda s: (s.location_key, s.date))
    return snaps[-500:]

def snapshot_from_report(r: WeatherReport) -> MemorySnapshot:
    return MemorySnapshot(
        date=Date.today().isoformat(),
        location_key=_loc_key(r.queried_location or r.location_label or "unknown"),
        location_label=r.location_label, queried_location=r.queried_location,
        desc=r.desc, temp_f=r.temp_f, feels_f=r.feels_f, humidity=r.humidity,
        wind_mph=r.wind_mph, wind_dir=r.wind_dir, uv_index=r.uv_index,
        precip_in=r.precip_in, cloud_cover=r.cloud_cover, obs_time=r.obs_time
    )

# ══════════════════════════════════════════════════════════════════════
#  Logic / Comparison
# ══════════════════════════════════════════════════════════════════════

def _safe_int(v: Any) -> Optional[int]:
    try: return int(str(v).strip())
    except Exception: return None

def compare(prev: Optional[MemorySnapshot], cur: WeatherReport) -> WeatherComparison:
    if prev is None:
        return WeatherComparison(found=False, summary="No prior weather memory for this location.")
    bullets: List[str] = []
    def _d(now: Any, old: Any, label: str, unit: str) -> None:
        a, b = _safe_int(now), _safe_int(old)
        if a is not None and b is not None:
            diff = a - b
            if diff: bullets.append(f"{label} {'up' if diff>0 else 'down'} {abs(diff)}{unit}")
            else: bullets.append(f"{label} unchanged")
    _d(cur.temp_f, prev.temp_f, "Temp", "°F")
    _d(cur.feels_f, prev.feels_f, "Feels-like", "°F")
    _d(cur.humidity, prev.humidity, "Humidity", " pp")
    _d(cur.wind_mph, prev.wind_mph, "Wind", " mph")
    if cur.desc.lower() != prev.desc.lower(): bullets.append(f"Conditions: '{prev.desc}' → '{cur.desc}'")
    if cur.wind_dir and prev.wind_dir and cur.wind_dir != prev.wind_dir:
        bullets.append(f"Wind direction: {prev.wind_dir} → {cur.wind_dir}")
    if not bullets: bullets.append(f"Conditions similar to {prev.date}")
    return WeatherComparison(found=True, previous_date=prev.date, summary=". ".join(bullets) + ".", bullets=bullets)

# ══════════════════════════════════════════════════════════════════════
#  Location Extraction
# ══════════════════════════════════════════════════════════════════════

_LOC_RE = [
    r"(?:current )?weather (?:in|for) (.+)$",
    r"(?:forecast|temperature|radar|satellite) (?:in|for) (.+)$",
    r"what(?:'s| is) (?:the )?(?:current )?weather (?:in|for) (.+)$",
    r"how (?:hot|cold|warm) is it in (.+)$",
    r"(?:rain|snow|storm) (?:in|near) (.+)$",
]

def extract_location(query: str) -> str:
    q = query.strip().rstrip(" ?.")
    for pat in _LOC_RE:
        m = re.search(pat, q, re.IGNORECASE)
        if m: return q[m.start(1):].strip().rstrip(" ?.")
    return ""

def _coord(lat: str, lon: str) -> str:
    try:
        la, lo = float(lat), float(lon)
        return f"{abs(la):.4f}°{'N' if la>=0 else 'S'}, {abs(lo):.4f}°{'E' if lo>=0 else 'W'}"
    except Exception: return f"{lat}, {lon}"

def _uv_label(uv: str) -> str:
    try:
        v = int(uv)
        w = ("Low","Moderate","High","Very High","Extreme")[0 if v<=2 else 1 if v<=5 else 2 if v<=7 else 3 if v<=10 else 4]
        return f"{v} {w}"
    except Exception: return uv or "N/A"
