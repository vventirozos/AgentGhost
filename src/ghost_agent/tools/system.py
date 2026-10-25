import asyncio
import datetime
import logging
import urllib.parse
import httpx
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None
from ..utils.logging import Icons, pretty_log, requester_is_member
from ..utils.helpers import request_new_tor_identity
from ..utils.egress_guard import resolve_egress_proxy

logger = logging.getLogger("GhostAgent")

def _server_name(resp) -> str:
    """The `Server:` header, bounded and printable — it is remote-controlled
    text that lands in a message the model reads (review, 2026-09-09)."""
    try:
        raw = str(resp.headers.get("server") or "").strip()
    except Exception:  # noqa: BLE001
        raw = ""
    raw = "".join(ch for ch in raw if ch.isprintable())[:40]
    return raw or "the server"


#: §4RE (2026-10-25): every weather answer carries a daily outlook, so "will
#: it rain tomorrow / at the weekend" needs no second call and no date
#: arithmetic by the model (each day is labelled with its weekday and date in
#: the PLACE's local calendar). `hourly=true` adds today and tomorrow in steps.
FORECAST_DAYS = 7
HOURLY_STEP_H = 3
_OM_CURRENT = "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
_OM_DAILY = ("weather_code,temperature_2m_max,temperature_2m_min,"
             "precipitation_probability_max,precipitation_sum,wind_speed_10m_max")
_OM_HOURLY = "temperature_2m,precipitation_probability,weather_code"


def _wants_hourly(value) -> bool:
    """A model may send the flag as a string."""
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("true", "1", "yes", "y")


def _http_client(proxy_url):
    """The one client every weather request uses (curl_cffi when present)."""
    if curl_requests:
        proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
        return curl_requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=20.0, verify=True)
    return httpx.AsyncClient(proxy=proxy_url, timeout=20.0, verify=True)


def _forecast_url(lat, lon, hourly: bool) -> str:
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&"
           f"current={_OM_CURRENT}&daily={_OM_DAILY}&forecast_days={FORECAST_DAYS}&"
           f"timezone=auto&wind_speed_unit=kmh")
    return url + (f"&hourly={_OM_HOURLY}" if hourly else "")


def _num(v, unit: str = "") -> str:
    """A reading as text, or "?" — a missing or non-numeric value never
    renders as "None" (a forecast array may carry nulls)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return "?"
    return f"{round(float(v), 1):g}{unit}"


def _cond(code) -> str:
    """A WMO code as words; a null code (a forecast array carries them) is "?"."""
    if code is None:
        return "?"
    return WMO_CONDITIONS.get(code, f"code {code}")


def _day_label(iso_date: str, index: int) -> str:
    """'Today Sun 25 Oct' — relative words from the forecast's own day order,
    so the label is the PLACE's calendar, not this host's."""
    try:
        d = datetime.date.fromisoformat(str(iso_date)[:10])
        stamp = d.strftime("%a %d %b").replace(" 0", " ")
    except (TypeError, ValueError):
        stamp = _wttr_text(iso_date, 10)
    return {0: "Today ", 1: "Tomorrow "}.get(index, "") + stamp


def _col(block: dict, key: str, i: int):
    try:
        return (block.get(key) or [])[i]
    except (IndexError, TypeError, AttributeError):
        return None


def _forecast_lines(js: dict, hourly: bool, now_s: str) -> list:
    """The daily outlook and (on request) the hourly rows of an Open-Meteo
    answer. Remote text is cut to bounded control-free lines (§4LV)."""
    lines = []
    tz = _wttr_text(js.get("timezone"), 40)
    daily = js.get("daily") or {}
    days = [str(d)[:10] for d in list(daily.get("time") or [])[:FORECAST_DAYS]]
    if days:
        lines.append(f"Daily forecast ({tz or 'local time'}):")
        for i, day in enumerate(days):
            lines.append(
                f"- {_day_label(day, i)}: {_cond(_col(daily, 'weather_code', i))}, "
                f"{_num(_col(daily, 'temperature_2m_min', i))}–{_num(_col(daily, 'temperature_2m_max', i))}°C, "
                f"chance of rain {_num(_col(daily, 'precipitation_probability_max', i), '%')}, "
                f"rain {_num(_col(daily, 'precipitation_sum', i), ' mm')}, "
                f"wind up to {_num(_col(daily, 'wind_speed_10m_max', i), ' km/h')}")
    if not hourly:
        return lines
    if len(now_s) < 13:                     # r1 m3: no "now" to start from — say so, show none
        lines.append("Hourly: not available (the source gave no local time).")
        return lines
    hb = js.get("hourly") or {}
    times = [str(t) for t in (hb.get("time") or [])]
    dates = days or sorted({t[:10] for t in times})          # r1 m4: bounded without the daily block
    first_day, last_day = (dates[0], dates[1] if len(dates) > 1 else dates[0]) if dates else ("", "")
    rows, seen = [], set()
    for i, t in enumerate(times):
        if (len(t) < 16 or t[:13] < now_s[:13] or not last_day or t[:10] > last_day
                or t[:16] in seen):                          # r1 m6: a repeated hour (DST) once
            continue
        try:
            if int(t[11:13]) % HOURLY_STEP_H:
                continue
        except ValueError:
            continue
        seen.add(t[:16])
        label = "Today" if t[:10] == first_day else ("Tomorrow" if t[:10] == last_day else _wttr_text(t[:10], 10))
        rows.append(f"- {label} {_wttr_text(t[11:16], 5)}: {_num(_col(hb, 'temperature_2m', i), '°C')}, "
                    f"{_cond(_col(hb, 'weather_code', i))}, "
                    f"chance of rain {_num(_col(hb, 'precipitation_probability', i), '%')}")
    if rows:
        lines.append(f"Hourly, today and tomorrow (every {HOURLY_STEP_H} h, local time):")
        lines.extend(rows[:2 * 24 // HOURLY_STEP_H])
    return lines


def _open_meteo_report(name: str, js: dict, hourly: bool) -> str:
    curr = js.get("current") or {}
    cond = WMO_CONDITIONS.get(curr.get("weather_code"), f"code {curr.get('weather_code')}")
    now_s = _wttr_text(curr.get("time"), 16)
    lines = [
        f"REPORT (Source: Open-Meteo): Weather in {_wttr_text(name, 120)}",
        f"Now{' (' + now_s[11:16] + ' local time)' if len(now_s) >= 16 else ''}:",
        f"Condition: {cond}",
        f"Temp: {curr.get('temperature_2m')}°C",
        f"Wind: {curr.get('wind_speed_10m')} km/h",
        f"Humidity: {curr.get('relative_humidity_2m')}%",
    ]
    try:
        lines.extend(_forecast_lines(js, hourly, now_s))
    except Exception as _fe:  # noqa: BLE001 — r1 m5: a bad forecast block never costs the current weather
        logger.debug("weather forecast block unreadable: %s", _fe)
        lines.append("Forecast: not available (the source's forecast could not be read).")
    return "\n".join(lines)


def _wttr_text(v, limit: int = 40) -> str:
    """A remote field: one short control-free line (§4LV)."""
    return _wttr_line(str(v if v is not None else ""))[:limit]


def _wttr_slot(h) -> str:
    """A j1 slot time ("0", "300" … "2100") as "HH:MM", or "" (r1 M1: the
    field went into the report raw)."""
    t = str((h or {}).get("time") if isinstance(h, dict) else "")
    return f"{t.zfill(4)[:2]}:{t.zfill(4)[2:]}" if t.isascii() and t.isdigit() and len(t) <= 4 and int(t) < 2400 else ""


def _wttr_report(js: dict, name: str, by_name: bool, hourly: bool) -> str:
    """wttr.in's j1 answer as the same report shape. Raises on an answer
    that is not the j1 shape (the caller treats it as a failed source)."""
    cur = (js.get("current_condition") or [])[0]
    days = list(js.get("weather") or [])[:FORECAST_DAYS]
    if not isinstance(cur, dict) or not days:
        raise ValueError("not a wttr.in forecast")
    desc = lambda h: _wttr_text(((h.get("weatherDesc") or [{}])[0] or {}).get("value"))
    if by_name:
        # r1 m8: name the place it matched, the failure this exists for
        area = ((js.get("nearest_area") or [{}])[0] or {})
        _v = lambda k: _wttr_text(((area.get(k) or [{}])[0] or {}).get("value"), 40)
        matched = ", ".join(x for x in (_v("areaName"), _v("region"), _v("country")) if x)
        where = (f"'{_wttr_text(name, 80)}' (wttr.in matched it by name only"
                 + (f", to {matched}" if matched else "") + " — it may not be the place meant)")
    else:
        where = _wttr_text(name, 120)
    obs = _wttr_text(cur.get("localObsDateTime"), 24)
    obs_h = None
    try:
        obs_h = datetime.datetime.strptime(obs, "%Y-%m-%d %I:%M %p").hour
    except ValueError:
        pass
    lines = [f"REPORT (Source: wttr.in, quoted data): Weather in {where}",
             f"Now{' (observed ' + obs + ' local time)' if obs else ''}:",
             f"Condition: {desc(cur)}",
             f"Temp: {_wttr_text(cur.get('temp_C'), 8)}°C",
             f"Wind: {_wttr_text(cur.get('windspeedKmph'), 8)} km/h",
             f"Humidity: {_wttr_text(cur.get('humidity'), 8)}%",
             "Daily forecast (local time):"]
    for i, d in enumerate(days):
        hrs = [h for h in (d.get("hourly") or [])[:8] if isinstance(h, dict)]   # j1: 8 slots a day
        noon = next((h for h in hrs if _wttr_slot(h) == "12:00"), hrs[len(hrs) // 2] if hrs else {})
        rain = [int(h.get("chanceofrain")) for h in hrs if str(h.get("chanceofrain") or "").isdigit()]
        lines.append(f"- {_day_label(_wttr_text(d.get('date'), 10), i)}: {desc(noon)}, "
                     f"{_wttr_text(d.get('mintempC'), 8)}–{_wttr_text(d.get('maxtempC'), 8)}°C, "
                     f"chance of rain {str(max(rain)) + '%' if rain else '?'}")
    if hourly:
        lines.append("Hourly, today and tomorrow (every 3 h, local time):")
        for i, d in enumerate(days[:2]):
            for h in (d.get("hourly") or [])[:8]:
                slot = _wttr_slot(h)
                if not slot:
                    continue
                if i == 0 and obs_h is not None and int(slot[:2]) < obs_h - obs_h % 3:   # r1 m2: past slots
                    continue
                lines.append(f"- {'Today' if i == 0 else 'Tomorrow'} {slot}: "
                             f"{_wttr_text(h.get('tempC'), 8)}°C, {desc(h)}, "
                             f"chance of rain {_wttr_text(h.get('chanceofrain'), 4)}%")
    return "\n".join(lines)


async def tool_get_weather(tor_proxy: str, profile_memory=None, location: str = None, hourly=False):
    hourly = _wants_hourly(hourly)
    if not location and profile_memory and not requester_is_member():   # a member's "weather?" gets no owner location (R3)
        # Narrow to Exception (was a bare `except:` catching BaseException —
        # it swallowed CancelledError/KeyboardInterrupt and hid every
        # profile-load failure).
        try:
            data = profile_memory.load()
            found_loc = _find_location_in_profile(data)
            if found_loc:
                location = found_loc
                pretty_log("Weather", f"Using profile location: {location}", icon=Icons.MEM_MATCH)
        except Exception as _ploc_err:
            logger.debug("profile location lookup failed: %s", _ploc_err)

    pretty_log("System Weather", f"Location: {location}{' · hourly' if hourly else ''}", icon=Icons.TOOL_SEARCH)
    if not location:
        return "SYSTEM ERROR: No location provided. You MUST specify a city (e.g., 'London') or update your profile."

    # Fail-closed (§4P): under --mandatory-tor a falsy proxy is replaced with
    # the loopback Tor default so weather (always a public API) never connects
    # cleartext — the socket guard can't backstop this curl_cffi call.
    proxy_url = resolve_egress_proxy(tor_proxy)
    # Accept a loopback Tor proxy named "localhost" too, not just the
    # 127.0.0.1 literal — otherwise socks5://localhost:9050 silently ran in
    # "WEB" mode (no identity rotation, wrong failure diagnosis).
    mode = "TOR" if proxy_url and any(h in proxy_url for h in ("127.0.0.1", "localhost", "::1")) else "WEB"
    if proxy_url and proxy_url.startswith("socks5://"):
        proxy_url = proxy_url.replace("socks5://", "socks5h://")

    last_error = None
    geo_not_found = False  # geocoder answered 200 but knew no such place
    place = None           # (lat, lon, name) once the geocoder resolved it — the fallback asks by it (§4RE)
    for attempt in range(3):
        try:
            async with _http_client(proxy_url) as client:
                if place is None:
                    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1&language=en&format=json"
                    geo_resp = await client.get(geo_url)
                    if geo_resp.status_code in [401, 403, 503] and mode == "TOR":
                        last_error = f"HTTP {geo_resp.status_code} from {_server_name(geo_resp)}"  # §4FS: record the refusal
                        await asyncio.to_thread(request_new_tor_identity)
                        await asyncio.sleep(5)
                        continue
                    if geo_resp.status_code == 200 and geo_resp.json().get("results"):
                        res = geo_resp.json()["results"][0]
                        # with region and country: an "Athens" match can be Georgia, US (§4LV)
                        place = (res["latitude"], res["longitude"],
                                 ", ".join(x for x in (res.get("name"), res.get("admin1"), res.get("country")) if x))
                    elif geo_resp.status_code == 200:
                        # 200 with no results = the geocoder knows no such
                        # place. Record it so the final message can say so
                        # instead of the opaque "failed: None".
                        geo_not_found = True
                        last_error = None  # a definitive answer outranks an earlier transient refusal (review, 2026-09-09)
                        break
                    else:
                        break
                lat, lon, name = place
                w_resp = await client.get(_forecast_url(lat, lon, hourly))
                if w_resp.status_code in [401, 403, 503] and mode == "TOR":
                    last_error = f"HTTP {w_resp.status_code} from {_server_name(w_resp)}"  # §4FS: record the refusal
                    await asyncio.to_thread(request_new_tor_identity)
                    await asyncio.sleep(5)
                    continue
                if w_resp.status_code == 200:
                    return _open_meteo_report(name, w_resp.json() or {}, hourly)
                break
        except Exception as e:
            last_error = e
            if mode == "TOR":
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(5)
                continue

    pretty_log("Weather Warn", f"Open-Meteo failed: {last_error}", level="WARNING", icon=Icons.WARN)

    # §4RE: by the geocoder's coordinates when it answered — by name, wttr.in's
    # "Athens" was Athens, Georgia (US) while Open-Meteo's was Greece
    w_query = f"{place[0]},{place[1]}" if place else urllib.parse.quote(location)
    for attempt in range(3):
        try:
            async with _http_client(proxy_url) as client:
                resp = await client.get(f"https://wttr.in/{w_query}?format=j1")
                if resp.status_code in [401, 403, 503] and mode == "TOR":
                    last_error = f"HTTP {resp.status_code} from {_server_name(resp)}"  # §4FS: record the refusal
                    await asyncio.to_thread(request_new_tor_identity)
                    await asyncio.sleep(5)
                    continue
                if resp.status_code == 200 and "<html" not in resp.text.lower():
                    try:
                        return _wttr_report(resp.json() or {}, place[2] if place else location,
                                            by_name=place is None, hourly=hourly)
                    except Exception as _shape:  # noqa: BLE001 — not the j1 shape: a failed source
                        if not geo_not_found:     # r1 m1: "no such place" stays the answer
                            last_error = f"wttr.in answered without a forecast ({type(_shape).__name__})"
                break
        except Exception as e:
            last_error = e
            if mode == "TOR":
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(5)
                continue

    pretty_log("Weather Error", str(last_error), level="ERROR", icon=Icons.FAIL)

    # Location genuinely not found (geocoder answered 200-empty and both
    # sources fell through) — surface the CAUSE, not "failed: None". A
    # retry of the identical call would just fail identically.
    if geo_not_found and last_error is None:
        return (
            f"SYSTEM ERROR: could not find a location named {location!r}. "
            "Check the spelling, or give a more specific place (e.g. add the "
            "country: 'Springfield, US'). Do NOT retry the same name."
        )

    # Distinguish "Tor itself is down" from "weather API rejected us / is down".
    # The last_error string lets the model decide whether to retry at all.
    if mode == "TOR":
        err_text = str(last_error or "unknown").lower()
        if any(tok in err_text for tok in ("proxy", "socks", "connection refused", "tor")):
            return (
                "SYSTEM ERROR: Weather lookup failed — the Tor proxy appears to be down. "
                f"Underlying: {last_error}. Try again after checking Tor, or ask the user "
                "to run non-anonymously."
            )
        return (
            "SYSTEM ERROR: Weather lookup failed after 3 Tor retries — the upstream weather APIs "
            f"(Open-Meteo, wttr.in) rejected the request. Underlying: {last_error}. Tor itself is "
            "likely fine; the remote services may be rate-limiting or down."
        )
    return (f"SYSTEM ERROR: Weather lookup failed: {last_error}" if last_error is not None
            else f"SYSTEM ERROR: Weather lookup for {location!r} failed — no data "
            "returned by Open-Meteo or wttr.in. The location may be unrecognised; "
            "try a more specific place name.")

def _find_location_in_profile(data: dict) -> str:
    """
    Robustly searches for a location string in the user profile.
    Prioritizes specific keys (location, city, address) across all categories.
    """
    if not data: return None

    # Priority 1: Explicit Root/Personal keys. `(data.get("root") or {})`,
    # NOT `data.get("root", {})` — a profile with an explicit "root": null
    # (key present, value null) makes the default-arg form return None, and
    # None.get(...) then AttributeError'd (crashing tool_check_location and,
    # via the bare-except in tool_get_weather, silently killing profile
    # location lookup).
    _root = data.get("root") or {}
    _personal = data.get("personal") or {}
    loc = (
        _root.get("location") or
        _root.get("city") or
        _personal.get("location")
    )
    if loc: return _locality(loc)

    # Priority 2: Broad Search in ALL categories, then an address — every
    # value reduced to its LOCALITY (§4LB r2: "home"/"residence" holding a
    # street went to the geocoder; an address-only profile lost its weather)
    search_keys = ["location", "city", "residence", "home"]
    for cat, subdata in data.items():
        if isinstance(subdata, dict):
            for k, v in subdata.items():
                if k.lower() in search_keys and isinstance(v, str) and _locality(v):
                    return _locality(v)
    for cat, subdata in data.items():
        if isinstance(subdata, dict):
            for k, v in subdata.items():
                if "address" in k.lower() and isinstance(v, str) and _locality(v):
                    return _locality(v)
    return None


def _locality(value):
    from ..memory.egress import locality
    return locality(value)


async def tool_check_location(profile_memory):
    if requester_is_member():
        return "User Location: not available for this channel."
    if not profile_memory: return "Error: Profile memory not loaded."
    try:
        data = profile_memory.load()
        loc = _find_location_in_profile(data)
        if loc:
            return f"User Location: {loc}"
        else:
            return "User Location: Unknown (Profile has no location data)."
    except Exception as e:
        return f"Error checking location: {e}"

import platform
import shutil
import os
import subprocess
import httpx
try:
    import psutil
except ImportError:
    psutil = None

async def tool_check_health(context=None):
    """
    Performs a real system health check including Docker, Internet, Tor, and Agent Internals.
    Returns:
        str: A formatted string containing system statistics.
    """
    from ..core.receipt_marks import HEALTH_CHECK_HEAD as _HCH  # §4QO: one home for the literal
    health_status = [
        _HCH,
        "System Status: Online"
    ]
    
    # 1. Platform Info
    health_status.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    
    # 2. CPU Load (Unix-like)
    try:
        load1, load5, load15 = os.getloadavg()
        health_status.append(f"CPU Load (1/5/15 min): {load1:.2f} / {load5:.2f} / {load15:.2f}")
    except OSError:
        pass # Not available on Windows

    if psutil:
        # cpu_percent(interval=0.1) is a BLOCKING 100ms sleep — off the loop
        # thread so a health check doesn't stall every other coroutine.
        _cpu_pct = await asyncio.to_thread(psutil.cpu_percent, 0.1)
        health_status.append(f"CPU Usage: {_cpu_pct}%")
        
        # 3. Memory
        mem = psutil.virtual_memory()
        health_status.append(f"Memory: {mem.percent}% used ({mem.used // (1024**2)}MB / {mem.total // (1024**2)}MB)")
        
        # 4. Disk
        disk = psutil.disk_usage('/')
        health_status.append(f"Disk (/): {disk.percent}% used ({disk.free // (1024**3)}GB free)")
    else:
        # Fallback for Disk if psutil missing
        try:
            total, used, free = shutil.disk_usage("/")
            health_status.append(f"Disk (/): {(used/total)*100:.1f}% used ({free // (1024**3)}GB free)")
        except: pass

    # 5. Docker Status
    try:
        def _run_docker_check():
            import os, shutil
            # Extend $PATH with standard package-manager install locations so
            # the lookup still works when launched from launchd/systemd, where
            # the inherited PATH is minimal. shutil.which still verifies the
            # binary exists and is executable, so this is no less safe than
            # trusting $PATH alone.
            extra = ["/usr/local/bin", "/opt/homebrew/bin", "/usr/bin", "/bin"]
            search_path = os.pathsep.join([os.environ.get("PATH", ""), *extra])
            docker_cmd = shutil.which("docker", path=search_path)
            if not docker_cmd:
                return subprocess.CompletedProcess(args=["docker"], returncode=127, stdout="", stderr="docker not in PATH")
            return subprocess.run([docker_cmd, "info", "--format", "{{.ServerVersion}}"], capture_output=True, text=True, timeout=5)
        docker_res = await asyncio.to_thread(_run_docker_check)
        if docker_res.returncode == 0:
            health_status.append(f"Docker: Active (Version {docker_res.stdout.strip()})")
        else:
            health_status.append("Docker: Inactive or Not Found")
    except Exception:
        health_status.append("Docker: Check Failed")

    # 6. Connectivity (Internet & Tor)
    try:
        # Use Tor Proxy for general internet check if available, to be safe.
        # Fail-closed (§4P): under --mandatory-tor a missing context proxy is
        # replaced with the loopback Tor default so this public probe (1.1.1.1)
        # can't itself leak cleartext; off-guard it stays None (direct probe).
        check_proxy = resolve_egress_proxy(getattr(context, "tor_proxy", None) if context else None)
        mode = "WEB"
        if check_proxy:
             check_proxy = check_proxy.replace("socks5://", "socks5h://")
             if "127.0.0.1" in check_proxy: mode = "TOR"

        # §4LV: a 3 s probe over Tor (the Tor check itself gets 5 s+) read a
        # slow circuit as "Internet: Disconnected" beside "Tor: Connected",
        # and every failed attempt forced a NEW Tor identity for the whole
        # process (3×, plus 15 s of sleeps). One circuit change, only for an
        # exit that is BLOCKED (401/403/503); a timeout is just retried.
        _renewed = False
        _last_err = ""
        for attempt in range(2):
            try:
                if curl_requests:
                    proxies = {"http": check_proxy, "https": check_proxy} if check_proxy else None
                    async with curl_requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=10.0, verify=True) as client:
                        resp = await client.get("https://1.1.1.1")
                else:
                    async with httpx.AsyncClient(timeout=10.0, proxy=check_proxy, verify=True) as client:
                        resp = await client.get("https://1.1.1.1")
                if resp.status_code in [401, 403, 503] and mode == "TOR" and not _renewed:
                    _renewed = True
                    await asyncio.to_thread(request_new_tor_identity)
                    await asyncio.sleep(3)
                    continue
                status_msg = f"Internet: Connected ({resp.status_code})"
                if check_proxy: status_msg += " [via Tor]"
                health_status.append(status_msg)
                break
            except Exception as _e:  # noqa: BLE001
                _last_err = type(_e).__name__
                if mode != "TOR":
                    health_status.append("Internet: Disconnected or Blocked")
                    break
        else:
            health_status.append(
                ("Internet: NOT reached via Tor within 10 s (" + (_last_err or "blocked exit")
                 + ") — the circuit may be slow; this is not proof the host is offline")
                if mode == "TOR" else "Internet: Disconnected or Blocked")
    except Exception:
        health_status.append("Internet: Disconnected or Blocked")
        
    if context and context.tor_proxy:
        check_proxy = context.tor_proxy.replace("socks5://", "socks5h://")
        mode = "TOR" if "127.0.0.1" in check_proxy else "WEB"
        for attempt in range(3):
            try:
                if curl_requests:
                    proxies = {"http": check_proxy, "https": check_proxy} if check_proxy else None
                    async with curl_requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=10.0, verify=True) as client:
                        resp = await client.get("https://check.torproject.org/api/ip")
                        if resp.status_code in [401, 403, 503] and mode == "TOR":
                            await asyncio.to_thread(request_new_tor_identity)
                            await asyncio.sleep(5)
                            continue
                        if resp.status_code == 200 and resp.json().get("IsTor", False):
                            health_status.append("Tor: Connected (Anonymous)")
                        else:
                            health_status.append("Tor: Connected but Not Anonymous (Check Config)")
                        break
                else:
                    async with httpx.AsyncClient(proxy=check_proxy, timeout=10.0, verify=True) as client:
                        resp = await client.get("https://check.torproject.org/api/ip")
                        if resp.status_code in [401, 403, 503] and mode == "TOR":
                            await asyncio.to_thread(request_new_tor_identity)
                            await asyncio.sleep(5)
                            continue
                        if resp.status_code == 200 and resp.json().get("IsTor", False):
                            health_status.append("Tor: Connected (Anonymous)")
                        else:
                            health_status.append("Tor: Connected but Not Anonymous (Check Config)")
                        break
            except Exception as e:
                if mode == "TOR":
                    # a timeout is a slow circuit, not a blocked exit: retry
                    # WITHOUT a process-wide identity change (§4LV)
                    await asyncio.sleep(2)
                    continue
                else:
                    health_status.append(f"Tor: Connection Failed ({str(e)})")
                    break
        else:
            health_status.append("Tor: Connection Failed (Retries exhausted)")
    else:
        health_status.append("Tor: Not Configured")

    # 7. Agent Internals
    if context:
        llm_status = "Active" if context.llm_client else "Offline"
        mem_status = "Active" if context.memory_system else "Offline"
        sandbox_status = "Active" if context.sandbox_dir else "Offline"
        
        # Biological Daemon Check (replaces the legacy APScheduler probe).
        # The watchdog is a long-lived asyncio.Task created in the FastAPI
        # lifespan and stashed on context.biological_task.
        bio_task = getattr(context, 'biological_task', None)
        if bio_task is None:
            bio_status = "Offline"
        elif getattr(bio_task, 'cancelled', lambda: False)():
            bio_status = "Cancelled"
        elif getattr(bio_task, 'done', lambda: False)():
            exc = None
            try:
                exc = bio_task.exception()
            except Exception:
                exc = None
            bio_status = f"Crashed ({type(exc).__name__})" if exc else "Stopped"
        else:
            bio_status = "Running"

        # Memory Bus Check
        bus_status = "Active" if getattr(context, 'memory_bus', None) else "Offline"

        # Graph Check
        graph_status = "Offline"
        if getattr(context, 'graph_memory', None):
            try:
                import sqlite3
                with context.graph_memory._lock:
                    with sqlite3.connect(context.graph_memory.db_path) as conn:
                        count = conn.execute("SELECT count(*) FROM triplets").fetchone()[0]
                graph_status = f"Active ({count} edges)"
            except Exception:
                graph_status = "Active (Count Error)"

        health_status.append(
            f"Agent Internals: LLM={llm_status}, Memory={mem_status}, "
            f"Graph={graph_status}, Sandbox={sandbox_status}, "
            f"Bus={bus_status}, Idle Scheduler={bio_status}"   # §4NN: was "Biological Daemon"
        )
        
    return "\n".join(health_status)

#: The full WMO weather-interpretation table Open-Meteo uses (§4LV: 9 codes
#: were mapped — drizzle, showers and most snow read "Variable", and moderate
#: rain (63) read "Heavy Rain").
WMO_CONDITIONS = {
    0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing Fog",
    51: "Light Drizzle", 53: "Drizzle", 55: "Heavy Drizzle",
    56: "Light Freezing Drizzle", 57: "Freezing Drizzle",
    61: "Light Rain", 63: "Rain", 65: "Heavy Rain",
    66: "Light Freezing Rain", 67: "Freezing Rain",
    71: "Light Snow", 73: "Snow", 75: "Heavy Snow", 77: "Snow Grains",
    80: "Light Showers", 81: "Showers", 82: "Violent Showers",
    85: "Light Snow Showers", 86: "Heavy Snow Showers",
    95: "Thunderstorm", 96: "Thunderstorm with Light Hail", 99: "Thunderstorm with Heavy Hail",
}


def _wttr_line(text) -> str:
    """wttr.in's `format=3` answer is ONE short line. Anything else (a
    hostile exit, an error page) is cut to one bounded, control-free line,
    so it can never be a page of instructions in the prompt (§4LV)."""
    import re as _re
    line = str(text or "").strip().splitlines()[0] if str(text or "").strip() else ""
    return _re.sub(r"[\x00-\x1f\x7f]", " ", line)[:160]


async def tool_system_utility(action: str = None, tor_proxy: str = None, profile_memory=None, location: str = None, context=None, **kwargs):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    # Normalise like the other action-dispatch tools: strip quotes AND
    # surrounding whitespace, then lowercase. Previously only quotes were
    # stripped, so "Check_Health" / "check_weather " fell through to the
    # unknown-action branch and a valid diagnostic silently didn't run.
    action = str(action).strip().strip("\"'").strip().lower()
    if action == "check_weather":
        return await tool_get_weather(tor_proxy, profile_memory, location, hourly=kwargs.get("hourly", False))
    elif action == "check_health":
        return await tool_check_health(context)
    elif action == "check_location":
        return await tool_check_location(profile_memory)
    else:
        return f"Error: Unknown action '{action}'"
