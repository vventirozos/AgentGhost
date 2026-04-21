import asyncio
import datetime
import urllib.parse
import httpx
try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None
from ..utils.logging import Icons, pretty_log
from ..utils.helpers import request_new_tor_identity

async def tool_get_weather(tor_proxy: str, profile_memory=None, location: str = None):
    if not location and profile_memory:
        try:
            data = profile_memory.load()
            found_loc = _find_location_in_profile(data)
            if found_loc:
                location = found_loc
                pretty_log("Weather", f"Using profile location: {location}", icon=Icons.MEM_MATCH)
        except: pass

    pretty_log("System Weather", f"Location: {location}", icon=Icons.TOOL_SEARCH)
    if not location:
        return "SYSTEM ERROR: No location provided. You MUST specify a city (e.g., 'London') or update your profile."

    proxy_url = tor_proxy
    mode = "TOR" if proxy_url and "127.0.0.1" in proxy_url else "WEB"
    if proxy_url and proxy_url.startswith("socks5://"):
        proxy_url = proxy_url.replace("socks5://", "socks5h://")
    
    last_error = None
    for attempt in range(3):
        try:
            if curl_requests:
                proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
                async with curl_requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=20.0, verify=False) as client:
                    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1&language=en&format=json"
                    geo_resp = await client.get(geo_url)
                    if geo_resp.status_code in [401, 403, 503] and mode == "TOR":
                        await asyncio.to_thread(request_new_tor_identity)
                        await asyncio.sleep(5)
                        continue
                    if geo_resp.status_code == 200 and geo_resp.json().get("results"):
                        res = geo_resp.json()["results"][0]
                        lat, lon, name = res["latitude"], res["longitude"], res["name"]
                        w_url = (
                            f"https://api.open-meteo.com/v1/forecast?"
                            f"latitude={lat}&longitude={lon}&"
                            f"current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&"
                            f"wind_speed_unit=kmh"
                        )
                        w_resp = await client.get(w_url)
                        if w_resp.status_code in [401, 403, 503] and mode == "TOR":
                            await asyncio.to_thread(request_new_tor_identity)
                            await asyncio.sleep(5)
                            continue
                        if w_resp.status_code == 200:
                            curr = w_resp.json().get("current", {})
                            wmo_map = {0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast", 45: "Fog", 61: "Rain", 63: "Heavy Rain", 71: "Snow", 95: "Thunderstorm"}
                            cond = wmo_map.get(curr.get("weather_code"), "Variable")
                            return (
                                f"REPORT (Source: Open-Meteo): Weather in {name}\n"
                                f"Condition: {cond}\n"
                                f"Temp: {curr.get('temperature_2m')}°C\n"
                                f"Wind: {curr.get('wind_speed_10m')} km/h\n"
                                f"Humidity: {curr.get('relative_humidity_2m')}%"
                            )
                    break
            else:
                async with httpx.AsyncClient(proxy=proxy_url, timeout=20.0, verify=False) as client:
                    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(location)}&count=1&language=en&format=json"
                    geo_resp = await client.get(geo_url)
                    if geo_resp.status_code in [401, 403, 503] and mode == "TOR":
                        await asyncio.to_thread(request_new_tor_identity)
                        await asyncio.sleep(5)
                        continue
                    if geo_resp.status_code == 200 and geo_resp.json().get("results"):
                        res = geo_resp.json()["results"][0]
                        lat, lon, name = res["latitude"], res["longitude"], res["name"]
                        w_url = (
                            f"https://api.open-meteo.com/v1/forecast?"
                            f"latitude={lat}&longitude={lon}&"
                            f"current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&"
                            f"wind_speed_unit=kmh"
                        )
                        w_resp = await client.get(w_url)
                        if w_resp.status_code in [401, 403, 503] and mode == "TOR":
                            await asyncio.to_thread(request_new_tor_identity)
                            await asyncio.sleep(5)
                            continue
                        if w_resp.status_code == 200:
                            curr = w_resp.json().get("current", {})
                            wmo_map = {0: "Clear", 1: "Mainly Clear", 2: "Partly Cloudy", 3: "Overcast", 45: "Fog", 61: "Rain", 63: "Heavy Rain", 71: "Snow", 95: "Thunderstorm"}
                            cond = wmo_map.get(curr.get("weather_code"), "Variable")
                            return (
                                f"REPORT (Source: Open-Meteo): Weather in {name}\n"
                                f"Condition: {cond}\n"
                                f"Temp: {curr.get('temperature_2m')}°C\n"
                                f"Wind: {curr.get('wind_speed_10m')} km/h\n"
                                f"Humidity: {curr.get('relative_humidity_2m')}%"
                            )
                    break 
        except Exception as e:
            last_error = e
            if mode == "TOR":
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(5)
                continue
            
    pretty_log("Weather Warn", f"Open-Meteo failed: {last_error}", level="WARN", icon=Icons.WARN)

    for attempt in range(3):
        try:
            url = f"https://wttr.in/{urllib.parse.quote(location)}?format=3"
            if curl_requests:
                proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
                async with curl_requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=20.0, verify=False) as client:
                    resp = await client.get(url)
                    if resp.status_code in [401, 403, 503] and mode == "TOR":
                        await asyncio.to_thread(request_new_tor_identity)
                        await asyncio.sleep(5)
                        continue
                    if resp.status_code == 200 and "<html" not in resp.text.lower():
                        return f"REPORT (Source: wttr.in): {resp.text.strip()}"
                    break
            else:
                async with httpx.AsyncClient(proxy=proxy_url, timeout=20.0, verify=False) as client:
                    resp = await client.get(url)
                    if resp.status_code in [401, 403, 503] and mode == "TOR":
                        await asyncio.to_thread(request_new_tor_identity)
                        await asyncio.sleep(5)
                        continue
                    if resp.status_code == 200 and "<html" not in resp.text.lower():
                        return f"REPORT (Source: wttr.in): {resp.text.strip()}"
                    break
        except Exception as e:
            last_error = e
            if mode == "TOR":
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(5)
                continue
                
    pretty_log("Weather Error", str(last_error), level="ERROR", icon=Icons.FAIL)

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
    return f"SYSTEM ERROR: Weather lookup failed: {last_error}"

def _find_location_in_profile(data: dict) -> str:
    """
    Robustly searches for a location string in the user profile.
    Prioritizes specific keys (location, city, address) across all categories.
    """
    if not data: return None
    
    # Priority 1: Explicit Root/Personal keys
    loc = (
        data.get("root", {}).get("location") or 
        data.get("root", {}).get("city") or 
        data.get("personal", {}).get("location")
    )
    if loc: return loc

    # Priority 2: Broad Search in ALL categories
    search_keys = ["location", "city", "address", "residence", "home"]
    for cat, subdata in data.items():
        if isinstance(subdata, dict):
            for k, v in subdata.items():
                if k.lower() in search_keys and isinstance(v, str):
                    return v
    return None

async def tool_check_location(profile_memory):
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
    health_status = [
        "--- SYSTEM HEALTH DIAGNOSTICS ---",
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
        health_status.append(f"CPU Usage: {psutil.cpu_percent(interval=0.1)}%")
        
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
        # Use Tor Proxy for general internet check if available, to be safe
        check_proxy = None
        mode = "WEB"
        if context and context.tor_proxy:
             check_proxy = context.tor_proxy.replace("socks5://", "socks5h://")
             if "127.0.0.1" in check_proxy: mode = "TOR"

        for attempt in range(3):
            try:
                if curl_requests:
                    proxies = {"http": check_proxy, "https": check_proxy} if check_proxy else None
                    async with curl_requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=3.0, verify=False) as client:
                        resp = await client.get("https://1.1.1.1")
                        if resp.status_code in [401, 403, 503] and mode == "TOR":
                            await asyncio.to_thread(request_new_tor_identity)
                            await asyncio.sleep(5)
                            continue
                        status_msg = f"Internet: Connected ({resp.status_code})"
                        if check_proxy: status_msg += " [via Tor]"
                        health_status.append(status_msg)
                        break
                else:
                    async with httpx.AsyncClient(timeout=3.0, proxy=check_proxy, verify=False) as client:
                        resp = await client.get("https://1.1.1.1")
                        if resp.status_code in [401, 403, 503] and mode == "TOR":
                            await asyncio.to_thread(request_new_tor_identity)
                            await asyncio.sleep(5)
                            continue
                        status_msg = f"Internet: Connected ({resp.status_code})"
                        if check_proxy: status_msg += " [via Tor]"
                        health_status.append(status_msg)
                        break
            except Exception:
                if mode == "TOR":
                    await asyncio.to_thread(request_new_tor_identity)
                    await asyncio.sleep(5)
                    continue
                else:
                    health_status.append("Internet: Disconnected or Blocked")
                    break
        else:
            health_status.append("Internet: Disconnected or Blocked")
    except Exception:
        health_status.append("Internet: Disconnected or Blocked")
        
    if context and context.tor_proxy:
        check_proxy = context.tor_proxy.replace("socks5://", "socks5h://")
        mode = "TOR" if "127.0.0.1" in check_proxy else "WEB"
        for attempt in range(3):
            try:
                if curl_requests:
                    proxies = {"http": check_proxy, "https": check_proxy} if check_proxy else None
                    async with curl_requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=5.0, verify=False) as client:
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
                    async with httpx.AsyncClient(proxy=check_proxy, timeout=5.0, verify=False) as client:
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
                    await asyncio.to_thread(request_new_tor_identity)
                    await asyncio.sleep(5)
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
            f"Bus={bus_status}, Biological Daemon={bio_status}"
        )
        
    return "\n".join(health_status)

async def tool_system_utility(action: str = None, tor_proxy: str = None, profile_memory=None, location: str = None, context=None, **kwargs):
    if not action:
        return "SYSTEM ERROR: The 'action' parameter is MANDATORY. You must specify it."
    action = action.strip("\"'")
    if action == "check_weather":
        return await tool_get_weather(tor_proxy, profile_memory, location)
    elif action == "check_health":
        return await tool_check_health(context)
    elif action == "check_location":
        return await tool_check_location(profile_memory)
    else:
        return f"Error: Unknown action '{action}'"
