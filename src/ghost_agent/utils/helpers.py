import datetime
import os
import asyncio
import httpx
import subprocess
import platform
from typing import List

import socket

def request_new_tor_identity(control_port=9051, password=""):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(("127.0.0.1", control_port))
            if password:
                s.sendall(f'AUTHENTICATE "{password}"\r\n'.encode())
            else:
                s.sendall(b'AUTHENTICATE\r\n')
            
            resp = s.recv(1024).decode()
            if not resp.startswith("250"):
                return False, f"Tor Auth failed: {resp.strip()}"
                
            s.sendall(b'SIGNAL NEWNYM\r\n')
            resp = s.recv(1024).decode()
            if not resp.startswith("250"):
                return False, f"Tor NEWNYM failed: {resp.strip()}"
                
            return True, "Identity renewed successfully"
    except Exception as e:
        # Fallback: try restarting the tor service directly
        try:
            if platform.system() == "Darwin":
                # macOS Homebrew
                subprocess.run(["brew", "services", "restart", "tor"], check=True, capture_output=True)
                return True, "Identity renewed successfully via brew services restart"
            else:
                # Linux systemd (might require sudo NOPASSWD or run as root, but worth trying)
                subprocess.run(["sudo", "-n", "systemctl", "restart", "tor"], check=True, capture_output=True)
                return True, "Identity renewed successfully via systemctl restart"
        except Exception as fallback_e:
            return False, f"Tor control port error: {e}. Fallback restart also failed: {fallback_e}"

async def helper_fetch_url_content(url: str) -> str:
    # 1. Setup Tor Proxy
    proxy_url = os.getenv("TOR_PROXY", "socks5://127.0.0.1:9050")
    if proxy_url and proxy_url.startswith("socks5://"): 
        proxy_url = proxy_url.replace("socks5://", "socks5h://")

    try:
        import curl_cffi.requests
    except ImportError:
        curl_cffi = None

    for attempt in range(3):
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            
            if curl_cffi:
                proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
                async with curl_cffi.requests.AsyncSession(impersonate="chrome110", proxies=proxies, timeout=20.0) as client:
                    resp = await client.get(url, headers=headers)
            else:
                # Fallback to httpx if curl_cffi is missing for some reason
                async with httpx.AsyncClient(proxy=proxy_url, timeout=20.0, follow_redirects=True) as client:
                    resp = await client.get(url, headers=headers)
            
            content_type = resp.headers.get("content-type", "").lower()
            if "application/pdf" in content_type or "application/octet-stream" in content_type or url.lower().endswith((".pdf", ".zip")):
                return "Error: URL points to a binary file. To read PDFs, download them using file_system and ingest them using knowledge_base."
                
            status_code = resp.status_code
            text = resp.text
            
            if status_code != 200:
                if status_code in [401, 403, 503] and proxy_url:
                    if attempt < 2:
                        await asyncio.to_thread(request_new_tor_identity)
                        await asyncio.sleep(5)
                        continue
                    return f"Error: Access Denied ({status_code}) via Tor. The site {url} likely blocks Tor exit nodes. Try a different source."
                return f"Error: Received status {status_code} from {url}"
            
            def _parse_html(html_content):
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                for script in soup(["script", "style", "nav", "footer", "iframe", "svg"]):
                    script.decompose()
                text_content = soup.get_text(separator=' ', strip=True)
                return " ".join(text_content.split()) if text_content else "Error: No text content extracted from page."
            
            return await asyncio.to_thread(_parse_html, text)
            
        except Exception as e:
            if attempt < 2 and proxy_url:
                await asyncio.to_thread(request_new_tor_identity)
                await asyncio.sleep(5)
                continue
            return f"Error reading {url}: {str(e)}"
            
    return f"Error fetching {url} after 3 retries."

def get_utc_timestamp():
    """Returns strict ISO8601 UTC timestamp for Go/iOS clients."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def recursive_split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 70) -> List[str]:
    if not text: return []
    if len(text) <= chunk_size: return [text]
    
    separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    final_chunks = []
    stack = [text]
    
    while stack:
        current_text = stack.pop()
        
        if len(current_text) <= chunk_size:
            final_chunks.append(current_text)
            continue
            
        found_sep = ""
        for sep in separators:
            if sep in current_text:
                found_sep = sep
                break
        
        if not found_sep:
            for i in range(0, len(current_text), chunk_size - chunk_overlap):
                final_chunks.append(current_text[i:i+chunk_size])
            continue
            
        parts = current_text.split(found_sep)
        buffer = ""
        temp_chunks = []
        
        for p in parts:
            fragment = p + found_sep if found_sep.strip() else p
            if len(buffer) + len(fragment) <= chunk_size:
                buffer += fragment
            else:
                if buffer:
                    temp_chunks.append(buffer.strip())
                buffer = fragment
        
        if buffer:
            temp_chunks.append(buffer.strip())

        for chunk in reversed(temp_chunks):
            if len(chunk) > chunk_size:
                if found_sep == "" or chunk == current_text:
                    for i in range(0, len(chunk), chunk_size - chunk_overlap):
                        final_chunks.append(chunk[i:i+chunk_size])
                else:
                    stack.append(chunk) 
            else:
                final_chunks.append(chunk)

    return final_chunks