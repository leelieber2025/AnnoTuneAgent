# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

"""One Q6_K base, two LoRA adapters, one enabled per request.

llama-server loads name and ml together and leaves them off.
A request sets that task's scale to 1 and the other to 0. Sequences
past ``maxlen`` raise. They are not truncated.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

_TASKS = ("name", "ml")
# One private directory per user, so two users never share a state file or a log.
# os.getuid does not exist on Windows. The package must still import there.
_USER = str(os.getuid()) if hasattr(os, "getuid") else getpass.getuser()
_RUN_DIR = (
    Path(os.environ["XDG_RUNTIME_DIR"]) / "annotune"
    if os.environ.get("XDG_RUNTIME_DIR")
    else Path(tempfile.gettempdir()) / f"annotune-{_USER}"
)
_STATE = _RUN_DIR / "llama-server.json"
_LOG = _RUN_DIR / "llama-server.log"
_DEFAULT_SERVICE = "http://127.0.0.1:8080"
# A long ML sample on a P1000 can take about two minutes. Health probes stay short.
_DEFAULT_TIMEOUT = 600
_SERVER: dict | None = None
_ADAPTER_IDS: dict[str, dict[str, int]] = {}
_LAST: dict = {}
_TOKENIZERS: dict = {}


class _Running:
    """A llama-server started by another process. ``poll`` is None while it lives."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int | None:
        return None if _alive(self.pid) else 1

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.time() + timeout
        while _alive(self.pid):
            if deadline is not None and time.time() > deadline:
                raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout)
            time.sleep(0.2)
        return 0


class _Remote:
    """An HTTP server this process did not start."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.pid = -1

    def poll(self) -> int | None:
        return None if _probe(self.url) is not None else 1

    def wait(self, timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired(cmd=self.url, timeout=timeout)


def _alive(pid: int) -> bool:
    """True when ``pid`` is still running.

    On Windows ``os.kill(pid, 0)`` terminates the process, so the check uses
    the process handle instead.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetExitCodeProcess.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        # Access denied means the process exists and this user cannot query it.
        return ctypes.get_last_error() == 5
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return ctypes.get_last_error() == 5
        return int(code.value) == 259
    finally:
        kernel.CloseHandle(handle)


class SequenceTooLong(RuntimeError):
    def __init__(self, tokens: int, maxlen: int):
        super().__init__(f"sequence is {tokens} tokens, MAXLEN={maxlen}; do not truncate")
        self.tokens = tokens
        self.maxlen = maxlen


def scales_for(task: str, ids: dict[str, int] | None = None) -> list[dict[str, float]]:
    """Enable one adapter. The other stays at scale 0 for this request.

    ``ids`` maps ``name`` and ``ml`` to the server's adapter ids. The default
    is the load order: name is 0 and ml is 1.
    """
    if task not in _TASKS:
        raise ValueError(f"unknown task {task}")
    chosen = ids or {name: index for index, name in enumerate(_TASKS)}
    return sorted(
        ({"id": int(chosen[name]), "scale": 1.0 if name == task else 0.0} for name in _TASKS),
        key=lambda item: item["id"],
    )


def _auth_headers() -> dict[str, str]:
    """``ANNOTUNE_API_KEY`` is sent as a bearer token, for llama-server ``--api-key``."""
    key = os.environ.get("ANNOTUNE_API_KEY", "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _adapter_ids(url: str) -> dict[str, int]:
    """Adapter ids on a server this process did not start.

    ``/lora-adapters`` lists each adapter's file. When exactly one file name
    has the token ``name`` and exactly one has ``ml``, those ids are used, so
    a server loaded with ``--lora ml.gguf,name.gguf`` is not called with the
    adapters swapped. Otherwise the load order is assumed.
    """
    url = _base_url(url)
    cached = _ADAPTER_IDS.get(url)
    if cached is not None:
        return cached
    ids = {name: index for index, name in enumerate(_TASKS)}
    listed = _get_json(url, "/lora-adapters")
    if isinstance(listed, list):
        found: dict[str, list[int]] = {}
        for item in listed:
            if not isinstance(item, dict) or "id" not in item:
                continue
            try:
                index = int(item["id"])
            except (TypeError, ValueError):
                continue
            filename = re.split(r"[\\/]", str(item.get("path") or ""))[-1].lower()
            tokens = set(re.split(r"[^a-z0-9]+", filename))
            for task in _TASKS:
                if task in tokens:
                    found.setdefault(task, []).append(index)
        if all(len(found.get(task, [])) == 1 for task in _TASKS):
            if found["name"][0] != found["ml"][0]:
                ids = {task: found[task][0] for task in _TASKS}
    _ADAPTER_IDS[url] = ids
    return ids


def last_call() -> dict:
    return dict(_LAST)


def packaged_tokenizer() -> str:
    """Qwen tokenizer shipped with the package."""
    return str(Path(__file__).resolve().parent / "tokenizer")


def _tokenizer_dir(path: str | None) -> str:
    return packaged_tokenizer() if path is None else str(path)


def service_url() -> str:
    """Server root. ``ANNOTUNE_URL`` overrides ``http://127.0.0.1:8080``."""
    return _base_url(os.environ.get("ANNOTUNE_URL", _DEFAULT_SERVICE))


def request_timeout(timeout: float | None = None) -> float:
    """Seconds for one model request. ``ANNOTUNE_TIMEOUT`` overrides the default."""
    if timeout is not None:
        return float(timeout)
    raw = os.environ.get("ANNOTUNE_TIMEOUT")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return float(_DEFAULT_TIMEOUT)


def attach(url: str | None = None, *, tokenizer: str | None = None) -> str:
    """Use a server that is already running. Does not load weights."""
    target = _base_url(url) if url else service_url()
    tokenizer = _tokenizer_dir(tokenizer)
    if _server_live() and _SERVER is not None and _SERVER.get("url") == target:
        _SERVER["tokenizer"] = tokenizer
        _SERVER["tok"] = None
        return target
    if _server_live() and _SERVER is not None and _SERVER.get("owned"):
        raise RuntimeError(
            f"this process already started {_SERVER['url']}; release() before using {target}"
        )
    _clear_client()
    if not _use_existing(target, tokenizer):
        raise RuntimeError(f"server is not ready: {target}")
    return target


def open_adapters(
    adapters: dict[str, str],
    *,
    gguf: str,
    tokenizer: str | None = None,
    ctx: int = 8192,
    url: str | None = None,
) -> None:
    """Use the server at ``url``, or load one base and two adapters there.

    A server that is already up is reused. This does not start a second one.
    """
    global _SERVER
    target = _base_url(url) if url else service_url()
    tokenizer = _tokenizer_dir(tokenizer)
    missing = [name for name in _TASKS if not adapters.get(name)]
    if missing:
        raise ValueError(f"shared base needs adapters for {', '.join(missing)}")
    paths = {name: str(Path(adapters[name]).resolve()) for name in _TASKS}
    base = str(Path(gguf).resolve())
    if _server_live() and _SERVER.get("url") == target:
        loaded = _SERVER
        if not _SERVER.get("owned"):
            loaded = _described_server(target)
            if loaded is None:
                raise RuntimeError(
                    f"server at {target} is already running but its GGUF and adapters cannot be verified"
                )
        _require_matching_model(loaded, base, paths, ctx)
        if not _SERVER.get("owned"):
            _SERVER["gguf"] = str(loaded.get("gguf") or "")
            _SERVER["adapters"] = dict(loaded.get("adapters") or {})
            _SERVER["ctx"] = int(loaded.get("ctx") or 0)
        if _SERVER["tokenizer"] != tokenizer:
            raise RuntimeError("server already uses a different tokenizer")
        return
    if _server_live() and _SERVER.get("owned"):
        raise RuntimeError(
            f"this process already started {_SERVER['url']}; release() before using {target}"
        )
    _clear_client()
    if _use_existing(target, tokenizer, gguf=base, adapters=paths, ctx=ctx):
        return
    other = _other_local(target)
    if other is not None:
        raise RuntimeError(
            f"a model is already loaded at {other}; not loading another. "
            f"Service address is {target}."
        )
    host, port = _split_host(target)
    if host not in {"127.0.0.1", "localhost"}:
        raise RuntimeError(f"server is not ready: {target}")
    base = _gguf_path(base)
    binary = _binary()
    _private_dir(_LOG.parent)
    log = _LOG.open("w")
    proc = subprocess.Popen(
        [
            binary,
            "-m",
            base,
            "--lora",
            ",".join(paths[name] for name in _TASKS),
            "--lora-init-without-apply",
            "-c",
            str(ctx),
            "-ngl",
            "99",
            "-np",
            "1",
            "--no-cont-batching",
            "--no-cache-prompt",
            "--slot-prompt-similarity",
            "0",
            "--no-jinja",
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=_child_env(binary),
        start_new_session=True,
    )
    _SERVER = _runtime(
        proc,
        url=target,
        port=port,
        gguf=base,
        adapters=paths,
        ctx=ctx,
        tokenizer=tokenizer,
        mode="llama",
        owned=True,
        log=log,
    )
    try:
        _wait_ready(proc, target, _LOG)
    except Exception:
        release()
        raise
    _write_state()


def complete(task: str, system: str, payload: dict | str, *, maxlen: int) -> str:
    """Run one greedy request with only ``task`` applied."""
    if task not in _TASKS:
        raise ValueError(f"unknown task {task}")
    if not _server_live():
        raise RuntimeError("shared base is not loaded")
    from .decide import user_text

    user = user_text(task, payload)
    prompt = render_prompt(system, user, runtime_tokenizer())
    tokens = prompt_tokens(prompt, runtime_tokenizer())
    if tokens + 32 > maxlen:
        raise SequenceTooLong(tokens, maxlen)
    runtime = _SERVER
    if runtime.get("mode") == "chat":
        from .decide import post_chat

        text = str(post_chat(runtime["url"], system, payload, request_timeout(), task=task)).strip()
        scales: list[dict[str, float]] = []
        evaluated = None
        predicted = None
    else:
        ids = None if runtime.get("owned") else _adapter_ids(runtime["url"])
        scales = scales_for(task, ids)
        text, evaluated, predicted = _completion(runtime["url"], prompt, scales)
    global _LAST
    _LAST = {
        "task": task,
        "lora": scales,
        "tokens": tokens,
        "tokens_evaluated": evaluated,
        "tokens_predicted": predicted,
    }
    return text


def complete_url(
    url: str,
    task: str,
    system: str,
    payload: dict | str,
    *,
    maxlen: int,
    timeout: float | None = None,
    tokenizer: str | None = None,
) -> str:
    """One greedy ``/completion``. ``task`` is enabled and the other adapter is off."""
    if task not in _TASKS:
        raise ValueError(f"unknown task {task}")
    from .decide import user_text

    # "http://host:8080/" would post to "//completion", which llama-server answers with 404.
    url = _base_url(url)
    prompt = render_prompt(system, user_text(task, payload), tokenizer)
    tokens = prompt_tokens(prompt, tokenizer)
    if tokens + 32 > maxlen:
        raise SequenceTooLong(tokens, maxlen)
    scales = scales_for(task, _adapter_ids(url))
    text, evaluated, predicted = _completion(url, prompt, scales, timeout=timeout)
    global _LAST
    _LAST = {
        "task": task,
        "lora": scales,
        "tokens": tokens,
        "tokens_evaluated": evaluated,
        "tokens_predicted": predicted,
    }
    return text


def local_server() -> str | None:
    """Base URL of a loaded server. None when nothing is running."""
    if _server_live():
        return str(_SERVER.get("url") or "")
    record = _read_state()
    if record is not None and _alive(int(record.get("pid", -1))):
        url = str(record.get("url") or f"http://127.0.0.1:{int(record['port'])}")
        if _health_ok(url):
            return url
    found = _loaded_servers()
    if len(found) == 1:
        return str(found[0]["url"])
    return None


def release() -> None:
    """Stop a server this process started."""
    global _SERVER
    runtime = _SERVER
    _SERVER = None
    _ADAPTER_IDS.clear()
    if runtime is None:
        return
    proc = runtime["proc"]
    log = runtime.get("log")
    if runtime.get("owned") and proc.poll() is None and getattr(proc, "pid", 0) > 0:
        try:
            os.killpg(proc.pid, 15)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
    if log is not None:
        log.close()
    if not runtime.get("owned"):
        return
    record = _read_state()
    pid = getattr(proc, "pid", None)
    if record is None or int(record.get("pid", -2)) == pid:
        _STATE.unlink(missing_ok=True)


def _private_dir(path: Path) -> None:
    """Create ``path`` for this user alone. Refuse one that someone else made or can use."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"{path} must be a plain directory")
    if hasattr(os, "getuid") and (info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise RuntimeError(f"{path} is not private to this user; remove it or set XDG_RUNTIME_DIR")


def _read_state() -> dict | None:
    if _STATE.parent.exists():
        _private_dir(_STATE.parent)
    try:
        record = json.loads(_STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return record if isinstance(record, dict) else None


def _health_ok(url: str) -> bool:
    try:
        request = urllib.request.Request(f"{url}/health", headers=_auth_headers())
        with urllib.request.urlopen(request, timeout=2) as resp:
            payload = json.loads(resp.read().decode())
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return False
    return isinstance(payload, dict) and payload.get("status") == "ok"


def _models_ok(url: str) -> bool:
    try:
        request = urllib.request.Request(f"{url}/v1/models", headers=_auth_headers())
        with urllib.request.urlopen(request, timeout=2) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _probe(url: str) -> str | None:
    """``llama`` for this server's health route, ``chat`` for an OpenAI-style API."""
    if _health_ok(url):
        return "llama"
    if _models_ok(url):
        return "chat"
    return None


def _base_url(url: str) -> str:
    text = url.strip().rstrip("/")
    if "://" not in text:
        text = "http://" + text
    return text


def _split_host(url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is not None:
        return host, parsed.port
    return host, 443 if parsed.scheme == "https" else 80


def _server_live() -> bool:
    runtime = _SERVER
    return runtime is not None and runtime["proc"].poll() is None


def _clear_client() -> None:
    global _SERVER
    runtime = _SERVER
    _SERVER = None
    if runtime is None:
        return
    log = runtime.get("log")
    if log is not None:
        log.close()


def _runtime(
    proc,
    *,
    url: str,
    port: int,
    gguf: str,
    adapters: dict[str, str],
    ctx: int,
    tokenizer: str,
    mode: str,
    owned: bool,
    log,
) -> dict:
    return {
        "proc": proc,
        "port": port,
        "log": log,
        "log_path": _LOG,
        "gguf": gguf,
        "adapters": adapters,
        "ctx": ctx,
        "tokenizer": tokenizer,
        "tok": None,
        "owned": owned,
        "url": url,
        "mode": mode,
    }


def _use_existing(
    url: str,
    tokenizer: str,
    *,
    gguf: str | None = None,
    adapters: dict[str, str] | None = None,
    ctx: int = 0,
) -> bool:
    """Attach to ``url`` when it is already serving. Does not start a process."""
    mode = _probe(url)
    if mode is None:
        return False
    loaded = _server_on(url) if mode == "llama" else None
    if loaded is None and mode == "llama" and adapters is not None:
        loaded = _described_server(url)
    if adapters is not None:
        if loaded is None:
            raise RuntimeError(
                f"server at {url} is already running but its GGUF and adapters cannot be verified"
            )
        _require_matching_model(loaded, str(gguf), adapters, ctx)
    if loaded is not None and int(loaded.get("pid") or 0) > 0:
        _connect_loaded(loaded, tokenizer)
    elif loaded is not None:
        _connect_described(loaded, tokenizer)
    else:
        _connect_remote(url, tokenizer, mode)
    print(f"server already loaded at {url}; not loading again", flush=True)
    return True


def _same_path(left: str, right: str) -> bool:
    if os.path.normcase(os.path.normpath(left)) == os.path.normcase(os.path.normpath(right)):
        return True
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False


def _require_matching_model(loaded: dict, gguf: str, adapters: dict[str, str], ctx: int) -> None:
    loaded_adapters = loaded.get("adapters") or {}
    same = set(loaded_adapters) == set(adapters) and all(
        _same_path(str(loaded_adapters.get(name) or ""), path) for name, path in adapters.items()
    )
    if (
        not _same_path(str(loaded.get("gguf") or ""), gguf)
        or not same
        or int(loaded.get("ctx") or 0) < ctx
    ):
        raise RuntimeError("server already runs different GGUF, adapters, or context")


def _connect_loaded(loaded: dict, tokenizer: str) -> None:
    global _SERVER
    _SERVER = _runtime(
        _Running(int(loaded["pid"])),
        url=str(loaded["url"]),
        port=int(loaded["port"]),
        gguf=str(loaded.get("gguf") or ""),
        adapters=dict(loaded.get("adapters") or {}),
        ctx=int(loaded.get("ctx") or 0),
        tokenizer=tokenizer,
        mode="llama",
        owned=False,
        log=None,
    )
    _write_state()


def _connect_described(loaded: dict, tokenizer: str) -> None:
    """Attach using paths from ``/props`` and ``/lora-adapters``. No process id."""
    global _SERVER
    url = str(loaded["url"])
    _host, port = _split_host(url)
    _SERVER = _runtime(
        _Remote(url),
        url=url,
        port=int(loaded.get("port") or port),
        gguf=str(loaded.get("gguf") or ""),
        adapters=dict(loaded.get("adapters") or {}),
        ctx=int(loaded.get("ctx") or 0),
        tokenizer=tokenizer,
        mode="llama",
        owned=False,
        log=None,
    )


def _connect_remote(url: str, tokenizer: str, mode: str) -> None:
    global _SERVER
    _host, port = _split_host(url)
    _SERVER = _runtime(
        _Remote(url),
        url=url,
        port=port,
        gguf="",
        adapters={},
        ctx=0,
        tokenizer=tokenizer,
        mode=mode,
        owned=False,
        log=None,
    )


def _other_local(target: str) -> str | None:
    for item in _loaded_servers():
        if item["url"] != target:
            return str(item["url"])
    return None


def _server_on(url: str) -> dict | None:
    for item in _loaded_servers():
        if item["url"] == url:
            return item
    return None


def _described_server(url: str) -> dict | None:
    """Base, adapter paths, and context reported by llama-server."""
    props = _get_json(url, "/props")
    listed = _get_json(url, "/lora-adapters")
    if not isinstance(props, dict) or not isinstance(listed, list):
        return None
    gguf = str(props.get("model_path") or "")
    settings = props.get("default_generation_settings")
    ctx = 0
    if isinstance(settings, dict):
        try:
            ctx = int(settings.get("n_ctx") or 0)
        except (TypeError, ValueError):
            ctx = 0
    ordered: list[tuple[int, str]] = []
    for item in listed:
        if not isinstance(item, dict) or "id" not in item:
            continue
        try:
            index = int(item["id"])
        except (TypeError, ValueError):
            continue
        ordered.append((index, str(item.get("path") or "")))
    ordered.sort()
    if not gguf or [index for index, _path in ordered] != [0, 1]:
        return None
    _host, port = _split_host(url)
    return {
        "port": port,
        "url": url,
        "gguf": gguf,
        "adapters": dict(zip(_TASKS, (path for _index, path in ordered), strict=True)),
        "ctx": ctx,
    }


def _get_json(url: str, path: str):
    try:
        request = urllib.request.Request(f"{url}{path}", headers=_auth_headers())
        with urllib.request.urlopen(request, timeout=2) as resp:
            body = json.loads(resp.read().decode())
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None
    return body


def _loaded_servers() -> list[dict]:
    """llama-server processes that are listening and healthy."""
    root = Path("/proc")
    if not root.is_dir():
        return []
    found = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        argv = _cmdline(int(entry.name))
        if not argv or Path(argv[0]).name != "llama-server":
            continue
        port = _listen_port(int(entry.name))
        if port is None:
            continue
        host = _arg(argv, "--host") or "127.0.0.1"
        url = f"http://{host}:{port}"
        if not _health_ok(url):
            continue
        # Both "--lora a,b" and "--lora a --lora b" load two adapters in that order.
        names = [part for value in _all_args(argv, "--lora") for part in value.split(",") if part]
        adapters = dict(zip(_TASKS, names, strict=True)) if len(names) == 2 else {}
        ctx_text = _arg(argv, "-c") or _arg(argv, "--ctx-size")
        ctx_text = ctx_text.strip() if ctx_text else ctx_text
        found.append(
            {
                "pid": int(entry.name),
                "port": port,
                "url": url,
                "gguf": _arg(argv, "-m") or _arg(argv, "--model") or "",
                "adapters": adapters,
                "ctx": int(ctx_text) if ctx_text and ctx_text.isdigit() else 0,
            }
        )
    return found


def _cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _arg(argv: list[str], flag: str) -> str | None:
    values = _all_args(argv, flag)
    return values[0] if values else None


def _all_args(argv: list[str], flag: str) -> list[str]:
    """Every value of ``flag``, as ``--flag value`` or ``--flag=value``."""
    values: list[str] = []
    for index, item in enumerate(argv):
        if item == flag and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif item.startswith(flag + "="):
            values.append(item[len(flag) + 1 :])
    return values


def _listen_port(pid: int) -> int | None:
    inodes: set[str] = set()
    try:
        for entry in Path(f"/proc/{pid}/fd").iterdir():
            try:
                target = os.readlink(entry)
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inodes.add(target[8:-1])
    except OSError:
        return None
    if not inodes:
        return None
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) < 10 or fields[3] != "0A" or fields[9] not in inodes:
                continue
            return int(fields[1].rsplit(":", 1)[-1], 16)
    return None


def _write_state() -> None:
    runtime = _SERVER
    if runtime is None:
        return
    pid = getattr(runtime["proc"], "pid", -1)
    if not isinstance(pid, int) or pid <= 0:
        return
    _private_dir(_STATE.parent)
    _STATE.write_text(
        json.dumps(
            {
                "pid": pid,
                "port": runtime["port"],
                "url": runtime["url"],
                "gguf": runtime["gguf"],
                "adapters": runtime["adapters"],
                "ctx": runtime["ctx"],
                "tokenizer": runtime["tokenizer"],
            }
        )
    )


def _gguf_path(gguf: str) -> str:
    path = Path(gguf)
    if not path.is_file():
        raise RuntimeError(f"Q6_K base is missing: {path}")
    return str(path)


def _binary() -> str:
    env = os.environ.get("LLAMA_SERVER")
    found = env or shutil.which("llama-server")
    if not found or not Path(found).is_file():
        raise RuntimeError("llama-server is not on PATH; set LLAMA_SERVER")
    return str(Path(found).resolve())


def _child_env(binary: str) -> dict[str, str]:
    env = os.environ.copy()
    root = Path(binary).resolve().parent
    libs = [str(root)]
    parent = root.parent
    libs.extend(str(path) for path in parent.glob("cudart-*") if path.is_dir())
    previous = env.get("LD_LIBRARY_PATH")
    if previous:
        libs.append(previous)
    env["LD_LIBRARY_PATH"] = ":".join(libs)
    return env


def _wait_ready(proc: subprocess.Popen, url: str, log_path: Path) -> None:
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited {proc.returncode}\n{_tail(log_path)}")
        if _health_ok(url):
            return
        time.sleep(0.4)
    raise RuntimeError(f"llama-server did not become ready\n{_tail(log_path)}")


def _tail(path: Path) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-40:])


def runtime_tokenizer() -> str | None:
    runtime = _SERVER
    if runtime is None:
        return None
    return runtime.get("tokenizer")


def load_tokenizer(path: str | None = None):
    key = _tokenizer_dir(path)
    tok = _TOKENIZERS.get(key)
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(key)
        _TOKENIZERS[key] = tok
    return tok


def render_prompt(system: str, user: str, tokenizer: str | None = None) -> str:
    """Chat prompt with thinking disabled. The server is started with ``--no-jinja``."""
    return load_tokenizer(tokenizer).apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def prompt_tokens(prompt: str, tokenizer: str | None = None) -> int:
    return len(load_tokenizer(tokenizer)(prompt, add_special_tokens=False).input_ids)


def _completion(
    url: str, prompt: str, scales: list[dict[str, float]], timeout: float | None = None
) -> tuple[str, object, object]:
    raw = _post_url(
        url,
        "/completion",
        {
            "prompt": prompt,
            "n_predict": 32,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "min_p": 0,
            "cache_prompt": False,
            "lora": scales,
        },
        timeout=timeout,
    )
    text = str(raw.get("content") or "").strip()
    if not text:
        raise RuntimeError("llama-server reply has no content")
    return text, raw.get("tokens_evaluated"), raw.get("tokens_predicted")


def _post_url(url: str, path: str, payload: dict, timeout: float | None = None) -> dict:
    data = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{url}{path}",
        data=data,
        headers={"Content-Type": "application/json", **_auth_headers()},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout(timeout)) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"llama-server {exc.code}: {detail}") from exc
    if not isinstance(body, dict):
        raise RuntimeError("llama-server reply is not an object")
    return body
