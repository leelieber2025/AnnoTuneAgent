# Copyright 2026 Zhao Li (李钊)
# Licensed under the Apache License, Version 2.0

import json
import os
from pathlib import Path

import pytest

import annotuneagent.local_lora as ll

_ADAPTERS = {"name": "name.gguf", "ml": "ml.gguf"}


class _Proc:
    def __init__(self, pid: int, code: int | None = None) -> None:
        self.pid = pid
        self.returncode = code

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return 0


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(ll, "_SERVER", None)
    monkeypatch.setattr(ll, "_STATE", tmp_path / "state.json")
    monkeypatch.setattr(ll, "_LOG", tmp_path / "server.log")
    monkeypatch.delenv("ANNOTUNE_URL", raising=False)


def test_reuses_a_loaded_server_without_starting(monkeypatch, tmp_path):
    base = tmp_path / "base.gguf"
    base.write_bytes(b"x")
    monkeypatch.setattr(ll, "_health_ok", lambda url: url == "http://127.0.0.1:8080")
    monkeypatch.setattr(ll, "_models_ok", lambda url: False)
    monkeypatch.setattr(ll, "_alive", lambda pid: pid == 42)
    monkeypatch.setattr(
        ll,
        "_loaded_servers",
        lambda: [
            {
                "pid": 42,
                "port": 8080,
                "url": "http://127.0.0.1:8080",
                "gguf": str(base.resolve()),
                "adapters": {name: str(Path(path).resolve()) for name, path in _ADAPTERS.items()},
                "ctx": 8192,
            }
        ],
    )

    def fail(*args, **kwargs):
        raise AssertionError("started a second server")

    monkeypatch.setattr(ll.subprocess, "Popen", fail)
    ll.open_adapters(_ADAPTERS, gguf=str(base), tokenizer="/tok")
    ll.open_adapters(_ADAPTERS, gguf=str(base), tokenizer="/tok")
    with pytest.raises(RuntimeError, match="different tokenizer"):
        ll.open_adapters(_ADAPTERS, gguf=str(base), tokenizer="/other-tokenizer")
    assert ll.local_server() == "http://127.0.0.1:8080"
    assert ll._SERVER["owned"] is False
    saved = json.loads(ll._STATE.read_text())
    assert saved["pid"] == 42
    assert saved["adapters"]["name"] == str(Path("name.gguf").resolve())


def test_refuses_to_reuse_a_different_model(monkeypatch, tmp_path):
    base = tmp_path / "base.gguf"
    base.write_bytes(b"x")
    monkeypatch.setattr(ll, "_health_ok", lambda url: True)
    monkeypatch.setattr(ll, "_models_ok", lambda url: False)
    monkeypatch.setattr(
        ll,
        "_loaded_servers",
        lambda: [
            {
                "pid": 42,
                "port": 8080,
                "url": "http://127.0.0.1:8080",
                "gguf": "/other.gguf",
                "adapters": {"name": "/other-name.gguf", "ml": "/other-ml.gguf"},
                "ctx": 8192,
            }
        ],
    )
    with pytest.raises(RuntimeError, match="different GGUF"):
        ll.open_adapters(_ADAPTERS, gguf=str(base))
    assert ll._SERVER is None


def test_remote_port_does_not_match_a_local_server(monkeypatch):
    monkeypatch.setattr(
        ll,
        "_loaded_servers",
        lambda: [{"url": "http://127.0.0.1:8080", "port": 8080}],
    )
    assert ll._server_on("http://remote.example:8080") is None


def test_refuses_to_start_while_another_server_is_loaded(monkeypatch, tmp_path):
    base = tmp_path / "base.gguf"
    base.write_bytes(b"x")
    monkeypatch.setattr(ll, "_health_ok", lambda url: False)
    monkeypatch.setattr(ll, "_models_ok", lambda url: False)
    monkeypatch.setattr(
        ll,
        "_loaded_servers",
        lambda: [
            {
                "pid": 7,
                "port": 39239,
                "url": "http://127.0.0.1:39239",
                "gguf": "/base.gguf",
                "adapters": {},
                "ctx": 8192,
            }
        ],
    )

    def fail(*args, **kwargs):
        raise AssertionError("started a second server")

    monkeypatch.setattr(ll.subprocess, "Popen", fail)
    with pytest.raises(RuntimeError, match="already loaded"):
        ll.open_adapters(_ADAPTERS, gguf=str(base), tokenizer="/tok")


def test_starts_on_the_service_port_when_nothing_is_loaded(monkeypatch, tmp_path):
    base = tmp_path / "base.gguf"
    base.write_bytes(b"x")
    monkeypatch.setattr(ll, "_health_ok", lambda url: False)
    monkeypatch.setattr(ll, "_models_ok", lambda url: False)
    monkeypatch.setattr(ll, "_loaded_servers", lambda: [])
    monkeypatch.setattr(ll, "_binary", lambda: "/bin/true")
    monkeypatch.setattr(ll, "_child_env", lambda binary: {})
    monkeypatch.setattr(ll, "_wait_ready", lambda *args: None)
    seen = {}

    def fake_popen(cmd, **kwargs):
        seen["cmd"] = cmd
        return _Proc(123)

    monkeypatch.setattr(ll.subprocess, "Popen", fake_popen)
    ll.open_adapters(_ADAPTERS, gguf=str(base), tokenizer="/tok", ctx=8192)
    port = seen["cmd"][seen["cmd"].index("--port") + 1]
    host = seen["cmd"][seen["cmd"].index("--host") + 1]
    assert host == "127.0.0.1"
    assert port == "8080"
    assert ll._SERVER["owned"] is True
    assert ll._SERVER["url"] == "http://127.0.0.1:8080"


def test_attaches_to_an_external_chat_api(monkeypatch):
    monkeypatch.setattr(ll, "_health_ok", lambda url: False)
    monkeypatch.setattr(ll, "_models_ok", lambda url: url == "http://127.0.0.1:8080")

    def fail(*args, **kwargs):
        raise AssertionError("started a server")

    monkeypatch.setattr(ll.subprocess, "Popen", fail)
    assert ll.attach("127.0.0.1:8080", tokenizer="/tok") == "http://127.0.0.1:8080"
    assert ll._SERVER["mode"] == "chat"
    assert ll._SERVER["owned"] is False
    assert ll.attach(tokenizer="/tok") == "http://127.0.0.1:8080"


def test_release_does_not_drop_another_servers_state(tmp_path):
    ll._STATE.write_text(json.dumps({"pid": 5, "port": 8080}))
    ll._SERVER = {
        "proc": _Proc(99, code=1),
        "log": None,
        "owned": True,
        "url": "http://127.0.0.1:8080",
    }
    ll.release()
    assert json.loads(ll._STATE.read_text())["pid"] == 5
    assert ll._SERVER is None


def test_action_is_not_a_runtime_task():
    with pytest.raises(ValueError, match="unknown task action"):
        ll.complete("action", "system", {}, maxlen=2048)


def test_package_imports_where_os_has_no_getuid():
    import subprocess
    import sys

    code = "import os; del os.getuid; import annotuneagent, annotuneagent.local_lora"
    src = str(Path(ll.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": src}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr


def test_private_dir_is_created_closed_and_reused(tmp_path):
    import stat

    target = tmp_path / "run" / "annotune"
    ll._private_dir(target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    ll._private_dir(target)


def test_private_dir_refuses_a_directory_others_can_use(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    with pytest.raises(RuntimeError, match="not private"):
        ll._private_dir(shared)


def test_private_dir_refuses_a_symlink(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(RuntimeError, match="plain directory"):
        ll._private_dir(link)


def test_state_in_a_shared_directory_is_not_read(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "state.json").write_text(json.dumps({"pid": 1, "port": 9}))
    shared.chmod(0o777)
    monkeypatch.setattr(ll, "_STATE", shared / "state.json")
    with pytest.raises(RuntimeError, match="not private"):
        ll._read_state()
