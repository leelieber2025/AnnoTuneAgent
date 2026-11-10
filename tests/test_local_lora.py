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


def test_reuses_a_server_described_over_http(monkeypatch, tmp_path):
    base = tmp_path / "base.gguf"
    name = tmp_path / "name.gguf"
    ml = tmp_path / "ml.gguf"
    for path in (base, name, ml):
        path.write_bytes(b"x")
    adapters = {"name": str(name), "ml": str(ml)}
    monkeypatch.setattr(ll, "_health_ok", lambda url: True)
    monkeypatch.setattr(ll, "_models_ok", lambda url: False)
    monkeypatch.setattr(ll, "_loaded_servers", lambda: [])

    def get(url, path):
        assert url == "http://127.0.0.1:8080"
        if path == "/props":
            return {
                "model_path": str(base.resolve()) + "/",
                "default_generation_settings": {"n_ctx": 8192},
            }
        if path == "/lora-adapters":
            return [
                {"id": 1, "path": str(ml.resolve()), "scale": 1.0},
                {"id": 0, "path": str(name.resolve()), "scale": 1.0},
            ]
        return None

    def fail(*_args, **_kwargs):
        raise AssertionError("started a second server")

    monkeypatch.setattr(ll, "_get_json", get)
    monkeypatch.setattr(ll.subprocess, "Popen", fail)
    ll.open_adapters(adapters, gguf=str(base), tokenizer="/tok")
    assert ll._SERVER["mode"] == "llama"
    assert ll._SERVER["owned"] is False
    assert ll._SERVER["proc"].pid == -1
    assert ll._SERVER["ctx"] == 8192
    assert ll.local_server() == "http://127.0.0.1:8080"
    assert not ll._STATE.exists()
    ll.open_adapters(adapters, gguf=str(base), tokenizer="/tok")


def test_http_description_rejects_a_different_context(monkeypatch, tmp_path):
    base = tmp_path / "base.gguf"
    base.write_bytes(b"x")
    monkeypatch.setattr(ll, "_health_ok", lambda url: True)
    monkeypatch.setattr(ll, "_loaded_servers", lambda: [])
    monkeypatch.setattr(
        ll,
        "_get_json",
        lambda url, path: {
            "/props": {
                "model_path": str(base.resolve()),
                "default_generation_settings": {"n_ctx": 2048},
            },
            "/lora-adapters": [
                {"id": 0, "path": str(Path("name.gguf").resolve())},
                {"id": 1, "path": str(Path("ml.gguf").resolve())},
            ],
        }[path],
    )
    with pytest.raises(RuntimeError, match="different GGUF"):
        ll.open_adapters(_ADAPTERS, gguf=str(base))
    assert ll._SERVER is None


def test_http_description_must_be_readable(monkeypatch, tmp_path):
    base = tmp_path / "base.gguf"
    base.write_bytes(b"x")
    monkeypatch.setattr(ll, "_health_ok", lambda url: True)
    monkeypatch.setattr(ll, "_loaded_servers", lambda: [])
    monkeypatch.setattr(ll, "_get_json", lambda url, path: None)
    with pytest.raises(RuntimeError, match="cannot be verified"):
        ll.open_adapters(_ADAPTERS, gguf=str(base))


def test_alive_on_windows_does_not_signal(monkeypatch):
    monkeypatch.setattr(ll.os, "name", "nt")
    monkeypatch.setattr(ll, "_windows_alive", lambda pid: pid == 7)

    def boom(*_a, **_k):
        raise AssertionError("os.kill")

    monkeypatch.setattr(ll.os, "kill", boom)
    assert ll._alive(7) is True
    assert ll._alive(3) is False
    assert ll._alive(0) is False


def test_request_timeout_is_configurable(monkeypatch):
    assert ll.request_timeout() == 600
    assert ll.request_timeout(10) == 10
    monkeypatch.setenv("ANNOTUNE_TIMEOUT", "300")
    assert ll.request_timeout() == 300
    monkeypatch.setenv("ANNOTUNE_TIMEOUT", "nope")
    assert ll.request_timeout() == 600


def test_complete_url_enables_one_adapter(monkeypatch):
    seen = {}

    def post(url, path, payload, timeout=None):
        seen["url"] = url
        seen["path"] = path
        seen["payload"] = payload
        seen["timeout"] = timeout
        return {"content": ' {"names": []} ', "tokens_evaluated": 4, "tokens_predicted": 2}

    monkeypatch.setattr(ll, "render_prompt", lambda system, user, tokenizer=None: "PROMPT")
    monkeypatch.setattr(ll, "prompt_tokens", lambda prompt, tokenizer=None: 8)
    monkeypatch.setattr(ll, "_post_url", post)
    text = ll.complete_url("http://127.0.0.1:8080/", "ml", "system", {"a": 1}, maxlen=8192)
    assert text == '{"names": []}'
    assert seen["path"] == "/completion"
    assert seen["payload"]["lora"] == [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}]
    assert seen["payload"]["prompt"] == "PROMPT"
    assert seen["timeout"] is None
    assert ll.last_call()["tokens"] == 8


def test_complete_url_refuses_a_long_prompt(monkeypatch):
    monkeypatch.setattr(ll, "render_prompt", lambda *_a, **_k: "PROMPT")
    monkeypatch.setattr(ll, "prompt_tokens", lambda *_a, **_k: 3000)

    def fail(*_args, **_kwargs):
        raise AssertionError("posted")

    monkeypatch.setattr(ll, "_post_url", fail)
    with pytest.raises(ll.SequenceTooLong):
        ll.complete_url("http://127.0.0.1:8080", "name", "system", "user", maxlen=2048)


def test_state_in_a_shared_directory_is_not_read(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "state.json").write_text(json.dumps({"pid": 1, "port": 9}))
    shared.chmod(0o777)
    monkeypatch.setattr(ll, "_STATE", shared / "state.json")
    with pytest.raises(RuntimeError, match="not private"):
        ll._read_state()
