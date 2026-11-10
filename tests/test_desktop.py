from __future__ import annotations

import tempfile

import httpx
import pytest

from neutrons.desktop import (
    API_BASE_URL,
    DesktopStartupError,
    ManagedBackend,
    backend_is_healthy,
    backend_is_ready,
    ensure_backend,
)


def test_backend_is_healthy_for_neutrons() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"status": "ok", "app": "NEUTRONS"},
            request=request,
        )
    )
    with httpx.Client(transport=transport, base_url=API_BASE_URL) as client:
        assert backend_is_healthy(client)


def test_backend_is_not_healthy_for_other_service() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"status": "ok", "app": "other"}, request=request)
    )
    with httpx.Client(transport=transport, base_url=API_BASE_URL) as client:
        assert not backend_is_healthy(client)


def test_backend_liveness_can_be_healthy_while_model_readiness_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok", "app": "NEUTRONS"}, request=request)
        return httpx.Response(
            503,
            json={"detail": "The configured model was not found in Ollama."},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url=API_BASE_URL) as client:
        ready, reason = backend_is_ready(client)
        assert backend_is_healthy(client)

    assert not ready
    assert reason == "Backend đang chạy nhưng Ollama/model chưa sẵn sàng."


def test_backend_readiness_succeeds_when_model_is_available() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"status": "ready", "model": "qwen3:4b"},
            request=request,
        )
    )
    with httpx.Client(transport=transport, base_url=API_BASE_URL) as client:
        assert backend_is_ready(client) == (True, "Model sẵn sàng.")


def test_backend_startup_reports_port_conflict(monkeypatch) -> None:
    monkeypatch.setattr("neutrons.desktop.backend_is_healthy", lambda client: client.is_closed)
    monkeypatch.setattr("neutrons.desktop._port_is_open", lambda: True)

    with pytest.raises(DesktopStartupError, match="đang được một ứng dụng khác sử dụng"):
        ensure_backend()


def test_model_readiness_failure_does_not_become_a_port_conflict(monkeypatch) -> None:
    monkeypatch.setattr("neutrons.desktop.backend_is_healthy", lambda client: not client.is_closed)
    monkeypatch.setattr(
        "neutrons.desktop.backend_is_ready",
        lambda client: (client.is_closed, "configured model is missing"),
    )
    monkeypatch.setattr(
        "neutrons.desktop._port_is_open",
        lambda: (_ for _ in ()).throw(AssertionError("port must not be checked as a conflict")),
    )

    assert ensure_backend() is None


def test_backend_startup_includes_captured_process_log(monkeypatch) -> None:
    class ExitedProcess:
        def poll(self) -> int:
            return 1

        def terminate(self) -> None:
            raise AssertionError("an exited process must not be terminated")

    def fake_popen(*args, stdout, **kwargs) -> ExitedProcess:
        assert args
        assert kwargs
        stdout.write(b"uvicorn startup error: address already in use\n")
        stdout.flush()
        return ExitedProcess()

    monkeypatch.setattr("neutrons.desktop.backend_is_healthy", lambda client: client.is_closed)
    monkeypatch.setattr("neutrons.desktop._port_is_open", lambda: False)
    monkeypatch.setattr("neutrons.desktop.subprocess.Popen", fake_popen)

    with pytest.raises(DesktopStartupError, match="address already in use"):
        ensure_backend()


def test_managed_backend_stops_child_process_and_closes_log() -> None:
    class RunningProcess:
        terminated = False

        def poll(self) -> int | None:
            return None if not self.terminated else 0

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float) -> int:
            assert timeout == 5
            return 0

        def kill(self) -> None:
            self.terminated = True

    output = tempfile.TemporaryFile(mode="w+b")
    process = RunningProcess()
    backend = ManagedBackend(process, output)

    backend.stop()
    backend.close_output()

    assert process.terminated
    assert output.closed
