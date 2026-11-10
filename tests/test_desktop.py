from __future__ import annotations

import tempfile

import httpx
import pytest

from neutrons.desktop import (
    API_BASE_URL,
    DesktopStartupError,
    ManagedBackend,
    backend_is_healthy,
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


def test_backend_startup_reports_port_conflict(monkeypatch) -> None:
    monkeypatch.setattr("neutrons.desktop.backend_is_healthy", lambda client: client.is_closed)
    monkeypatch.setattr("neutrons.desktop._port_is_open", lambda: True)

    with pytest.raises(DesktopStartupError, match="đang được một ứng dụng khác sử dụng"):
        ensure_backend()


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
