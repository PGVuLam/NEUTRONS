from __future__ import annotations

import httpx

from neutrons.desktop import API_BASE_URL, backend_is_healthy


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
