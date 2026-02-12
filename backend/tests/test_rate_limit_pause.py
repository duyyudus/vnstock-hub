import pytest

import app.services.vnstock_service.rate_limit_pause as pause_module
from app.services.vnstock_service.rate_limit_pause import SharedRateLimitPauseController


@pytest.mark.asyncio
async def test_register_rate_limit_wait_uses_longest_timer(monkeypatch):
    class _FakeStatus:
        @property
        def rate_limit_seconds_remaining(self):
            return 60.0

    class _FakeCircuit:
        @property
        def time_until_half_open(self):
            return 20.0

    monkeypatch.setattr(pause_module, "sync_status", _FakeStatus())
    monkeypatch.setattr(pause_module, "api_circuit_breaker", _FakeCircuit())
    monkeypatch.setattr(pause_module.time, "monotonic", lambda: 100.0)

    controller = SharedRateLimitPauseController()
    wait_seconds = await controller.register_rate_limit_and_get_wait(30.0)
    assert wait_seconds == pytest.approx(60.0, abs=1e-9)


@pytest.mark.asyncio
async def test_wait_if_paused_rechecks_when_deadline_extended(monkeypatch):
    class _FakeStatus:
        @property
        def rate_limit_seconds_remaining(self):
            return 0.0

    class _FakeCircuit:
        @property
        def time_until_half_open(self):
            return 0.0

    fake_clock = {"now": 0.0}
    sleep_calls = []
    did_extend = {"value": False}

    def _fake_monotonic() -> float:
        return fake_clock["now"]

    controller = SharedRateLimitPauseController()

    async def _fake_sleep(seconds: float):
        sleep_calls.append(seconds)
        fake_clock["now"] += seconds
        if not did_extend["value"]:
            did_extend["value"] = True
            controller._pause_deadline_monotonic = fake_clock["now"] + 10.0

    monkeypatch.setattr(pause_module, "sync_status", _FakeStatus())
    monkeypatch.setattr(pause_module, "api_circuit_breaker", _FakeCircuit())
    monkeypatch.setattr(pause_module.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(pause_module.asyncio, "sleep", _fake_sleep)

    wait_seconds = await controller.register_rate_limit_and_get_wait(5.0)
    assert wait_seconds == pytest.approx(5.0, abs=1e-9)

    await controller.wait_if_paused()
    assert sleep_calls == [5.0, 10.0]
