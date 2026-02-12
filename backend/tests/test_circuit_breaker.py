import app.core.circuit_breaker as circuit_breaker_module
from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


def test_half_open_probe_timeout_reopens_and_allows_future_probe(monkeypatch):
    fake_clock = {"now": 100.0}
    monkeypatch.setattr(circuit_breaker_module.time, "time", lambda: fake_clock["now"])

    breaker = CircuitBreaker(
        CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=5.0,
            half_open_max_calls=1,
            half_open_probe_timeout=2.0,
        ),
        name="test_breaker",
    )

    breaker.record_failure(reset_timeout=5.0)
    assert breaker.state == CircuitState.OPEN

    fake_clock["now"] += 5.1
    assert breaker.can_proceed() is True
    assert breaker.state == CircuitState.HALF_OPEN

    # Probe slot is consumed; without a success/failure record this would
    # previously block forever in HALF_OPEN.
    assert breaker.can_proceed() is False

    fake_clock["now"] += 2.1
    assert breaker.can_proceed() is False
    assert breaker.state == CircuitState.OPEN

    fake_clock["now"] += 5.1
    assert breaker.can_proceed() is True
    assert breaker.state == CircuitState.HALF_OPEN

