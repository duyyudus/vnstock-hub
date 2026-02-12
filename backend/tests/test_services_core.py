import pytest
import asyncio
from unittest.mock import MagicMock, patch
from app.core.circuit_breaker import CircuitState
from app.services.vnstock_service.core import (
    _is_rate_limit_error,
    retry_with_backoff,
    async_retry_with_backoff,
    RateLimitError,
    CircuitOpenError
)

def test_is_rate_limit_error():
    assert _is_rate_limit_error(RateLimitError("test")) is True
    assert _is_rate_limit_error(CircuitOpenError("test")) is True
    assert _is_rate_limit_error(SystemExit("Rate limit reached")) is True
    assert _is_rate_limit_error(Exception("429 Too Many Requests")) is True
    assert _is_rate_limit_error(Exception("Normal error")) is False

def test_retry_with_backoff_success():
    mock_func = MagicMock(return_value="success")
    with patch("app.services.vnstock_service.core.api_circuit_breaker") as mock_cb:
        mock_cb.can_proceed.return_value = True
        result = retry_with_backoff(mock_func)
        assert result == "success"
        assert mock_func.call_count == 1
        mock_cb.record_success.assert_called_once()

def test_retry_with_backoff_regular_error():
    mock_func = MagicMock(side_effect=Exception("Normal error"))
    with patch("app.services.vnstock_service.core.api_circuit_breaker") as mock_cb:
        mock_cb.can_proceed.return_value = True
        with pytest.raises(Exception, match="Normal error"):
            retry_with_backoff(mock_func, max_retries=2)
        
        # 1 initial call + 2 retries = 3 calls total
        assert mock_func.call_count == 3

def test_retry_with_backoff_rate_limit_fail_fast():
    mock_func = MagicMock(side_effect=Exception("Rate limit exceeded"))
    with patch("app.services.vnstock_service.core.api_circuit_breaker") as mock_cb:
        with patch("app.services.vnstock_service.core.sync_status") as mock_sync:
            mock_cb.can_proceed.return_value = True
            with pytest.raises(Exception, match="Rate limit exceeded"):
                retry_with_backoff(mock_func, max_retries=2)
            
            # Should fail fast after first call
            assert mock_func.call_count == 1
            mock_cb.record_failure.assert_called_once()
            mock_sync.set_rate_limited.assert_called_once()

@pytest.mark.asyncio
async def test_async_retry_with_backoff_success():
    mock_func = MagicMock(return_value="success")
    with patch("app.services.vnstock_service.core.api_circuit_breaker") as mock_cb:
        mock_cb.can_proceed.return_value = True
        result = await async_retry_with_backoff(mock_func)
        assert result == "success"
        mock_cb.record_success.assert_called_once()

@pytest.mark.asyncio
async def test_async_retry_with_backoff_rate_limit_retry():
    mock_func = MagicMock(side_effect=[Exception("Rate limit"), "success"])
    with patch("app.services.vnstock_service.core.api_circuit_breaker") as mock_cb:
        with patch("app.services.vnstock_service.core.sync_status") as mock_sync:
            with patch("asyncio.sleep", return_value=None) as mock_sleep:
                mock_cb.can_proceed.return_value = True
                result = await async_retry_with_backoff(mock_func, initial_delay=1.0)
                
                assert result == "success"
                assert mock_func.call_count == 2
                mock_sleep.assert_called_once_with(1.0)
                assert mock_cb.record_failure.call_count == 1
                assert mock_cb.record_success.call_count == 1


def test_is_rate_limit_error_half_open_non_rate_limit_error():
    with patch("app.services.vnstock_service.core.api_circuit_breaker") as mock_cb:
        mock_cb.state = CircuitState.HALF_OPEN
        assert _is_rate_limit_error(Exception("database exploded")) is True


def test_is_rate_limit_error_closed_non_rate_limit_error_stays_false():
    with patch("app.services.vnstock_service.core.api_circuit_breaker") as mock_cb:
        mock_cb.state = CircuitState.CLOSED
        assert _is_rate_limit_error(Exception("database exploded")) is False
