import pytest
import httpx

from tools.cafef_scraper.downloader import HttpDownloader, RateLimiter, is_pdf_response


@pytest.mark.asyncio
async def test_rate_limiter_adaptive_decreases_and_recovers():
    limiter = RateLimiter(
        rate_limit_rps=4.0,
        adaptive=True,
        min_rps=1.0,
        recovery_multiplier=1.05,
    )
    initial = limiter.current_rps

    await limiter.report_result(success=False, status_code=429, retryable=True)
    dropped = limiter.current_rps
    assert dropped < initial

    await limiter.report_result(success=True, status_code=200, retryable=False)
    assert limiter.current_rps > dropped


@pytest.mark.asyncio
async def test_rate_limiter_non_adaptive_no_change():
    limiter = RateLimiter(rate_limit_rps=3.0, adaptive=False)
    initial = limiter.current_rps
    await limiter.report_result(success=False, status_code=429, retryable=True)
    assert limiter.current_rps == initial


@pytest.mark.asyncio
async def test_rate_limiter_cooldown_streak_pushes_to_min_rps():
    limiter = RateLimiter(
        rate_limit_rps=2.0,
        adaptive=True,
        min_rps=0.6,
        cooldown_seconds=5.0,
        cooldown_trigger_streak=2,
    )
    await limiter.report_result(success=False, status_code=None, retryable=True)
    assert limiter.current_rps <= 2.0
    await limiter.report_result(success=False, status_code=None, retryable=True)
    assert limiter.current_rps == pytest.approx(0.6)


def test_is_pdf_response_accepts_pdf_content_type():
    assert is_pdf_response("application/pdf", b"not even checked")


def test_is_pdf_response_accepts_pdf_signature():
    assert is_pdf_response("application/octet-stream", b"%PDF-1.7\n...")


def test_is_pdf_response_rejects_html():
    assert not is_pdf_response("text/html", b"<html>error</html>")


class _FailingClient:
    def __init__(self):
        self.calls = 0

    async def get(self, *args, **kwargs):
        self.calls += 1
        raise httpx.RequestError("boom", request=httpx.Request("GET", str(args[0])))


class _RecordingClient:
    def __init__(self):
        self.calls = 0
        self.timeouts: list[float] = []

    async def get(self, *args, **kwargs):
        self.calls += 1
        self.timeouts.append(float(kwargs["timeout"]))
        return httpx.Response(200, text="ok", request=httpx.Request("GET", str(args[0])))


@pytest.mark.asyncio
async def test_fetch_text_stage_override_max_retries():
    client = _FailingClient()
    downloader = HttpDownloader(client=client, limiter=RateLimiter(rate_limit_rps=100.0), max_retries=5)
    value = await downloader.fetch_text(
        url="https://example.com",
        stage="id_scan_probe",
        max_retries=2,
    )
    assert value is None
    assert client.calls == 2


@pytest.mark.asyncio
async def test_fetch_text_stage_override_timeout():
    client = _RecordingClient()
    downloader = HttpDownloader(client=client, limiter=RateLimiter(rate_limit_rps=100.0), timeout_seconds=30.0)
    value = await downloader.fetch_text(
        url="https://example.com",
        stage="id_scan_probe",
        timeout_seconds=7.0,
    )
    assert value == "ok"
    assert client.calls == 1
    assert client.timeouts == [7.0]
