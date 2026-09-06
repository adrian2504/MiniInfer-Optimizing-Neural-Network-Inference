import pytest

from miniinfer.metrics import percentile, summarize


def test_aggregate_rates_use_total_time():
    result = summarize([
        {"latency_ms": 100, "ttft_ms": 20},
        {"latency_ms": 300, "ttft_ms": 40},
    ], batch_size=2, new_tokens=5)
    assert result["output_tokens_per_second"] == pytest.approx(50)
    assert result["requests_per_second"] == pytest.approx(10)
    assert result["decode_tokens_per_second"] == pytest.approx(16 / 0.34)
    assert result["latency_p50_ms"] == 200


def test_one_token_has_no_decode_rate():
    assert summarize([{"latency_ms": 10, "ttft_ms": 10}], 1, 1)["decode_tokens_per_second"] is None


def test_percentile_interpolation_and_invalid_input():
    assert percentile([30, 10, 20], 0.95) == 29
    with pytest.raises(ValueError):
        percentile([], 0.5)
