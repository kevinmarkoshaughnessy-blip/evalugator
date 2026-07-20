import json
import pytest

from evalugator.api import Api
from evalugator.api.providers import openrouter
from evalugator.api.requests import GetProbsRequest, GetTextRequest, Message


TEXT_REQUEST_1 = GetTextRequest(
    prompt=[Message("user", "foo")],
    temperature=1,
    max_tokens=66,
    context=None,
)
TEXT_REQUEST_SYSTEM = GetTextRequest(
    prompt=[
        Message("system", "hellooo"),
        Message("user", "foo"),
        Message("assistant", "bar"),
    ],
    temperature=0,
    max_tokens=17,
    context=None,
)
TEXT_REQUEST_SMALL_TOKENS = GetTextRequest(
    prompt=[Message("user", "foo")],
    temperature=1,
    max_tokens=10,  # smaller than reasoning minimums, to test max_tokens boosting
    context=None,
)
PROBS_REQUEST_1 = GetProbsRequest(
    prompt=[Message("user", "foo")],
    min_top_n=3,
    num_samples=7,  # ignored by openrouter, like openai
    context=None,
)

TEST_DATA = [
    #   Plain chat — no reasoning
    (
        "openrouter/anthropic/claude-3.5-sonnet",
        TEXT_REQUEST_1,
        {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [{"role": "user", "content": "foo"}],
            "temperature": 1,
            "max_tokens": 66,
        },
    ),
    #   Multi-turn prompt — no reasoning
    (
        "openrouter/meta-llama/llama-3-70b-instruct",
        TEXT_REQUEST_SYSTEM,
        {
            "model": "meta-llama/llama-3-70b-instruct",
            "messages": [
                {"role": "system", "content": "hellooo"},
                {"role": "user", "content": "foo"},
                {"role": "assistant", "content": "bar"},
            ],
            "temperature": 0,
            "max_tokens": 17,
        },
    ),
    #   Probs request
    (
        "openrouter/anthropic/claude-3.5-sonnet",
        PROBS_REQUEST_1,
        {
            "model": "anthropic/claude-3.5-sonnet",
            "messages": [{"role": "user", "content": "foo"}],
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 3,
        },
    ),
]


@pytest.mark.parametrize("model_id, req, expected_api_kwargs", TEST_DATA)
def test_provider(openrouter_echo_requests, model_id, req, expected_api_kwargs):
    response = Api(model_id).execute(req).result()
    kwargs = json.loads(response.raw_responses[0]._echo)
    assert kwargs == expected_api_kwargs


def test_provides_model():
    assert openrouter.provides_model("openrouter/anthropic/claude-3.5-sonnet")
    assert not openrouter.provides_model("anthropic/claude-3.5-sonnet")
    assert not openrouter.provides_model("gpt-4o")


# ──────────────────────────────────────────────────────────
#   Reasoning effort tests
# ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=False)
def reset_effort():
    """Ensure _effort is restored to 'none' after each reasoning test."""
    yield
    openrouter.set_effort("none")


def test_budget_thinking_high_effort(openrouter_echo_requests, reset_effort):
    openrouter.set_effort("high")
    response = Api("openrouter/anthropic/claude-haiku-4-5").execute(TEXT_REQUEST_SMALL_TOKENS).result()
    kwargs = json.loads(response.raw_responses[0]._echo)
    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["temperature"] == 1  # forced to 1 when thinking is enabled
    assert kwargs["max_tokens"] >= 11024  # min for high effort budget thinking


def test_budget_thinking_low_effort(openrouter_echo_requests, reset_effort):
    openrouter.set_effort("low")
    response = Api("openrouter/anthropic/claude-haiku-4-5").execute(TEXT_REQUEST_1).result()
    kwargs = json.loads(response.raw_responses[0]._echo)
    assert kwargs["reasoning"] == {"effort": "low"}
    assert kwargs["temperature"] == 1
    assert kwargs["max_tokens"] >= 2048


def test_budget_thinking_xhigh_caps_to_high(openrouter_echo_requests, reset_effort):
    openrouter.set_effort("xhigh")
    response = Api("openrouter/anthropic/claude-haiku-4-5").execute(TEXT_REQUEST_SMALL_TOKENS).result()
    kwargs = json.loads(response.raw_responses[0]._echo)
    # xhigh has no higher budget tier for Haiku 4.5, so it caps to "high"
    assert kwargs["reasoning"] == {"effort": "high"}


def test_adaptive_thinking_high_effort(openrouter_echo_requests, reset_effort):
    openrouter.set_effort("high")
    response = Api("openrouter/anthropic/claude-sonnet-4-6").execute(TEXT_REQUEST_SMALL_TOKENS).result()
    kwargs = json.loads(response.raw_responses[0]._echo)
    assert kwargs["reasoning"] == {"effort": "high"}
    assert "temperature" not in kwargs  # omitted for adaptive thinking
    assert kwargs["max_tokens"] >= 4096  # OPENROUTER_ADAPTIVE_MIN_TOKENS["high"]


def test_adaptive_thinking_max_effort(openrouter_echo_requests, reset_effort):
    openrouter.set_effort("max")
    response = Api("openrouter/anthropic/claude-sonnet-4-6").execute(TEXT_REQUEST_SMALL_TOKENS).result()
    kwargs = json.loads(response.raw_responses[0]._echo)
    assert kwargs["reasoning"] == {"effort": "max"}
    assert kwargs["max_tokens"] >= 16384  # OPENROUTER_ADAPTIVE_MIN_TOKENS["max"]


def test_adaptive_thinking_xhigh_on_opus(openrouter_echo_requests, reset_effort):
    openrouter.set_effort("xhigh")
    response = Api("openrouter/anthropic/claude-opus-4-7").execute(TEXT_REQUEST_SMALL_TOKENS).result()
    kwargs = json.loads(response.raw_responses[0]._echo)
    assert kwargs["reasoning"] == {"effort": "xhigh"}
    assert kwargs["max_tokens"] >= 8192


def test_adaptive_thinking_xhigh_rejected_on_sonnet(openrouter_echo_requests, reset_effort):
    openrouter.set_effort("xhigh")
    with pytest.raises(ValueError, match="xhigh.*Opus 4.7"):
        Api("openrouter/anthropic/claude-sonnet-4-6").execute(TEXT_REQUEST_1).result()


def test_adaptive_thinking_medium_rejected(openrouter_echo_requests, reset_effort):
    openrouter.set_effort("medium")
    with pytest.raises(ValueError, match="medium"):
        Api("openrouter/anthropic/claude-sonnet-4-6").execute(TEXT_REQUEST_1).result()


def test_no_reasoning_when_effort_none(openrouter_echo_requests, reset_effort):
    openrouter.set_effort("none")
    response = Api("openrouter/anthropic/claude-haiku-4-5").execute(TEXT_REQUEST_1).result()
    kwargs = json.loads(response.raw_responses[0]._echo)
    assert "reasoning" not in kwargs
    assert kwargs["temperature"] == 1  # from the request
    assert kwargs["max_tokens"] == 66


def test_non_reasoning_model_unaffected(openrouter_echo_requests, reset_effort):
    openrouter.set_effort("high")
    response = Api("openrouter/anthropic/claude-3.5-sonnet").execute(TEXT_REQUEST_1).result()
    kwargs = json.loads(response.raw_responses[0]._echo)
    # claude-3.5-sonnet is not in any thinking model set, so reasoning should not be set
    assert "reasoning" not in kwargs
    assert kwargs["temperature"] == 1
    assert kwargs["max_tokens"] == 66
