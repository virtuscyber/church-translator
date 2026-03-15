"""Tests for the text refinement layer."""

from __future__ import annotations

import pytest

from src.refiner import Refiner


@pytest.mark.asyncio
async def test_refiner_disabled_passes_through():
    """When disabled, text passes through unchanged without any API call."""
    refiner = Refiner(api_key="fake-key", enabled=False)
    result = await refiner.refine("Um, so like, we need to pray, you know.")
    assert result == "Um, so like, we need to pray, you know."


@pytest.mark.asyncio
async def test_refiner_empty_input():
    """Empty text returns empty without API call."""
    refiner = Refiner(api_key="fake-key", enabled=True)
    result = await refiner.refine("")
    assert result == ""


@pytest.mark.asyncio
async def test_refiner_none_input():
    """None input returns None without API call (falsy bypass)."""
    refiner = Refiner(api_key="fake-key", enabled=True)
    result = await refiner.refine(None)
    assert result is None


@pytest.mark.asyncio
async def test_refiner_api_failure_returns_original(monkeypatch):
    """On API failure, returns original text (graceful degradation)."""
    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs):
                    raise RuntimeError("API error")

    refiner = Refiner(api_key="fake-key", enabled=True)
    refiner.client = FakeClient()

    result = await refiner.refine("We need to— actually, let me say it differently.")
    assert result == "We need to— actually, let me say it differently."


@pytest.mark.asyncio
async def test_refiner_context_window():
    """Context window stores refined outputs for continuity (maxlen enforcement)."""
    refiner = Refiner(api_key="fake-key", enabled=True, context_window=3)

    # Manually populate context to test deque maxlen (avoids API calls)
    refiner._context.append("First sentence.")
    refiner._context.append("Second sentence.")
    refiner._context.append("Third sentence.")

    assert len(refiner._context) == 3
    assert list(refiner._context) == ["First sentence.", "Second sentence.", "Third sentence."]

    # Adding a 4th should drop the first (deque maxlen=3)
    refiner._context.append("Fourth sentence.")
    assert len(refiner._context) == 3
    assert list(refiner._context) == ["Second sentence.", "Third sentence.", "Fourth sentence."]


@pytest.mark.asyncio
async def test_refiner_disabled_skips_context():
    """When disabled, context is not tracked (no unnecessary state)."""
    refiner = Refiner(api_key="fake-key", enabled=False)
    await refiner.refine("Some text.")
    assert len(refiner._context) == 0


@pytest.mark.asyncio
async def test_refiner_calls_api_with_context(monkeypatch):
    """When enabled, sends context and text to the API."""
    captured_messages = []

    class FakeResponse:
        class choices:
            pass

    class FakeChoice:
        class message:
            content = "Cleaned text."

    FakeResponse.choices = [FakeChoice()]

    class FakeCompletions:
        @staticmethod
        async def create(**kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return FakeResponse

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    refiner = Refiner(api_key="fake-key", enabled=True, context_window=2)
    refiner.client = FakeClient()

    # First call — no context yet
    result = await refiner.refine("Um, we should pray.")
    assert result == "Cleaned text."
    assert len(captured_messages) == 1
    # User message should contain the text
    user_msg = captured_messages[0][-1]["content"]
    assert "Um, we should pray." in user_msg
    # No context block on first call
    assert "Recent context" not in user_msg

    # Second call — should include context from first
    await refiner.refine("So, like, amen.")
    assert len(captured_messages) == 2
    user_msg2 = captured_messages[1][-1]["content"]
    assert "Cleaned text." in user_msg2  # Previous refined output as context
    assert "So, like, amen." in user_msg2
