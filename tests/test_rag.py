import pytest
from unittest.mock import AsyncMock, patch
from app.RAG.nodes import (
    parse_json_bool_array,
    faithfulness_checker,
    relevance_grader,
    _MAX_RETRIES,
)
from app.RAG.graph import route_faithfulness


def test_parse_json_bool_array():
    assert parse_json_bool_array("[true, false, true]", 3) == [True, False, True]
    assert parse_json_bool_array("```json\n[false, true]\n```", 2) == [False, True]
    assert parse_json_bool_array("invalid json", 3) == [True, True, True]


def test_route_faithfulness():
    state_faithful = {"is_faithful": True, "retry_count": 0}
    assert route_faithfulness(state_faithful) == "end"

    state_unfaithful_1 = {"is_faithful": False, "retry_count": 1}
    assert route_faithfulness(state_unfaithful_1) == "rewrite"

    state_unfaithful_max = {"is_faithful": False, "retry_count": _MAX_RETRIES}
    assert route_faithfulness(state_unfaithful_max) == "end"


@pytest.mark.anyio
async def test_faithfulness_checker_increments_retry_count():
    state = {
        "context": "The sky is blue.",
        "answer": "The sky is green.",
        "retry_count": 0,
    }

    with patch("app.RAG.nodes.faithfulness_prompt") as mock_prompt, patch(
        "app.RAG.nodes.llm"
    ) as mock_llm:
        chain_mock = AsyncMock()
        chain_mock.ainvoke.return_value = "no"
        mock_prompt.__or__.return_value.__or__.return_value = chain_mock

        res = await faithfulness_checker(state)
        assert res["is_faithful"] is False
        assert res["faithfulness"] == "unfaithful"
        assert res["retry_count"] == 1


@pytest.mark.anyio
async def test_relevance_grader_batched():
    state = {
        "question": "What color is the sky?",
        "retrieved_chunks": [
            "The sky is blue.",
            "Water boils at 100 degrees Celsius.",
        ],
        "context": "The sky is blue.\n\nWater boils at 100 degrees Celsius.",
    }

    with patch("app.RAG.nodes.relevance_prompt") as mock_prompt, patch(
        "app.RAG.nodes.llm"
    ) as mock_llm:
        chain_mock = AsyncMock()
        chain_mock.ainvoke.return_value = "[true, false]"
        mock_prompt.__or__.return_value.__or__.return_value = chain_mock

        res = await relevance_grader(state)
        assert res["relevance_scores"] == [True, False]
        assert res["context"] == "The sky is blue."
