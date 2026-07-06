"""
image_analyzer.py — Legacy image description utility.

This module is preserved for backwards compatibility.
New code should use app.utils.chunking_strategies.chunk_image() directly,
which provides richer descriptions and image type classification.
"""

from langchain_core.messages import HumanMessage
from app.services.llm_config import fast_llm, FAST_LLM_LIMITER


async def describe_image(base64_data: str, alt_text: str = "") -> str:
    """
    Generate a detailed description of an image using the fast LLM.

    Args:
        base64_data: Base64-encoded image data URI.
        alt_text:    Original alt attribute from the <img> tag.

    Returns:
        A string description of the image content.
    """
    if not base64_data:
        return alt_text or "No description available."

    await FAST_LLM_LIMITER.acquire(500)

    prompt = (
        "Describe this image in detail. It is from a webpage. "
        "If it is a chart, graph, or diagram, explain the data and insights it conveys. "
        "If there is text in the image, include it verbatim. "
        f"Original alt text: {alt_text}"
    )

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": base64_data}},
        ]
    )

    try:
        response = await fast_llm.ainvoke([message])
        return response.content
    except Exception as e:
        print(f"Error describing image: {e}")
        return alt_text or "Image could not be described."
