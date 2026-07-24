import asyncio
from app.utils.chunks import process_elements

async def main():
    url = "https://example.com"
    title = "Test"
    elements = [
        {"type": "text", "content": "hello world", "heading": "Intro"}
    ]
    res = await process_elements(url, title, elements)
    print("RES:", res)

asyncio.run(main())
