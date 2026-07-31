import asyncio

async def inner():
    await asyncio.sleep(1)
    return "done"

async def outer():
    t = asyncio.create_task(inner())
    return t

async def main():
    task = asyncio.create_task(outer())
    await asyncio.sleep(0.1)
    if task.done():
        print(task.result())
    task.cancel()
    
asyncio.run(main())
