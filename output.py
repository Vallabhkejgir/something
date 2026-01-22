from Graph import app

async def Output(inputs):
    # Use 'async for' and 'astream' because your nodes are async
    async for output in app.astream(inputs):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            # print(value) 
    print("\n")
    