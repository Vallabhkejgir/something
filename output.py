from Graph import app

def Output():
    from app import inputs
    for output in app.stream(inputs):
    # The output of each node is printed as it executes
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            # print(value) # Uncomment to see the full state at each step
    print("\n")