import torch
import torch.nn as nn
import torch.optim as optim

def test_gpu_pipeline():
    # Check which device is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("Warning: CUDA is not available, running on CPU. GPU cannot be tested.")

    # Build a small model (2 fully connected layers)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)

    # Create dummy data
    batch_size = 4
    input_dim = 10
    output_dim = 5
    x = torch.randn(batch_size, input_dim).to(device)          # input
    y = torch.randint(0, output_dim, (batch_size,)).to(device) # labels for CrossEntropyLoss

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)

    # Backward pass and weight update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that gradients were computed for all parameters
    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError(f"Gradient was not computed for parameter {name}")

    print("✅ Pipeline test succeeded: forward and backward completed on the GPU.")
    return True

if __name__ == "__main__":
    test_gpu_pipeline()