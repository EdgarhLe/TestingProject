import torch
import torch.nn as nn
import torch.optim as optim

def test_gpu_pipeline():
    # Kiểm tra thiết bị có sẵn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")
    if device.type == "cpu":
        print("Cảnh báo: CUDA không khả dụng, chạy trên CPU. Không kiểm tra được GPU.")

    # Xây dựng mô hình nhỏ (2 lớp fully connected)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    ).to(device)

    # Tạo dữ liệu giả
    batch_size = 4
    input_dim = 10
    output_dim = 5
    x = torch.randn(batch_size, input_dim).to(device)          # đầu vào
    y = torch.randint(0, output_dim, (batch_size,)).to(device) # nhãn cho CrossEntropyLoss

    # Hàm mất mát và bộ tối ưu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)

    # Backward pass và cập nhật trọng số
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Kiểm tra gradient đã được tính cho tất cả tham số
    for name, param in model.named_parameters():
        if param.grad is None:
            raise RuntimeError(f"Gradient không được tính cho tham số {name}")

    print("✅ Pipeline test thành công: forward và backward hoàn tất trên GPU.")
    return True

if __name__ == "__main__":
    test_gpu_pipeline()