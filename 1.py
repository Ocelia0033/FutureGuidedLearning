import torch

print(f"PyTorch版本: {torch.__version__}")

# 检查MPS（Metal Performance Shaders）是否可用，这是Apple芯片的GPU加速后端
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS可用，将使用Apple Silicon GPU进行加速。")
else:
    device = torch.device("cpu")
    print("MPS不可用，将使用CPU。")

# 进一步检查CUDA是否可用（在你的Mac上通常不可用）
print(f"CUDA可用: {torch.cuda.is_available()}")