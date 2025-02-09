import torch
print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA版本:", torch.version.cuda)
    print("GPU数量:", torch.cuda.device_count())
    print("当前GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))