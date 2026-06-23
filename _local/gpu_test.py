import torch

print("PyTorch 버전:", torch.__version__)
print("GPU 사용 가능 여부:", torch.cuda.is_available())
print("내 그래픽카드 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU 없음")