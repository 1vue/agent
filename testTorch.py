import sys
import torch

print("Python version:", sys.version)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Total memory GB:", torch.cuda.get_device_properties(0).total_memory / 1024 ** 3)