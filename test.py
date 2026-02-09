import sys
print("Python:", sys.version)

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version (torch):", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("PyTorch error:", e)

try:
    import numpy as np
    print("NumPy:", np.__version__)
except Exception as e:
    print("NumPy error:", e)
