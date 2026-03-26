import torch
import sys

sys.stderr.write("DEBUG: Start CUDA test\n"); sys.stderr.flush()
sys.stderr.write(f"DEBUG: CUDA available: {torch.cuda.is_available()}\n"); sys.stderr.flush()
if torch.cuda.is_available():
    sys.stderr.write(f"DEBUG: Device count: {torch.cuda.device_count()}\n"); sys.stderr.flush()
    sys.stderr.write(f"DEBUG: Current device: {torch.cuda.current_device()}\n"); sys.stderr.flush()
    
    try:
        sys.stderr.write("DEBUG: Creating tensor on CPU\n"); sys.stderr.flush()
        x = torch.tensor([1.0, 2.0])
        sys.stderr.write("DEBUG: Moving tensor to CUDA\n"); sys.stderr.flush()
        x = x.to('cuda')
        sys.stderr.write("DEBUG: Tensor moved to CUDA successfully\n"); sys.stderr.flush()
        sys.stderr.write(f"DEBUG: x on device: {x.device}\n"); sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"DEBUG: CUDA error: {e}\n"); sys.stderr.flush()
else:
    sys.stderr.write("DEBUG: CUDA not available\n"); sys.stderr.flush()
