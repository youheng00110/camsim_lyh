import argparse
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--minutes", type=int, default=10)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--mem_gb", type=float, default=16.0)
parser.add_argument("--matmul_size", type=int, default=16384)
parser.add_argument("--inner_iters", type=int, default=20)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

torch.cuda.set_device(args.device)
device = torch.device(f"cuda:{args.device}")

print("device:", torch.cuda.get_device_name(device), flush=True)
print("target memory GB:", args.mem_gb, flush=True)
print("matmul_size:", args.matmul_size, flush=True)
print("inner_iters:", args.inner_iters, flush=True)

num_fp16_elements = int(args.mem_gb * 1024**3 / 2)
buffer = torch.empty(num_fp16_elements, dtype=torch.float16, device=device)
buffer.fill_(1.0)

a = torch.randn((args.matmul_size, args.matmul_size), dtype=torch.float16, device=device)
b = torch.randn((args.matmul_size, args.matmul_size), dtype=torch.float16, device=device)
c = torch.empty((args.matmul_size, args.matmul_size), dtype=torch.float16, device=device)

end_time = time.time() + args.minutes * 60
step = 0

while time.time() < end_time:
    for _ in range(args.inner_iters):
        torch.matmul(a, b, out=c)
        a, c = c, a

    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    print(
        f"step={step}, allocated={allocated:.2f}GB, reserved={reserved:.2f}GB",
        flush=True,
    )

    step += 1

print("finished", flush=True)
