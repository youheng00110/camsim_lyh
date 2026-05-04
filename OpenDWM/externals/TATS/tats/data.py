# run_fvd_fid_eval_stream.py
import os, sys, glob
import torch
import torch.nn.functional as F
from torchvision.io import read_video
from tqdm.auto import tqdm

# ==== 1) 让 Python 找到 TATS 的 I3D ====
sys.path.append("/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/ggearth/OpenDWM/externals/TATS")
from tats.fvd.pytorch_i3d import InceptionI3d

# ==== 2) 数据根目录 ====
ROOT   = "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/MDDIT/MagicDrive-V2/outputs/eval/CogVAE-848-17f/MagicDriveSTDiT3-XL-2_original_test_first_test_20251024-1738/generation"
GT_DIR  = os.path.join(ROOT, "gt_video")
GEN_DIR = os.path.join(ROOT, "gen_video")

# ==== 3) I3D 权重（Kinetics-400）====
I3D_WEIGHTS = "/inspire/hdd/project/wuliqifa/chenxinyan-240108120066/songbur/ggearth/OpenDWM/externals/TATS/tats/fvd/i3d_pretrained_400.pt"

# ==== 4) 参数 ====
SEQ_LEN   = 16
I3D_RES   = (224, 224)
FID_RES   = (299, 299)
BATCH_I3D = 8
BATCH_FID = 64
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 工具函数 =================
def list_videos(d):
    return sorted(glob.glob(os.path.join(d, "**", "*.mp4"), recursive=True))

def sample_frames_uint8(path, seq_len=16):
    # 返回 [T,H,W,C] uint8 的均匀采样帧
    v, _, _ = read_video(path, pts_unit="sec")
    if v.numel() == 0:
        raise RuntimeError(f"空视频或解码失败: {path}")
    T = v.shape[0]
    idx = torch.linspace(0, T-1, steps=seq_len).long()
    return v[idx]  # [seq,H,W,C] uint8

def prep_i3d_input(frames_uint8, target_res):
    # [T,H,W,C] uint8 -> [1,C,T,H,W] float in [-1,1]
    x = frames_uint8.float().permute(0,3,1,2).contiguous()     # [T,3,H,W]
    x = F.interpolate(x, size=target_res, mode="bilinear", align_corners=False)
    x = x.unsqueeze(0).transpose(1,2).contiguous()             # [1,C,T,H,W]
    x = x * 2 / 255. - 1
    return x

def frames_to_imgs(frames_uint8, target_res):
    # [T,H,W,C] uint8 -> [T,3,H,W] float[0,1] (用于 FID)
    x = frames_uint8.float().permute(0,3,1,2).contiguous()
    x = F.interpolate(x, size=target_res, mode="bilinear", align_corners=False)
    x = x / 255.0
    return x

# ============== I3D 特征（流式，从路径批量提取到 1024D） ==============
@torch.no_grad()
def i3d_features_from_paths(paths):
    if not os.path.exists(I3D_WEIGHTS):
        raise FileNotFoundError(f"I3D 权重不存在: {I3D_WEIGHTS}")
    i3d = InceptionI3d(400, in_channels=3).to(DEVICE).eval()
    i3d.load_state_dict(torch.load(I3D_WEIGHTS, map_location="cpu"))

    feats = []
    for i in tqdm(range(0, len(paths), BATCH_I3D), desc="I3D feats", leave=False):
        batch_paths = paths[i:i+BATCH_I3D]
        batch = []
        for p in batch_paths:
            fr = sample_frames_uint8(p, SEQ_LEN)                  # [T,H,W,C]
            x  = prep_i3d_input(fr, I3D_RES).to(DEVICE)           # [1,C,T,H,W]
            batch.append(x)
        Z = torch.cat(batch, 0)                                   # [B,C,T,H,W]
        f = i3d.extract_features(Z)                               # [B,1024,T',H',W']
        f = F.adaptive_avg_pool3d(f, (1,1,1)).flatten(1).cpu()    # [B,1024] on CPU
        feats.append(f)
        del Z, f
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
    return torch.cat(feats, 0)                                     # [N,1024]

# ============== FVD 计算（只用 1024D 特征做 Frechet） ==============
def frechet_from_features(Fx, Fy):
    def _cov(m):
        m = m - m.mean(0, keepdim=True)
        return (m.T @ m) / (m.shape[0]-1 + 1e-6)

    mu1, mu2 = Fx.mean(0), Fy.mean(0)
    s1, s2   = _cov(Fx), _cov(Fy)
    a = torch.sum((mu1-mu2)**2)
    b = torch.trace(s1) + torch.trace(s2)
    P = s1 @ s2
    P = 0.5 * (P + P.T)                     # 对称化更稳
    eig = torch.linalg.eigvalsh(P).clamp_min(0)
    c = eig.sqrt().sum()
    return float((a + b - 2*c).item())

# ============== FID（流式，逐小批帧 update） ==============
@torch.no_grad()
def fid_from_paths(paths_real, paths_fake):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_dev = DEVICE if (DEVICE == "cuda") else "cpu"
    fid = FrechetInceptionDistance(feature=2048).to(fid_dev)

    # real
    buf = []
    for p in tqdm(paths_real, desc="FID[real] sampling", leave=False):
        fr = sample_frames_uint8(p, SEQ_LEN)
        im = frames_to_imgs(fr, FID_RES)                  # [T,3,H,W]
        buf.append(im)
        if sum(x.shape[0] for x in buf) >= BATCH_FID:     # 累够若干帧再 update
            X = torch.cat(buf, 0).to(fid_dev)
            fid.update(X, real=True)
            buf.clear()
    if buf:
        X = torch.cat(buf, 0).to(fid_dev); fid.update(X, real=True); buf.clear()

    # fake
    for p in tqdm(paths_fake, desc="FID[fake] sampling", leave=False):
        fr = sample_frames_uint8(p, SEQ_LEN)
        im = frames_to_imgs(fr, FID_RES)
        buf.append(im)
        if sum(x.shape[0] for x in buf) >= BATCH_FID:
            Y = torch.cat(buf, 0).to(fid_dev)
            fid.update(Y, real=False)
            buf.clear()
    if buf:
        Y = torch.cat(buf, 0).to(fid_dev); fid.update(Y, real=False); buf.clear()

    v = fid.compute()
    return float(v.cpu())

# ================= 主流程 =================
def main():
    gt_paths  = list_videos(GT_DIR)
    gen_paths = list_videos(GEN_DIR)
    print(f"[info] gt={len(gt_paths)}  gen={len(gen_paths)}", flush=True)
    if len(gt_paths) == 0 or len(gen_paths) == 0:
        raise RuntimeError("未找到 mp4，请检查 GT_DIR/GEN_DIR 路径。")

    # === FVD ===
    print("[stage] computing I3D/FVD (streaming) ...", flush=True)
    Fx = i3d_features_from_paths(gt_paths)     # [N,1024] on CPU
    Fy = i3d_features_from_paths(gen_paths)    # [N,1024] on CPU
    fvd = frechet_from_features(Fx, Fy)
    print(f"FVD = {fvd:.4f}", flush=True)

    # === FID ===
    print("[stage] computing FID (streaming) ...", flush=True)
    fid = fid_from_paths(gt_paths, gen_paths)
    print(f"FID(frames) = {fid:.4f}", flush=True)

if __name__ == "__main__":
    # 如果显存紧：把 BATCH_I3D=4, BATCH_FID=32；仍不行就 DEVICE='cpu' 验证流程
    main()
