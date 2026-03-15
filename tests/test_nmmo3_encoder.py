"""Equivalence test: CUDA NMMO3 encoder vs PyTorch reference implementation."""

import subprocess
import ctypes
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Build the shared library
SRC = os.path.join(os.path.dirname(__file__), "test_nmmo3_cuda.cu")
SO = os.path.join(os.path.dirname(__file__), "ocean_test.so")


def build():
    cmd = [
        "nvcc",
        "-shared",
        "-o",
        SO,
        SRC,
        "-I",
        os.path.join(os.path.dirname(__file__), "..", "pufferlib", "src"),
        "-lcublas",
        "-lcudnn",
        "-lcurand",
        "--compiler-options",
        "-fPIC",
        "-Xcompiler",
        "-O2",
    ]
    print(f"Building: {' '.join(cmd)}")
    subprocess.check_call(cmd)


# ============================================================================
# PyTorch reference
# ============================================================================


class NMMO3EncoderRef(nn.Module):
    def __init__(self):
        super().__init__()
        self.factors = np.array([4, 4, 17, 5, 3, 5, 5, 5, 7, 4])
        offsets = torch.tensor([0] + list(np.cumsum(self.factors)[:-1])).view(
            1, -1, 1, 1
        )
        self.register_buffer("offsets", offsets)
        self.multihot_dim = int(self.factors.sum())  # 59

        self.conv1 = nn.Conv2d(self.multihot_dim, 128, 5, stride=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 3, stride=1)
        self.embed = nn.Embedding(128, 32)
        self.proj = nn.Linear(1817, 512)
        self.relu_proj = nn.ReLU()

    def forward(self, observations):
        batch = observations.shape[0]
        ob_map = observations[:, :1650].view(batch, 11, 15, 10)
        map_buf = torch.zeros(
            batch, 59, 11, 15, dtype=torch.float32, device=observations.device
        )
        codes = ob_map.long().permute(0, 3, 1, 2) + self.offsets
        map_buf.scatter_(1, codes, 1)
        ob_map = self.conv1(map_buf)
        ob_map = self.relu1(ob_map)
        ob_map = self.conv2(ob_map)
        ob_map = ob_map.flatten(1)
        ob_player = observations[:, 1650:-10]
        player_discrete = self.embed(ob_player.int()).flatten(1)
        ob_reward = observations[:, -10:]
        obs_cat = torch.cat(
            [ob_map, player_discrete, ob_player.float(), ob_reward.float()], dim=1
        )
        return self.relu_proj(self.proj(obs_cat))


# ============================================================================
# ctypes wrapper
# ============================================================================

ENCODER_ARGTYPES = [
    ctypes.c_void_p,
    ctypes.c_void_p,  # output, obs
    ctypes.c_void_p,
    ctypes.c_void_p,  # conv1_w, conv1_b
    ctypes.c_void_p,
    ctypes.c_void_p,  # conv2_w, conv2_b
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,  # embed_w, proj_w, proj_b
    ctypes.c_void_p,
    ctypes.c_int,  # workspace, B
    ctypes.c_void_p,
    ctypes.c_void_p,  # cublas handle, stream
]


def load_lib():
    lib = ctypes.CDLL(SO)
    lib.nmmo3_workspace_size.restype = ctypes.c_int
    lib.nmmo3_workspace_size.argtypes = [ctypes.c_int]
    lib.nmmo3_encoder_setup.restype = None
    lib.nmmo3_encoder_setup.argtypes = [ctypes.c_int]
    lib.nmmo3_encoder_forward.restype = None
    lib.nmmo3_encoder_forward.argtypes = ENCODER_ARGTYPES

    # Conv test helpers
    lib.conv2d_test_forward.restype = None
    lib.conv2d_test_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,  # out, in, w, b
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,  # B, conv_idx, stream
    ]
    lib.conv2d_test_backward.restype = None
    lib.conv2d_test_backward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,  # wgrad, bgrad, input_grad
        ctypes.c_void_p,
        ctypes.c_void_p,  # grad_output, saved_output
        ctypes.c_void_p,
        ctypes.c_void_p,  # input, weight
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,  # B, conv_idx, stream
    ]
    return lib


def extract_weights(model):
    return [
        t.data.contiguous()
        for t in [
            model.conv1.weight,
            model.conv1.bias,
            model.conv2.weight,
            model.conv2.bias,
            model.embed.weight,
            model.proj.weight,
            model.proj.bias,
        ]
    ]


def run_encoder(lib, model, obs_uint8, B):
    device = obs_uint8.device
    output = torch.zeros(B, 512, dtype=torch.float32, device=device)
    ws_size = lib.nmmo3_workspace_size(B)
    workspace = torch.zeros(ws_size, dtype=torch.float32, device=device)
    weights = extract_weights(model)
    lib.nmmo3_encoder_forward(
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(obs_uint8.data_ptr()),
        *[ctypes.c_void_p(w.data_ptr()) for w in weights],
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    return output


def generate_valid_obs(B, device):
    factors = [4, 4, 17, 5, 3, 5, 5, 5, 7, 4]
    obs = torch.zeros(B, 1707, dtype=torch.uint8, device=device)
    for h in range(11):
        for w in range(15):
            for f in range(10):
                idx = (h * 15 + w) * 10 + f
                obs[:, idx] = torch.randint(0, factors[f], (B,), dtype=torch.uint8)
    obs[:, 1650:1697] = torch.randint(0, 128, (B, 47), dtype=torch.uint8)
    obs[:, 1697:1707] = torch.randint(0, 256, (B, 10), dtype=torch.uint8)
    return obs


def check_match(name, out, ref_out, B, atol=1e-4, rtol=1e-4):
    max_diff = (out - ref_out).abs().max().item()
    mean_diff = (out - ref_out).abs().mean().item()
    ref_norm = ref_out.abs().mean().item()
    print(
        f"  [{name}] max={max_diff:.6e} mean={mean_diff:.6e} rel={mean_diff / (ref_norm + 1e-8):.6e}"
    )
    ok = torch.allclose(out, ref_out, atol=atol, rtol=rtol)
    if not ok:
        diff = (out - ref_out).abs()
        flat_idx = diff.argmax().item()
        idx = np.unravel_index(flat_idx, out.shape)
        print(
            f"    Worst at {idx}: got={out[idx].item():.6f}, ref={ref_out[idx].item():.6f}"
        )
    assert ok, f"{name} mismatch for B={B}"


# ============================================================================
# Tests
# ============================================================================


def test_encoder(B):
    print(f"\n--- Testing encoder B={B} ---")
    device = torch.device("cuda")
    torch.manual_seed(42)
    model = NMMO3EncoderRef().to(device).float()
    model.eval()
    obs = generate_valid_obs(B, device)
    lib = load_lib()
    lib.nmmo3_encoder_setup(ctypes.c_int(B))

    with torch.no_grad():
        ref_out = model(obs.float())
    cuda_out = run_encoder(lib, model, obs, B)
    check_match("cudnn", cuda_out, ref_out, B)
    print(f"  PASSED")


def test_conv_backward(B):
    """Test conv1 backward (NCHW, relu) against PyTorch autograd."""
    print(f"\n--- Testing conv backward B={B} ---")
    device = torch.device("cuda")
    torch.manual_seed(42)
    lib = load_lib()
    lib.nmmo3_encoder_setup(ctypes.c_int(B))

    conv = nn.Conv2d(59, 128, 5, stride=3, bias=True).cuda().float()
    w = conv.weight.data.contiguous()
    b = conv.bias.data.contiguous()
    x = torch.randn(B, 59, 11, 15, device=device)

    # CUDA forward (ground truth — cuDNN fused conv+bias+relu)
    cuda_out = torch.zeros(B, 128, 3, 4, device=device)
    lib.conv2d_test_forward(
        ctypes.c_void_p(cuda_out.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_int(0),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()

    # PyTorch backward using the CUDA forward output as the relu mask
    # This avoids mismatch from cuDNN fused vs separate conv→relu
    x_ag = x.detach().requires_grad_(True)
    conv_out_raw = conv(x_ag)  # no relu
    # Apply the same relu mask as cuDNN produced
    relu_mask = (cuda_out > 0).float()
    out_masked = conv_out_raw * relu_mask
    grad_out = torch.randn(B, 128, 3, 4, device=device)
    out_masked.backward(grad_out)
    ref_wgrad = conv.weight.grad.clone()
    ref_bgrad = conv.bias.grad.clone()
    ref_xgrad = x_ag.grad.clone()

    # CUDA backward
    wgrad = torch.zeros_like(w)
    bgrad = torch.zeros(128, device=device)
    xgrad = torch.zeros_like(x)
    grad_out_c = grad_out.clone()

    lib.conv2d_test_backward(
        ctypes.c_void_p(wgrad.data_ptr()),
        ctypes.c_void_p(bgrad.data_ptr()),
        ctypes.c_void_p(xgrad.data_ptr()),
        ctypes.c_void_p(grad_out_c.data_ptr()),
        ctypes.c_void_p(cuda_out.data_ptr()),
        ctypes.c_void_p(x.data_ptr()),
        ctypes.c_void_p(w.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_int(0),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()

    check_match("wgrad", wgrad, ref_wgrad, B, atol=1e-3, rtol=1e-3)
    check_match(
        "bgrad", bgrad.unsqueeze(0), ref_bgrad.unsqueeze(0), B, atol=1e-3, rtol=1e-3
    )
    check_match("xgrad", xgrad, ref_xgrad, B, atol=1e-3, rtol=1e-3)
    print(f"  PASSED")


def test_encoder_backward(B):
    """Test full encoder backward against PyTorch autograd.

    Uses CUDA forward outputs as relu masks for the PyTorch reference so both
    paths agree on which elements are zeroed — cuDNN fused conv+bias+relu can
    differ from separate conv→relu near zero.
    """
    print(f"\n--- Testing encoder backward B={B} ---")
    device = torch.device("cuda")
    torch.manual_seed(42)
    lib = load_lib()
    lib.nmmo3_encoder_setup(ctypes.c_int(B))

    model = NMMO3EncoderRef().to(device).float()
    obs_uint8 = generate_valid_obs(B, device)

    weights = extract_weights(model)
    conv1_w, conv1_b, conv2_w, conv2_b, embed_w, proj_w, proj_b = weights

    # --- CUDA forward to get ground-truth intermediate values ---
    ws_size = lib.nmmo3_workspace_size(B)
    workspace = torch.zeros(ws_size, dtype=torch.float32, device=device)
    output = torch.zeros(B, 512, dtype=torch.float32, device=device)
    lib.nmmo3_encoder_forward(
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(obs_uint8.data_ptr()),
        *[ctypes.c_void_p(w.data_ptr()) for w in weights],
        ctypes.c_void_p(workspace.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_void_p(0),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()

    # Extract workspace regions (NCHW layout for conv outputs)
    off = 0
    multihot = workspace[off : off + B * 59 * 11 * 15].view(B, 59, 11, 15).clone()
    off += B * 59 * 11 * 15
    conv1_out = workspace[off : off + B * 128 * 3 * 4].view(B, 128, 3, 4).clone()
    off += B * 128 * 3 * 4
    conv2_out = workspace[off : off + B * 128 * 1 * 2].view(B, 128, 1, 2).clone()
    off += B * 128 * 1 * 2
    off += B * 47 * 32  # skip embed
    concat = workspace[off : off + B * 1817].view(B, 1817).clone()

    # --- PyTorch reference backward using CUDA relu masks ---
    obs_f = obs_uint8.float()

    # Enable grads on conv/proj weights
    for w in weights:
        w.requires_grad_(True)

    # Rebuild forward with autograd but apply CUDA's relu masks
    mh = multihot.detach()  # multihot is discrete, no grad needed

    # Conv1: raw conv (no relu), then apply CUDA's relu mask
    conv1_raw = F.conv2d(mh, conv1_w, conv1_b, stride=3)
    conv1_relu_mask = (conv1_out > 0).float()
    conv1_masked = conv1_raw * conv1_relu_mask

    # Conv2: no relu in original model
    conv2_masked = F.conv2d(conv1_masked, conv2_w, conv2_b, stride=1)

    # Player branch (no grads flow here for conv test)
    ob_player = obs_f[:, 1650:-10]
    player_discrete = model.embed(ob_player.int()).flatten(1).detach()
    ob_reward = obs_f[:, -10:].detach()

    # Concat + proj with CUDA's relu mask
    obs_cat = torch.cat(
        [conv2_masked.flatten(1), player_discrete, ob_player.detach(), ob_reward], dim=1
    )
    proj_raw = F.linear(obs_cat, proj_w, proj_b)
    proj_relu_mask = (output > 0).float()
    proj_masked = proj_raw * proj_relu_mask

    grad_out = torch.randn_like(proj_masked)
    proj_masked.backward(grad_out)

    ref_conv1_wgrad = conv1_w.grad.clone()
    ref_conv1_bgrad = conv1_b.grad.clone()
    ref_conv2_wgrad = conv2_w.grad.clone()
    ref_conv2_bgrad = conv2_b.grad.clone()
    ref_proj_wgrad = proj_w.grad.clone()
    ref_proj_bgrad = proj_b.grad.clone()
    # Zero grads for reuse
    for w in weights:
        if w.grad is not None:
            w.grad.zero_()

    # --- CUDA backward (manual, matching integrated encoder backward) ---

    # Proj backward
    grad = grad_out.clone()
    grad[output <= 0] = 0
    cuda_proj_bgrad = grad.sum(0)
    cuda_proj_wgrad = grad.t() @ concat
    check_match("proj_bgrad", cuda_proj_bgrad, ref_proj_bgrad, B, atol=1e-3, rtol=1e-3)
    check_match("proj_wgrad", cuda_proj_wgrad, ref_proj_wgrad, B, atol=1e-2, rtol=1e-2)

    # grad_concat = grad @ proj_w
    grad_concat = grad @ proj_w
    conv2_grad = grad_concat[:, :256].view(B, 128, 1, 2).contiguous()

    # Conv2 backward
    conv2_wgrad = torch.zeros_like(conv2_w)
    conv2_bgrad = torch.zeros(128, device=device)
    conv1_input_grad = torch.zeros(B, 128, 3, 4, device=device)
    lib.conv2d_test_backward(
        ctypes.c_void_p(conv2_wgrad.data_ptr()),
        ctypes.c_void_p(conv2_bgrad.data_ptr()),
        ctypes.c_void_p(conv1_input_grad.data_ptr()),
        ctypes.c_void_p(conv2_grad.data_ptr()),
        ctypes.c_void_p(conv2_out.data_ptr()),
        ctypes.c_void_p(conv1_out.data_ptr()),
        ctypes.c_void_p(conv2_w.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_int(1),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    check_match("conv2_wgrad", conv2_wgrad, ref_conv2_wgrad, B, atol=1e-2, rtol=1e-2)
    check_match(
        "conv2_bgrad",
        conv2_bgrad.unsqueeze(0),
        ref_conv2_bgrad.unsqueeze(0),
        B,
        atol=1e-2,
        rtol=1e-2,
    )

    # Conv1 backward
    conv1_wgrad = torch.zeros_like(conv1_w)
    conv1_bgrad = torch.zeros(128, device=device)
    lib.conv2d_test_backward(
        ctypes.c_void_p(conv1_wgrad.data_ptr()),
        ctypes.c_void_p(conv1_bgrad.data_ptr()),
        ctypes.c_void_p(0),  # no input grad needed
        ctypes.c_void_p(conv1_input_grad.data_ptr()),
        ctypes.c_void_p(conv1_out.data_ptr()),
        ctypes.c_void_p(multihot.data_ptr()),
        ctypes.c_void_p(conv1_w.data_ptr()),
        ctypes.c_int(B),
        ctypes.c_int(0),
        ctypes.c_void_p(0),
    )
    torch.cuda.synchronize()
    check_match("conv1_wgrad", conv1_wgrad, ref_conv1_wgrad, B, atol=1e-2, rtol=1e-2)
    check_match(
        "conv1_bgrad",
        conv1_bgrad.unsqueeze(0),
        ref_conv1_bgrad.unsqueeze(0),
        B,
        atol=1e-2,
        rtol=1e-2,
    )

    print(f"  PASSED")


# ============================================================================
# Benchmarks
# ============================================================================


def bench_encoder(B, warmup=10, iters=100):
    print(f"\n--- Benchmark B={B} ({iters} iters) ---")
    device = torch.device("cuda")
    torch.manual_seed(42)
    model = NMMO3EncoderRef().to(device).float()
    model.eval()
    obs = generate_valid_obs(B, device)
    lib = load_lib()
    lib.nmmo3_encoder_setup(ctypes.c_int(B))

    weights = extract_weights(model)
    output = torch.zeros(B, 512, dtype=torch.float32, device=device)
    ws_size = lib.nmmo3_workspace_size(B)
    workspace = torch.zeros(ws_size, dtype=torch.float32, device=device)

    def run_cuda():
        lib.nmmo3_encoder_forward(
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(obs.data_ptr()),
            *[ctypes.c_void_p(w.data_ptr()) for w in weights],
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_int(B),
            ctypes.c_void_p(0),
            ctypes.c_void_p(0),
        )

    def run_torch():
        with torch.no_grad():
            model(obs.float())

    runners = [("cudnn", run_cuda), ("torch", run_torch)]

    for _, fn in runners:
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = {}
    for name, fn in runners:
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times[name] = start.elapsed_time(end) / iters

    for name, ms in times.items():
        print(f"  {name:8s} {ms:.3f} ms")
    print(f"  cudnn vs torch: {times['torch'] / times['cudnn']:.2f}x")


if __name__ == "__main__":
    build()
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("test", "all"):
        for B in [1, 8, 64]:
            test_encoder(B)
            test_conv_backward(B)
            test_encoder_backward(B)
        print("\nAll tests passed!")

    if mode in ("bench", "all"):
        for B in [1, 8, 64, 256]:
            bench_encoder(B)
