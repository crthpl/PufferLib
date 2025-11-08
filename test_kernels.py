import torch

import pufferlib
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')


B = 2048
T = 64
H = 128

def assert_close(a, b, rtol=1e-5, atol=1e-5):
    max_diff = (a - b).abs().max()
    assert torch.allclose(a, b, rtol=rtol, atol=atol), 'Max diff: {}'.format(max_diff)

def mingru_gate(state, gate, hidden):
    hidden = torch.where(hidden >= 0, hidden + 0.5, hidden.sigmoid())
    gate = gate.sigmoid()
    out = torch.lerp(state, hidden, gate)
    return out

def test_mingru_gate():
    state = torch.randn(B, T, H).cuda()
    gate = torch.randn(B, T, H).cuda()
    hidden = torch.randn(B, T, H).cuda()

    python = mingru_gate(state, gate, hidden)
    cpp = _C.mingru_gate(state, gate, hidden)

    assert_close(python, cpp)

def log_coeffs_and_values(gate, hidden):
    log_coeffs = -torch.nn.functional.softplus(gate)
    log_z = -torch.nn.functional.softplus(-gate)
    log_tilde_h = torch.where(hidden >= 0,
        (torch.nn.functional.relu(hidden) + 0.5).log(),
        -torch.nn.functional.softplus(-hidden))
    log_values = log_z + log_tilde_h
    return log_coeffs, log_values

def test_log_coeffs_and_values():
    gate = torch.randn(B, T, H).cuda()
    hidden = torch.randn(B, T, H).cuda()

    python = log_coeffs_and_values(gate, hidden)
    cpp = _C.log_coeffs_and_values(gate, hidden)

    assert_close(python[0], cpp[0])
    assert_close(python[1], cpp[1])

def fused_scan(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1)
    log_h = a_star + log_h0_plus_b_star
    out = log_h.exp()
    return out

def test_fused_scan():
    log_coeffs = torch.randn(B, T+1, H).cuda()
    log_values = torch.randn(B, T+1, H).cuda()

    python = fused_scan(log_coeffs, log_values)
    cpp = _C.fused_scan(log_coeffs, log_values)[0]

    assert_close(python, cpp)

if __name__ == '__main__':
    test_mingru_gate()
    test_log_coeffs_and_values()
    test_fused_scan()
