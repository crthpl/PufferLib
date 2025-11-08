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
    passed = torch.allclose(a, b, rtol=rtol, atol=atol)
    if not passed:
        raise AssertionError('Max diff: {}'.format(max_diff))
    else:
        print(f'PASSED: {max_diff}')

def test_kernel(py_func, cpp_func, *args, backward=True):
    py_tensors = [arg.clone().detach().double().cuda().requires_grad_(backward) for arg in args]
    cpp_tensors = [arg.clone().detach().cuda().requires_grad_(backward) for arg in args]

    py_out = py_func(*py_tensors)
    cpp_out = cpp_func(*cpp_tensors)

    if not isinstance(py_out, (tuple, list)):
        py_out = [py_out]
    if not isinstance(cpp_out, (tuple, list)):
        cpp_out = [cpp_out]

    for py_o, cpp_o in zip(py_out, cpp_out):
        assert_close(py_o.float(), cpp_o)

    if not backward:
        return

    py_loss = sum([o.mean() for o in py_out])/len(py_out)
    cpp_loss = sum([o.mean() for o in cpp_out])/len(cpp_out)

    py_loss.backward()
    cpp_loss.backward()

    for py_t, cpp_t in zip(py_tensors, cpp_tensors):
        assert_close(py_t.grad.float(), cpp_t.grad)

def mingru_gate(state, gate, hidden):
    hidden = torch.where(hidden >= 0, hidden + 0.5, hidden.sigmoid())
    gate = gate.sigmoid()
    out = torch.lerp(state, hidden, gate)
    return out

def test_mingru_gate():
    state = torch.randn(B, T, H)
    gate = torch.randn(B, T, H)
    hidden = torch.randn(B, T, H)
    test_kernel(mingru_gate, _C.mingru_gate, state, gate, hidden, backward=False)

def log_coeffs_and_values(gate, hidden):
    log_coeffs = -torch.nn.functional.softplus(gate)
    log_z = -torch.nn.functional.softplus(-gate)
    log_tilde_h = torch.where(hidden >= 0,
        (torch.nn.functional.relu(hidden) + 0.5).log(),
        -torch.nn.functional.softplus(-hidden))
    log_values = log_z + log_tilde_h
    return log_coeffs, log_values

def test_log_coeffs_and_values():
    gate = torch.randn(B, T, H, requires_grad=True).cuda()
    hidden = torch.randn(B, T, H, requires_grad=True).cuda()
    test_kernel(log_coeffs_and_values, _C.log_coeffs_and_values, gate, hidden)

def fused_scan(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(1)
    log_h = a_star + log_h0_plus_b_star
    out = log_h.exp()
    return out

def test_fused_scan():
    # Numerically unstable function. Must be called with the distribution
    # that is used in the full network.
    log_coeffs = -torch.nn.functional.softplus(torch.randn(B, T+1, H))
    log_values = -torch.nn.functional.softplus(torch.randn(B, T+1, H))
    test_kernel(fused_scan, _C.fused_scan, log_coeffs, log_values)

def logcumsumexp(x):
    return torch.log(torch.exp(x).cumsum(1))

def test_logcumsumexp():
    x = torch.randn(B, T, H)
    test_kernel(logcumsumexp, _C.logcumsumexp_cuda, x)

if __name__ == '__main__':
    test_mingru_gate()
    test_log_coeffs_and_values()
    test_logcumsumexp()
    test_fused_scan()
