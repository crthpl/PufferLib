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

def test_kernel(py_func, cpp_func, *args):
    py_args = []
    cpp_args = []
    backward = False
    for arg in args:
        if isinstance(arg, torch.Tensor):
            if arg.requires_grad:
                backward = True

            dtype = torch.double if arg.dtype == torch.float32 else arg.dtype
            py_args.append(arg.clone().detach().to(dtype).cuda().requires_grad_(arg.requires_grad))
            cpp_args.append(arg.clone().detach().cuda().requires_grad_(arg.requires_grad))
        else:
            py_args.append(arg)
            cpp_args.append(arg)

    py_out = py_func(*py_args)
    cpp_out = cpp_func(*cpp_args)

    if not isinstance(py_out, (tuple, list)):
        py_out = [py_out]
    if not isinstance(cpp_out, (tuple, list)):
        cpp_out = [cpp_out]

    for py_o, cpp_o in zip(py_out, cpp_out):
        print(py_o.float(), cpp_o)
        assert_close(py_o.float(), cpp_o)

    if not backward:
        return

    py_loss = sum([o.mean() for o in py_out])/len(py_out)
    cpp_loss = sum([o.mean() for o in cpp_out])/len(cpp_out)

    py_loss.backward()
    cpp_loss.backward()

    for py_arg, cpp_arg in zip(py_args, cpp_args):
        if isinstance(py_arg, torch.Tensor) and py_arg.grad is not None:
            assert_close(py_arg.grad.float(), cpp_arg.grad)

def mingru_gate(state, gate, hidden):
    hidden = torch.where(hidden >= 0, hidden + 0.5, hidden.sigmoid())
    gate = gate.sigmoid()
    out = torch.lerp(state, hidden, gate)
    return out

def test_mingru_gate():
    state = torch.randn(B, T, H)
    gate = torch.randn(B, T, H)
    hidden = torch.randn(B, T, H)
    test_kernel(mingru_gate, _C.mingru_gate, state, gate, hidden)

def log_coeffs_and_values(gate, hidden):
    log_coeffs = -torch.nn.functional.softplus(gate)
    log_z = -torch.nn.functional.softplus(-gate)
    log_tilde_h = torch.where(hidden >= 0,
        (torch.nn.functional.relu(hidden) + 0.5).log(),
        -torch.nn.functional.softplus(-hidden))
    log_values = log_z + log_tilde_h
    return log_coeffs, log_values

def test_log_coeffs_and_values():
    gate = torch.randn(B, T, H, requires_grad=True)
    hidden = torch.randn(B, T, H, requires_grad=True)
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
    log_coeffs = -torch.nn.functional.softplus(torch.randn(B, T+1, H), requires_grad=True)
    log_values = -torch.nn.functional.softplus(torch.randn(B, T+1, H), requires_grad=True)
    test_kernel(fused_scan, _C.fused_scan, log_coeffs, log_values)

def logcumsumexp(x):
    return torch.log(torch.exp(x).cumsum(1))

def test_logcumsumexp():
    x = torch.randn(B, T, H, requires_grad=True)
    test_kernel(logcumsumexp, _C.logcumsumexp_cuda, x)

def fused_ppo_loss(logits, newvalue, actions, old_logprobs,
        advantages, prio, values, returns, adv_mean, adv_std,
        clip_coef, vf_clip_coef, vf_coef, ent_coef):

    segments, horizon, _ = logits.shape

    flat_logits = logits.reshape(-1, logits.size(-1));
    flat_actions = actions.reshape(-1);
    logprobs_new = torch.log_softmax(flat_logits, 1);

    probs_new = logprobs_new.exp();

    newlogprob_flat = logprobs_new.gather(1, flat_actions.unsqueeze(1)).squeeze(1);

    newlogprob = newlogprob_flat.reshape(segments, horizon);
    entropy = - (probs_new * logprobs_new).sum(1).mean();

    logratio = newlogprob - old_logprobs;
    ratio_new = logratio.exp();

    adv_normalized = prio.unsqueeze(1) * (advantages - adv_mean) / (adv_std + 1e-8);

    pg_loss1 = -adv_normalized * ratio_new;
    pg_loss2 = -adv_normalized * torch.clamp(ratio_new, 1.0 - clip_coef, 1.0 + clip_coef);
    pg_loss = torch.max(pg_loss1, pg_loss2).mean();


    newvalue = newvalue.view(returns.shape)
    v_clipped = values + torch.clamp(newvalue - values, -vf_clip_coef, vf_clip_coef);
    v_loss_unclipped = (newvalue - returns).pow(2);
    v_loss_clipped = (v_clipped - returns).pow(2);
    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean();

    #loss = pg_loss + vf_coef*v_loss - ent_coef*entropy
    loss = vf_coef*v_loss
    return loss

def test_fused_ppo_loss():
    A = 4
    logits = torch.randn(B, T, A, requires_grad=True)
    values_pred = torch.randn(B, T, requires_grad=True).contiguous()
    actions = torch.randint(0, A, (B, T))
    old_logprobs = torch.randn(B, T)
    advantages = torch.randn(B, T)
    prio = torch.rand(B)
    values = torch.randn(B, T)
    returns = torch.randn(B, T)
    clip_coef = 0.1
    vf_clip_coef = 0.1
    vf_coef = 0.1
    ent_coef = 0.1
    '''

    B, T, A = 2, 2, 4

    vf_coef = 0.5
    vf_clip_coef = 0.5

    # Fixed inputs for value loss only
    values_pred = torch.tensor([[1.0]], requires_grad=True)  # (B, T)
    values = torch.tensor([[0.5]])                           # (B, T)
    returns = torch.tensor([[2.0]])                          # (B, T)

    # Dummy inputs (not used when isolating value loss)
    logits = torch.randn(B, T, A, requires_grad=True)
    actions = torch.randint(0, A, (B, T))
    old_logprobs = torch.randn(B, T)
    advantages = torch.randn(B, T)
    prio = torch.rand(B)  # (B,)
    ent_coef = 0.1
    clip_coef = 0.1
    adv_mean = advantages.mean()
    adv_std = advantages.std()

    
    vf_coef = 0.5
    vf_clip_coef = 0.5

    # We'll define val_pred, values, returns explicitly for each (b,t)
    # Format: [B, T]

    values_pred = torch.tensor([
        [1.0, 2.0],   # b=0: 1.0 (in range), 2.0 (out of range)
        [0.0, 1.5]    # b=1: 0.0 (out), 1.5 (in)
    ], requires_grad=True)

    values = torch.tensor([
        [0.5, 1.0],
        [1.0, 1.0]
    ])

    returns = torch.tensor([
        [2.0, 3.0],
        [0.5, 1.0]
    ])
    '''


    test_kernel(fused_ppo_loss, _C.fused_ppo_loss, logits, values_pred, actions,
        old_logprobs, advantages, prio, values, returns, advantages.mean(), advantages.std(),
        clip_coef, vf_clip_coef, vf_coef, ent_coef)

if __name__ == '__main__':
    #test_mingru_gate()
    #test_log_coeffs_and_values()
    #test_logcumsumexp()
    #test_fused_scan()
    test_fused_ppo_loss()
