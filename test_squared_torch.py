import torch
import torch.utils.cpp_extension
try:
    from pufferlib import _C
except ImportError:
    raise ImportError('Failed to import C/CUDA advantage kernel. If you have non-default PyTorch, try installing with --no-build-isolation')


if __name__ == '__main__':
    # THIS IS HARDCODED IN CUDA. DO NOT CHANGE
    num_envs = 2048

    steps = 10000
    grid_size = 9
    dummy = torch.zeros(5).cuda()
    indices = torch.arange(num_envs).cuda().int()
    envs, obs, actions, rewards, terminals = torch.ops.squared.create_squared_environments(num_envs, grid_size, dummy)
    atns = torch.randint(0, 5, (num_envs,)).cuda()
    actions[:] = atns

    torch.ops.squared.reset_environments(envs, indices)

    import time
    start = time.time()
    torch.cuda.synchronize()
    for i in range(steps):
        torch.ops.squared.step_environments(envs)

    torch.cuda.synchronize()
    end = time.time()

    print('Steps/sec:', num_envs * steps / (end - start))


