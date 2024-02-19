import torch
import time
from flash_attn_v100 import flash_attn_func
Z, H, N_CTX, D_HEAD, causal, dtype = 2, 40, 2048, 128, True, torch.float16
torch.manual_seed(20)
q = torch.empty((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=1).requires_grad_()
k = torch.empty((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=1).requires_grad_()
v = torch.empty((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=1).requires_grad_()
sm_scale = 1 / 10
dout = torch.empty((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=1)
begin = time.time()
for i in range(1):
    q_transposed = q.transpose(1, 2)
    k_transposed = k.transpose(1, 2)
    v_transposed = v.transpose(1, 2)
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q_transposed, k_transposed.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    ref_out = torch.matmul(p, v_transposed)
    ref_out = ref_out.transpose(1, 2)
    ref_out.backward(dout)
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None
torch.cuda.synchronize(device="cuda:0")
end = time.time()
print(f"torch cost : {end - begin}")
begin = time.time()
# triton implementation
for i in range(1):
    cuda_out = flash_attn_func(q, k, v, sm_scale, causal)
    cuda_out.backward(dout)
    dq, q.grad = q.grad.clone(), None
    dk, k.grad = k.grad.clone(), None
    dv, v.grad = v.grad.clone(), None
torch.cuda.synchronize(device="cuda:0")
end = time.time()
print(f"triton cost : {end - begin}")
# compare
assert torch.allclose(ref_out, cuda_out, atol=1e-2, rtol=0)
assert torch.allclose(ref_dq, dq, atol=1e-2, rtol=0)
assert torch.allclose(ref_dk, dk, atol=1e-2, rtol=0)
assert torch.allclose(ref_dv, dv, atol=1e-2, rtol=0)