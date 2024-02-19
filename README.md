# Flash_Attention_V100

flash attention只支持Ampere架构以上的显卡，对于V100这个Volta架构的显卡并不支持，所以出于兴趣，我按照cutlass教程以及flash attention2的论文，写了这个适用于V100的版本，不过由于工作繁忙以及硬件条件限制，不能细致地进行性能调试，本Repo的性能并不能比得上pytorch的attention计算。当前forward的耗时相比于pytorch大约降低了40%，但是backward的耗时大约比pytorch多20%，两者相消。另外，该实现没有考虑边界条件，因此句子的长度要用right padding的方式，pad到32的倍数。这对正常训练并不会有影响，只需在计算loss时，将padding的地方忽略即可。

## 安装
在安装前，你需要确保：

- PyTorch >= 2.0.1
- CUDA >= 11.6
- Linux OS
- Cutlass源码

修改setup.py的146行，将这一行改为你下载的cutlass源码的位置

```py
include_dirs=[
    Path(this_dir) / "include",
    "/home/user/cutlass/include",
],
```

修改完毕后，执行命令进行源码安装
```bash
python setup.py install --user
```

## 用法

```python
from flash_attn_v100 import flash_attn_func
q = torch.empty((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=1).requires_grad_()
k = torch.empty((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=1).requires_grad_()
v = torch.empty((Z, N_CTX, H, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=1).requires_grad_()
cuda_out = flash_attn_func(q, k, v, sm_scale, causal)
```

## 参考
- [Flash-Attention](https://github.com/Dao-AILab/flash-attention)
- [CUTLASS](https://github.com/NVIDIA/cutlass)