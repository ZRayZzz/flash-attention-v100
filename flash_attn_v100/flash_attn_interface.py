# Copyright (c) 2023, Tri Dao.

import os
from typing import Optional, Union

import torch
import torch.nn as nn
# We need to import the CUDA kernels after importing torch
import flash_attn_v100_cuda as flash_attn_cuda

def _flash_attn_forward(
    q, k, v, softmax_scale, causal
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_max, softmax_sum = flash_attn_cuda.fwd(
        q,
        k,
        v,
        None,
        softmax_scale,
        causal
    )
    return out, q, k, v, softmax_max, softmax_sum

def _flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_max,
    softmax_sum,
    dq,
    dk,
    dv,
    softmax_scale,
    causal,
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    D = out * dout
    D = D.sum(-1)
    dq, dk, dv, = flash_attn_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        D,
        softmax_sum,
        softmax_max,
        dq,
        dk,
        dv,
        softmax_scale,
        causal,
    )
    return dq, dk, dv

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, k, v, softmax_scale, causal
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, softmax_max, softmax_sum = _flash_attn_forward(
            q,
            k,
            v,
            softmax_scale,
            causal=causal,
        )
        ctx.save_for_backward(q, k, v, out, softmax_max, softmax_sum)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_max, softmax_sum = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_max,
            softmax_sum,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
        )
        return dq, dk, dv, None, None

def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
):
    return FlashAttnFunc.apply(
        q, k, v, softmax_scale, causal
    )