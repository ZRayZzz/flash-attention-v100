#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>
#include "fused_mha.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x num_heads x seqlen x head_size
        const at::Tensor &k,         // batch_size x num_heads x seqlen x head_size
        const at::Tensor &v,         // batch_size x num_heads x seqlen x head_size
        c10::optional<at::Tensor> &out_,             // batch_size x num_heads x seqlen x head_size
        const float softmax_scale,
        bool is_causal)
{
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm70 = dprops->major == 7 && dprops->minor == 0;
    TORCH_CHECK(is_sm70, "This repo only supports Volta GPUs.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16,
                "This repo only supports fp16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int num_heads = sizes[2];
    const int seqlen_q = sizes[1];
    const int head_size = sizes[3];
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size == 128, "current repo only supports head dimension 128, we will support more in the fulture");
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_q, num_heads, head_size);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    } else {
        out = torch::empty_like(q);
    }
    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto opts = q.options();

    auto softmax_sum = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    auto softmax_max = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fused_mha_forward(q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(), softmax_max.data_ptr(), softmax_sum.data_ptr(), 
        batch_size, num_heads, seqlen_q, seqlen_q, head_size, softmax_scale, is_causal, stream);

    return {out, softmax_max, softmax_sum};
}

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen_q x num_heads, x head_size_og
        const at::Tensor &q,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen_k x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &D,   // batch_size x seqlen_q x num_heads x head_size
        const at::Tensor &softmax_sum,     // b x h x seqlen_q
        const at::Tensor &softmax_max,     // b x h x seqlen_q
        c10::optional<at::Tensor> &dq_,   // batch_size x seqlen_q x num_heads x head_size
        c10::optional<at::Tensor> &dk_,   // batch_size x seqlen_k x num_heads_k x head_size
        c10::optional<at::Tensor> &dv_,   // batch_size x seqlen_k x num_heads_k x head_size
        const float softmax_scale,
        const bool is_causal)
{
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm70 = dprops->major == 7 && dprops->minor == 0;
    TORCH_CHECK(is_sm70, "This repo only supports Volta GPUs.");

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16,
                "This repo only supports fp16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_sum); CHECK_DEVICE(softmax_max);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int num_heads = sizes[2];
    const int seqlen_q = sizes[1];
    const int head_size = sizes[3];
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(head_size == 128, "current repo only supports head dimension 128, we will support more in the fulture");
    CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(v, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(dout, batch_size, seqlen_q, num_heads, head_size);
    CHECK_SHAPE(softmax_sum, batch_size, num_heads, seqlen_q);
    CHECK_SHAPE(softmax_max, batch_size, num_heads, seqlen_q);
    CHECK_SHAPE(D, batch_size, seqlen_q, num_heads);
    auto opts = q.options();
    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        CHECK_SHAPE(dq, batch_size, seqlen_q, num_heads, head_size);
    } else {
        // dq = torch::empty_like(q);
        dq = torch::zeros({batch_size, seqlen_q, num_heads, head_size}, opts.dtype(at::kHalf));
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        CHECK_SHAPE(dk, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        CHECK_SHAPE(dv, batch_size, seqlen_q, num_heads, head_size);
    } else {
        dv = torch::empty_like(k);
    }

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fused_mha_backward(q.data_ptr(), k.data_ptr(), v.data_ptr(), 
            out.data_ptr(), dout.data_ptr(), D.data_ptr(), softmax_max.data_ptr(), softmax_sum.data_ptr(), 
            dq.data_ptr(), dk.data_ptr(), dv.data_ptr(), batch_size, num_heads, seqlen_q, seqlen_q, head_size, softmax_scale, is_causal, stream);
    return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("bwd", &mha_bwd, "Backward pass");
}
