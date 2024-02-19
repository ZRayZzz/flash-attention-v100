#include <cuda.h>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// 从flash attention复制过来，对register中的数据类型做转换
template <typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_type(cute::Tensor<Engine, Layout> const &tensor) {
    using namespace cute;
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename Config>
__global__ void fused_mha_forward_kernel(const void *query_ptr, const void *key_ptr, const void *value_ptr, void *output_ptr, void *max_ptr, void *sum_ptr, int head, int m, int n, int k, float scale, bool causal)
{
    using namespace cute;
    using X = Underscore;
    using ElementType = typename Config::ElementType;
    using ComputeType = typename Config::ComputeType;

    using SmemLayoutQuery = typename Config::SmemLayoutQuery;
    using SmemLayoutKey = typename Config::SmemLayoutKey;
    using SmemLayoutValue = typename Config::SmemLayoutValue;
    using SmemLayoutValueTransposed = typename Config::SmemLayoutValueTransposed;
    using SmemLayoutAcc = typename Config::SmemLayoutAcc;
    using SmemLayoutOutput = typename Config::SmemLayoutOutput;
    using TiledMMA_TN = typename Config::MMA_TN;

    using Global2SharedCopyQuery = typename Config::Global2SharedCopyQuery;
    using Global2SharedCopyKey = typename Config::Global2SharedCopyKey;
    using Global2SharedCopyValue = typename Config::Global2SharedCopyValue;
    using Shared2RegisterCopyAcc = typename Config::Shared2RegisterCopyAcc;
    using Shared2RegisterCopyFp16Acc = typename Config::Shared2RegisterCopyFp16Acc;
    using Register2SharedCopyFp16Acc = typename Config::Register2SharedCopyFp16Acc;
    using Global2SharedCopyAtom = typename Config::Global2SharedCopyAtom;
    using SmemLayoutMax = typename Config::SmemLayoutMax;
    using SmemLayoutSum = typename Config::SmemLayoutSum;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;

    extern __shared__ ElementType shm_data[];

    int idx = threadIdx.x;
    int b = blockIdx.y / head;
    int h = blockIdx.y % head;
    int row_m = blockIdx.x;

    Tensor global_query = make_tensor(make_gmem_ptr((ElementType *)query_ptr + b * head * m * k + h * k + row_m * kTileM * head * k), 
                            Shape<Int<kTileM>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
    Tensor global_output = make_tensor(make_gmem_ptr((ElementType *)output_ptr + b * head * m * k + h * k + row_m * kTileM * head * k), Shape<Int<kTileM>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
    Tensor global_max = make_tensor(make_gmem_ptr((ComputeType*)max_ptr + b * head * m + h * m + row_m * kTileM),
                            Shape<Int<kTileM>, Int<1>>{},
                            make_stride(1, Int<1>{}));
    Tensor global_sum = make_tensor(make_gmem_ptr((ComputeType*)sum_ptr + b * head * m + h * m + row_m * kTileM),
                            Shape<Int<kTileM>, Int<1>>{},
                            make_stride(1, Int<1>{}));
    ComputeType *max_shm = (ComputeType*)((char*)shm_data + Config::kShmSizeQKVO);
    auto shared_max = make_tensor(make_smem_ptr(max_shm), SmemLayoutMax{});
    
    if (idx <32)
    shared_max(idx, 0) = -INFINITY;//global_max(idx, 0);
    __syncthreads();
    ComputeType *old_max_shm = max_shm + cute::cosize(SmemLayoutMax{});
    auto shared_old_max = make_tensor(make_smem_ptr(old_max_shm), SmemLayoutMax{});
    if (idx < 32)
    shared_old_max(idx, 0) = -INFINITY;
    __syncthreads();
    ComputeType *sum_shm = old_max_shm + cute::cosize(SmemLayoutMax{});
    auto shared_sum = make_tensor(make_smem_ptr(sum_shm), SmemLayoutSum{});
    clear(shared_sum);
    ElementType *query_shm = shm_data;
    auto shared_query = make_tensor(make_smem_ptr(query_shm), SmemLayoutQuery{});  // (kTileM, kTileK)
    Global2SharedCopyQuery global_2_shared_query_copy_tile;
    auto global_2_shared_query_copy_thread = global_2_shared_query_copy_tile.get_slice(idx);
    auto global_query_thread_copy_tensor = global_2_shared_query_copy_thread.partition_S(global_query);  // (CPY, CPY_M, CPY_K)
    auto shared_query_thread_copy_tensor = global_2_shared_query_copy_thread.partition_D(shared_query);  // (CPY, CPY_M, CPY_K)
    float4* shared_ptr = reinterpret_cast<float4*>(shm_data);
    float4* global_ptr = reinterpret_cast<float4*>((ElementType *)query_ptr + b * head * m * k + h * k + row_m * kTileM * head * k);
    for (int i = 0; i < 4; i++) {
        shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16] = global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16];
    }
    __syncthreads();
    TiledMMA_TN tiled_mma_tn;
    auto thr_mma_tn = tiled_mma_tn.get_slice(idx);
    auto mma_thread_query_register_tensor = thr_mma_tn.partition_fragment_A(global_query);  // (MMA, MMA_M, MMA_K)
    auto mma_thread_query_shared_tensor = thr_mma_tn.partition_A(shared_query);
    cute::copy(mma_thread_query_shared_tensor, mma_thread_query_register_tensor);
    __syncthreads();
    auto shared_key = make_tensor(make_smem_ptr(query_shm), SmemLayoutKey{});  // (kTileN, kTileK)

    Tensor global_key = make_tensor(make_gmem_ptr((ElementType *)key_ptr + b * head * n * k + h * k), 
                            Shape<Int<kTileN>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
    auto mma_thread_key_register_tensor = thr_mma_tn.partition_fragment_B(global_key);  // (MMA, MMA_N, MMA_K)
    auto mma_thread_key_shared_tensor = thr_mma_tn.partition_B(shared_key);

    ElementType *acc_shm = (ElementType*)shm_data;
    auto shared_acc = make_tensor(make_smem_ptr(acc_shm), SmemLayoutAcc{});
    auto mma_thread_acc_register_tensor = thr_mma_tn.partition_fragment_C(shared_acc);
    auto mma_thread_acc_shared_tensor = thr_mma_tn.partition_C(shared_acc);
    Shared2RegisterCopyAcc shared_2_register_acc_copy_tile;
    auto shared_2_register_acc_copy_thread = shared_2_register_acc_copy_tile.get_slice(idx);
    auto shared_2_register_acc_thread_shared_tensor = shared_2_register_acc_copy_thread.partition_D(shared_acc);  // (CPY, CPY_M, CPY_K)
    auto shared_2_register_acc_thread_register_tensor   = make_fragment_like(shared_2_register_acc_thread_shared_tensor);
    ElementType *fp16_acc_shm = shm_data;
    auto shared_fp16_acc = make_tensor(make_smem_ptr(fp16_acc_shm), SmemLayoutAcc{});
    Shared2RegisterCopyFp16Acc shared_2_register_fp16_acc_copy_tile;
    auto shared_2_register_fp16_acc_copy_thread = shared_2_register_fp16_acc_copy_tile.get_slice(idx);
    auto register_2_shared_fp16_acc_thread_shared_tensor = shared_2_register_fp16_acc_copy_thread.partition_D(shared_fp16_acc);  // (CPY, CPY_M, CPY_K)
    auto mma_thread_acc_shared_tensor_a_matrix = thr_mma_tn.partition_A(shared_fp16_acc);
    auto mma_thread_acc_register_tensor_a_matrix = thr_mma_tn.partition_fragment_A(shared_fp16_acc);
    ElementType *value_shm = shm_data;
    auto shared_value = make_tensor(make_smem_ptr(value_shm), SmemLayoutValue{});
    auto shared_value_transposed = make_tensor(make_smem_ptr(value_shm), SmemLayoutValueTransposed{});
    auto mma_thread_value_register_tensor = thr_mma_tn.partition_fragment_B(shared_value_transposed);  // (MMA, MMA_N, MMA_K)
    auto mma_thread_value_shared_tensor = thr_mma_tn.partition_B(shared_value_transposed);

    ElementType *output_shm = (ElementType*)shm_data;
    auto shared_output = make_tensor(make_smem_ptr(output_shm), SmemLayoutOutput{});
    auto mma_thread_output_register_tensor = thr_mma_tn.partition_fragment_C(global_output); // (MMA, MMA_M, MMA_K)
    auto mma_thread_output_shared_tensor = thr_mma_tn.partition_C(shared_output);
    auto mma_thread_output_global_tensor = thr_mma_tn.partition_C(global_output);
    clear(mma_thread_output_register_tensor);
    ElementType *o_scale_shm = (ElementType *)(value_shm + cute::cosize(SmemLayoutValue{}));
    auto shared_o_scale = make_tensor(make_smem_ptr(o_scale_shm), SmemLayoutOutput{});
    auto mma_thread_o_scale_shared_tensor = thr_mma_tn.partition_C(shared_o_scale);
    ElementType *sum_scale_shm = (ElementType *)shm_data;
    auto shared_sum_scale = make_tensor(make_smem_ptr(sum_scale_shm), SmemLayoutOutput{});
    auto mma_thread_sum_scale_shared_tensor = thr_mma_tn.partition_C(shared_sum_scale);
    int end_tile = n / kTileN - 1;
    if (causal) {
        end_tile = row_m;
    }
    #pragma unroll
    for (int n_tile = 0; n_tile < end_tile; n_tile++) {
        clear(mma_thread_acc_register_tensor);
        Tensor global_key = make_tensor(make_gmem_ptr((ElementType *)key_ptr + b * head * n * k + h * k + n_tile * kTileN * head * k), 
                            Shape<Int<kTileN>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
        auto mma_thread_key_global_tensor = thr_mma_tn.partition_B(global_key);
        cute::copy(mma_thread_key_global_tensor, mma_thread_key_register_tensor);
        cute::gemm(tiled_mma_tn, mma_thread_acc_register_tensor, 
                              mma_thread_query_register_tensor, 
                              mma_thread_key_register_tensor, 
                              mma_thread_acc_register_tensor);
        cute::copy(mma_thread_acc_register_tensor, mma_thread_acc_shared_tensor);
        __syncthreads();
        if (idx < 32) {
            ComputeType thread_max = shared_acc(idx, 0);
            #pragma unroll
            for (int j = 1; j < kTileN; j++) {
                ComputeType tmp  = shared_acc(idx, j);
                thread_max = thread_max < tmp ? tmp : thread_max;
            }
            thread_max *= scale;
            ComputeType old_thread_max = shared_max(idx, 0);
            ComputeType new_thread_max = old_thread_max < thread_max ? thread_max : old_thread_max;
            ComputeType block_thread_sum = 0;
            #pragma unroll
            for (int j = 0; j < kTileN; j++) {
                ComputeType thread_acc = shared_acc(idx, j);
                ComputeType new_thread_acc = __expf(thread_acc * scale - new_thread_max);
                block_thread_sum += new_thread_acc;
                shared_acc(idx, j) = new_thread_acc;
            }
            shared_sum(idx, 0) = __expf(old_thread_max - new_thread_max) * shared_sum(idx, 0) + block_thread_sum;
            shared_max(idx, 0) = new_thread_max;
            // shared_old_max(idx, 0) = __expf(old_thread_max - new_thread_max);
            shared_old_max(idx, 0) = __expf(old_thread_max - new_thread_max);
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < kTileN; i++) {
            shared_o_scale(i, idx) = shared_old_max(i, 0);
        }
        cute::copy(mma_thread_acc_shared_tensor_a_matrix, mma_thread_acc_register_tensor_a_matrix);
        __syncthreads();
        Tensor global_value = make_tensor(make_gmem_ptr((ElementType *)value_ptr + b * head * n * k + h * k + n_tile * kTileN * head * k), 
                            Shape<Int<kTileN>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));

        Global2SharedCopyValue global_2_shared_value_copy_tile;
        auto global_2_shared_value_copy_thread = global_2_shared_value_copy_tile.get_slice(idx);
        auto global_value_thread_copy_tensor = global_2_shared_value_copy_thread.partition_S(global_value);  // (CPY, CPY_N, CPY_K)
        auto shared_value_thread_copy_tensor = global_2_shared_value_copy_thread.partition_D(shared_value);  // (CPY, CPY_N , CPY_K)
        shared_ptr = reinterpret_cast<float4*>(shm_data);
        global_ptr = reinterpret_cast<float4*>((ElementType *)value_ptr + b * head * n * k + h * k + n_tile * kTileN * head * k);
        for (int i = 0; i < 4; i++) {
            shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16] = global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16];
        }
        __syncthreads();
        cute::copy(mma_thread_value_shared_tensor, mma_thread_value_register_tensor);
        #pragma unroll
        for (int i = 0; i < size<0>(mma_thread_output_register_tensor); i++) {
            #pragma unroll
            for (int j = 0; j < size<1>(mma_thread_output_register_tensor); j++) {
                #pragma unroll
                for (int kk = 0; kk < size<2>(mma_thread_output_register_tensor); kk++) {
                    mma_thread_output_register_tensor(i, j, kk) *= mma_thread_o_scale_shared_tensor(i, j, kk);
                }
            }
        }
        cute::gemm(thr_mma_tn, mma_thread_output_register_tensor, 
                                 mma_thread_acc_register_tensor_a_matrix,
                                 mma_thread_value_register_tensor,
                                 mma_thread_output_register_tensor);
    }
    
    clear(mma_thread_acc_register_tensor);
    global_key = make_tensor(make_gmem_ptr((ElementType *)key_ptr + b * head * n * k + h * k + end_tile * kTileN * head * k), 
                        Shape<Int<kTileN>, Int<kTileK>>{},
                        make_stride(head * k, Int<1>{}));
    auto mma_thread_key_global_tensor = thr_mma_tn.partition_B(global_key);
    cute::copy(mma_thread_key_global_tensor, mma_thread_key_register_tensor);
    cute::gemm(tiled_mma_tn, mma_thread_acc_register_tensor, 
                            mma_thread_query_register_tensor, 
                            mma_thread_key_register_tensor, 
                            mma_thread_acc_register_tensor);
    cute::copy(mma_thread_acc_register_tensor, mma_thread_acc_shared_tensor);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < kTileN; i+=4) {
        shared_acc(idx / kTileN + i, idx % kTileN) = (idx / kTileN + i >= idx % kTileN ? shared_acc(idx / kTileN + i, idx % kTileN) : -INFINITY);
    }
    __syncthreads();
    if (idx < 32) {
        ComputeType thread_max = shared_acc(idx, 0);
        #pragma unroll
        for (int j = 1; j < kTileN; j++) {
            ComputeType tmp  = shared_acc(idx, j);
            thread_max = thread_max < tmp ? tmp : thread_max;
        }
        thread_max *= scale;
        ComputeType old_thread_max = shared_max(idx, 0);
        ComputeType new_thread_max = old_thread_max < thread_max ? thread_max : old_thread_max;
        ComputeType block_thread_sum = 0;
        #pragma unroll
        for (int j = 0; j < kTileN; j++) {
            ComputeType thread_acc = shared_acc(idx, j);
            ComputeType new_thread_acc = __expf(thread_acc * scale - new_thread_max);
            block_thread_sum += new_thread_acc;
            shared_acc(idx, j) = new_thread_acc;
        }
        shared_sum(idx, 0) = __expf(old_thread_max - new_thread_max) * shared_sum(idx, 0) + block_thread_sum;
        shared_max(idx, 0) = new_thread_max;
        shared_old_max(idx, 0) = __expf(old_thread_max - new_thread_max);
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < kTileN; i++) {
        shared_o_scale(i, idx) = shared_old_max(i, 0);
    }
    
    cute::copy(mma_thread_acc_shared_tensor_a_matrix, mma_thread_acc_register_tensor_a_matrix);
    __syncthreads();
    Tensor global_value = make_tensor(make_gmem_ptr((ElementType *)value_ptr + b * head * n * k + h * k + end_tile * kTileN * head * k), 
                        Shape<Int<kTileN>, Int<kTileK>>{},
                        make_stride(head * k, Int<1>{}));

    Global2SharedCopyValue global_2_shared_value_copy_tile;
    auto global_2_shared_value_copy_thread = global_2_shared_value_copy_tile.get_slice(idx);
    auto global_value_thread_copy_tensor = global_2_shared_value_copy_thread.partition_S(global_value);  // (CPY, CPY_N, CPY_K)
    auto shared_value_thread_copy_tensor = global_2_shared_value_copy_thread.partition_D(shared_value);  // (CPY, CPY_N , CPY_K)
    shared_ptr = reinterpret_cast<float4*>(shm_data);
    global_ptr = reinterpret_cast<float4*>((ElementType *)value_ptr + b * head * n * k + h * k + end_tile * kTileN * head * k);
    for (int i = 0; i < 4; i++) {
        shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16] = global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16];
    }
    __syncthreads();
    cute::copy(mma_thread_value_shared_tensor, mma_thread_value_register_tensor);
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < size<0>(mma_thread_output_register_tensor); i++) {
        #pragma unroll
        for (int j = 0; j < size<1>(mma_thread_output_register_tensor); j++) {
            #pragma unroll
            for (int kk = 0; kk < size<2>(mma_thread_output_register_tensor); kk++) {
                mma_thread_output_register_tensor(i, j, kk) *= mma_thread_o_scale_shared_tensor(i, j, kk);
            }
        }
    }
    __syncthreads();
    cute::gemm(thr_mma_tn, mma_thread_output_register_tensor, 
                                mma_thread_acc_register_tensor_a_matrix,
                                mma_thread_value_register_tensor,
                                mma_thread_output_register_tensor);
    if (idx < 32) {
        global_max(idx, 0) = shared_max(idx, 0);
    }
    else if (idx < 64) {
        global_sum(idx - 32, 0) = shared_sum(idx - 32, 0);
    }
    #pragma unroll
    for (int i = 0; i < kTileN; i++) {
            shared_sum_scale(i, idx) = 1.f / shared_sum(i, 0);
    }
    __syncthreads();
    output_shm = (ElementType*)(sum_scale_shm + cute::cosize(SmemLayoutOutput{}));
    shared_output = make_tensor(make_smem_ptr(output_shm), SmemLayoutOutput{});
    mma_thread_output_shared_tensor = thr_mma_tn.partition_C(shared_output);
    cute::copy(mma_thread_output_register_tensor, mma_thread_output_shared_tensor);
    __syncthreads();
    for (int i = 0; i < kTileM; i++) {
        shared_output(i, idx) = shared_output(i, idx) * shared_sum_scale(i, idx);
    }
    __syncthreads();
    shared_ptr = reinterpret_cast<float4*>(output_shm);
    global_ptr = reinterpret_cast<float4*>((ElementType *)output_ptr + b * head * m * k + h * k + row_m * kTileM * head * k);
    for (int i = 0; i < 4; i++) {
        global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16] = shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16];
    }
}

template <typename Config>
__global__ void 
fused_mha_backward_kernel(const void *query_ptr, const void *key_ptr, const void *value_ptr, 
            void *output_ptr, void *d_output_ptr, void *d_ptr, void *max_ptr, void *sum_ptr, 
            void *d_query_ptr, void *d_key_ptr, void *d_value_ptr, int head, int m, int n, int k, float scale, bool causal)
{
    using namespace cute;
    using X = Underscore;
    using ElementType = typename Config::ElementType;
    using ComputeType = typename Config::ComputeType;

    using SmemLayoutQuery = typename Config::SmemLayoutQuery;
    using SmemLayoutQueryTransposed = typename Config::SmemLayoutQueryTransposed;
    using SmemLayoutKey = typename Config::SmemLayoutKey;
    using SmemLayoutKeyTransposed = typename Config::SmemLayoutKeyTransposed;
    using SmemLayoutValue = typename Config::SmemLayoutValue;
    using SmemLayoutValueTransposed = typename Config::SmemLayoutValueTransposed;
    using SmemLayoutAcc = typename Config::SmemLayoutAcc;
    using SmemLayoutAccTransposed = typename Config::SmemLayoutAccTransposed;
    using SmemLayoutOutput = typename Config::SmemLayoutOutput;
    using SmemLayoutDOutput = typename Config::SmemLayoutDOutput;
    using SmemLayoutDOutputTransposed = typename Config::SmemLayoutDOutputTransposed;
    using TiledMMA_TN = typename Config::MMA_TN;

    using Global2SharedCopyQuery = typename Config::Global2SharedCopyQuery;
    using Global2SharedCopyKey = typename Config::Global2SharedCopyKey;
    using Global2SharedCopyValue = typename Config::Global2SharedCopyValue;
    using Global2SharedCopyOutput = typename Config::Global2SharedCopyOutput;
    using Global2SharedCopyDOutput = typename Config::Global2SharedCopyDOutput;
    using Shared2RegisterCopyAcc = typename Config::Shared2RegisterCopyAcc;
    using Shared2RegisterCopyFp16Acc = typename Config::Shared2RegisterCopyFp16Acc;
    using Register2SharedCopyFp16Acc = typename Config::Register2SharedCopyFp16Acc;
    using Global2SharedCopyAtom = typename Config::Global2SharedCopyAtom;
    using SmemLayoutMax = typename Config::SmemLayoutMax;
    using SmemLayoutSum = typename Config::SmemLayoutSum;

    constexpr int kTileM = Config::kTileM;
    constexpr int kTileN = Config::kTileN;
    constexpr int kTileK = Config::kTileK;

    extern __shared__ ElementType shm_data[];

    ElementType *qkvodo_shm = shm_data;
    ComputeType *max_shm = (ComputeType*)(qkvodo_shm + 2 * cosize(SmemLayoutQuery{}));
    auto shared_max = make_tensor(make_smem_ptr(max_shm), SmemLayoutMax{});

    ComputeType *sum_shm = (ComputeType*)(max_shm + cosize(SmemLayoutMax{}));
    auto shared_sum = make_tensor(make_smem_ptr(sum_shm), SmemLayoutSum{});

    int idx = threadIdx.x;
    int b = blockIdx.y / head;
    int h = blockIdx.y % head;
    int col_n = blockIdx.x;
    TiledMMA_TN tiled_mma_tn;
    auto thr_mma_tn = tiled_mma_tn.get_slice(idx);

    Tensor global_d_key = make_tensor(make_gmem_ptr((ElementType *)d_key_ptr + b * head * n * k + h * k + col_n * kTileN * head * k), 
                            Shape<Int<kTileN>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
    auto mma_thread_d_key_register_tensor = thr_mma_tn.partition_fragment_C(global_d_key);
    auto mma_thread_d_key_global_tensor = thr_mma_tn.partition_C(global_d_key);
    clear(mma_thread_d_key_register_tensor);
    Tensor global_d_value = make_tensor(make_gmem_ptr((ElementType *)d_value_ptr + b * head * n * k + h * k + col_n * kTileN * head * k), 
                            Shape<Int<kTileN>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
    auto mma_thread_d_value_global_tensor = thr_mma_tn.partition_C(global_d_value);
    auto mma_thread_d_value_register_tensor = thr_mma_tn.partition_fragment_C(global_d_value);
    clear(mma_thread_d_value_register_tensor);
    int begin_m_tile = 0;
    ElementType* value_shm = shm_data;
    auto shared_value = make_tensor(make_smem_ptr(value_shm), SmemLayoutValue{});
    float4* shared_ptr1 = reinterpret_cast<float4*>(value_shm);
    float4* global_ptr1 = reinterpret_cast<float4*>((ElementType *)value_ptr + b * head * n * k + h * k + col_n * kTileN * head * k);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        shared_ptr1[(idx / 16 + i * 8) * 16 + idx % 16] = global_ptr1[(idx / 16 + i * 8) * head * 16 + idx % 16];
    }
    __syncthreads();
    auto mma_thread_value_register_tensor = thr_mma_tn.partition_fragment_B(shared_value);  // (MMA, MMA_N, MMA_K)
    auto mma_thread_value_shared_tensor = thr_mma_tn.partition_B(shared_value);
    cute::copy(mma_thread_value_shared_tensor, mma_thread_value_register_tensor);
    __syncthreads();
    if (causal) {
        begin_m_tile = col_n;
    }
    {
        ElementType* key_shm = shm_data;
        Tensor global_key = make_tensor(make_gmem_ptr((ElementType *)key_ptr + b * head * n * k + h * k + col_n * kTileN * head * k), 
                            Shape<Int<kTileN>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
        float4* shared_ptr = reinterpret_cast<float4*>(key_shm);
        float4* global_ptr = reinterpret_cast<float4*>((ElementType *)key_ptr + b * head * n * k + h * k + col_n * kTileN * head * k);
        auto mma_thread_key_register_tensor = thr_mma_tn.partition_fragment_B(global_key);  // (MMA, MMA_N, MMA_K)
        ElementType* query_shm = key_shm + cosize(SmemLayoutKey{});
        Tensor global_query = make_tensor(make_gmem_ptr((ElementType *)query_ptr + b * head * m * k + h * k + begin_m_tile * kTileM * head * k), 
                            Shape<Int<kTileM>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
        auto mma_thread_query_register_tensor = thr_mma_tn.partition_fragment_A(global_query);  // (MMA, MMA_M, MMA_K)
        ComputeType *acc_shm = (ComputeType*)(sum_shm + cosize(SmemLayoutSum{}));
        auto shared_acc = make_tensor(make_smem_ptr(acc_shm), SmemLayoutAcc{});
        auto mma_thread_acc_shared_tensor = thr_mma_tn.partition_C(shared_acc);
        auto mma_thread_acc_register_tensor = thr_mma_tn.partition_fragment_C(shared_acc);
        clear(mma_thread_acc_register_tensor);
        auto mma_thread_key_global_tensor = thr_mma_tn.partition_B(global_key);
        auto mma_thread_query_global_tensor = thr_mma_tn.partition_A(global_query);
        cute::copy(mma_thread_key_global_tensor, mma_thread_key_register_tensor);
        cute::copy(mma_thread_query_global_tensor, mma_thread_query_register_tensor);
        cute::gemm(tiled_mma_tn, mma_thread_acc_register_tensor, 
                              mma_thread_query_register_tensor, 
                              mma_thread_key_register_tensor, 
                              mma_thread_acc_register_tensor);
        cute::copy(mma_thread_acc_register_tensor, mma_thread_acc_shared_tensor);
        __syncthreads();
        if (causal) {
            if (idx < 32) {
                #pragma unroll
                for (int j = 0; j < kTileN; j++) {
                    shared_acc(idx, j) = (idx >= j ? shared_acc(idx, j) : -INFINITY);
                }
            }
            __syncthreads();
        }
        Tensor global_max = make_tensor(make_gmem_ptr((ComputeType *)max_ptr + b * head * m + h * m + begin_m_tile * kTileM), 
                            Shape<Int<kTileM>, Int<1>>{},
                            make_stride(1, Int<1>{}));
        Tensor global_sum = make_tensor(make_gmem_ptr((ComputeType *)sum_ptr + b * head * m + h * m + begin_m_tile * kTileM), 
                            Shape<Int<kTileM>, Int<1>>{},
                            make_stride(1, Int<1>{}));
        #pragma unroll
        for (int i = 0; i < kTileM; i += 4) {
            auto thread_max = global_max(idx / kTileN + i, 0);
            auto thread_sum = 1.f / global_sum(idx / kTileN + i, 0);
            shared_acc(idx / kTileN + i, idx % kTileN) = __expf(scale * shared_acc(idx / kTileN + i, idx % kTileN) - thread_max) * thread_sum;
        }
        __syncthreads();

        ElementType *fp16_acc_shm = (ElementType*)(acc_shm + cosize(SmemLayoutAcc{}));
        ElementType* d_output_shm = key_shm;
        auto shared_d_output = make_tensor(make_smem_ptr(d_output_shm), SmemLayoutDOutput{});
        auto shared_d_output_transposed = make_tensor(make_smem_ptr(d_output_shm), SmemLayoutDOutputTransposed{});
        shared_ptr = reinterpret_cast<float4*>(d_output_shm);
        global_ptr = reinterpret_cast<float4*>((ElementType *)d_output_ptr + b * head * m * k + h * k + begin_m_tile * kTileM * head * k);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16] = global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16];
        }
        __syncthreads();

        auto mma_thread_d_output_shared_tensor_b_matrix = thr_mma_tn.partition_B(shared_d_output_transposed);
        auto mma_thread_d_output_register_tensor_b_matrix = thr_mma_tn.partition_fragment_B(shared_d_output_transposed);  // (MMA, MMA_M, MMA_K)
        cute::copy(mma_thread_d_output_shared_tensor_b_matrix, mma_thread_d_output_register_tensor_b_matrix);
        __syncthreads();
        auto shared_acc_transposed = make_tensor(make_smem_ptr(acc_shm), SmemLayoutAccTransposed{});
        auto mma_thread_acc_transposed_register_tensor_a_matrix = thr_mma_tn.partition_fragment_A(shared_acc_transposed);
        auto mma_thread_acc_transposed_shared_tensor_a_matrix = thr_mma_tn.partition_A(shared_acc_transposed);
        cute::copy(mma_thread_acc_transposed_shared_tensor_a_matrix, mma_thread_acc_transposed_register_tensor_a_matrix);
        cute::gemm(thr_mma_tn, mma_thread_d_value_register_tensor, 
                                 mma_thread_acc_transposed_register_tensor_a_matrix,
                                 mma_thread_d_output_register_tensor_b_matrix,
                                 mma_thread_d_value_register_tensor);
        auto mma_thread_d_output_register_tensor_a_matrix = thr_mma_tn.partition_fragment_A(shared_d_output);  // (MMA, MMA_M, MMA_K)
        auto mma_thread_d_output_shared_tensor_a_matrix = thr_mma_tn.partition_A(shared_d_output);
        cute::copy(mma_thread_d_output_shared_tensor_a_matrix, mma_thread_d_output_register_tensor_a_matrix);
        __syncthreads();
        ComputeType *d_p_shm = (ComputeType*)(fp16_acc_shm);
        auto shared_d_p = make_tensor(make_smem_ptr(d_p_shm), SmemLayoutAcc{});
        auto mma_thread_d_p_register_tensor = thr_mma_tn.partition_fragment_C(shared_d_p);
        auto mma_thread_d_p_shared_tensor = thr_mma_tn.partition_C(shared_d_p);
        clear(mma_thread_d_p_register_tensor);
        cute::gemm(thr_mma_tn, mma_thread_d_p_register_tensor, 
                                 mma_thread_d_output_register_tensor_a_matrix,
                                 mma_thread_value_register_tensor,
                                 mma_thread_d_p_register_tensor);
        cute::copy(mma_thread_d_p_register_tensor, mma_thread_d_p_shared_tensor);
        __syncthreads();
        Tensor global_d = make_tensor(make_gmem_ptr((ElementType *)d_ptr + b * head * m + h + begin_m_tile * kTileM * head), 
                            Shape<Int<kTileM>, Int<1>>{},
                            make_stride(head, Int<1>{}));
        #pragma unroll
        for (int i = 0; i < kTileM; i+=4) {
            shared_acc(i + idx / 32, idx % 32) *= (shared_d_p(i + idx / 32, idx % 32) - global_d(i + idx / 32, 0)) * scale;
        }
        
        __syncthreads();
        auto mma_thread_d_s_fp32_shared_tensor_a_matrix = thr_mma_tn.partition_A(shared_acc);
        auto mma_thread_d_s_fp16_register_tensor_a_matrix = thr_mma_tn.partition_fragment_A(shared_acc);
        cute::copy(mma_thread_d_s_fp32_shared_tensor_a_matrix, mma_thread_d_s_fp16_register_tensor_a_matrix);
        Tensor global_d_query = make_tensor(make_gmem_ptr((ElementType *)d_query_ptr + b * head * m * k + h * k + begin_m_tile * kTileM * head * k), 
                            Shape<Int<kTileM>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
        auto mma_thread_d_query_register_tensor = thr_mma_tn.partition_fragment_C(global_d_query);
        auto mma_thread_d_query_global_tensor = thr_mma_tn.partition_C(global_d_query);
        clear(mma_thread_d_query_register_tensor);
        auto shared_key_transposed = make_tensor(make_smem_ptr(key_shm), SmemLayoutKeyTransposed{});
        shared_ptr = reinterpret_cast<float4*>(key_shm);
        global_ptr = reinterpret_cast<float4*>((ElementType *)key_ptr + b * head * n * k + h * k + col_n * kTileN * head * k);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16] = global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16];
        }
        __syncthreads();
        auto mma_thread_transposed_key_register_tensor = thr_mma_tn.partition_fragment_B(shared_key_transposed);  // (MMA, MMA_N, MMA_K)
        auto mma_thread_transposed_key_shared_tensor = thr_mma_tn.partition_B(shared_key_transposed);
        cute::copy(mma_thread_transposed_key_shared_tensor, mma_thread_transposed_key_register_tensor);
        __syncthreads();
        cute::gemm(thr_mma_tn, mma_thread_d_query_register_tensor, 
                                 mma_thread_d_s_fp16_register_tensor_a_matrix,
                                 mma_thread_transposed_key_register_tensor,
                                 mma_thread_d_query_register_tensor);
        auto shared_d_query = make_tensor(make_smem_ptr((ComputeType*)shm_data), SmemLayoutQuery{});  // (kTileM, kTileK)
        auto mma_thread_d_query_shared_tensor = thr_mma_tn.partition_C(shared_d_query);
        cute::copy(mma_thread_d_query_register_tensor, mma_thread_d_query_shared_tensor);
        __syncthreads();
        cute::half_t d_query_register_half2[2];
        #pragma unroll
        for (int i = 0; i < kTileM; i+=2) {
            d_query_register_half2[0] = (cute::half_t)shared_d_query(i + idx / 64, (idx % 64) * 2);
            d_query_register_half2[1] = (cute::half_t)shared_d_query(i + idx / 64, (idx % 64) * 2 + 1);
            atomicAdd(reinterpret_cast<__half2*>(&global_d_query(i + idx / 64, (idx % 64) * 2)), *(reinterpret_cast<__half2*>(d_query_register_half2)));
        }
        cute::copy(mma_thread_acc_transposed_shared_tensor_a_matrix, mma_thread_acc_transposed_register_tensor_a_matrix);
        __syncthreads();
        auto shared_query_transposed = make_tensor(make_smem_ptr(query_shm), SmemLayoutQueryTransposed{});
        shared_ptr = reinterpret_cast<float4*>(query_shm);
        global_ptr = reinterpret_cast<float4*>((ElementType *)query_ptr + b * head * m * k + h * k + begin_m_tile * kTileM * head * k);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16] = global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16];
        }
        __syncthreads();
        auto mma_thread_query_register_tensor_b_matrix = thr_mma_tn.partition_fragment_B(shared_query_transposed);  // (MMA, MMA_M, MMA_K)
        auto mma_thread_query_shared_tensor_b_matrix = thr_mma_tn.partition_B(shared_query_transposed);  // (MMA, MMA_M, MMA_K)
        cute::copy(mma_thread_query_shared_tensor_b_matrix, mma_thread_query_register_tensor_b_matrix);
        cute::gemm(thr_mma_tn, mma_thread_d_key_register_tensor, 
                                 mma_thread_acc_transposed_register_tensor_a_matrix,
                                 mma_thread_query_register_tensor_b_matrix,
                                 mma_thread_d_key_register_tensor);
    }
    #pragma unroll
    for (int m_tile = begin_m_tile + 1; m_tile < m / kTileM; m_tile++) {
        ElementType* key_shm = shm_data;
        Tensor global_key = make_tensor(make_gmem_ptr((ElementType *)key_ptr + b * head * n * k + h * k + col_n * kTileN * head * k), 
                            Shape<Int<kTileN>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
        float4* shared_ptr = reinterpret_cast<float4*>(key_shm);
        float4* global_ptr = reinterpret_cast<float4*>((ElementType *)key_ptr + b * head * n * k + h * k + col_n * kTileN * head * k);
        ElementType* query_shm = key_shm + cosize(SmemLayoutKey{});
        Tensor global_query = make_tensor(make_gmem_ptr((ElementType *)query_ptr + b * head * m * k + h * k + m_tile * kTileM * head * k), 
                            Shape<Int<kTileM>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
        ComputeType *acc_shm = (ComputeType*)(sum_shm + cute::cosize(SmemLayoutSum{}));
        auto shared_acc = make_tensor(make_smem_ptr(acc_shm), SmemLayoutAcc{});
        auto mma_thread_acc_shared_tensor = thr_mma_tn.partition_C(shared_acc);
        auto mma_thread_acc_register_tensor = thr_mma_tn.partition_fragment_C(shared_acc);
        clear(mma_thread_acc_register_tensor);
        auto mma_thread_key_global_tensor = thr_mma_tn.partition_B(global_key);
        auto mma_thread_query_global_tensor = thr_mma_tn.partition_A(global_query);
        auto mma_thread_key_register_tensor = thr_mma_tn.partition_fragment_B(global_key);
        auto mma_thread_query_register_tensor = thr_mma_tn.partition_fragment_A(global_query);
        cute::copy(mma_thread_query_global_tensor, mma_thread_query_register_tensor);
        cute::copy(mma_thread_key_global_tensor, mma_thread_key_register_tensor);
        cute::gemm(thr_mma_tn, mma_thread_acc_register_tensor, 
                                 mma_thread_query_register_tensor,
                                 mma_thread_key_register_tensor,
                                 mma_thread_acc_register_tensor);
        cute::copy(mma_thread_acc_register_tensor, mma_thread_acc_shared_tensor);
        __syncthreads();
        Tensor global_max = make_tensor(make_gmem_ptr((ComputeType *)max_ptr + b * head * m + h * m + m_tile * kTileM), 
                            Shape<Int<kTileM>, Int<1>>{},
                            make_stride(1, Int<1>{}));
        Tensor global_sum = make_tensor(make_gmem_ptr((ComputeType *)sum_ptr + b * head * m + h * m + m_tile * kTileM), 
                            Shape<Int<kTileM>, Int<1>>{},
                            make_stride(1, Int<1>{}));
        #pragma unroll
        for (int i = 0; i < kTileM; i += 4) {
            auto thread_max = global_max(idx / kTileN + i, 0);
            auto thread_sum = 1.f / global_sum(idx / kTileN + i, 0);
            shared_acc(idx / kTileN + i, idx % kTileN) = __expf(scale * shared_acc(idx / kTileN + i, idx % kTileN) - thread_max) * thread_sum;
        }
        __syncthreads();

        ElementType *fp16_acc_shm = (ElementType*)(acc_shm + cosize(SmemLayoutAcc{}));
        ElementType* d_output_shm = key_shm;
        auto shared_d_output = make_tensor(make_smem_ptr(d_output_shm), SmemLayoutDOutput{});
        auto shared_d_output_transposed = make_tensor(make_smem_ptr(d_output_shm), SmemLayoutDOutputTransposed{});
        shared_ptr = reinterpret_cast<float4*>(d_output_shm);
        global_ptr = reinterpret_cast<float4*>((ElementType *)d_output_ptr + b * head * m * k + h * k + m_tile * kTileM * head * k);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16] = global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16];
        }
        __syncthreads();

        auto mma_thread_d_output_shared_tensor_b_matrix = thr_mma_tn.partition_B(shared_d_output_transposed);
        auto mma_thread_d_output_register_tensor_b_matrix = thr_mma_tn.partition_fragment_B(shared_d_output_transposed);  // (MMA, MMA_M, MMA_K)
        cute::copy(mma_thread_d_output_shared_tensor_b_matrix, mma_thread_d_output_register_tensor_b_matrix);
        __syncthreads();
        auto shared_acc_transposed = make_tensor(make_smem_ptr(acc_shm), SmemLayoutAccTransposed{});
        auto mma_thread_acc_transposed_register_tensor_a_matrix = thr_mma_tn.partition_fragment_A(shared_acc_transposed);
        auto mma_thread_acc_transposed_shared_tensor_a_matrix = thr_mma_tn.partition_A(shared_acc_transposed);
        cute::copy(mma_thread_acc_transposed_shared_tensor_a_matrix, mma_thread_acc_transposed_register_tensor_a_matrix);
        cute::gemm(thr_mma_tn, mma_thread_d_value_register_tensor, 
                                 mma_thread_acc_transposed_register_tensor_a_matrix,
                                 mma_thread_d_output_register_tensor_b_matrix,
                                 mma_thread_d_value_register_tensor);
                                 
        auto mma_thread_d_output_register_tensor_a_matrix = thr_mma_tn.partition_fragment_A(shared_d_output);  // (MMA, MMA_M, MMA_K)
        auto mma_thread_d_output_shared_tensor_a_matrix = thr_mma_tn.partition_A(shared_d_output);
        cute::copy(mma_thread_d_output_shared_tensor_a_matrix, mma_thread_d_output_register_tensor_a_matrix);
        __syncthreads();
        
        ComputeType *d_p_shm = (ComputeType*)(fp16_acc_shm);
        auto shared_d_p = make_tensor(make_smem_ptr(d_p_shm), SmemLayoutAcc{});
        auto mma_thread_d_p_register_tensor = thr_mma_tn.partition_fragment_C(shared_d_p);
        auto mma_thread_d_p_shared_tensor = thr_mma_tn.partition_C(shared_d_p);
        clear(mma_thread_d_p_register_tensor);
        cute::gemm(thr_mma_tn, mma_thread_d_p_register_tensor, 
                                 mma_thread_d_output_register_tensor_a_matrix,
                                 mma_thread_value_register_tensor,
                                 mma_thread_d_p_register_tensor);
                                 
        cute::copy(mma_thread_d_p_register_tensor, mma_thread_d_p_shared_tensor);
        __syncthreads();

        Tensor global_d = make_tensor(make_gmem_ptr((ElementType *)d_ptr + b * head * m + h + m_tile * kTileM * head), 
                            Shape<Int<kTileM>, Int<1>>{},
                            make_stride(head, Int<1>{}));
        
        #pragma unroll
        for (int i = 0; i < kTileM; i+=4) {
            shared_acc(i + idx / 32, idx % 32) *= (shared_d_p(i + idx / 32, idx % 32) - global_d(i + idx / 32, 0)) * scale;
        }

        __syncthreads();
        auto mma_thread_d_s_fp32_shared_tensor_a_matrix = thr_mma_tn.partition_A(shared_acc);
        auto mma_thread_d_s_fp16_register_tensor_a_matrix = thr_mma_tn.partition_fragment_A(shared_acc);
        cute::copy(mma_thread_d_s_fp32_shared_tensor_a_matrix, mma_thread_d_s_fp16_register_tensor_a_matrix);
        Tensor global_d_query = make_tensor(make_gmem_ptr((ElementType *)d_query_ptr + b * head * m * k + h * k + m_tile * kTileM * head * k), 
                            Shape<Int<kTileM>, Int<kTileK>>{},
                            make_stride(head * k, Int<1>{}));
        auto mma_thread_d_query_register_tensor = thr_mma_tn.partition_fragment_C(global_d_query);
        auto mma_thread_d_query_global_tensor = thr_mma_tn.partition_C(global_d_query);
        clear(mma_thread_d_query_register_tensor);
        auto shared_key_transposed = make_tensor(make_smem_ptr(key_shm), SmemLayoutKeyTransposed{});
        shared_ptr = reinterpret_cast<float4*>(key_shm);
        global_ptr = reinterpret_cast<float4*>((ElementType *)key_ptr + b * head * n * k + h * k + col_n * kTileN * head * k);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16] = global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16];
        }
        __syncthreads();
        auto mma_thread_transposed_key_register_tensor = thr_mma_tn.partition_fragment_B(shared_key_transposed);  // (MMA, MMA_N, MMA_K)
        auto mma_thread_transposed_key_shared_tensor = thr_mma_tn.partition_B(shared_key_transposed);
        cute::copy(mma_thread_transposed_key_shared_tensor, mma_thread_transposed_key_register_tensor);
        __syncthreads();
        cute::gemm(thr_mma_tn, mma_thread_d_query_register_tensor, 
                                 mma_thread_d_s_fp16_register_tensor_a_matrix,
                                 mma_thread_transposed_key_register_tensor,
                                 mma_thread_d_query_register_tensor);
        cute::half_t d_query_register_half2[2];
        #pragma unroll
        for (int i = 0; i < size<1>(mma_thread_d_query_register_tensor); i++) {
            #pragma unroll
            for (int j = 0; j < size<2>(mma_thread_d_query_register_tensor); j++) {
                #pragma unroll
                for (int kk = 0; kk < size<0>(mma_thread_d_query_register_tensor); kk+=2) {
                    d_query_register_half2[0] = (cute::half_t)mma_thread_d_query_register_tensor(kk, i, j);
                    d_query_register_half2[1] = (cute::half_t)mma_thread_d_query_register_tensor(kk + 1, i, j);
                    atomicAdd(reinterpret_cast<__half2*>(&mma_thread_d_query_global_tensor(kk, i, j)), *(reinterpret_cast<__half2*>(d_query_register_half2)));
                }
            }
        }
        cute::copy(mma_thread_acc_transposed_shared_tensor_a_matrix, mma_thread_acc_transposed_register_tensor_a_matrix);
        auto shared_query_transposed = make_tensor(make_smem_ptr(query_shm), SmemLayoutQueryTransposed{});
        shared_ptr = reinterpret_cast<float4*>(query_shm);
        global_ptr = reinterpret_cast<float4*>((ElementType *)query_ptr + b * head * m * k + h * k + m_tile * kTileM * head * k);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16] = global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16];
        }
        __syncthreads();
        auto mma_thread_query_register_tensor_b_matrix = thr_mma_tn.partition_fragment_B(shared_query_transposed);  // (MMA, MMA_M, MMA_K)
        auto mma_thread_query_shared_tensor_b_matrix = thr_mma_tn.partition_B(shared_query_transposed);  // (MMA, MMA_M, MMA_K)
        cute::copy(mma_thread_query_shared_tensor_b_matrix, mma_thread_query_register_tensor_b_matrix);
        cute::gemm(thr_mma_tn, mma_thread_d_key_register_tensor, 
                                 mma_thread_acc_transposed_register_tensor_a_matrix,
                                 mma_thread_query_register_tensor_b_matrix,
                                 mma_thread_d_key_register_tensor);
    }
    ElementType *d_key_shm = (ElementType*)shm_data;
    auto shared_d_key = make_tensor(make_smem_ptr(d_key_shm), SmemLayoutKey{});
    auto mma_thread_d_key_shared_tensor = thr_mma_tn.partition_C(shared_d_key);
    cute::copy(mma_thread_d_key_register_tensor, mma_thread_d_key_shared_tensor);
    __syncthreads();
    float4* shared_ptr = reinterpret_cast<float4*>(d_key_shm);
    float4* global_ptr = reinterpret_cast<float4*>((ElementType *)d_key_ptr + b * head * n * k + h * k + col_n * kTileN * head * k);
    for (int i = 0; i < 4; i++) {
        global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16] = shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16];
    }
    __syncthreads();
    ElementType *d_value_shm = (ElementType*)shm_data;
    auto shared_d_value = make_tensor(make_smem_ptr(d_value_shm), SmemLayoutValue{});
    auto mma_thread_d_value_shared_tensor = thr_mma_tn.partition_C(shared_d_value);
    cute::copy(mma_thread_d_value_register_tensor, mma_thread_d_value_shared_tensor);
    __syncthreads();
    shared_ptr = reinterpret_cast<float4*>(d_value_shm);
    global_ptr = reinterpret_cast<float4*>((ElementType *)d_value_ptr + b * head * n * k + h * k + col_n * kTileN * head * k);
    for (int i = 0; i < 4; i++) {
        global_ptr[(idx / 16 + i * 8) * head * 16 + idx % 16] = shared_ptr[(idx / 16 + i * 8) * 16 + idx % 16];
    }
}

namespace config {
    using namespace cute;
    template <typename ElementType_, int kTileM_ = 32, int kTileN_ = 32, int kTileK_ = 128, typename ComputeType_ = ElementType_>
    struct GemmConfig {
        using ElementType = ElementType_;
        using ComputeType = ComputeType_;
        // tile configuration
        static constexpr int kTileM = kTileM_;
        static constexpr int kTileN = kTileN_;
        static constexpr int kTileK = kTileK_;
        
        using SmemLayoutQuery = decltype(make_layout(
            make_shape(Int<kTileM>{}, Int<kTileK>{}),
            make_stride(Int<kTileK>{}, Int<1>{})
        ));

        using SmemLayoutQueryTransposed = decltype(make_layout(
            make_shape(Int<kTileK>{}, Int<kTileM>{}),
            make_stride(Int<1>{}, Int<kTileK>{})
        ));

        using SmemLayoutKey = decltype(make_layout(
            make_shape(Int<kTileN>{}, Int<kTileK>{}),
            make_stride(Int<kTileK>{}, Int<1>{})
        ));

        using SmemLayoutKeyTransposed = decltype(make_layout(
            make_shape(Int<kTileK>{}, Int<kTileN>{}),
            make_stride(Int<1>{}, Int<kTileK>{})
        ));

        using SmemLayoutValue = decltype(make_layout(
            make_shape(Int<kTileN>{}, Int<kTileK>{}),
            make_stride(Int<kTileK>{}, Int<1>{})
        ));

        using SmemLayoutValueTransposed = decltype(make_layout(
            make_shape(Int<kTileK>{}, Int<kTileN>{}),
            make_stride(Int<1>{}, Int<kTileK>{})
        ));

        using SmemLayoutAcc = decltype(make_layout(
            make_shape(Int<kTileM>{}, Int<kTileN>{}),
            make_stride(Int<kTileN>{}, Int<1>{})
        ));

        using SmemLayoutAccTransposed = decltype(make_layout(
            make_shape(Int<kTileN>{}, Int<kTileM>{}),
            make_stride(Int<1>{}, Int<kTileN>{})
        ));

        using SmemLayoutOutput = SmemLayoutQuery;

        using SmemLayoutDOutput = SmemLayoutQuery;

        using SmemLayoutDOutputTransposed = decltype(make_layout(
            make_shape(Int<kTileK>{}, Int<kTileM>{}),
            make_stride(Int<1>{}, Int<kTileK>{})
        ));

        using SmemLayoutMax = decltype(make_layout(
            make_shape(Int<kTileM>{}, Int<1>{}),
            make_stride(Int<1>{}, Int<1>{})
        ));

        using SmemLayoutSum = SmemLayoutMax;

        using mma_tn_op = SM70_8x8x4_F32F16F16F32_TN;
        using mma_tn_traits = MMA_Traits<mma_tn_op>;
        using mma_tn_atom = MMA_Atom<mma_tn_traits>;

        static constexpr int kMmaEURepeatM = 4;
        static constexpr int kMmaEURepeatN = 4;

        static constexpr int kMmaVRepeatM = 1;
        static constexpr int kMmaVRepeatN = 1;

        using MMA_EU_RepeatT = decltype(
            make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}))
        );
        using MMA_V_RepeatT = decltype(
            make_layout(make_shape(Int<kMmaVRepeatM>{}, Int<kMmaVRepeatN>{}))
        );
        using MMA_TN = decltype(make_tiled_mma(mma_tn_atom{}, MMA_EU_RepeatT{}, MMA_V_RepeatT{}));

        // global mem to shared mem copy
        using Global2SharedCopyAtom = Copy_Atom<DefaultCopy, ElementType>;

        using Global2SharedCopyQuery = decltype(
            make_tiled_copy(Global2SharedCopyAtom{}, 
                make_layout(
                    make_shape(Int<1>{}, Int<128>{}),
                    make_stride(Int<128>{}, Int<1>{})
                ),
                make_layout(
                    make_shape(Int<1>{}, Int<1>{})
                )
            )
        );

        using Global2SharedCopyKey = Global2SharedCopyQuery;
        using Global2SharedCopyDOutput = Global2SharedCopyQuery;
        using Global2SharedCopyOutput = Global2SharedCopyQuery;
        using Global2SharedCopyValue = decltype(
            make_tiled_copy(Global2SharedCopyAtom{}, 
                make_layout(
                    make_shape(Int<1>{}, Int<128>{}),
                    make_stride(Int<128>{}, Int<1>{})
                ),
                make_layout(
                    make_shape(Int<1>{}, Int<1>{})
                )
            )
        );
        using Shared2RegisterCopyAtomAcc = Copy_Atom<DefaultCopy, ComputeType>;
        using Shared2RegisterCopyAcc = decltype(
            make_tiled_copy(Shared2RegisterCopyAtomAcc{}, 
                make_layout(
                    make_shape(Int<4>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{})
                ),
                make_layout(
                    make_shape(Int<1>{}, Int<1>{})
                )
            )
        );
        using Shared2RegisterCopyFp16Acc = decltype(
            make_tiled_copy(Shared2RegisterCopyAtomAcc{}, 
                make_layout(
                    make_shape(Int<4>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{})
                ),
                make_layout(
                    make_shape(Int<1>{}, Int<1>{})
                )
            )
        );
        using Register2SharedCopyFp16Acc = Global2SharedCopyQuery;

        // register to global via shared memory
        using MNK = typename MMA_TN::TiledShape_MNK;

        static constexpr int kThreadNum = size(MMA_TN{});
        static constexpr int backward_kShmSize = sizeof(ComputeType) * cosize(SmemLayoutAcc{}) * 2 + 
                            sizeof(ElementType) * cosize(SmemLayoutQuery{}) * 2 + 
                            sizeof(ComputeType) * cosize(SmemLayoutMax{}) * 2;
        static constexpr int shm_size_query_key = cute::max(cute::cosize(SmemLayoutQuery{}), cute::cosize(SmemLayoutKey{})) * sizeof(ElementType);
        static constexpr int shm_size_query_key_acc = cute::max(shm_size_query_key, cute::cosize(SmemLayoutAcc{}) * sizeof(ElementType));
        static constexpr int kShmSizeQKVO = cute::max(shm_size_query_key_acc , cute::cosize(SmemLayoutValue{}) * sizeof(ElementType)) 
                                + cute::cosize(SmemLayoutOutput{}) * sizeof(ElementType);
        static constexpr int forward_kShmSize = kShmSizeQKVO + (cute::cosize(SmemLayoutMax{}) * 2 + cute::cosize(SmemLayoutSum{})) * sizeof(ComputeType);
    };
}

void fused_mha_forward(const void *query_ptr, const void *key_ptr, const void *value_ptr, void *output_ptr, void *max_ptr, void *sum_ptr, 
    int batch, int head, int m, int n, int k, float scale, bool causal, cudaStream_t stream)
{
    config::GemmConfig<cute::half_t, 32, 32, 128, float> gemm_config;
    // print(typename decltype(gemm_config)::MMA{});
    dim3 block = gemm_config.kThreadNum;
    dim3 grid((m + gemm_config.kTileM - 1) / gemm_config.kTileM, head * batch);
    int shm_size = gemm_config.forward_kShmSize;
    fused_mha_forward_kernel<decltype(gemm_config)>
            <<<grid, block, shm_size, stream>>>(query_ptr, key_ptr, value_ptr, 
            output_ptr, max_ptr, sum_ptr, head, m, n, k, scale, causal);
}

void fused_mha_backward(const void *query_ptr, const void *key_ptr, const void *value_ptr, 
            void *output_ptr, void *d_output_ptr, void *d_ptr, void *max_ptr, void *sum_ptr, 
            void *d_query_ptr, void *d_key_ptr, void *d_value_ptr, int batch, int head, int m, int n, int k, float scale, bool causal, cudaStream_t stream)
{
    config::GemmConfig<cute::half_t, 32, 32, 128, float> gemm_config;
    // print(typename decltype(gemm_config)::MMA{});
    dim3 block = gemm_config.kThreadNum;
    dim3 grid((m + gemm_config.kTileM - 1) / gemm_config.kTileM, head * batch);
    int shm_size = gemm_config.backward_kShmSize;
    fused_mha_backward_kernel<decltype(gemm_config)>
            <<<grid, block, shm_size, stream>>>(query_ptr, key_ptr, value_ptr, output_ptr, d_output_ptr, d_ptr,
                max_ptr, sum_ptr, d_query_ptr, d_key_ptr, d_value_ptr, head, m, n, k, scale, causal);
}