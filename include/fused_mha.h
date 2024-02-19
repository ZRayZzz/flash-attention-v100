void fused_mha_forward(const void *query_ptr, const void *key_ptr, const void *value_ptr, void *output_ptr, void *max_ptr, void *sum_ptr, 
    int batch, int head, int m, int n, int k, float scale, bool causal, cudaStream_t stream);

void fused_mha_backward(const void *query_ptr, const void *key_ptr, const void *value_ptr, 
            void *output_ptr, void *d_output_ptr, void *d_ptr, void *max_ptr, void *sum_ptr, 
            void *d_query_ptr, void *d_key_ptr, void *d_value_ptr, int batch, int head, int m, int n, int k, float scale, bool causal, cudaStream_t stream);