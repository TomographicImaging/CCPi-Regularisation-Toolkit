/*shared macros*/
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return (float)(x*x);
        }
};



/*checks CUDA call, should be used in functions returning <int> value
if error happens, writes to standard error and explicitly returns -1*/
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        return -1;                                                             \
    }                                                                          \
}

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        return -1;                                                                \
    }                                                                          \
}
/*#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        return;
    }
}
*/

