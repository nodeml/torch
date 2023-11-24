#include <napi.h>
#include <addon/Tensor.h>
#include <addon/utils.h>
#include <addon/types.h>
#include <addon/aten.h>
#include <addon/nn/nn.h>
#include <addon/jit/jit.h>
#include <addon/vision/vision.h>
#include <addon/cuda/cuda.h>

Napi::Object InitModule(Napi::Env env, Napi::Object exports)
{
    nodeml_torch::Tensor::Init(env, exports);
    nodeml_torch::utils::Init(env, exports);
    nodeml_torch::types::Init(env, exports);
    nodeml_torch::aten::Init(env, exports);
    nodeml_torch::nn::Init(env, exports);
    nodeml_torch::jit::Init(env, exports);
    nodeml_torch::vision::Init(env, exports);
    nodeml_torch::cuda::Init(env,exports);
    return exports;
}

NODE_API_MODULE(nodeml_torch, InitModule);