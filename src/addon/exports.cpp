#include <napi.h>
#include <addon/Tensor.hpp>
#include <addon/utils.hpp>
#include <addon/types.hpp>
#include <addon/aten.hpp>
#include <addon/nn/nn.hpp>
#include <addon/jit/jit.hpp>
#include <addon/vision/vision.hpp>
#include <addon/cuda/cuda.hpp>

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