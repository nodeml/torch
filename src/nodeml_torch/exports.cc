#include <napi.h>
#include <nodeml_torch/Tensor.h>
#include <nodeml_torch/utils.h>
#include <nodeml_torch/types.h>
#include <nodeml_torch/aten.h>
#include <nodeml_torch/nn/nn.h>
#include <nodeml_torch/jit/jit.h>

Napi::Object InitModule(Napi::Env env, Napi::Object exports)
{
    nodeml_torch::Tensor::Init(env, exports);
    nodeml_torch::utils::Init(env, exports);
    nodeml_torch::types::Init(env, exports);
    nodeml_torch::aten::Init(env, exports);
    nodeml_torch::nn::Init(env, exports);
    nodeml_torch::jit::Init(env, exports);
    return exports;
}

NODE_API_MODULE(nodeml_torch, InitModule);