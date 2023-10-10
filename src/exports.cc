#include <napi.h>
#include "Tensor.h"
#include "utils.h"
#include "types.h"
#include "aten.h"

Napi::Object InitModule(Napi::Env env, Napi::Object exports)
{
    nodeml_torch::Tensor::Init(env, exports);
    nodeml_torch::utils::Init(env, exports);
    nodeml_torch::types::Init(env, exports);
    nodeml_torch::aten::Init(env, exports);
    return exports;
}

NODE_API_MODULE(nodeml_torch, InitModule);