#include <nodeml_torch/nn/functional.h>
#include <nodeml_torch/Tensor.h>
#include <nodeml_torch/utils.h>
#include <torch/torch.h>
#include "functional.h"

namespace nodeml_torch
{
    namespace nn
    {
        namespace functional
        {
            Napi::Value interpolate(const Napi::CallbackInfo &info)
            {

                auto env = info.Env();
                try
                {
                    auto tensor = nodeml_torch::Tensor::FromObject(info[0]);

                    return nodeml_torch::Tensor::FromTorchTensor(env,
                                                                 torch::nn::functional::interpolate(tensor->torchTensor,
                                                                                                    torch::nn::functional::InterpolateFuncOptions()
                                                                                                        .size(utils::napiArrayToVector<int64_t>(info[1].As<Napi::Array>()))
                                                                                                        .mode(torch::kNearest)));
                }
                catch (const std::exception &e)
                {
                    throw Napi::Error::New(env, e.what());
                }
            }

            Napi::Value pad(const Napi::CallbackInfo &info)
            {
                auto env = info.Env();
                try
                {
                    auto tensor = nodeml_torch::Tensor::FromObject(info[0]);
                    auto options = torch::nn::functional::PadFuncOptions(utils::napiArrayToVector<int64_t>(info[1].As<Napi::Array>()));

                    return nodeml_torch::Tensor::FromTorchTensor(env,
                                                                 torch::nn::functional::pad(tensor->torchTensor, options));
                }
                catch (const std::exception &e)
                {
                    throw Napi::Error::New(env, e.what());
                }
            }
            Napi::Object Init(Napi::Env env)
            {
                auto exports = Napi::Object::New(env);

                exports.Set("interpolate", Napi::Function::New(env, interpolate));
                exports.Set("pad", Napi::Function::New(env, pad));
                return exports;
            }
        }
    }
}
