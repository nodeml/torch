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
                    auto options = torch::nn::functional::InterpolateFuncOptions();
                    options.size(utils::napiArrayToVector<int64_t>(info[1].As<Napi::Array>()));
                    auto mode = info[2].ToString().Utf8Value();

                    //'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
                    if (mode == "nearest")
                    {
                        options.mode(torch::kNearest);
                    }
                    else if (mode == "linear")
                    {
                        options.mode(torch::kLinear);
                    }
                    else if (mode == "bilinear")
                    {
                        options.mode(torch::kBilinear);
                    }
                    else if (mode == "bicubic")
                    {
                        options.mode(torch::kBicubic);
                    }
                    else if (mode == "trilinear")
                    {
                        options.mode(torch::kTrilinear);
                    }
                    else if (mode == "area")
                    {
                        options.mode(torch::kArea);
                    }
                    else if (mode == "nearest-exact")
                    {
                        options.mode(torch::kNearestExact);
                    }

                    if (info.Length() >= 4 && info[3].IsObject())
                    {
                        auto extraOptions = info[3].ToObject();
                        if (extraOptions.Has("alignCorners"))
                        {
                            options.align_corners(extraOptions.Get("alignCorners").ToBoolean().Value());
                        }

                        if (extraOptions.Has("antiAlias"))
                        {
                            options.antialias(extraOptions.Get("antiAlias").ToBoolean().Value());
                        }
                    }
                    return nodeml_torch::Tensor::FromTorchTensor(env,
                                                                 torch::nn::functional::interpolate(tensor->torchTensor, options));
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
