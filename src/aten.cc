#include "aten.h"

#include "tensor.h"
#include "utils.h"

namespace nodeml_torch
{
    namespace aten
    {

        using namespace nodeml_torch::utils;

        Napi::Value randTensor(const Napi::CallbackInfo &info)
        {
            auto env = info.Env();
            if (info.Length() >= 1 && info[0].IsArray())
            {
                auto shape = utils::napiArrayToVector<std::int64_t>(info[0].As<Napi::Array>());
                c10::TensorOptions options;
                if (info.Length() >= 2 && info[1].IsString())
                {
                    auto type = info[1].As<Napi::String>().Utf8Value();
                    auto scalarType = utils::stringToScalarType(type);
                    options.dtype(scalarType);
                }
                else
                {
                    options.dtype(torch::ScalarType::Float);
                }

                return Tensor::FromTorchTensor(env, torch::rand(shape, options));
            }
            else
            {
                throw Napi::Error::New(env, "Tensor shape is required");
            }
            return Napi::Value();
        }

        Napi::Value arange(const Napi::CallbackInfo &info)
        {
            auto env = info.Env();
            auto dtypeIndex = info.Length() - 1;

            if (dtypeIndex <= 0)
            {
                throw Napi::Error::New(env, "Missing Arguments");
            }

            c10::ScalarType dtype;

            if (!info[dtypeIndex].IsNumber())
            {
                auto type = info[dtypeIndex].As<Napi::String>().Utf8Value();
                dtype = utils::stringToScalarType(type);
            }
            else
            {
                dtype = torch::ScalarType::Float;
                dtypeIndex++;
            }

            if (dtypeIndex == 1)
            {
                return Tensor::FromTorchTensor(env, torch::arange(info[0].ToNumber().FloatValue()).toType(dtype));
            }
            else if (dtypeIndex == 2)
            {
                return Tensor::FromTorchTensor(
                    env, torch::arange(info[0].ToNumber().FloatValue(), info[1].ToNumber().FloatValue()).toType(dtype));
            }
            else if (dtypeIndex >= 3)
            {
                return Tensor::FromTorchTensor(
                    env, torch::arange(info[0].ToNumber().FloatValue(), info[1].ToNumber().FloatValue(), info[2].ToNumber().FloatValue()).toType(dtype));
            }

            return Napi::Value();
        }

        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            exports.Set("rand", Napi::Function::New(env, randTensor));
            exports.Set("arange", Napi::Function::New(env, arange));
            return exports;
        }
    }
}
