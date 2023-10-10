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

                return Tensor::FromTorchTensor(info, torch::rand(shape, options));
            }
            else
            {
                throw Napi::Error::New(env, "Tensor shape is required");
            }
            return Napi::Value();
        }

        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            exports.Set("rand", Napi::Function::New(env, randTensor));
            return exports;
        }
    }
}
