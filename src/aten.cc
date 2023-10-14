#include "aten.h"

#include "tensor.h"
#include "utils.h"

namespace nodeml_torch
{
    namespace aten
    {

        using namespace nodeml_torch::utils;

        Napi::Value rand(const Napi::CallbackInfo &info)
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

        Napi::Value greater(const Napi::CallbackInfo &info)
        {
            try
            {
                auto a = Tensor::FromObject(info[0]);
                auto b = Tensor::FromObject(info[1]);

                return Tensor::FromTorchTensor(info.Env(), a->torchTensor > b->torchTensor);
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
        }

        Napi::Value greaterEqual(const Napi::CallbackInfo &info)
        {
            try
            {
                auto a = Tensor::FromObject(info[0]);
                auto b = Tensor::FromObject(info[1]);

                return Tensor::FromTorchTensor(info.Env(), a->torchTensor >= b->torchTensor);
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
        }

        Napi::Value less(const Napi::CallbackInfo &info)
        {
            try
            {
                auto a = Tensor::FromObject(info[0]);
                auto b = Tensor::FromObject(info[1]);

                return Tensor::FromTorchTensor(info.Env(), a->torchTensor < b->torchTensor);
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
        }

        Napi::Value lessEqual(const Napi::CallbackInfo &info)
        {
            try
            {
                auto a = Tensor::FromObject(info[0]);
                auto b = Tensor::FromObject(info[1]);

                return Tensor::FromTorchTensor(info.Env(), a->torchTensor <= b->torchTensor);
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
        }

        Napi::Value equal(const Napi::CallbackInfo &info)
        {
            try
            {
                auto a = Tensor::FromObject(info[0]);
                auto b = Tensor::FromObject(info[1]);

                return Tensor::FromTorchTensor(info.Env(), a->torchTensor == b->torchTensor);
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
        }

        Napi::Value zeros(const Napi::CallbackInfo &info)
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

                return Tensor::FromTorchTensor(env, torch::zeros(shape, options));
            }
            else
            {
                throw Napi::Error::New(env, "Tensor shape is required");
            }
        }

        Napi::Value cat(const Napi::CallbackInfo &info)
        {
            try
            {
                auto env = info.Env();

                auto tensors = utils::napiArrayToVector<torch::Tensor>(info[0].As<Napi::Array>());

                return Tensor::FromTorchTensor(env, torch::cat(tensors, info.Length() >= 2 ? info[1].ToNumber().Int64Value() : 0i64));
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
        }

        Napi::Value where(const Napi::CallbackInfo &info)
        {

            try
            {
                auto env = info.Env();

                auto result = torch::where(Tensor::FromObject(info[0])->torchTensor);
                auto arr = Napi::Array::New(env, result.size());

                for (auto i = 0; i < result.size(); i++)
                {
                    arr.Set(uint32_t(i), Tensor::FromTorchTensor(env, result.at(i)));
                }

                return arr;
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
        }

        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            exports.Set("rand", Napi::Function::New(env, rand));
            exports.Set("arange", Napi::Function::New(env, arange));
            exports.Set("greater", Napi::Function::New(env, greater));
            exports.Set("greaterEqual", Napi::Function::New(env, greaterEqual));
            exports.Set("less", Napi::Function::New(env, less));
            exports.Set("lessEqual", Napi::Function::New(env, lessEqual));
            exports.Set("equal", Napi::Function::New(env, equal));
            exports.Set("zeros", Napi::Function::New(env, zeros));
            exports.Set("cat", Napi::Function::New(env, cat));
            exports.Set("where", Napi::Function::New(env, where));
            return exports;
        }
    }
}
