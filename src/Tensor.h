#pragma once

#include <napi.h>

#include <memory>
#include <torch/torch.h>
#include <string>

namespace nodeml_torch
{

    class Tensor : public Napi::ObjectWrap<Tensor>
    {

    public:
        static Napi::FunctionReference constructor;

        torch::Tensor tensor;

        static Napi::Object Init(Napi::Env env, Napi::Object exports);

        Tensor(const Napi::CallbackInfo &info);

        static Napi::Object FromTorchTensor(const Napi::CallbackInfo &info, const torch::Tensor &torchTensor);

        static Napi::Value FromTypedArray(const Napi::CallbackInfo &info);

        Napi::Value Shape(const Napi::CallbackInfo &info);

        Napi::Value ToArray(const Napi::CallbackInfo &info);

        Napi::Value Reshape(const Napi::CallbackInfo &info);

        Napi::Value Transpose(const Napi::CallbackInfo &info);

        Napi::Value Permute(const Napi::CallbackInfo &info);

        Napi::Value Unsqueeze(const Napi::CallbackInfo &info);

        Napi::Value Squeeze(const Napi::CallbackInfo &info);

        Napi::Value Slice(const Napi::CallbackInfo &info);

        Napi::Value Type(const Napi::CallbackInfo &info);

        Napi::Value DType(const Napi::CallbackInfo &info);

        static Napi::Function GetClass(Napi::Env env);

        Napi::Value toString(const Napi::CallbackInfo &info);
    };

}