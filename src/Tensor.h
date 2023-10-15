#pragma once

#include <napi.h>

#include <memory>
#include <string>
#include <torch/torch.h>

namespace nodeml_torch
{

    class Tensor : public Napi::ObjectWrap<Tensor>
    {

    public:
        static Napi::FunctionReference constructor;

        torch::Tensor torchTensor;

        static Napi::Object Init(Napi::Env env, Napi::Object exports);

        static bool IsInstance(Napi::Object &obj);

        Tensor(const Napi::CallbackInfo &info);

        static Napi::Object FromTorchTensor(Napi::Env env, const torch::Tensor &targetTorchTensor);

        static Tensor *FromObject(Napi::Value value);

        static Napi::Value FromTypedArray(const Napi::CallbackInfo &info);

        Napi::Value Shape(const Napi::CallbackInfo &info);

        Napi::Value ToArray(const Napi::CallbackInfo &info);

        Napi::Value Reshape(const Napi::CallbackInfo &info);

        Napi::Value Transpose(const Napi::CallbackInfo &info);

        Napi::Value Permute(const Napi::CallbackInfo &info);

        Napi::Value Unsqueeze(const Napi::CallbackInfo &info);

        Napi::Value Squeeze(const Napi::CallbackInfo &info);

        Napi::Value Type(const Napi::CallbackInfo &info);

        Napi::Value DType(const Napi::CallbackInfo &info);

        Napi::Value Clone(const Napi::CallbackInfo &info);

        Napi::Value Add(const Napi::CallbackInfo &info);

        Napi::Value Sub(const Napi::CallbackInfo &info);

        Napi::Value Mul(const Napi::CallbackInfo &info);

        Napi::Value Div(const Napi::CallbackInfo &info);

        Napi::Value Index(const Napi::CallbackInfo &info);

        Napi::Value IndexPut(const Napi::CallbackInfo &info);

        Napi::Value MatMul(const Napi::CallbackInfo &info);

        Napi::Value AMax(const Napi::CallbackInfo &info);

        Napi::Value Split(const Napi::CallbackInfo &info);

        Napi::Value Argsort(const Napi::CallbackInfo &info);

        Napi::Value Max(const Napi::CallbackInfo &info);

        static Napi::Function GetClass(Napi::Env env);

        Napi::Value toString(const Napi::CallbackInfo &info);
    };

}