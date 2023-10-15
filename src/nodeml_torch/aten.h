#pragma once

#include <napi.h>
#include <torch/torch.h>

namespace nodeml_torch
{
    namespace aten
    {
        Napi::Value rand(const Napi::CallbackInfo &info);

        Napi::Value arange(const Napi::CallbackInfo &info);

        Napi::Value greater(const Napi::CallbackInfo &info);

        Napi::Value greaterEqual(const Napi::CallbackInfo &info);

        Napi::Value less(const Napi::CallbackInfo &info);

        Napi::Value lessEqual(const Napi::CallbackInfo &info);

        Napi::Value equal(const Napi::CallbackInfo &info);

        Napi::Value zeros(const Napi::CallbackInfo &info);

        Napi::Value cat(const Napi::CallbackInfo &info);

        Napi::Value where(const Napi::CallbackInfo &info);

        Napi::Object Init(Napi::Env env, Napi::Object exports);
    }
}