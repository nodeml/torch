#pragma once

#include <napi.h>
#include <torch/torch.h>

namespace nodeml_torch
{
    namespace aten
    {
        Napi::Value randTensor(const Napi::CallbackInfo &info);

        Napi::Value arange(const Napi::CallbackInfo &info);

        Napi::Object Init(Napi::Env env, Napi::Object exports);
    }
}