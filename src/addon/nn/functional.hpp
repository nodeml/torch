#pragma once

#include <napi.h>
#include <torch/torch.h>
namespace nodeml_torch
{
    namespace nn
    {
        namespace functional
        {

            Napi::Value interpolate(const Napi::CallbackInfo &info);

            Napi::Value pad(const Napi::CallbackInfo &info);

            Napi::Object Init(Napi::Env env);
        }
    }
}