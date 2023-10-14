#pragma once

#include <napi.h>

#include <memory>
#include <string>
#include <torch/torch.h>

namespace nodeml_torch
{
    namespace jit
    {
        Napi::Value load(const Napi::CallbackInfo &info);
        Napi::Object Init(Napi::Env env, Napi::Object exports);
    }
}