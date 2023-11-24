#pragma once

#include <napi.h>

namespace nodeml_torch
{
    namespace cuda
    {

        Napi::Value deviceCount(const Napi::CallbackInfo &info);

        Napi::Value isAvailable(const Napi::CallbackInfo &info);

        Napi::Object Init(Napi::Env env, Napi::Object exports);
    }
}
