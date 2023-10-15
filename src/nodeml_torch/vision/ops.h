#pragma once

#include <napi.h>

namespace nodeml_torch
{
    namespace vision
    {
        namespace ops
        {

            Napi::Value nms(const Napi::CallbackInfo &info);

            Napi::Object Init(Napi::Env env, Napi::Object exports);
        }
    }
}
