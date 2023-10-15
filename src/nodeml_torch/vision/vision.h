#pragma once

#include <napi.h>

namespace nodeml_torch
{
    namespace vision
    {

        Napi::Object Init(Napi::Env env, Napi::Object exports);
    }
}
