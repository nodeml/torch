#pragma once

#include <napi.h>
#include <torch/torch.h>
namespace nodeml_torch
{
    namespace nn
    {
        Napi::Object Init(Napi::Env env, Napi::Object exports);
    }
}