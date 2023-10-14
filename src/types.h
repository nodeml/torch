#pragma once

#include <napi.h>
#include <string>

namespace nodeml_torch
{
    namespace types
    {
        static const std::string torchFloatType = "float";
        static const std::string torchInt32Type = "int32";
        static const std::string torchDoubleType = "double";
        static const std::string torchUint8Type = "uint8";
        static const std::string torchLongType = "long";
        static const std::string torchBooleanType = "bool";

        Napi::Object Init(Napi::Env env, Napi::Object exports);
    }
}