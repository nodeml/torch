#pragma once

#include <napi.h>
#include <torch/torch.h>
namespace nodeml_torch
{
    namespace utils
    {
        template <typename T>
        std::vector<T> napiArrayToVector(const Napi::Array &arr);

        template <typename T>
        std::vector<T> vectorToNapiArray(const Napi::CallbackInfo &info);

        template <typename T>
        torch::ScalarType scalarType();

        torch::ScalarType stringToScalarType(std::string typeString);

        template <typename T>
        Napi::Array tensorToNestedArray(Napi::Env env, torch::Tensor &tensor, const std::function<T(Napi::Env, Napi::Number)> &convertNumber);

        Napi::Object Init(Napi::Env env, Napi::Object exports);
    }
}
