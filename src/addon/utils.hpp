#pragma once

#include <napi.h>
#include <torch/torch.h>
namespace nodeml_torch
{
    namespace utils
    {
        
        template <typename T>
        std::vector<T> napiArrayToVector(const Napi::Array &arr, int stopIndex = -1);

        template <typename T>
        std::vector<T> napiArrayToVector(const Napi::Array &arr, std::function<T(Napi::Value)> converter, int stopIndex = -1);

        template <typename T>
        Napi::Array vectorToNapiArray(Napi::Env env,std::vector<T> vec);

        template <typename T>
        torch::ScalarType scalarType();

        torch::ScalarType stringToScalarType(std::string typeString);

        template <typename T>
        Napi::Array tensorToNestedArray(Napi::Env env, torch::Tensor &tensor,
                                        const std::function<T(Napi::Env, Napi::Number)> &convertNumber);

        // https://github.com/nodejs/node-addon-api/issues/265#issuecomment-552145007
        bool isNapiValueInt(Napi::Env env, Napi::Value num);

        c10::optional<c10::SymInt> intIndexOrNone(const Napi::Value &value);

        torch::indexing::TensorIndex napiValueToTorchIndex(Napi::Env env, const Napi::Value &value);

        Napi::Object Init(Napi::Env env, Napi::Object exports);
    }
}
