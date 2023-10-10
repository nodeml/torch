#include "utils.h"
#include "types.h"

namespace nodeml_torch
{
    namespace utils
    {
        template <>
        std::vector<std::int64_t> napiArrayToVector(const Napi::Array &arr)
        {
            std::vector<std::int64_t> result;

            for (int i = 0; i < arr.Length(); i++)
            {
                result.push_back(arr.Get(i).As<Napi::Number>().Int64Value());
            }

            return result;
        }

        template <>
        std::vector<float> napiArrayToVector(const Napi::Array &arr)
        {
            std::vector<float> result;

            for (int i = 0; i < arr.Length(); i++)
            {
                result.push_back(arr.Get(i).As<Napi::Number>().FloatValue());
            }

            return result;
        }

        template <>
        std::vector<double> napiArrayToVector(const Napi::Array &arr)
        {
            std::vector<double> result;

            for (int i = 0; i < arr.Length(); i++)
            {
                result.push_back(arr.Get(i).As<Napi::Number>().DoubleValue());
            }

            return result;
        }

        template <typename T>
        std::vector<T> vectorToNapiArray(const Napi::CallbackInfo &info)
        {
            return std::vector<T>;
        }

        template <>
        torch::ScalarType scalarType<float>() { return torch::kFloat32; }

        template <>
        torch::ScalarType scalarType<double>() { return torch::kFloat64; }
        template <>
        torch::ScalarType scalarType<int32_t>() { return torch::kInt32; }
        template <>
        torch::ScalarType scalarType<int64_t>() { return torch::kInt64; }
        template <>
        torch::ScalarType scalarType<uint8_t>() { return torch::kUInt8; }

        template <typename T>
        Napi::Array tensorToNestedArray(Napi::Env env, torch::Tensor &tensor, const std::function<T(Napi::Env, Napi::Number)> &convertNumber)
        {
            std::vector<int64_t> dims = tensor.sizes();
            std::vector<int64_t> counter;

            for (auto x : dims)
            {
                counter.push_back(0);
            }

            auto result = Napi::Array::New(env, dims.at(0));

            while (counter.at(0) != dims.at(0) - 1)
            {
                        }
            return Napi::Array();
        }

        torch::ScalarType stringToScalarType(std::string typeString)
        {
            if (typeString == types::torchFloatType)
            {
                return torch::kFloat32;
            }
            else if (typeString == types::torchDoubleType)
            {
                return torch::kFloat64;
            }
            else if (typeString == types::torchInt32Type)
            {
                return torch::kInt32;
            }
            else if (typeString == types::torchLongType)
            {
                return torch::kInt64;
            }
            else if (typeString == types::torchUint8Type)
            {
                return torch::kUInt8;
            }

            return torch::kFloat32;
        }

        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            return exports;
        }
    }
}