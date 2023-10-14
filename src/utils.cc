#include "utils.h"
#include "types.h"

namespace nodeml_torch {
namespace utils {
    template <> std::vector<std::int64_t> napiArrayToVector(const Napi::Array& arr, int stopIndex)
    {
        std::vector<std::int64_t> result;

        auto stopAt = stopIndex == -1 ? arr.Length() : stopIndex;

        for (int i = 0; i < stopAt; i++) {
            result.push_back(arr.Get(i).As<Napi::Number>().Int64Value());
        }

        return result;
    }

    template <> std::vector<float> napiArrayToVector(const Napi::Array& arr, int stopIndex)
    {
        std::vector<float> result;

        auto stopAt = stopIndex == -1 ? arr.Length() : stopIndex;

        for (int i = 0; i < stopAt; i++) {
            result.push_back(arr.Get(i).As<Napi::Number>().FloatValue());
        }

        return result;
    }

    template <> std::vector<double> napiArrayToVector(const Napi::Array& arr, int stopIndex)
    {
        std::vector<double> result;

        auto stopAt = stopIndex == -1 ? arr.Length() : stopIndex;

        for (int i = 0; i < stopAt; i++) {
            result.push_back(arr.Get(i).As<Napi::Number>().DoubleValue());
        }

        return result;
    }

    template <typename T> std::vector<T> vectorToNapiArray(const Napi::CallbackInfo& info)
    {
        return std::vector<T>;
    }

    template <> torch::ScalarType scalarType<float>() { return torch::kFloat32; }

    template <> torch::ScalarType scalarType<double>() { return torch::kFloat64; }
    template <> torch::ScalarType scalarType<int32_t>() { return torch::kInt32; }
    template <> torch::ScalarType scalarType<int64_t>() { return torch::kInt64; }
    template <> torch::ScalarType scalarType<uint8_t>() { return torch::kUInt8; }

    template <typename T>
    Napi::Array tensorToNestedArray(Napi::Env env, torch::Tensor& tensor,
        const std::function<T(Napi::Env, Napi::Number)>& convertNumber)
    {
        std::vector<int64_t> dims = tensor.sizes();
        std::vector<int64_t> counter;

        for (auto x : dims) {
            counter.push_back(0);
        }

        auto result = Napi::Array::New(env, dims.at(0));

        while (counter.at(0) != dims.at(0) - 1) { }
        return Napi::Array();
    }

    torch::ScalarType stringToScalarType(std::string typeString)
    {
        if (typeString == types::torchFloatType) {
            return torch::kFloat32;
        } else if (typeString == types::torchDoubleType) {
            return torch::kFloat64;
        } else if (typeString == types::torchInt32Type) {
            return torch::kInt32;
        } else if (typeString == types::torchLongType) {
            return torch::kInt64;
        } else if (typeString == types::torchUint8Type) {
            return torch::kUInt8;
        }

        return torch::kFloat32;
    }

    bool isNapiValueInt(Napi::Env& env, Napi::Value& num)
    {
        return env.Global()
            .Get("Number")
            .ToObject()
            .Get("isInteger")
            .As<Napi::Function>()
            .Call({ num })
            .ToBoolean()
            .Value();
    }

    c10::optional<c10::SymInt> intIndexOrNone(const Napi::Value& value)
    {
        if (value.IsNull()) {
            return torch::indexing::None;
        }

        return value.ToNumber().Int64Value();
    }

    torch::indexing::TensorIndex napiValueToTorchIndex(Napi::Env& env, const Napi::Value& value)
    {
        // https: // pytorch.org/cppdocs/notes/tensor_indexing.html#setter

        if (value.IsNull()) {
            return torch::indexing::None;
        }

        if (value.IsString() && value.ToString().Utf8Value() == "...") {
            return torch::indexing::Ellipsis;
        }

        if (value.IsArray()) {
            auto asArray = value.As<Napi::Array>();

            if (asArray.Length() == 0) {
                return torch::indexing::Slice();
            } else if (asArray.Length() == 2) {
                return torch::indexing::Slice(intIndexOrNone(asArray.Get(uint32_t(0))),
                    intIndexOrNone(asArray.Get(uint32_t(1))));
            } else if (asArray.Length() == 3) {
                return torch::indexing::Slice(intIndexOrNone(asArray.Get(uint32_t(0))),
                    intIndexOrNone(asArray.Get(uint32_t(1))),
                    intIndexOrNone(asArray.Get(uint32_t(2))));
            }
        }

        if (value.IsBoolean()) {
            return value.ToBoolean().Value();
        }

        if (value.IsNumber()) {
            return value.ToNumber().Int32Value();
        }

        // if (value.IsArray()) {
        //     return torch::Tensor(napiArrayToVector<int>(value.As<Napi::Array>()));
        // }
        throw Napi::Error::New(env, "Failed To Parse Tensor Index");
    }

    std::vector<torch::indexing::TensorIndex> napiArrayToTorchIndex(const Napi::Array& arr)
    {
        return std::vector<torch::indexing::TensorIndex>();
    }

    Napi::Object Init(Napi::Env env, Napi::Object exports) { return exports; }
}
}