#include <addon/Tensor.hpp>
#include <addon/types.hpp>
#include <addon/utils.hpp>
#include <exception>
#include <iostream>

namespace nodeml_torch
{
    using namespace Napi;
    using namespace nodeml_torch::utils;
    using namespace nodeml_torch::types;

    Napi::FunctionReference Tensor::constructor;

    template <typename T>
    Napi::Value tensorToArray(Napi::Env env, const torch::Tensor &torchTensor, std::function<Napi::Value(Napi::Env, T)> converter = nullptr)
    {
        if (converter == nullptr)
        {
            Napi::EscapableHandleScope scope(env);
            assert(torchTensor.is_contiguous());
            auto typed_array = Napi::TypedArrayOf<T>::New(env, torchTensor.numel());
            memcpy(typed_array.Data(), torchTensor.data_ptr<T>(), sizeof(T) * torchTensor.numel());
            return scope.Escape(typed_array);
        }
        else
        {
            auto arr = Napi::Array::New(env, torchTensor.numel());

            assert(torchTensor.is_contiguous());

            T *ptr = (T *)torchTensor.data_ptr();

            for (auto i = 0; i < torchTensor.numel(); i++)
            {
                arr.Set(uint32_t(i), converter(env, *ptr++));
            }

            return arr;
        }
    }

    template <typename T>
    torch::Tensor arrayToTensor(
        Napi::Env env, const Napi::TypedArray &data, const Napi::Array &shape_array)
    {
        auto *data_ptr = data.As<Napi::TypedArrayOf<T>>().Data();
        auto shape = napiArrayToVector<std::int64_t>(shape_array);
        torch::TensorOptions options(scalarType<T>());
        auto torch_tensor = torch::empty(shape, options);
        memcpy(torch_tensor.data_ptr<T>(), data_ptr, sizeof(T) * torch_tensor.numel());
        return torch_tensor;
    }

    torch::Tensor typedArrayToTensor(
        Napi::Env env, const Napi::TypedArray &data, const Napi::Array &shape)
    {

        auto arrayType = data.TypedArrayType();

        switch (arrayType)
        {
        case napi_float32_array:
            return arrayToTensor<float>(env, data, shape);
        case napi_float64_array:
            return arrayToTensor<double>(env, data, shape);
        case napi_int32_array:
            return arrayToTensor<int32_t>(env, data, shape);
        case napi_uint8_array:
            return arrayToTensor<uint8_t>(env, data, shape);
        default:
            throw Napi::TypeError::New(env, "Unsupported type");
        }
    }

    Napi::Object Tensor::Init(Napi::Env env, Napi::Object exports)
    {
        auto func = DefineClass(env, "Tensor",
                                {Tensor::InstanceMethod("toArray", &Tensor::ToArray),
                                 Tensor::InstanceAccessor("shape", &Tensor::Shape, nullptr),
                                 Tensor::InstanceMethod("reshape", &Tensor::Reshape),
                                 Tensor::InstanceMethod("toString", &Tensor::toString),
                                 Tensor::StaticMethod("fromTypedArray", &Tensor::FromTypedArray),
                                 Tensor::InstanceMethod("type", &Tensor::Type),
                                 Tensor::InstanceMethod("transpose", &Tensor::Transpose),
                                 Tensor::InstanceAccessor("dtype", &Tensor::DType, nullptr),
                                 Tensor::InstanceMethod("squeeze", &Tensor::Squeeze),
                                 Tensor::InstanceMethod("unsqueeze", &Tensor::Unsqueeze),
                                 Tensor::InstanceMethod("add", &Tensor::Add),
                                 Tensor::InstanceMethod("sub", &Tensor::Sub),
                                 Tensor::InstanceMethod("mul", &Tensor::Mul),
                                 Tensor::InstanceMethod("div", &Tensor::Div),
                                 Tensor::InstanceMethod("get", &Tensor::Index),
                                 Tensor::InstanceMethod("set", &Tensor::IndexPut),
                                 Tensor::InstanceMethod("clone", &Tensor::Clone),
                                 Tensor::InstanceMethod("matmul", &Tensor::MatMul),
                                 Tensor::InstanceMethod("amax", &Tensor::AMax),
                                 Tensor::InstanceMethod("split", &Tensor::Split),
                                 Tensor::InstanceMethod("argsort", &Tensor::Argsort),
                                 Tensor::InstanceMethod("view", &Tensor::View),
                                 Tensor::InstanceMethod("any", &Tensor::Any),
                                 Tensor::InstanceMethod("max", &Tensor::Max),
                                 Tensor::InstanceMethod("clamp", &Tensor::Clamp),
                                 Tensor::InstanceMethod("sigmoid", &Tensor::Sigmoid), Tensor::InstanceMethod("cpu", &Tensor::Cpu),
                                 Tensor::InstanceMethod("cuda", &Tensor::Cuda), Tensor::InstanceMethod("detach", &Tensor::Detach), Tensor::InstanceMethod("backward", &Tensor::Backward)});

        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();
        exports.Set("Tensor", func);
        return exports;
    }

    bool Tensor::IsInstance(Napi::Object &obj)
    {
        return obj.InstanceOf(constructor.Value());
    }

    Tensor::Tensor(const Napi::CallbackInfo &info)
        : ObjectWrap(info)
    {
        torchTensor = torch::empty(0);
    }

    Napi::Object Tensor::FromTorchTensor(Napi::Env env, const torch::Tensor &targetTorchTensor)
    {
        try
        {
            Napi::EscapableHandleScope scope(env);
            auto newTensor = Tensor::constructor.New({});
            Napi::ObjectWrap<Tensor>::Unwrap(newTensor)->torchTensor = targetTorchTensor;
            return scope.Escape(newTensor).ToObject();
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }

        return Napi::Object();
    }

    Tensor *Tensor::FromObject(Napi::Value value)
    {
        return Napi::ObjectWrap<Tensor>::Unwrap(value.ToObject());
    }

    Napi::Value Tensor::FromTypedArray(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();

        if (info.Length() > 1 && info[0].IsTypedArray() && info[1].IsArray())
        {
            return Tensor::FromTorchTensor(env,
                                           typedArrayToTensor(env, info[0].As<Napi::TypedArray>(), info[1].As<Napi::Array>()));
        }
        else if (info.Length() > 0 && info[0].IsTypedArray())
        {
            Tensor::FromTorchTensor(env,
                                    typedArrayToTensor(env, info[0].As<Napi::TypedArray>(),
                                                       Napi::Array::New(env, {info[0].As<Napi::TypedArray>().ElementLength()})));
        }
        return Napi::Value();
    }

    Napi::Value Tensor::Shape(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        auto shape = torchTensor.sizes();
        auto nodeReturn = Napi::Array::New(env, shape.size());
        int idx = 0;
        for (auto dim : shape)
        {
            nodeReturn.Set(idx, dim);
            idx++;
        }

        return nodeReturn;
    }

    Napi::Value Tensor::ToArray(const Napi::CallbackInfo &info)
    {

        auto env = info.Env();

        auto st = torchTensor.scalar_type();

        switch (st)
        {
        case torch::ScalarType::Float:
            return tensorToArray<float>(env, torchTensor);
        case torch::ScalarType::Double:
            return tensorToArray<double>(env, torchTensor);
        case torch::ScalarType::Int:
            return tensorToArray<int32_t>(env, torchTensor);
        case torch::ScalarType::Byte:
            return tensorToArray<uint8_t>(env, torchTensor);
        case torch::ScalarType::Long:
            return tensorToArray<int64_t>(env, torchTensor);
        case torch::ScalarType::Bool:
            return tensorToArray<bool>(env, torchTensor, [=](Napi::Env env, bool value) -> Napi::Boolean
                                       { return Napi::Boolean::New(env, value); });
        default:
            throw Napi::TypeError::New(env, "Unsupported type");
        }
    }

    // Napi::Value Tensor::ToMultiArray(const Napi::CallbackInfo &info)
    // {

    //     auto env = info.Env();

    //     try
    //     {
    //         auto st = torchTensor.scalar_type();

    //         switch (st)
    //         {
    //         case torch::ScalarType::Float:
    //             return tensorToMultiArray<float>(env, torchTensor, [=](Napi::Env env, float value) -> Napi::Number
    //                                              { return Napi::Number::New(env, value); });
    //         case torch::ScalarType::Double:
    //             return tensorToMultiArray<double>(env, torchTensor, [=](Napi::Env env, double value) -> Napi::Number
    //                                               { return Napi::Number::New(env, value); });
    //         case torch::ScalarType::Int:
    //             return tensorToMultiArray<int32_t>(env, torchTensor, [=](Napi::Env env, int32_t value) -> Napi::Number
    //                                                { return Napi::Number::New(env, value); });
    //         case torch::ScalarType::Byte:
    //             return tensorToMultiArray<uint8_t>(env, torchTensor, [=](Napi::Env env, uint8_t value) -> Napi::Number
    //                                                { return Napi::Number::New(env, value); });
    //         case torch::ScalarType::Long:
    //             return tensorToMultiArray<int64_t>(env, torchTensor, [=](Napi::Env env, int64_t value) -> Napi::Number
    //                                                { return Napi::Number::New(env, value); });
    //         case torch::ScalarType::Bool:
    //             return tensorToMultiArray<bool>(env, torchTensor, [=](Napi::Env env, bool value) -> Napi::Boolean
    //                                             { return Napi::Boolean::New(env, value); });
    //         default:
    //             throw Napi::TypeError::New(env, "Unsupported type");
    //         }
    //     }
    //     catch (const std::exception &e)
    //     {
    //         throw Napi::Error::New(env, e.what());
    //     }
    // }
    Napi::Value Tensor::Reshape(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();

        try
        {
            return FromTorchTensor(
                env, torchTensor.reshape(napiArrayToVector<int64_t>(info[0].As<Array>())));
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Permute(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();

        try
        {
            return FromTorchTensor(
                env, torchTensor.permute(napiArrayToVector<int64_t>(info[0].As<Array>())));
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Unsqueeze(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        if (info.Length() >= 1 && info[0].IsNumber())
        {
            return FromTorchTensor(env, torchTensor.unsqueeze(info[0].ToNumber().Int64Value()));
        }
        else
        {
            throw Napi::Error::New(env, "Why have you done this ?");
        }
    }

    Napi::Value Tensor::Squeeze(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        if (info.Length() >= 1 && info[0].IsNumber())
        {
            return FromTorchTensor(env, torchTensor.squeeze(info[0].ToNumber().Int64Value()));
        }
        else
        {
            throw Napi::Error::New(env, "Why have you done this ?");
        }
    }

    Napi::Value Tensor::Transpose(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();

        try
        {
            return FromTorchTensor(env,
                                   torchTensor.transpose(
                                       info[0].As<Napi::Number>().Int64Value(), info[1].As<Napi::Number>().Int64Value()));
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Type(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        if (info.Length() == 0 || !info[0].IsString())
        {
            throw Napi::Error::New(env, "Unknown Type");
        }

        auto targetType = info[0].As<Napi::String>().Utf8Value();

        if (targetType == torchFloatType)
        {
            return Tensor::FromTorchTensor(env, torchTensor.toType(torch::ScalarType::Float));
        }
        else if (targetType == torchDoubleType)
        {
            return Tensor::FromTorchTensor(env, torchTensor.toType(torch::ScalarType::Double));
        }
        else if (targetType == torchInt32Type)
        {
            return Tensor::FromTorchTensor(env, torchTensor.toType(torch::ScalarType::Int));
        }
        else if (targetType == torchLongType)
        {
            return Tensor::FromTorchTensor(env, torchTensor.toType(torch::ScalarType::Long));
        }
        else if (targetType == torchUint8Type)
        {
            return Tensor::FromTorchTensor(env, torchTensor.toType(torch::ScalarType::Byte));
        }

        else if (targetType == torchBooleanType)
        {
            return Tensor::FromTorchTensor(env, torchTensor.toType(torch::ScalarType::Bool));
        }

        throw Napi::Error::New(env, "Unknown Type");
    }

    Napi::Value Tensor::DType(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();

        auto st = torchTensor.scalar_type();

        switch (st)
        {
        case torch::ScalarType::Float:
            return Napi::String::New(env, torchFloatType);
        case torch::ScalarType::Double:
            return Napi::String::New(env, torchDoubleType);
        case torch::ScalarType::Int:
            return Napi::String::New(env, torchInt32Type);
        case torch::ScalarType::Long:
            return Napi::String::New(env, torchLongType);
        case torch::ScalarType::Byte:
            return Napi::String::New(env, torchUint8Type);
        case torch::ScalarType::Bool:
            return Napi::String::New(env, torchBooleanType);
        default:
            throw Napi::TypeError::New(env, "Unsupported type");
        }
    }

    Napi::Value Tensor::Clone(const Napi::CallbackInfo &info)
    {
        return Tensor::FromTorchTensor(info.Env(), torchTensor.clone());
    }

    Napi::Value Tensor::Add(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            auto a = torchTensor;

            if (info[0].IsNumber())
            {
                auto b = info[0].ToNumber();

                return Tensor::FromTorchTensor(
                    env, a + (utils::isNapiValueInt(env, b) ? b.Int32Value() : b.FloatValue()));
            }

            auto b = FromObject(info[0])->torchTensor;

            return Tensor::FromTorchTensor(env, a + b);
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Sub(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            auto a = torchTensor;

            if (info[0].IsNumber())
            {
                auto b = info[0].ToNumber();

                return Tensor::FromTorchTensor(
                    env, a - (utils::isNapiValueInt(env, b) ? b.Int32Value() : b.FloatValue()));
            }

            auto b = FromObject(info[0])->torchTensor;

            return Tensor::FromTorchTensor(env, a - b);
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Mul(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            auto a = torchTensor;

            if (info[0].IsNumber())
            {
                auto b = info[0].ToNumber();

                return Tensor::FromTorchTensor(
                    env, a * (utils::isNapiValueInt(env, b) ? b.Int32Value() : b.FloatValue()));
            }

            auto b = FromObject(info[0])->torchTensor;

            return Tensor::FromTorchTensor(env, a * b);
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Div(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();

        try
        {
            auto a = torchTensor;

            if (info[0].IsNumber())
            {
                auto b = info[0].ToNumber();

                return Tensor::FromTorchTensor(
                    env, a / (utils::isNapiValueInt(env, b) ? b.Int32Value() : b.FloatValue()));
            }

            auto b = FromObject(info[0])->torchTensor;

            return Tensor::FromTorchTensor(env, a / b);
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Index(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();

        try
        {
            std::vector<torch::indexing::TensorIndex> indexes;

            for (int i = 0; i < info.Length(); i++)
            {
                indexes.push_back(utils::napiValueToTorchIndex(env, info[i]));
            }

            return Tensor::FromTorchTensor(env, torchTensor.index(indexes));
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::IndexPut(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();

        try
        {
            std::vector<torch::indexing::TensorIndex> indexes;

            for (int i = 1; i < info.Length(); i++)
            {
                indexes.push_back(utils::napiValueToTorchIndex(env, info[i]));
            }

            auto b = FromObject(info[0])->torchTensor;

            torchTensor.index_put_(indexes, b);

            return Napi::Value();
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::MatMul(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            auto a = torchTensor;

            auto b = FromObject(info[0])->torchTensor;

            return Tensor::FromTorchTensor(env, a.matmul(b));
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::AMax(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            return Tensor::FromTorchTensor(env, torchTensor.amax(info[0].As<Napi::Number>().Int64Value()));
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Split(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            if (info[0].IsArray())
            {
                if (info.Length() >= 1)
                {
                    return utils::vectorToNapiArray(env, torchTensor.split(utils::napiArrayToVector<int64_t>(info[0].As<Napi::Array>()), info[1].As<Napi::Number>().Int64Value()));
                }
                else
                {
                    return utils::vectorToNapiArray(env, torchTensor.split(utils::napiArrayToVector<int64_t>(info[0].As<Napi::Array>())));
                }
            }
            else
            {
                if (info.Length() >= 1)
                {
                    return utils::vectorToNapiArray(env, torchTensor.split(info[0].As<Napi::Number>().Int64Value(),info[1].As<Napi::Number>().Int64Value()));
                }
                else
                {
                    return utils::vectorToNapiArray(env, torchTensor.split(info[0].As<Napi::Number>().Int64Value()));
                }
            }
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Argsort(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            if (info.Length() >= 2)
            {
                return Tensor::FromTorchTensor(env, torchTensor.argsort(info[0].As<Napi::Number>().Int64Value(), info[1].As<Napi::Boolean>().Value()));
            }
            else
            {
                return Tensor::FromTorchTensor(env, torchTensor.argsort(info[0].As<Napi::Number>().Int64Value()));
            }
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Max(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            if (info.Length() >= 2)
            {
                auto [a, b] = torchTensor.max(info[0].As<Napi::Number>().Int64Value(), info[1].As<Napi::Boolean>().Value());

                std::vector<torch::Tensor> tuple = {a, b};

                return utils::vectorToNapiArray(env, tuple);
            }
            else
            {
                auto [a, b] = torchTensor.max(info[0].As<Napi::Number>().Int64Value());
                std::vector<torch::Tensor> tuple = {a, b};
                return utils::vectorToNapiArray(env, tuple);
            }
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::View(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            std::vector<int64_t> dims;

            for (auto i = 0; i < info.Length(); i++)
            {
                dims.push_back(info[i].As<Napi::Number>().Int64Value());
            }

            return FromTorchTensor(env, torchTensor.view(dims));
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Any(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            if (info.Length() >= 2)
            {
                return Tensor::FromTorchTensor(env, torchTensor.argsort(info[0].As<Napi::Number>().Int64Value(), info[1].As<Napi::Boolean>().Value()));
            }
            else
            {
                return Tensor::FromTorchTensor(env, torchTensor.argsort(info[0].As<Napi::Number>().Int64Value()));
            }
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Clamp(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {

            return Tensor::FromTorchTensor(env, torchTensor.clamp(info[0].As<Napi::Number>().Int64Value(), info[1].As<Napi::Number>().Int64Value()));
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Sigmoid(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {

            return Tensor::FromTorchTensor(env, torchTensor.sigmoid());
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Cuda(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {

            return Tensor::FromTorchTensor(env, torchTensor.cuda());
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Cpu(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {

            return Tensor::FromTorchTensor(env, torchTensor.cpu());
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Detach(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            return Tensor::FromTorchTensor(env, torchTensor.detach());
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::Backward(const Napi::CallbackInfo &info)
    {
        auto env = info.Env();
        try
        {
            torchTensor.backward();

            return Napi::Value();
        }
        catch (const std::exception &e)
        {
            throw Napi::Error::New(env, e.what());
        }
    }

    Napi::Value Tensor::toString(const Napi::CallbackInfo &info)
    {
        return Napi::String::New(info.Env(), torchTensor.toString());
    }
}
