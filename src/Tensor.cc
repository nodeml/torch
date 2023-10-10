#include "Tensor.h"
#include <torch/torch.h>
#include <exception>
#include "utils.h"
#include "types.h"

namespace nodeml_torch
{
	using namespace Napi;
	using namespace nodeml_torch::utils;
	using namespace nodeml_torch::types;

	Napi::FunctionReference Tensor::constructor;

	template <typename T>
	Napi::Value tensorToArray(Napi::Env env, const torch::Tensor &tensor)
	{
		Napi::EscapableHandleScope scope(env);
		assert(tensor.is_contiguous());
		auto typed_array = Napi::TypedArrayOf<T>::New(env, tensor.numel());
		memcpy(typed_array.Data(), tensor.data_ptr<T>(), sizeof(T) * tensor.numel());
		return scope.Escape(typed_array);
	}

	template <typename T>
	torch::Tensor arrayToTensor(Napi::Env env, const Napi::TypedArray &data,
								const Napi::Array &shape_array)
	{
		auto *data_ptr = data.As<Napi::TypedArrayOf<T>>().Data();
		auto shape = napiArrayToVector<std::int64_t>(shape_array);
		torch::TensorOptions options(scalarType<T>());
		auto torch_tensor = torch::empty(shape, options);
		memcpy(torch_tensor.data<T>(), data_ptr, sizeof(T) * torch_tensor.numel());
		return torch_tensor;
	}

	torch::Tensor typedArrayToTensor(Napi::Env env, const Napi::TypedArray &data,
									 const Napi::Array &shape)
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
		auto func = DefineClass(env, "Tensor", {Tensor::InstanceMethod("toArray", &Tensor::ToArray), Tensor::InstanceMethod("shape", &Tensor::Shape), Tensor::InstanceMethod("reshape", &Tensor::Reshape), Tensor::InstanceMethod("toString", &Tensor::toString), Tensor::StaticMethod("fromTypedArray", &Tensor::FromTypedArray), Tensor::InstanceMethod("slice", &Tensor::Slice), Tensor::InstanceMethod("type", &Tensor::Type), Tensor::InstanceMethod("dtype", &Tensor::DType)});

		constructor = Napi::Persistent(func);
		constructor.SuppressDestruct();
		exports.Set("Tensor", func);
		return exports;
	}

	Tensor::Tensor(const Napi::CallbackInfo &info) : ObjectWrap(info)
	{
		tensor = torch::empty(0);
	}

	Napi::Object Tensor::FromTorchTensor(const Napi::CallbackInfo &info, const torch::Tensor &torchTensor)
	{
		auto env = info.Env();
		try
		{
			Napi::EscapableHandleScope scope(env);
			auto newTensor = Tensor::constructor.New({});
			Napi::ObjectWrap<Tensor>::Unwrap(newTensor)->tensor = torchTensor;
			return scope.Escape(newTensor).ToObject();
		}
		catch (const std::exception &e)
		{
			throw Napi::Error::New(env, e.what());
		}

		return Napi::Object();
	}

	Napi::Value Tensor::FromTypedArray(const Napi::CallbackInfo &info)
	{
		auto env = info.Env();

		if (info.Length() > 1 && info[0].IsTypedArray() && info[1].IsArray())
		{
			return Tensor::FromTorchTensor(info, typedArrayToTensor(env, info[0].As<Napi::TypedArray>(), info[1].As<Napi::Array>()));
		}
		else if (info.Length() > 0 && info[0].IsTypedArray())
		{
			Tensor::FromTorchTensor(info, typedArrayToTensor(env, info[0].As<Napi::TypedArray>(), Napi::Array::New(env, {info[0].As<Napi::TypedArray>().ElementLength()})));
		}
		return Napi::Value();
	}

	Napi::Value Tensor::Shape(const Napi::CallbackInfo &info)
	{
		auto env = info.Env();
		auto shape = tensor.sizes();
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

		auto st = tensor.scalar_type();

		switch (st)
		{
		case torch::ScalarType::Float:
			return tensorToArray<float>(env, tensor);
		case torch::ScalarType::Double:
			return tensorToArray<double>(env, tensor);
		case torch::ScalarType::Int:
			return tensorToArray<int32_t>(env, tensor);
		case torch::ScalarType::Byte:
			return tensorToArray<uint8_t>(env, tensor);
		case torch::ScalarType::Long:
			return tensorToArray<int64_t>(env, tensor);
		default:
			throw Napi::TypeError::New(env, "Unsupported type");
		}
	}

	Napi::Value Tensor::Reshape(const Napi::CallbackInfo &info)
	{
		auto env = info.Env();

		if (info.Length() > 0 && info[0].IsArray())
		{
			return FromTorchTensor(info, tensor.reshape(napiArrayToVector<int64_t>(info[0].As<Array>())));
		}
		else
		{
			throw Napi::Error::New(env, "Why have you done this ?");
		}
		return Napi::Object();
	}

	Napi::Value Tensor::Permute(const Napi::CallbackInfo &info)
	{
		auto env = info.Env();

		if (info.Length() > 0 && info[0].IsArray())
		{
			return FromTorchTensor(info, tensor.permute(napiArrayToVector<int64_t>(info[0].As<Array>())));
		}
		else
		{
			throw Napi::Error::New(env, "Why have you done this ?");
		}
		return Napi::Object();
	}

	Napi::Value Tensor::Transpose(const Napi::CallbackInfo &info)
	{
		auto env = info.Env();

		if (info.Length() >= 2 && info[0].IsNumber() && info[1].IsNumber())
		{
			return FromTorchTensor(info, tensor.transpose(info[0].As<Napi::Number>().Int64Value(), info[1].As<Napi::Number>().Int64Value()));
		}
		else
		{
			throw Napi::Error::New(env, "Why have you done this ?");
		}
		return Napi::Object();
	}

	Napi::Value Tensor::Slice(const Napi::CallbackInfo &info)
	{
		auto env = info.Env();
		auto sliceOptions = info[0].As<Napi::Object>();
		auto sliceDims = info.Length() >= 1 && info[0].IsNumber() ? info[0].As<Napi::Number>().Int64Value() : 0;
		auto sliceStart = info.Length() >= 2 && info[1].IsNumber() ? info[1].As<Napi::Number>().Int64Value() : NULL;
		auto sliceEnd = info.Length() >= 3 && info[2].IsNumber() ? info[2].As<Napi::Number>().Int64Value() : NULL;
		return Tensor::FromTorchTensor(info, tensor.slice(sliceDims, sliceStart, sliceEnd));
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
			return Tensor::FromTorchTensor(info, tensor.toType(torch::ScalarType::Float));
		}
		else if (targetType == torchDoubleType)
		{
			return Tensor::FromTorchTensor(info, tensor.toType(torch::ScalarType::Double));
		}
		else if (targetType == torchInt32Type)
		{
			return Tensor::FromTorchTensor(info, tensor.toType(torch::ScalarType::Int));
		}
		else if (targetType == torchLongType)
		{
			return Tensor::FromTorchTensor(info, tensor.toType(torch::ScalarType::Long));
		}
		else if (targetType == torchUint8Type)
		{
			return Tensor::FromTorchTensor(info, tensor.toType(torch::ScalarType::Byte));
		}

		throw Napi::Error::New(env, "Unknown Type");
	}

	Napi::Value Tensor::DType(const Napi::CallbackInfo &info)
	{
		auto env = info.Env();

		auto st = tensor.scalar_type();

		switch (st)
		{
		case torch::ScalarType::Float:
			return Napi::String::New(env, torchFloatType);
		case torch::ScalarType::Double:
			return Napi::String::New(env, torchDoubleType);
		case torch::ScalarType::Int:
			return Napi::String::New(env, torchInt32Type);
		case torch::ScalarType::Byte:
			return Napi::String::New(env, torchUint8Type);
		case torch::ScalarType::Long:
			return Napi::String::New(env, torchLongType);
		default:
			throw Napi::TypeError::New(env, "Unsupported type");
		}
	}

	Napi::Value Tensor::toString(const Napi::CallbackInfo &info)
	{
		return Napi::String::New(info.Env(), tensor.toString());
	}
}
