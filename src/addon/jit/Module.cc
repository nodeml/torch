#include <addon/jit/Module.h>
#include <addon/FunctionWorker.h>
#include <addon/Tensor.h>
#include "Module.h"
namespace nodeml_torch
{
    namespace jit
    {

        Napi::FunctionReference JitModule::constructor;

        Napi::Object JitModule::Init(Napi::Env env, Napi::Object exports)
        {
            auto func = DefineClass(env, "Module",
                                    {
                                        JitModule::InstanceMethod("forward", &JitModule::Forward),
                                    });

            constructor = Napi::Persistent(func);
            constructor.SuppressDestruct();
            exports.Set("Module", func);
            return exports;
        }

        JitModule::JitModule(const Napi::CallbackInfo &info) : ObjectWrap(info)
        {
        }

        Napi::Object JitModule::FromTorchJitModule(Napi::Env env, const torch::jit::Module &torchJitModule)
        {
            try
            {
                Napi::EscapableHandleScope scope(env);
                auto newModule = JitModule::constructor.New({});
                Napi::ObjectWrap<JitModule>::Unwrap(newModule)->torchModule = torchJitModule;
                return scope.Escape(newModule).ToObject();
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(env, e.what());
            }

            return Napi::Object();
        }
        Napi::Value JitModule::Forward(const Napi::CallbackInfo &info)
        {
            try
            {
                torch::NoGradGuard no_grad;
                torchModule.eval();
                auto env = info.Env();

                auto len = info.Length();

                std::vector<torch::jit::IValue> inputs;
                for (size_t i = 0; i < len; ++i)
                {
                    inputs.push_back(JSTypeToIValue(env, info[i]));
                }

                auto worker = new FunctionWorker<c10::IValue>(
                    info.Env(),
                    [=]() -> c10::IValue
                    {
                        torch::NoGradGuard no_grad;
                        return torchModule.forward(inputs);
                    },
                    [=](Napi::Env env, c10::IValue value) -> Napi::Value
                    {
                        return IValueToJSType(env, value);
                    });

                worker->Queue();
                return worker->GetPromise();
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
        }

        Napi::Value JitModule::Eval(const Napi::CallbackInfo &info)
        {
            try
            {
                torchModule.eval();
                return Napi::Value();
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
            return Napi::Value();
        }

        Napi::Value JitModule::toString(const Napi::CallbackInfo &info)
        {
            return Napi::Value();
        }
        Napi::Value JitModule::IValueToJSType(Napi::Env env, const c10::IValue &iValue)
        {
            // From https://github.com/arition/torch-js/blob/c94aa01ee2a45921f2cb461c5b0b3e0323f3fc9d/src/ScriptModule.cc
            Napi::EscapableHandleScope scope(env);
            if (iValue.isTensor())
            {
                return scope.Escape(Tensor::FromTorchTensor(env, iValue.toTensor()));
            }
            else if (iValue.isList())
            {
                auto list = iValue.toList();
                auto jsList = Napi::Array::New(env);
                for (auto i = 0; i < list.size(); i++)
                {
                    jsList[i] = IValueToJSType(env, list[i]);
                }
                return scope.Escape(jsList);
            }
            else if (iValue.isGenericDict())
            {
                auto dict = iValue.toGenericDict();
                auto jsDict = Napi::Object::New(env);
                for (auto iter = dict.begin(); iter != dict.end(); iter++)
                {
                    auto key = IValueToJSType(env, iter->key());
                    auto value = IValueToJSType(env, iter->value());
                    jsDict.Set(key, value);
                }
                return scope.Escape(jsDict);
            }
            else if (iValue.isInt())
            {
                return scope.Escape(Napi::Number::New(env, iValue.toInt()));
            }
            else if (iValue.isDouble())
            {
                return scope.Escape(Napi::Number::New(env, iValue.toDouble()));
            }
            else if (iValue.isBool())
            {
                return scope.Escape(Napi::Boolean::New(env, iValue.toBool()));
            }
            else if (iValue.isString())
            {
                return scope.Escape(Napi::String::New(env, iValue.toString().get()->string()));
            }
            else if (iValue.isTuple())
            {
                auto list = iValue.toTuple()->elements();
                auto jsList = Napi::Array::New(env);
                for (auto i = 0; i < list.size(); i++)
                {
                    jsList[i] = IValueToJSType(env, list[i]);
                }
                return scope.Escape(jsList);
            }
            throw Napi::Error::New(env, "Unsupported output type from ScriptModule");
        }
        c10::IValue JitModule::JSTypeToIValue(Napi::Env env, const Napi::Value &jsValue)
        {
            // From https://github.com/arition/torch-js/blob/c94aa01ee2a45921f2cb461c5b0b3e0323f3fc9d/src/ScriptModule.cc
            Napi::HandleScope scope(env);
            if (jsValue.IsArray())
            {
                auto jsList = jsValue.As<Napi::Array>();
                auto len = jsList.Length();
                if (len == 0)
                {
                    throw Napi::Error::New(env, "Empty array is not supported");
                }
                auto firstElement = JSTypeToIValue(env, jsList[(uint32_t)0]);
                c10::List<c10::IValue> list(firstElement.type());
                for (uint32_t i = 1; i < len; ++i)
                {
                    list.push_back(JSTypeToIValue(env, jsList[i]));
                }
                return list;
            }
            else if (jsValue.IsObject())
            {
                auto jsObject = jsValue.As<Napi::Object>();
                if (Tensor::IsInstance(jsObject))
                {
                    return c10::IValue(Napi::ObjectWrap<Tensor>::Unwrap(jsObject)->torchTensor);
                }
                throw Napi::Error::New(env, "Object/Dict input is not implemented");
            }
            else if (jsValue.IsNumber())
            {
                auto jsNumber = jsValue.As<Napi::Number>().DoubleValue();
                return c10::IValue(jsNumber);
            }
            else if (jsValue.IsBoolean())
            {
                auto jsBool = jsValue.As<Napi::Boolean>().Value();
                return c10::IValue(jsBool);
            }
            else if (jsValue.IsString())
            {
                auto jsString = jsValue.As<Napi::String>().Utf8Value();
                return c10::IValue(jsString);
            }
            throw Napi::Error::New(env, "Unsupported javascript input type");
        }
    }
}


