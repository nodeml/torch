#pragma once

#include <napi.h>

#include <memory>
#include <string>
#include <torch/torch.h>

namespace nodeml_torch
{
    namespace jit
    {
        class Module : public Napi::ObjectWrap<Module>
        {

        public:
            static Napi::FunctionReference constructor;

            torch::jit::Module torchModule;

            static Napi::Object Init(Napi::Env env, Napi::Object exports);

            Module(const Napi::CallbackInfo &info);

            static Napi::Object FromTorchJitModule(
                const Napi::CallbackInfo &info, const torch::jit::Module &torchJitModule);

            Napi::Value Forward(const Napi::CallbackInfo &info);

            Napi::Value Eval(const Napi::CallbackInfo &info);

            Napi::Value toString(const Napi::CallbackInfo &info);



            static Napi::Value IValueToJSType(Napi::Env env, const c10::IValue &iValue);
            static c10::IValue JSTypeToIValue(Napi::Env env, const Napi::Value &jsValue);
        };
    }
}