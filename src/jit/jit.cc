#include "jit.h"
#include <torch/script.h>
#include "Module.h"

namespace nodeml_torch
{
    namespace jit
    {
        Napi::Value load(const Napi::CallbackInfo &info)
        {

            try
            {
                auto env = info.Env();

                if (!info[0].IsString())
                {
                    throw Napi::Error::New(env, "Path Must Be A String");
                }

                return Module::FromTorchJitModule(info, torch::jit::load(info[0].ToString().Utf8Value()));
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
        }

        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            auto myExports = Napi::Object::New(env);

            Module::Init(env, myExports);

            myExports.Set("load", Napi::Function::New(env, load));

            exports.Set("jit", myExports);

            return exports;
        }
    }
}