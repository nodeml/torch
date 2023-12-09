#include <addon/jit/jit.hpp>
#include <torch/script.h>
#include <addon/FunctionWorker.hpp>
#include <addon/jit/Module.hpp>

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

                auto modulePath = info[0].ToString().Utf8Value();

                auto worker = new FunctionWorker<torch::jit::Module>(
                    info.Env(),
                    [=]() -> torch::jit::Module
                    {
                        return torch::jit::load(modulePath);
                    },
                    [=](Napi::Env env, torch::jit::Module value) -> Napi::Value
                    {
                        return JitModule::FromTorchJitModule(env, value);
                    });

                worker->Queue();
                return worker->GetPromise();
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(info.Env(), e.what());
            }
        }

        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            auto myExports = Napi::Object::New(env);

            JitModule::Init(env, myExports);

            myExports.Set("load", Napi::Function::New(env, load));

            exports.Set("jit", myExports);

            return exports;
        }
    }
}
