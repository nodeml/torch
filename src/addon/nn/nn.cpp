#include <addon/nn/nn.hpp>
#include <addon/nn/functional.hpp>

namespace nodeml_torch
{
    namespace nn
    {
        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            auto myExports = Napi::Object::New(env);

            myExports.Set("functional", functional::Init(env));
            exports.Set("nn", myExports);
            return exports;
        }
    }
}
