#include <napi.h>
#include <addon/cuda/cuda.hpp>
#include <torch/torch.h>

namespace nodeml_torch
{
    namespace cuda
    {

        Napi::Value deviceCount(const Napi::CallbackInfo &info)
        {
            auto env = info.Env();

            try
            {
                return Napi::Number::New(env,torch::cuda::device_count());
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(env, e.what());
            }
        }
        Napi::Value isAvailable(const Napi::CallbackInfo &info)
        {
            auto env = info.Env();

            try
            {
                return Napi::Boolean::New(env,torch::cuda::is_available());
            }
            catch (const std::exception &e)
            {
                throw Napi::Error::New(env, e.what());
            }
        }

        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            auto myExports = Napi::Object::New(env);
            myExports.Set("isAvailable",Napi::Function::New(env,isAvailable));
            myExports.Set("deviceCount",Napi::Function::New(env,deviceCount));
            return exports;
        }
    }
}
