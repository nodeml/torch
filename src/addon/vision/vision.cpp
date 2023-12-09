#include <napi.h>
#include <addon/vision/vision.hpp>
#include <torch/torch.h>
#include <addon/vision/ops.hpp>
#include <addon/vision/io.hpp>

namespace nodeml_torch
{
    namespace vision
    {

        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            auto myExports = Napi::Object::New(env);

            ops::Init(env, myExports);
            io::Init(env, myExports);

            exports.Set("vision", myExports);

            return exports;
        }
    }
}
