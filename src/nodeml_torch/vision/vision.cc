#include <napi.h>
#include <nodeml_torch/vision/vision.h>
#include <torch/torch.h>
#include <nodeml_torch/vision/ops.h>

namespace nodeml_torch
{
    namespace vision
    {

        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            auto myExports = Napi::Object::New(env);

            ops::Init(env, myExports);

            exports.Set("vision", myExports);

            return exports;
        }
    }
}
