#include <napi.h>
#include <nodeml_torch/Tensor.h>
#include <nodeml_torch/vision/ops.h>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/ops/ops.h>

namespace torchvision_ops = vision::ops;
namespace nodeml_torch
{
    namespace vision
    {
        namespace ops
        {
            Napi::Value nms(const Napi::CallbackInfo &info)
            {
                auto env = info.Env();
                try
                {
                    auto boxes = nodeml_torch::Tensor::FromObject(info[0])->torchTensor;
                    auto scores = nodeml_torch::Tensor::FromObject(info[1])->torchTensor;
                    auto iouThreshold = info[2].As<Napi::Number>().FloatValue();
                    auto tensor = torchvision_ops::nms(boxes, scores, iouThreshold);

                    return nodeml_torch::Tensor::FromTorchTensor(env, tensor);
                }
                catch (const std::exception &e)
                {
                    throw Napi::Error::New(env, e.what());
                }
            }

            Napi::Object Init(Napi::Env env, Napi::Object exports)
            {
                auto myExports = Napi::Object::New(env);

                myExports.Set("nms", Napi::Function::New(env, nms));

                exports.Set("ops", myExports);

                return exports;
            }
        }
    }
}
