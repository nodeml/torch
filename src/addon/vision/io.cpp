#include <napi.h>
#include <addon/Tensor.hpp>
#include <addon/vision/ops.hpp>
#include <torch/torch.h>
#include <torchvision/vision.hpp>
#include <addon/FunctionWorker.hpp>
#include <torchvision/io/image/image.hpp>
#include "io.hpp"

namespace torchvision_io = vision::image;
namespace nodeml_torch
{
    namespace vision
    {
        namespace io
        {
            Napi::Value readFile(const Napi::CallbackInfo &info)
            {
                try
                {
                    auto env = info.Env();

                    if (!info[0].IsString())
                    {
                        throw Napi::Error::New(env, "Path Must Be A String");
                    }

                    auto pathInput = info[0].ToString().Utf8Value();

                    auto worker = new FunctionWorker<torch::Tensor>(
                        info.Env(),
                        [=]() -> torch::Tensor
                        {
                            return torchvision_io::read_file(pathInput);
                        },
                        [=](Napi::Env env, torch::Tensor value) -> Napi::Value
                        {
                            return Tensor::FromTorchTensor(env, value);
                        });

                    worker->Queue();

                    return worker->GetPromise();
                }
                catch (const std::exception &e)
                {
                    throw Napi::Error::New(info.Env(), e.what());
                }
            }

            Napi::Value writeFile(const Napi::CallbackInfo &info)
            {
                try
                {
                    auto env = info.Env();

                    auto tensor = Tensor::FromObject(info[0])->torchTensor.clone();

                    auto filePath = info[1].ToString().Utf8Value();

                    auto worker = new FunctionWorker<int>(
                        info.Env(),
                        [=, &tensor]() -> int
                        {
                            torchvision_io::write_file(filePath, tensor);
                            return 0;
                        },
                        [=](Napi::Env env, int value) -> Napi::Value
                        {
                            return Napi::Value();
                        });

                    worker->Queue();

                    return worker->GetPromise();
                }
                catch (const std::exception &e)
                {
                    throw Napi::Error::New(info.Env(), e.what());
                }
            }

            Napi::Value readImage(const Napi::CallbackInfo &info)
            {
                try
                {
                    auto env = info.Env();

                    if (!info[0].IsString())
                    {
                        throw Napi::Error::New(env, "Path Must Be A String");
                    }

                    auto pathInput = info[0].ToString().Utf8Value();

                    auto worker = new FunctionWorker<torch::Tensor>(
                        info.Env(),
                        [=]() -> torch::Tensor
                        {
                            return torchvision_io::decode_image(torchvision_io::read_file(pathInput));
                        },
                        [=](Napi::Env env, torch::Tensor value) -> Napi::Value
                        {
                            return Tensor::FromTorchTensor(env, value);
                        });

                    worker->Queue();

                    return worker->GetPromise();
                }
                catch (const std::exception &e)
                {
                    throw Napi::Error::New(info.Env(), e.what());
                }
            }

            Napi::Value encodeJpeg(const Napi::CallbackInfo &info)
            {
                try
                {
                    auto env = info.Env();

                    auto tensor = Tensor::FromObject(info[0])->torchTensor.clone();

                    auto quality = info[1].ToNumber().Int64Value();

                    auto worker = new FunctionWorker<torch::Tensor>(
                        info.Env(),
                        [=, &tensor]() -> torch::Tensor
                        {
                            return torchvision_io::encode_jpeg(tensor, quality);
                        },
                        [=](Napi::Env env, torch::Tensor value) -> Napi::Value
                        {
                            return Tensor::FromTorchTensor(env, value);
                        });

                    worker->Queue();

                    return worker->GetPromise();
                }
                catch (const std::exception &e)
                {
                    throw Napi::Error::New(info.Env(), e.what());
                }
            }

            // Napi::Value encodePng(const Napi::CallbackInfo &info)
            // {
            //     try
            //     {
            //         auto env = info.Env();

            //         auto tensor = Tensor::FromObject(info[0])->torchTensor.clone();

            //         auto compressionLevel = info[1].ToNumber().Int64Value();

            //         auto worker = new FunctionWorker<torch::Tensor>(
            //             info.Env(),
            //             [=, &tensor]() -> torch::Tensor
            //             {
            //                 return torchvision_io::encode_png(tensor, compressionLevel);
            //             },
            //             [=](Napi::Env env, torch::Tensor value) -> Napi::Value
            //             {
            //                 return Tensor::FromTorchTensor(env, value);
            //             });

            //         worker->Queue();

            //         return worker->GetPromise();
            //     }
            //     catch (const std::exception &e)
            //     {
            //         throw Napi::Error::New(info.Env(), e.what());
            //     }
            // }

            Napi::Value decodeImage(const Napi::CallbackInfo &info)
            {
                try
                {
                    auto env = info.Env();

                    auto tensor = Tensor::FromObject(info[0])->torchTensor;

                    auto worker = new FunctionWorker<torch::Tensor>(
                        info.Env(),
                        [=]() -> torch::Tensor
                        {
                            return torchvision_io::decode_image(tensor);
                        },
                        [=](Napi::Env env, torch::Tensor value) -> Napi::Value
                        {
                            return Tensor::FromTorchTensor(env, value);
                        });

                    worker->Queue();

                    return worker->GetPromise();
                }
                catch (const std::exception &e)
                {
                    throw Napi::Error::New(info.Env(), e.what());
                }
            }

            Napi::Value decodeJpeg(const Napi::CallbackInfo &info)
            {
                try
                {
                    auto env = info.Env();

                    auto tensor = Tensor::FromObject(info[0])->torchTensor.clone();

                    auto worker = new FunctionWorker<torch::Tensor>(
                        info.Env(),
                        [=, &tensor]() -> torch::Tensor
                        {
                            return torchvision_io::decode_jpeg(tensor);
                        },
                        [=](Napi::Env env, torch::Tensor value) -> Napi::Value
                        {
                            return Tensor::FromTorchTensor(env, value);
                        });

                    worker->Queue();

                    return worker->GetPromise();
                }
                catch (const std::exception &e)
                {
                    throw Napi::Error::New(info.Env(), e.what());
                }
            }

            Napi::Value decodePng(const Napi::CallbackInfo &info)
            {
                try
                {
                    auto env = info.Env();

                    auto tensor = Tensor::FromObject(info[0])->torchTensor.clone();

                    auto worker = new FunctionWorker<torch::Tensor>(
                        info.Env(),
                        [=, &tensor]() -> torch::Tensor
                        {
                            return torchvision_io::decode_png(tensor);
                        },
                        [=](Napi::Env env, torch::Tensor value) -> Napi::Value
                        {
                            return Tensor::FromTorchTensor(env, value);
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

                myExports.Set("readFile", Napi::Function::New(env, readFile));

                myExports.Set("writeFile", Napi::Function::New(env, writeFile));

                myExports.Set("readImage", Napi::Function::New(env, readImage));

                //myExports.Set("encodePng", Napi::Function::New(env, encodePng));

                myExports.Set("encodeJpeg", Napi::Function::New(env, encodeJpeg));

                myExports.Set("decodeImage", Napi::Function::New(env, decodeImage));

                myExports.Set("decodePng", Napi::Function::New(env, decodePng));

                myExports.Set("decodeJpeg", Napi::Function::New(env, decodeJpeg));

                exports.Set("io", myExports);

                return exports;
            }
        }
    }
}
