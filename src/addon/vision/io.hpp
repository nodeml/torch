#pragma once

#include <napi.h>

namespace nodeml_torch
{
    namespace vision
    {
        namespace io
        {

            Napi::Value readFile(const Napi::CallbackInfo &info);

            Napi::Value writeFile(const Napi::CallbackInfo &info);

            Napi::Value readImage(const Napi::CallbackInfo &info);

            Napi::Value encodeJpeg(const Napi::CallbackInfo &info);

            // Napi::Value encodePng(const Napi::CallbackInfo &info);

            Napi::Value decodeImage(const Napi::CallbackInfo &info);

            Napi::Value decodeJpeg(const Napi::CallbackInfo &info);

            Napi::Value decodePng(const Napi::CallbackInfo &info);

            Napi::Object Init(Napi::Env env, Napi::Object exports);
        }
    }
}
