// From https://github.com/arition/torch-js/blob/c94aa01ee2a45921f2cb461c5b0b3e0323f3fc9d/src/FunctionWorker.h
#pragma once

#include <napi.h>
#include <functional>

namespace nodeml_torch
{
    template <typename T>
    class FunctionWorker : public Napi::AsyncWorker
    {
    public:
        FunctionWorker(Napi::Env env, std::function<T()> _workFunction, std::function<Napi::Value(Napi::Env, T)> _postWorkFunction);
        ~FunctionWorker() {}
        void Execute() override;
        void OnOK() override;
        void OnError(const Napi::Error &e) override;
        Napi::Promise GetPromise();

    private:
        Napi::Promise::Deferred promise;
        std::function<T()> workFunction;
        std::function<Napi::Value(Napi::Env, T)> postWorkFunction;
        T value;
    };

    template <typename T>
    FunctionWorker<T>::FunctionWorker(Napi::Env env, std::function<T()> _workFunction, std::function<Napi::Value(Napi::Env, T)> _postWorkFunction)
        : promise(Napi::Promise::Deferred::New(env)), workFunction(_workFunction), postWorkFunction(_postWorkFunction), AsyncWorker(env) {}

    template <typename T>
    void FunctionWorker<T>::Execute()
    {
        value = workFunction();
    }

    template <typename T>
    void FunctionWorker<T>::OnOK()
    {
        auto result = postWorkFunction(Env(), value);
        promise.Resolve(result);
    }

    template <typename T>
    void FunctionWorker<T>::OnError(const Napi::Error &e)
    {
        promise.Reject(e.Value());
    }

    template <typename T>
    Napi::Promise FunctionWorker<T>::GetPromise()
    {
        return promise.Promise();
    }
}