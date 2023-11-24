#include <addon/types.h>

namespace nodeml_torch
{

    namespace types
    {
        Napi::Object Init(Napi::Env env, Napi::Object exports)
        {
            auto TypeObject = Napi::Object::New(env);
            TypeObject.Set("int32", torchInt32Type);
            TypeObject.Set("double", torchDoubleType);
            TypeObject.Set("float", torchFloatType);
            TypeObject.Set("uint8", torchUint8Type);
            TypeObject.Set("long", torchLongType);
            TypeObject.Set("bool", torchBooleanType);
            exports.Set("types", TypeObject);
            return exports;
        }
    }
}