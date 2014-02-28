#pragma once

#include <amp.h>

namespace deep_learning_lib
{
    float atomic_fetch_add(float *_Dest, const float _Value) restrict(amp)
    {
        float oldVal = *_Dest;
        float newVal;
        do {
            newVal = oldVal + _Value;
        } while (!concurrency::atomic_compare_exchange(
            reinterpret_cast<unsigned int*>(_Dest),
            reinterpret_cast<unsigned int*>(&oldVal),
            (*(reinterpret_cast<unsigned int*>(&newVal)))));

        return newVal;
    }
}

