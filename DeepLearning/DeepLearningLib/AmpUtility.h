#pragma once

#include <array>
#include <tuple>
#include <limits>
#include <type_traits>
#include <vector>

#include <amp.h>

namespace deep_learning_lib
{
    inline float atomic_fetch_add(float *_Dest, const float _Value) restrict(amp)
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

    template<typename T, int Rank>
    std::vector<T> CopyToVector(const concurrency::array_view<T, Rank>& arr)
    {
        std::vector<T> v(arr.extent.size());
        concurrency::copy(arr, v.begin());
        return v;
    }

    template<typename T, int Rank>
    inline void fill(concurrency::array<T, Rank>& arr, T initValue)
    {
        concurrency::parallel_for_each(arr.extent,
            [&arr, initValue](concurrency::index<Rank> idx) restrict(amp)
        {
            arr[idx] = initValue;
        });
    }

    template<typename T, int Rank>
    inline void fill(const concurrency::array_view<T, Rank>& arr, T initValue)
    {
        arr.discard_data();
        concurrency::parallel_for_each(arr.extent,
            [=](concurrency::index<Rank> idx) restrict(amp)
        {
            arr[idx] = initValue;
        });
    }

    template<typename T, int Rank>
    std::pair<concurrency::index<Rank>, T> min(const concurrency::array_view<T, Rank>& arr)
    {
        static_assert(std::is_same<T, int>::value || std::is_same<T, float>::value, "only allow atomic operation for int and float type.");

        typedef typename std::remove_const<T>::type TT;

        concurrency::array_view<TT> min_value_view(1);
        min_value_view(0) = std::numeric_limits<TT>::max();
        
        concurrency::parallel_for_each(arr.extent, [=](concurrency::index<Rank> idx) restrict(amp)
        {
            TT old_min_value = min_value_view(0);
            TT new_min_value;
            T current_value = arr[idx];
            do
            {
                new_min_value = current_value < old_min_value ? current_value : old_min_value;
            } while (!concurrency::atomic_compare_exchange(
                reinterpret_cast<unsigned int*>(&min_value_view(0)),
                reinterpret_cast<unsigned int*>(&old_min_value),
                (*(reinterpret_cast<unsigned int*>(&new_min_value)))));
        });

        concurrency::array_view<int> min_index_view(1);// used to communicate the min index back
        min_index_view(0) = arr.extent.size();
        concurrency::parallel_for_each(arr.extent, [=](concurrency::index<Rank> idx) restrict(amp)
        {
            T current_value = arr[idx];
            if (current_value == min_value_view(0))
            {
                int old_min_index = min_index_view(0);
                int new_min_index;
                // flatten the index, take the smallest upon a draw
                int current_index = idx[0];
                for (int i = 1; i < Rank; ++i)
                {
                    current_index = current_index * arr.extent[i] + idx[i];
                }
                do
                {
                    new_min_index = current_index < old_min_index ? current_index : old_min_index;
                } while (!concurrency::atomic_compare_exchange(
                    reinterpret_cast<unsigned int*>(&min_index_view(0)),
                    reinterpret_cast<unsigned int*>(&old_min_index),
                    (*(reinterpret_cast<unsigned int*>(&new_min_index)))));
            }
        });

        T min_value = min_value_view(0);
        int min_flat_index = min_index_view(0);
        concurrency::index<Rank> min_index;
        for (int i = Rank - 1; i >= 0; --i)
        {
            min_index[i] = min_flat_index % arr.extent[i];
            min_flat_index = (min_flat_index - min_index[i]) / arr.extent[i];
        }
        
        assert(arr[min_index] == min_value);

        return std::make_pair(min_index, min_value);
    }

    template<typename T, int Rank>
    std::pair<concurrency::index<Rank>, T> max(const concurrency::array_view<T, Rank>& arr)
    {
        typedef typename std::remove_const<T>::type TT;

        static_assert(std::is_same<TT, int>::value || std::is_same<TT, float>::value, "only allow atomic operation for int and float type.");

        concurrency::array_view<TT> max_value_view(1);
        max_value_view(0) = std::numeric_limits<TT>::min();

        concurrency::parallel_for_each(arr.extent, [=](concurrency::index<Rank> idx) restrict(amp)
        {
            TT old_max_value = max_value_view(0);
            TT new_max_value;
            T current_value = arr[idx];
            do
            {
                new_max_value = current_value > old_max_value ? current_value : old_max_value;
            } while (!concurrency::atomic_compare_exchange(
                reinterpret_cast<unsigned int*>(&max_value_view(0)),
                reinterpret_cast<unsigned int*>(&old_max_value),
                (*(reinterpret_cast<unsigned int*>(&new_max_value)))));
        });

        concurrency::array_view<int> max_index_view(1);// used to communicate the min index back
        max_index_view(0) = arr.extent.size();
        concurrency::parallel_for_each(arr.extent, [=](concurrency::index<Rank> idx) restrict(amp)
        {
            T current_value = arr[idx];
            if (current_value == max_value_view(0))
            {
                int old_max_index = max_index_view(0);
                int new_max_index;
                // flatten the index, take the smallest upon a draw
                int current_index = idx[0];
                for (int i = 1; i < Rank; ++i)
                {
                    current_index = current_index * arr.extent[i] + idx[i];
                }
                do
                {
                    new_max_index = current_index < old_max_index ? current_index : old_max_index;
                } while (!concurrency::atomic_compare_exchange(
                    reinterpret_cast<unsigned int*>(&max_index_view(0)),
                    reinterpret_cast<unsigned int*>(&old_max_index),
                    (*(reinterpret_cast<unsigned int*>(&new_max_index)))));
            }
        });

        T max_value = max_value_view(0);
        int max_flat_index = max_index_view(0);
        concurrency::index<Rank> max_index;
        for (int i = Rank - 1; i >= 0; --i)
        {
            max_index[i] = max_flat_index % arr.extent[i];
            max_flat_index = (max_flat_index - max_index[i]) / arr.extent[i];
        }

        assert(arr[max_index] == max_value);

        return std::make_pair(max_index, max_value);
    }

    // until c++14, we cannot use initializer_list.size to remove the dependency on size template parameter
    inline concurrency::extent<4> make_extent(int e0, int e1, int e2, int e3)
    {
        return concurrency::extent<4>(std::array < int, 4 > {{e0, e1, e2, e3}}.data());
    }

    inline concurrency::index<4> make_index(int e0, int e1, int e2, int e3)
    {
        return concurrency::index<4>(std::array < int, 4 > {{e0, e1, e2, e3}}.data());
    }

    inline concurrency::extent<5> make_extent(int e0, int e1, int e2, int e3, int e4)
    {
        return concurrency::extent<5>(std::array < int, 5 > {{e0, e1, e2, e3, e4}}.data());
    }

    inline concurrency::index<5> make_index(int e0, int e1, int e2, int e3, int e4)
    {
        return concurrency::index<5>(std::array < int, 5 > {{e0, e1, e2, e3, e4}}.data());
    }

    inline int max(const int a, const int b) restrict(amp)
    {
        return a >= b ? a : b;
    }

    inline int min(const int a, const int b) restrict(amp)
    {
        return a <= b ? a : b;
    }
}

