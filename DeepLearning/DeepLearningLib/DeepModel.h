#pragma once
#include <amp.h>
#include <vector>

namespace deep_learning_lib
{
    class DataLayer
    {
    private:
        // this vector is initialzed before data_view_
        std::vector<float> data_;

    public:
        concurrency::array_view<float, 3> data_view_;

    public:
        DataLayer(int width, int height, int depth);
    };

    class ModelLayer
    {
    private:
        // this vector is initialized before weight_view_
        std::vector<float> weights_;

    public:
        concurrency::array_view<float, 4> weight_view_;

    public:
        ModelLayer(int num_neuron, int neuron_width, int neuron_height, int neuron_depth);
    };

    class DeepModel
    {
    public:

    };
}

