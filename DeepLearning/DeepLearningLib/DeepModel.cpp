#include "DeepModel.h"

#include <array>

namespace deep_learning_lib
{
    DataLayer::DataLayer(int width, int height, int depth)
        : data_(width * height * depth), data_view_(width, height, depth, data_)
    {
    }

    ModelLayer::ModelLayer(int num_neuron, int neuron_width, int neuron_height, int neuron_depth)
        : weights_(num_neuron * neuron_width * neuron_height * neuron_depth),
        weight_view_(concurrency::extent<4>(
        std::array<int, 4>{{ num_neuron, neuron_width, neuron_height, neuron_depth }}.data()), weights_)
    {
    }
}
