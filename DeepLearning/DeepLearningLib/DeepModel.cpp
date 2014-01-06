#include "DeepModel.h"

#include <array>
#include <assert.h>

namespace deep_learning_lib
{
    DataLayer::DataLayer(int height, int width, int depth)
        : data_(height * width * depth), data_view_(height, width, depth, data_)
    {
    }


    ModelLayer::ModelLayer(int num_neuron, int neuron_height, int neuron_width, int neuron_depth)
        : weights_(num_neuron * neuron_height * neuron_width * neuron_depth),
        weight_view_(concurrency::extent<4>(
        std::array<int, 4>{{ num_neuron, neuron_height, neuron_width, neuron_depth }}.data()), weights_)
    {
    }

    void ModelLayer::PassUp(const DataLayer& bottom_layer, DataLayer& top_layer) const
    {
        
    }
}
