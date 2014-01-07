#include "DeepModel.h"

#include <array>
#include <assert.h>

namespace deep_learning_lib
{
    using namespace concurrency;

    DataLayer::DataLayer(int depth, int width, int height)
        : data_(depth * width * height), data_view_(depth, width, height, data_)
    {
    }


    ModelLayer::ModelLayer(int num_neuron, int neuron_depth, int neuron_width, int neuron_height)
        : weights_(num_neuron * neuron_depth * neuron_width * neuron_height),
        weight_view_(extent<4>(std::array<int, 4>{{ num_neuron, neuron_depth, neuron_width, neuron_height }}.data()), weights_)
    {
    }

    void ModelLayer::PassUp(const DataLayer& bottom_layer, DataLayer& top_layer) const
    {
        assert(top_layer.depth() == this->neuron_num());

        // readonly
        array_view<const float, 3> bottom_layer_data = bottom_layer.data_view_;
        array_view<const float, 4> neuron_weights = weight_view_;
        // writeonly
        array_view<float, 3>& top_layer_data = top_layer.data_view_;
        top_layer_data.discard_data();

        // non-tiled version
        parallel_for_each(top_layer_data.extent,
            [=](index<3> idx) restrict(amp)
        {
            array_view<const float, 3> current_neuron = neuron_weights[idx[0]];// projection
            float convolution = 0.0f;

            for (int depth_idx = 0; depth_idx < current_neuron.extent[0]; depth_idx++)
            {
                for (int width_idx = 0; width_idx < current_neuron.extent[1]; width_idx++)
                {
                    for (int height_idx = 0; height_idx < current_neuron.extent[2]; height_idx++)
                    {
                        index<3> neuron_idx(depth_idx, width_idx, height_idx);
                        convolution += bottom_layer_data[idx + neuron_idx] * current_neuron[neuron_idx];
                    }
                }
            }

            top_layer_data[idx] = convolution;
        });
    }
}
