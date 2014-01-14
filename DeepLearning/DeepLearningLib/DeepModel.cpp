#include "DeepModel.h"

#include <array>
#include <assert.h>
#include <random>
#include <iostream>

namespace deep_learning_lib
{
    using namespace concurrency;

    DataLayer::DataLayer(int depth, int width, int height)
        : data_(depth * width * height), data_view_(depth, width, height, data_), 
        data_generated_(data_view_.extent)
    {
        memory_.reserve(kMemorySize);
        for (int i = 0; i < kMemorySize; i++)
        {
            memory_.emplace_back(data_view_.extent);
        }
    }

    DataLayer::DataLayer(DataLayer&& other)
        : data_(std::move(other.data_)), data_view_(other.data_view_),
        data_generated_(std::move(other.data_generated_)), memory_(std::move(other.memory_))
    {

    }

    void DataLayer::SetData(const std::vector<float>& data)
    {
        assert(data.size() == data_.size());

        // Disgard the data on GPU
        data_view_.discard_data();
        // Copy the data
        data_ = data;
        data_view_.refresh();
    }


    ConvolveLayer::ConvolveLayer(int num_neuron, int neuron_depth, int neuron_width, int neuron_height)
        : weights_(num_neuron * neuron_depth * neuron_width * neuron_height),
        weight_view_(extent<4>(std::array<int, 4>{{ num_neuron, neuron_depth, neuron_width, neuron_height }}.data()), weights_)
    {
    }

    ConvolveLayer::ConvolveLayer(ConvolveLayer&& other)
        : weights_(std::move(other.weights_)), weight_view_(other.weight_view_)
    {
    }

    void ConvolveLayer::PassUp(concurrency::array_view<const float, 3> bottom_layer,
        concurrency::array_view<float, 3> top_layer) const
    {
        assert(top_layer.extent[0] /* depth */ == this->neuron_num());

        // readonly
        array_view<const float, 4> neuron_weights = weight_view_;
        // writeonly
        top_layer.discard_data();

        // non-tiled version
        parallel_for_each(top_layer.extent,
            [=](index<3> idx) restrict(amp)
        {
            array_view<const float, 3> current_neuron = neuron_weights[idx[0]];// projection
            float result = 0.0f;

            for (int depth_idx = 0; depth_idx < current_neuron.extent[0]; depth_idx++)
            {
                for (int width_idx = 0; width_idx < current_neuron.extent[1]; width_idx++)
                {
                    for (int height_idx = 0; height_idx < current_neuron.extent[2]; height_idx++)
                    {
                        index<3> neuron_idx(depth_idx, width_idx, height_idx);
                        result += bottom_layer[idx + neuron_idx] * current_neuron[neuron_idx];
                    }
                }
            }

            top_layer[idx] = result;
        });
    }

    void ConvolveLayer::PassDown(concurrency::array_view<const float, 3> top_layer,
        concurrency::array_view<float, 3> bottom_layer) const
    {
        assert(top_layer.extent[0] == this->neuron_num());

        // readonly
        array_view<const float, 4> neuron_weights = weight_view_;
        // writeonly
        bottom_layer.discard_data();

        // non-tiled version
        parallel_for_each(bottom_layer.extent,
            [=](index<3> idx) restrict(amp)
        {
            float result = 0.0f;
            int cur_depth_idx = idx[0];
            int cur_width_idx = idx[1];
            int cur_height_idx = idx[2];

            for (int neuron_idx = 0; neuron_idx < neuron_weights.extent[0]; neuron_idx++)
            {
                array_view<const float, 3> current_neuron = neuron_weights[neuron_idx];

                for (int width_idx = 0; width_idx < neuron_weights.extent[2]; width_idx++)
                {
                    for (int height_idx = 0; height_idx < neuron_weights.extent[3]; height_idx++)
                    {
                        if (cur_width_idx - width_idx >= 0 && cur_height_idx - height_idx >= 0)
                        {
                            result += current_neuron(cur_depth_idx, cur_width_idx, cur_height_idx) * 
                                top_layer(neuron_idx, cur_width_idx - width_idx, cur_height_idx - height_idx);
                        }
                    }
                }
            }

            bottom_layer[idx] = result;
        });
    }

    void ConvolveLayer::RandomizeParams(unsigned int seed)
    {
        std::default_random_engine generator(seed);
        std::normal_distribution<float> distribution;

        for (float& w : weights_)
        {
            w = distribution(generator);
        }

        weight_view_.discard_data();
        weight_view_.refresh();
    }

    void DeepModel::AddDataLayer(int depth, int width, int height)
    {
        data_layers_.emplace_back(depth, width, height);
    }

    void DeepModel::AddConvolveLayer(int num_neuron, int neuron_depth, int neuron_width, int neuron_height, unsigned int rand_seed)
    {
        convolve_layers_.emplace_back(num_neuron, neuron_depth, neuron_width, neuron_height);
        convolve_layers_.back().RandomizeParams(rand_seed);
    }

    void DeepModel::PassUp(const std::vector<float>& data)
    {
        auto& bottom_layer = data_layers_.front();
        bottom_layer.SetData(data);

        for (int conv_layer_idx = 0; conv_layer_idx < convolve_layers_.size(); conv_layer_idx++)
        {
            auto& conv_layer = convolve_layers_[conv_layer_idx];
            auto& bottom_data_layer = data_layers_[conv_layer_idx];
            auto& top_data_layer = data_layers_[conv_layer_idx + 1];

            conv_layer.PassUp(bottom_data_layer.data_view_, top_data_layer.data_view_);

            top_data_layer.data_view_.synchronize();
        }
    }
}
