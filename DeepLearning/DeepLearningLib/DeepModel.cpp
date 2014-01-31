#include "DeepModel.h"

#include <array>
#include <assert.h>
#include <random>
#include <amp_math.h>

#include <iostream>

namespace deep_learning_lib
{
    using namespace concurrency;

    DataLayer::DataLayer(int depth, int width, int height, int seed)
        : value_(depth * width * height),
        value_view_(depth, width, height, value_),
        expect_(value_.size()),
        expect_view_(value_view_.extent, expect_),
        next_value_(value_.size()),
        next_value_view_(value_view_.extent, next_value_),
        next_expect_(value_.size()),
        next_expect_view_(value_view_.extent, next_expect_),
        rand_collection_(value_view_.extent, seed)
    {
        memory_.reserve(kMemorySize);
        for (int i = 0; i < kMemorySize; i++)
        {
            memory_.emplace_back(value_view_.extent);
        }
    }

    DataLayer::DataLayer(DataLayer&& other)
        : value_(std::move(other.value_)),
        value_view_(other.value_view_),
        expect_(std::move(other.expect_)),
        expect_view_(other.expect_view_),
        next_value_(std::move(other.next_value_)),
        next_value_view_(other.next_value_view_),
        next_expect_(std::move(other.next_expect_)),
        next_expect_view_(other.next_expect_view_),
        memory_(std::move(other.memory_)),
        rand_collection_(other.rand_collection_)
    {

    }

    void DataLayer::SetValue(const std::vector<float>& data)
    {
        assert(data.size() == value_.size());

        // Disgard the data on GPU
        value_view_.discard_data();
        // Copy the data
        value_ = data;
        value_view_.refresh();
    }

    float DataLayer::ReconstructionError() const
    {
        value_view_.synchronize();
        next_expect_view_.synchronize();

        float result = 0.0f;
        for (int i = 0; i < value_.size(); i++)
        {
            result += std::powf(value_[i] - next_expect_[i], 2);
        }

        return std::sqrtf(result);
    }


    ConvolveLayer::ConvolveLayer(int num_neuron, int neuron_depth, int neuron_width, int neuron_height)
        : weights_(num_neuron * neuron_depth * neuron_width * neuron_height),
        weights_view_(extent<4>(std::array<int, 4>{{ num_neuron, neuron_depth, neuron_width, neuron_height }}.data()), weights_),
        weights_delta_(weights_view_.extent),
        vbias_(neuron_depth),
        vbias_view_(neuron_depth, vbias_),
        vbias_delta_(vbias_view_.extent),
        hbias_(num_neuron),
        hbias_view_(num_neuron, hbias_),
        hbias_delta_(hbias_view_.extent)
    {
        auto& weights_delta = weights_delta_;
        parallel_for_each(weights_delta.extent, 
            [&](index<4> idx) restrict(amp)
        {
            weights_delta[idx] = 0.0f;
        });

        auto& vbias_delta = vbias_delta_;
        parallel_for_each(vbias_delta.extent,
            [&](index<1> idx) restrict(amp)
        {
            vbias_delta[idx] = 0.0f;
        });

        auto& hbias_delta = hbias_delta_;
        parallel_for_each(hbias_delta.extent,
            [&](index<1> idx) restrict(amp)
        {
            hbias_delta[idx] = 0.0f;
        });
    }

    ConvolveLayer::ConvolveLayer(ConvolveLayer&& other)
        : weights_(std::move(other.weights_)),
        weights_view_(other.weights_view_),
        weights_delta_(std::move(other.weights_delta_)),
        vbias_(std::move(other.vbias_)),
        vbias_view_(other.vbias_view_),
        vbias_delta_(std::move(other.vbias_delta_)),
        hbias_(std::move(other.hbias_)),
        hbias_view_(other.hbias_view_),
        hbias_delta_(std::move(hbias_delta_))
    {
    }

    void ConvolveLayer::PassUp(const DataLayer& bottom_layer, bool bottom_switcher,
        DataLayer& top_layer, bool top_switcher) const
    {
        assert(top_layer.depth() == this->neuron_num());
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1);
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1);

        // readonly
        array_view<const float, 4> neuron_weights = weights_view_;
        array_view<const float> hbias = hbias_view_;
        array_view<const float, 3> bottom_layer_value = bottom_switcher ? bottom_layer.value_view_ : bottom_layer.next_value_view_;

        // writeonly
        array_view<float, 3> top_layer_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;
        array_view<float, 3> top_layer_expect = top_switcher ? top_layer.expect_view_ : top_layer.next_expect_view_;
        top_layer_value.discard_data();
        top_layer_expect.discard_data();

        auto& rand_collection = top_layer.rand_collection_;

        // non-tiled version
        parallel_for_each(top_layer_value.extent,
            [=](index<3> idx) restrict(amp)
        {
            array_view<const float, 3> current_neuron = neuron_weights[idx[0]];// projection
            index<3> base_idx(0, idx[1], idx[2]);

            float result = hbias[idx[0]];

            for (int depth_idx = 0; depth_idx < current_neuron.extent[0]; depth_idx++)
            {
                for (int width_idx = 0; width_idx < current_neuron.extent[1]; width_idx++)
                {
                    for (int height_idx = 0; height_idx < current_neuron.extent[2]; height_idx++)
                    {
                        index<3> neuron_idx(depth_idx, width_idx, height_idx);
                        result += bottom_layer_value[base_idx + neuron_idx] * current_neuron[neuron_idx];
                    }
                }
            }

            // Logistic activation function. Maybe more types of activation function later.
            float prob = 1.0f / (1.0f + fast_math::expf(-result));
            top_layer_expect[idx] = prob;
            top_layer_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;
        });

        // for debug
        top_layer_value.synchronize();
        top_layer_expect.synchronize();
    }

    void ConvolveLayer::PassDown(const DataLayer& top_layer, bool top_switcher,
        DataLayer& bottom_layer, bool bottom_switcher) const
    {
        assert(top_layer.depth() == this->neuron_num());
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1);
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1);

        // readonly
        array_view<const float, 4> neuron_weights = weights_view_;
        array_view<const float> vbias = vbias_view_;
        array_view<const float, 3> top_layer_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;

        // writeonly
        array_view<float, 3> bottom_layer_value = bottom_switcher ? bottom_layer.value_view_ : bottom_layer.next_value_view_;
        array_view<float, 3> bottom_layer_expect = bottom_switcher ? bottom_layer.expect_view_ : bottom_layer.next_expect_view_;
        bottom_layer_value.discard_data();
        bottom_layer_expect.discard_data();

        auto& rand_collection = bottom_layer.rand_collection_;

        // non-tiled version
        parallel_for_each(bottom_layer_value.extent,
            [=](index<3> idx) restrict(amp)
        {
            int cur_depth_idx = idx[0];
            int cur_width_idx = idx[1];
            int cur_height_idx = idx[2];

            float result = vbias[cur_depth_idx];

            for (int neuron_idx = 0; neuron_idx < neuron_weights.extent[0]; neuron_idx++)
            {
                array_view<const float, 3> current_neuron = neuron_weights[neuron_idx];

                for (int width_idx = 0; width_idx < current_neuron.extent[1]; width_idx++)
                {
                    for (int height_idx = 0; height_idx < current_neuron.extent[2]; height_idx++)
                    {
                        // make sure the convolve window fits in the bottom layer
                        if (cur_width_idx - width_idx >= 0 && cur_height_idx - height_idx >= 0 &&
                            cur_width_idx - width_idx + current_neuron.extent[1] <= bottom_layer_value.extent[1] &&
                            cur_height_idx - height_idx + current_neuron.extent[2] <= bottom_layer_value.extent[2])
                        {
                            result += current_neuron(cur_depth_idx, width_idx, height_idx) * 
                                top_layer_value(neuron_idx, cur_width_idx - width_idx, cur_height_idx - height_idx);
                        }
                    }
                }
            }

            // Logistic activation function. Maybe more types of activation function later.
            float prob = 1.0f / (1.0f + fast_math::expf(-result));
            bottom_layer_expect[idx] = prob;
            bottom_layer_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;
        });

        // for debug
        bottom_layer_value.synchronize();
        bottom_layer_expect.synchronize();
    }

    void ConvolveLayer::Train(const DataLayer& bottom_layer, const DataLayer& top_layer,
        float learning_rate, bool buffered_update)
    {
        array_view<float, 4> weights = buffered_update ? weights_delta_ : weights_view_;
        array_view<float> vbias = buffered_update ? vbias_delta_ : vbias_view_;
        array_view<float> hbias = buffered_update ? hbias_delta_ : hbias_view_;

        array_view<const float, 3> top_layer_expect = top_layer.expect_view_;
        array_view<const float, 3> top_layer_next_expect = top_layer.next_expect_view_;
        array_view<const float, 3> bottom_layer_value = bottom_layer.value_view_;
        array_view<const float, 3> bottom_layer_next_value = bottom_layer.next_value_view_;

        // non-tiled version
        parallel_for_each(weights.extent, [=](index<4> idx) restrict(amp)
        {
            float delta = 0.0f;

            int neuron_idx = idx[0];

            for (int top_width_idx = 0; top_width_idx < top_layer_expect.extent[1]; top_width_idx++)
            {
                for (int top_height_idx = 0; top_height_idx < top_layer_expect.extent[2]; top_height_idx++)
                {
                    float top_expect = top_layer_expect(neuron_idx, top_width_idx, top_height_idx);
                    float top_next_expect = top_layer_next_expect(neuron_idx, top_width_idx, top_height_idx);

                    float bottom_value = bottom_layer_value(idx[1], idx[2] + top_width_idx, idx[3] + top_height_idx);
                    float bottom_next_value = bottom_layer_next_value(idx[1], idx[2] + top_width_idx, idx[3] + top_height_idx);

                    delta += bottom_value * top_expect - bottom_next_value * top_next_expect;
                }
            }

            weights[idx] += delta / (top_layer_expect.extent[1] * top_layer_expect.extent[2]) * learning_rate;
        });

        parallel_for_each(vbias.extent, [=](index<1> idx) restrict(amp)
        {
            float delta = 0.0f;

            int depth_idx = idx[0];

            for (int bottom_width_idx = 0; bottom_width_idx < bottom_layer_value.extent[1]; bottom_width_idx++)
            {
                for (int bottom_height_idx = 0; bottom_height_idx < bottom_layer_value.extent[2]; bottom_height_idx++)
                {
                    float bottom_value = bottom_layer_value(depth_idx, bottom_width_idx, bottom_height_idx);
                    float bottom_next_value = bottom_layer_next_value(depth_idx, bottom_width_idx, bottom_height_idx);

                    delta += bottom_value - bottom_next_value;
                }
            }

            vbias[idx] += delta / (bottom_layer_value.extent[1] * bottom_layer_value.extent[2]) * learning_rate;
        });

        parallel_for_each(hbias.extent, [=](index<1> idx) restrict(amp)
        {
            float delta = 0.0f;

            int neuron_idx = idx[0];

            for (int top_width_idx = 0; top_width_idx < top_layer_expect.extent[1]; top_width_idx++)
            {
                for (int top_height_idx = 0; top_height_idx < top_layer_expect.extent[2]; top_height_idx++)
                {
                    float top_expect = top_layer_expect(neuron_idx, top_width_idx, top_height_idx);
                    float top_next_expect = top_layer_next_expect(neuron_idx, top_width_idx, top_height_idx);

                    delta += top_expect - top_next_expect;
                }
            }

            hbias[idx] += delta / (top_layer_expect.extent[1] * top_layer_expect.extent[2]) * learning_rate;
        });

        weights.synchronize();
        vbias.synchronize();
        hbias.synchronize();
    }

    void ConvolveLayer::ApplyBufferedUpdate(int buffer_size)
    {
        auto& weights = weights_view_;
        auto& weights_delta = weights_delta_;

        parallel_for_each(weights.extent, [=, &weights_delta](index<4> idx) restrict(amp)
        {
            weights[idx] += weights_delta[idx] / buffer_size;
            weights_delta[idx] = 0.0f;
        });

        auto& vbias = vbias_view_;
        auto& vbias_delta = vbias_delta_;
        parallel_for_each(vbias.extent, [=, &vbias_delta](index<1> idx) restrict(amp)
        {
            vbias[idx] += vbias_delta[idx] / buffer_size;
            vbias_delta[idx] = 0.0f;
        });

        auto& hbias = hbias_view_;
        auto& hbias_delta = hbias_delta_;
        parallel_for_each(hbias.extent, [=, &hbias_delta](index<1> idx) restrict(amp)
        {
            hbias[idx] += hbias_delta[idx] / buffer_size;
            hbias_delta[idx] = 0.0f;
        });
    }

    void ConvolveLayer::RandomizeParams(unsigned int seed)
    {
        std::default_random_engine generator(seed);
        std::normal_distribution<float> distribution(0.0f, 0.05f);

        for (float& w : weights_)
        {
            w = distribution(generator);
        }

        weights_view_.discard_data();
        weights_view_.refresh();
    }

    PoolingLayer::PoolingLayer(int block_width, int block_height)
        : block_width_(block_width), block_height_(block_height)
    {

    }

    void PoolingLayer::PassUp(const DataLayer& bottom_layer, bool bottom_switcher,
        DataLayer& top_layer, bool top_switcher) const
    {
        assert(top_layer.width() * block_width_ == bottom_layer.width());
        assert(top_layer.height() * block_height_ == bottom_layer.height());

        // readonly
        int block_width = block_width_;
        int block_height = block_height_;
        array_view<const float, 3> bottom_layer_value = bottom_switcher ? bottom_layer.value_view_ : bottom_layer.next_value_view_;
        array_view<const float, 3> bottom_layer_expect = bottom_switcher ? bottom_layer.expect_view_ : bottom_layer.next_expect_view_;

        // writeonly
        array_view<float, 3> top_layer_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;
        array_view<float, 3> top_layer_expect = top_switcher ? top_layer.expect_view_ : top_layer.next_expect_view_;
        top_layer_value.discard_data();
        top_layer_expect.discard_data();

        parallel_for_each(top_layer_value.extent, [=](index<3> idx) restrict(amp)
        {
            float max_value = 0.0f;
            float max_expect = 1.0f;
            for (int width_idx = 0; width_idx < block_width; width_idx++)
            {
                for (int height_idx = 0; height_idx < block_height; height_idx++)
                {
                    float value = bottom_layer_value(idx[0], idx[1] * block_width + width_idx, idx[2] * block_height + height_idx);
                    float expect = bottom_layer_expect(idx[0], idx[1] * block_width + width_idx, idx[2] * block_height + height_idx);

                    max_value = fast_math::fmaxf(max_value, value);
                    max_expect *= (1.0f - expect); // the probability that all nodes are 0
                }
            }
            max_expect = 1.0f - max_expect;// the probability that at least one node is 1.

            top_layer_value[idx] = max_value;
            top_layer_expect[idx] = max_expect;
        });
    }

    void PoolingLayer::PassDown(const DataLayer& top_layer, bool top_switcher,
        DataLayer& bottom_layer, bool bottom_switcher) const
    {
        assert(top_layer.width() * block_width_ == bottom_layer.width());
        assert(top_layer.height() * block_height_ == bottom_layer.height());

        // readonly
        int block_width = block_width_;
        int block_height = block_height_;
        array_view<const float, 3> top_layer_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;
        array_view<const float, 3> top_layer_expect = top_switcher ? top_layer.expect_view_ : top_layer.next_expect_view_;

        // writeonly
        array_view<float, 3> bottom_layer_value = bottom_switcher ? bottom_layer.value_view_ : bottom_layer.next_value_view_;
        array_view<float, 3> bottom_layer_expect = bottom_switcher ? bottom_layer.expect_view_ : bottom_layer.next_expect_view_;
        bottom_layer_value.discard_data();
        bottom_layer_expect.discard_data();

        auto& rand_collection = bottom_layer.rand_collection_;

        parallel_for_each(bottom_layer_value.extent, [=](index<3> idx) restrict(amp)
        {
            // when we have memory, the bottom_layer can activate according to its memory. 
            // But now we just use uniform activation.

            int width_idx = idx[1] / block_width;// trunc towards zero
            int height_idx = idx[2] / block_height;

            bottom_layer_expect[idx] = 1.0f - fast_math::powf(1.0f - 
                top_layer_expect(idx[0], width_idx, height_idx), -1.0f * block_width * block_height);
            bottom_layer_value[idx] = 0.0f;// clear the value
        });

        parallel_for_each(top_layer_value.extent, [=](index<3> idx) restrict(amp)
        {
            if (top_layer_value[idx] == 1.0f)
            {
                // randomly select a node in bottom_layer to activate
                int width_idx = rand_collection[idx].next_uint() % block_width;
                int height_idx = rand_collection[idx].next_uint() % block_height;
                bottom_layer_value(idx[0], idx[1] * block_width + width_idx, idx[2] * block_height + height_idx) = 1.0f;
            }
        });
    }

    DeepModel::DeepModel(unsigned int model_seed) : random_engine_(model_seed)
    {

    }

    void DeepModel::AddDataLayer(int depth, int width, int height, int seed)
    {
        data_layers_.emplace_back(depth, width, height, seed);
    }

    void DeepModel::AddConvolveLayer(int num_neuron, int neuron_depth, int neuron_width, int neuron_height, unsigned int rand_seed)
    {
        convolve_layers_.emplace_back(num_neuron, neuron_depth, neuron_width, neuron_height);
        convolve_layers_.back().RandomizeParams(rand_seed);
    }

    void DeepModel::PassUp(const std::vector<float>& data)
    {
        auto& bottom_layer = data_layers_.front();
        bottom_layer.SetValue(data);

        for (int conv_layer_idx = 0; conv_layer_idx < convolve_layers_.size(); conv_layer_idx++)
        {
            auto& conv_layer = convolve_layers_[conv_layer_idx];
            auto& bottom_data_layer = data_layers_[conv_layer_idx];
            auto& top_data_layer = data_layers_[conv_layer_idx + 1];

            conv_layer.PassUp(bottom_data_layer, true, top_data_layer, true);
        }
    }

    void DeepModel::PassDown()
    {
        // prepare top layer for passing down
        auto& roof_data_layer = data_layers_.back();
        roof_data_layer.value_view_.copy_to(roof_data_layer.next_value_view_);

        for (int conv_layer_idx = (int)convolve_layers_.size() - 1; conv_layer_idx >= 0; conv_layer_idx--)
        {
            auto& conv_layer = convolve_layers_[conv_layer_idx];
            auto& bottom_data_layer = data_layers_[conv_layer_idx];
            auto& top_data_layer = data_layers_[conv_layer_idx + 1];

            conv_layer.PassDown(top_data_layer, false, bottom_data_layer, false);
        }
    }

    float DeepModel::TrainLayer(const std::vector<float>& data, int layer_idx, float learning_rate)
    {
        auto& bottom_layer = data_layers_[layer_idx];
        auto& top_layer = data_layers_[layer_idx + 1];

        auto& conv_layer = convolve_layers_[layer_idx];

        // train with contrastive divergence (CD) algorithm to maximize likelihood on dataset
        bottom_layer.SetValue(data);

        conv_layer.PassUp(bottom_layer, true, top_layer, true);
        conv_layer.PassDown(top_layer, true, bottom_layer, false);
        conv_layer.PassUp(bottom_layer, false, top_layer, false);

        conv_layer.Train(bottom_layer, top_layer, learning_rate, false);

        return bottom_layer.ReconstructionError();
    }

    void DeepModel::TrainLayer(const std::vector<const std::vector<float>>& dataset,
        int layer_idx, int mini_batch_size, float learning_rate, int iter_count)
    {
        std::uniform_int_distribution<int> uniform_dist(0, (int)dataset.size() - 1);

        auto& bottom_layer = data_layers_[layer_idx];
        auto& top_layer = data_layers_[layer_idx + 1];

        auto& conv_layer = convolve_layers_[layer_idx];

        for (int iter = 0; iter < iter_count; iter++)
        {
            // sample mini-batch, sample with replacement
            for (int mini_batch_idx = 0; mini_batch_idx < mini_batch_size; mini_batch_idx++)
            {
                auto& data = dataset[uniform_dist(random_engine_)];
                bottom_layer.SetValue(data);

                conv_layer.PassUp(bottom_layer, true, top_layer, true);
                conv_layer.PassDown(top_layer, true, bottom_layer, false);
                conv_layer.PassUp(bottom_layer, false, top_layer, false);

                conv_layer.Train(bottom_layer, top_layer, learning_rate, true);
            }

            conv_layer.ApplyBufferedUpdate(mini_batch_size);

            std::cout << "iter = " << iter << "\t err = " << bottom_layer.ReconstructionError() << std::endl;
        }
    }
}
