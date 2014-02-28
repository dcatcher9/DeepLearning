#include "DeepModel.h"

#include <array>
#include <tuple>
#include <assert.h>
#include <random>
#include <amp_math.h>

#include <iostream>

#include "AmpUtility.h"

namespace deep_learning_lib
{
    using namespace concurrency;

    DataLayer::DataLayer(int depth, int height, int width, int seed, int memory_pool_size)
        : value_(depth * height * width),
        value_view_(depth, height, width, value_),
        expect_(value_.size()),
        expect_view_(value_view_.extent, expect_),
        next_value_(value_.size()),
        next_value_view_(value_view_.extent, next_value_),
        next_expect_(value_.size()),
        next_expect_view_(value_view_.extent, next_expect_),
        active_prob_(1.0f),
        active_(value_.size(), 1),
        active_view_(value_view_.extent, active_),
        memory_pool_(memory_pool_size * depth * height * width),
        memory_pool_view_(extent<4>(std::array<int, 4>{{memory_pool_size, depth, height, width}}.data()), memory_pool_),
        rand_collection_(value_view_.extent, seed)
    {
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
        active_prob_(other.active_prob_),
        active_(std::move(other.active_)),
        active_view_(other.active_view_),
        memory_pool_(std::move(other.memory_pool_)),
        memory_pool_view_(other.memory_pool_view_),
        rand_collection_(other.rand_collection_)
    {
    }

    void DataLayer::SetValue(const std::vector<float>& data)
    {
        assert(data.size() == value_.size());

        // Copy the data
        value_ = data;
        value_view_.refresh();

        Activate();
    }

    void DataLayer::Activate(float probability)
    {
        if (probability == active_prob_ && 
            (active_prob_ == 1.0f || active_prob_ == 0.0f))
        {
            return;
        }

        array_view<int, 3> active_view = this->active_view_;
        auto& rand_collection = rand_collection_;

        parallel_for_each(active_view.extent, 
            [=](index<3> idx) restrict(amp)
        {
            active_view[idx] = rand_collection[idx].next_single() <= probability ? 1 : 0;
        });

        active_prob_ = probability;
    }

    float DataLayer::ReconstructionError() const
    {
        array_view<float> result(1);
        array_view<const float, 3> value_view = value_view_;
        array_view<const float, 3> next_expect_view = next_expect_view_;

        value_view.synchronize();
        next_expect_view.synchronize();

        // TODO: compare with reduce method for performance
        parallel_for_each(value_view.extent,
            [=](index<3> idx) restrict(amp)
        {
            atomic_fetch_add(&result(0), fast_math::powf(value_view[idx] - next_expect_view[idx], 2));
        });

        return std::sqrtf(result(0));
    }

    void DataLayer::Memorize()
    {
        array_view<float> diffs_view(memory_pool_size());
        array_view<const float, 3> value_view = value_view_;
        array_view<const float, 4> memory_pool_view = memory_pool_view_;

        parallel_for_each(value_view.extent, 
            [=](index<3> idx) restrict(amp)
        {
            for (int i = 0; i < diffs_view.extent[0]; i++)
            {
                atomic_fetch_add(&diffs_view(i), fast_math::powf(memory_pool_view[i][idx] - value_view[idx], 2));
            }
        });

        float min_diff = std::numeric_limits<float>::max();
        int min_idx = -1;

        for (int i = 0; i < diffs_view.extent[0]; i++)
        {
            if (diffs_view(i) < min_diff)
            {
                min_diff = diffs_view(i);
                min_idx = i;
            }
        }
        
        value_view_.copy_to(memory_pool_view_[min_idx]);
    }

    bitmap_image DataLayer::GenerateImage() const
    {
        value_view_.synchronize();
        expect_view_.synchronize();
        next_value_view_.synchronize();
        next_expect_view_.synchronize();

        bitmap_image image;

        const int block_size = 2;

        if (width() == 1 && height() == 1)
        {
            image.setwidth_height(depth() * (block_size + 1), 4 * (block_size + 1), true);
            for (int i = 0; i < value_.size(); i++)
            {
                image.set_region(i * (block_size + 1), 0, block_size, block_size, 
                    value_[i] == 0.0f ? 0 : 255);
            }

            for (int i = 0; i < next_value_.size(); i++)
            {
                image.set_region(i * (block_size + 1), block_size + 1, block_size, block_size, 
                    next_value_[i] == 0.0f ? 0 : 255);
            }

            for (int i = 0; i < expect_.size(); i++)
            {
                image.set_region(i * (block_size + 1), 2 * (block_size + 1), block_size, block_size, 
                    static_cast<unsigned char>(255.0 * expect_[i]));
            }

            for (int i = 0; i < next_expect_.size(); i++)
            {
                image.set_region(i * (block_size + 1), 3 * (block_size + 1), block_size, block_size,
                    static_cast<unsigned char>(255.0 * next_expect_[i]));
            }
        }
        else
        {
            image.setwidth_height(depth() * (width() + 1) * (block_size + 1), 
                4 * (height() + 1) * (block_size + 1), true);
            for (int depth_idx = 0; depth_idx < depth(); depth_idx++)
            {
                for (int height_idx = 0; height_idx < height(); height_idx++)
                {
                    for (int width_idx = 0; width_idx < width(); width_idx++)
                    {
                        image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                            height_idx * (block_size + 1), block_size, block_size, 
                            value_[depth_idx * width() * height() + height_idx * width() + width_idx] == 0.0f ? 0 : 255);
                    }
                }
            }

            for (int depth_idx = 0; depth_idx < depth(); depth_idx++)
            {
                for (int height_idx = 0; height_idx < height(); height_idx++)
                {
                    for (int width_idx = 0; width_idx < width(); width_idx++)
                    {
                        image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                            (height() + 1 + height_idx) * (block_size + 1), block_size, block_size,
                            next_value_[depth_idx * width() * height() + height_idx * width() + width_idx] == 0.0f ? 0 : 255);
                    }
                }
            }

            for (int depth_idx = 0; depth_idx < depth(); depth_idx++)
            {
                for (int height_idx = 0; height_idx < height(); height_idx++)
                {
                    for (int width_idx = 0; width_idx < width(); width_idx++)
                    {
                        image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                            (2 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                            static_cast<unsigned char>(255.0 * expect_[depth_idx * width() * height() + height_idx * width() + width_idx]));
                    }
                }
            }

            for (int depth_idx = 0; depth_idx < depth(); depth_idx++)
            {
                for (int height_idx = 0; height_idx < height(); height_idx++)
                {
                    for (int width_idx = 0; width_idx < width(); width_idx++)
                    {
                        image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                            (3 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                            static_cast<unsigned char>(255.0 * next_expect_[depth_idx * width() * height() + height_idx * width() + width_idx]));
                    }
                }
            }
        }

        return image;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    OutputLayer::OutputLayer(int output_num, int input_depth, int input_height, int input_width)
        : outputs_(output_num),
        outputs_view_(output_num, outputs_),
        next_outputs_(output_num),
        next_outputs_view_(output_num, next_outputs_),
        bias_(output_num),
        bias_view_(output_num, bias_),
        bias_delta_(bias_view_.extent),
        weights_(output_num * input_depth * input_height * input_width),
        weights_view_(extent<4>(std::array<int, 4>{{output_num, input_depth, input_height, input_width}}.data()), weights_),
        weights_delta_(weights_view_.extent)
    {
        auto& weights_delta = weights_delta_;
        parallel_for_each(weights_delta.extent,
            [&](index<4> idx) restrict(amp)
        {
            weights_delta[idx] = 0.0f;
        });

        auto& bias_delta = bias_delta_;
        parallel_for_each(bias_delta.extent,
            [&](index<1> idx) restrict(amp)
        {
            bias_delta[idx] = 0.0f;
        });
    }

    OutputLayer::OutputLayer(OutputLayer&& other)
        : outputs_(std::move(other.outputs_)),
        outputs_view_(other.outputs_view_),
        next_outputs_(std::move(other.next_outputs_)),
        next_outputs_view_(other.next_outputs_view_),
        bias_(std::move(other.bias_)),
        bias_view_(other.bias_view_),
        bias_delta_(std::move(other.bias_delta_)),
        weights_(std::move(other.weights_)),
        weights_view_(other.weights_view_),
        weights_delta_(std::move(other.weights_delta_))
    {

    }

    void OutputLayer::SetLabel(const int label)
    {
        std::fill(outputs_.begin(), outputs_.end(), 0.0f);
        outputs_[label] = 1.0f;

        outputs_view_.refresh();
    }

    void OutputLayer::ApplyBufferedUpdate(int buffer_size)
    {
        auto& weights = weights_view_;
        auto& weights_delta = weights_delta_;

        parallel_for_each(weights.extent, [=, &weights_delta](index<4> idx) restrict(amp)
        {
            weights[idx] += weights_delta[idx] / buffer_size;
            weights_delta[idx] = 0.0f;
        });

        auto& bias = bias_view_;
        auto& bias_delta = bias_delta_;
        parallel_for_each(bias.extent, [=, &bias_delta](index<1> idx) restrict(amp)
        {
            bias[idx] += bias_delta[idx] / buffer_size;
            bias_delta[idx] = 0.0f;
        });

#ifdef DEBUG_SYNC
        weights.synchronize();
        bias.synchronize();
#endif
    }

    void OutputLayer::RandomizeParams(unsigned int seed)
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

    int OutputLayer::PredictLabel(const DataLayer& bottom_layer, bool bottom_switcher,
        DataLayer& top_layer, bool top_switcher,
        const ConvolveLayer& conv_layer, const float dropout_prob)
    {
        assert(top_layer.depth() == conv_layer.neuron_num() && top_layer.depth() == this->input_depth());
        assert(top_layer.width() == bottom_layer.width() - conv_layer.neuron_width() + 1 && top_layer.width() == this->input_width());
        assert(top_layer.height() == bottom_layer.height() - conv_layer.neuron_height() + 1 && top_layer.height() == this->input_height());

        // read only
        array_view<const float, 3> bottom_layer_value = bottom_switcher ? bottom_layer.value_view_ : bottom_layer.next_value_view_;
        array_view<const float, 4> neuron_weights = conv_layer.weights_view_;
        array_view<const float> hbias = conv_layer.hbias_view_;
        array_view<const float> output_bias = this->bias_view_;
        array_view<const float, 4> output_weights = this->weights_view_;

        // read write
        array_view<float, 3> top_layer_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;
        array_view<float> outputs = this->outputs_view_;

        // calculate base score, ignore top layer activation
        parallel_for_each(top_layer_value.extent, [=](index<3> idx) restrict(amp)
        {
            array_view<const float, 3> current_neuron = neuron_weights[idx[0]];// projection
            index<3> base_idx(0, idx[1], idx[2]);

            float result = hbias[idx[0]];

            for (int depth_idx = 0; depth_idx < current_neuron.extent[0]; depth_idx++)
            {
                for (int height_idx = 0; height_idx < current_neuron.extent[1]; height_idx++)
                {
                    for (int width_idx = 0; width_idx < current_neuron.extent[2]; width_idx++)
                    {
                        index<3> neuron_idx(depth_idx, height_idx, width_idx);
                        result += bottom_layer_value[base_idx + neuron_idx] * current_neuron[neuron_idx];
                    }
                }
            }

            top_layer_value[idx] = result;
        });

        parallel_for_each(outputs.extent, [=](index<1> idx) restrict(amp)
        {
            float result = output_bias[idx];

            auto& current_output_weights = output_weights[idx[0]];

            for (int depth_idx = 0; depth_idx < top_layer_value.extent[0]; depth_idx++)
            {
                for (int height_idx = 0; height_idx < top_layer_value.extent[1]; height_idx++)
                {
                    for (int width_idx = 0; width_idx < top_layer_value.extent[2]; width_idx++)
                    {
                        float score = top_layer_value(depth_idx, height_idx, width_idx) 
                            + current_output_weights(depth_idx, height_idx, width_idx);
                        result += fast_math::logf((fast_math::expf(score) + 1.0f) * (1.0f - dropout_prob) + 2.0f * dropout_prob);
                    }
                }
            }

            outputs[idx] = result;
        });

        outputs.synchronize();
        int max_idx = 0;
        float max_value = outputs_[max_idx];

        for (int i = 1; i < outputs_.size(); i++)
        {
            if (outputs_[i] > max_value)
            {
                max_value = outputs_[i];
                max_idx = i;
            }
        }

        return max_idx;;
    }

    void OutputLayer::PassDown(const DataLayer& top_layer, bool top_switcher, bool output_switcher)
    {
        assert(top_layer.depth() == this->input_depth());
        assert(top_layer.width() == this->input_width());
        assert(top_layer.height() == this->input_height());

        // readonly
        array_view<const float, 3> top_layer_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;
        array_view<const float> output_layer_bias = bias_view_;
        array_view<const float, 4> output_layer_weights = weights_view_;

        // writeonly
        array_view<float> output_layer_value = output_switcher ? outputs_view_ : next_outputs_view_;
        output_layer_value.discard_data();

        // non-tiled version
        parallel_for_each(output_layer_value.extent,
            [=](index<1> idx) restrict(amp)
        {
            float result = output_layer_bias[idx];

            const auto& weights = output_layer_weights[idx[0]];

            for (int depth_idx = 0; depth_idx < top_layer_value.extent[0]; depth_idx++)
            {
                for (int height_idx = 0; height_idx < top_layer_value.extent[1]; height_idx++)
                {
                    for (int width_idx = 0; width_idx < top_layer_value.extent[2]; width_idx++)
                    {
                        result += top_layer_value(depth_idx, height_idx, width_idx)
                            * weights(depth_idx, height_idx, width_idx);
                    }
                }
            }

            output_layer_value[idx] = 1.0f / (1.0f + fast_math::expf(-result));
        });

#ifdef DEBUG_SYNC
        output_layer_value.synchronize();
#endif
    }

    bitmap_image OutputLayer::GenerateImage() const
    {
        weights_view_.synchronize();
        bias_view_.synchronize();
        
        bitmap_image image;

        const int block_size = 2;

        float max_abs_weight = std::numeric_limits<float>::min();
        for (float weight : weights_)
        {
            max_abs_weight = std::max(max_abs_weight, std::abs(weight));
        }

        float max_abs_bias = std::numeric_limits<float>::min();
        for (float bias : bias_)
        {
            max_abs_bias = std::max(max_abs_bias, std::abs(bias));
        }

        if (input_width() == 1 && input_height() == 1)
        {
            image.setwidth_height((2 + input_depth()) * (block_size + 1), (2 + output_num()) * (block_size + 1), true);

            for (int i = 0; i < bias_.size(); i++)
            {
                image.set_region((2 + i) * (block_size + 1), 0, block_size, block_size,
                    bias_[i] >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(std::abs(bias_[i]) / max_abs_bias * 255.0));
            }

            for (int output_idx = 0; output_idx < output_num(); output_idx++)
            {
                for (int depth_idx = 0; depth_idx < input_depth(); depth_idx++)
                {
                    float value = weights_[output_idx * input_depth() + depth_idx];
                    image.set_region((2 + depth_idx) * (block_size + 1), (2 + output_idx) * (block_size + 1), block_size, block_size,
                        value >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                        static_cast<unsigned char>(std::abs(value) / max_abs_weight * 255.0));
                }
            }
        }
        else
        {
            image.setwidth_height((2 + input_depth() * (input_width() + 1)) * (block_size + 1),
                (2 + output_num() * (input_height() + 1)) * (block_size + 1), true);

            for (int i = 0; i < bias_.size(); i++)
            {
                image.set_region((2 + input_width() / 2 + (input_width() + 1) * i) * (block_size + 1), 0, block_size, block_size,
                    bias_[i] >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(std::abs(bias_[i]) / max_abs_bias * 255.0));
            }

            for (int output_idx = 0; output_idx < output_num(); output_idx++)
            {
                for (int depth_idx = 0; depth_idx < input_depth(); depth_idx++)
                {
                    for (int height_idx = 0; height_idx < input_height(); height_idx++)
                    {
                        for (int width_idx = 0; width_idx < input_width(); width_idx++)
                        {
                            float value = weights_[output_idx * input_depth() * input_height() * input_width()
                                + depth_idx * input_height() * input_width() + height_idx * input_width() + width_idx];

                            image.set_region((2 + width_idx + depth_idx * (input_width() + 1)) * (block_size + 1),
                                (2 + output_idx * (input_height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                                value >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                                static_cast<unsigned char>(std::abs(value) / max_abs_weight * 255.0));
                        }
                    }
                }
            }
        }

        return image;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    ConvolveLayer::ConvolveLayer(int num_neuron, int neuron_depth, int neuron_height, int neuron_width)
        : weights_(num_neuron * neuron_depth * neuron_height * neuron_width),
        weights_view_(extent<4>(std::array<int, 4>{{ num_neuron, neuron_depth, neuron_height, neuron_width }}.data()), weights_),
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

        array_view<const int, 3> top_layer_active = top_layer.active_view_;

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
            if (top_layer_active[idx] == 0)
            {
                top_layer_expect[idx] = 0.0f;
                top_layer_value[idx] = 0.0f;
            }
            else
            {
                array_view<const float, 3> current_neuron = neuron_weights[idx[0]];// projection
                index<3> base_idx(0, idx[1], idx[2]);

                float result = hbias[idx[0]];

                for (int depth_idx = 0; depth_idx < current_neuron.extent[0]; depth_idx++)
                {
                    for (int height_idx = 0; height_idx < current_neuron.extent[1]; height_idx++)
                    {
                        for (int width_idx = 0; width_idx < current_neuron.extent[2]; width_idx++)
                        {
                            index<3> neuron_idx(depth_idx, height_idx, width_idx);
                            result += bottom_layer_value[base_idx + neuron_idx] * current_neuron[neuron_idx];
                        }
                    }
                }

                // Logistic activation function. Maybe more types of activation function later.
                float prob = 1.0f / (1.0f + fast_math::expf(-result));
                top_layer_expect[idx] = prob;
                top_layer_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;
            }
        });

#ifdef DEBUG_SYNC
        top_layer_value.synchronize();
        top_layer_expect.synchronize();
#endif
    }

    void ConvolveLayer::PassUp(const DataLayer& bottom_layer, bool bottom_switcher,
        const OutputLayer& output_layer, bool output_switcher,
        DataLayer& top_layer, bool top_switcher) const
    {
        assert(top_layer.depth() == this->neuron_num() && top_layer.depth() == output_layer.input_depth());
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1 && top_layer.width() == output_layer.input_width());
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1 && top_layer.height() == output_layer.input_height());

        // readonly
        array_view<const float, 4> neuron_weights = weights_view_;
        array_view<const float> hbias = hbias_view_;
        array_view<const float, 3> bottom_layer_value = bottom_switcher ? bottom_layer.value_view_ : bottom_layer.next_value_view_;
        array_view<const float> output_layer_value = output_switcher ? output_layer.outputs_view_ : output_layer.next_outputs_view_;
        array_view<const float> output_layer_bias = output_layer.bias_view_;
        array_view<const float, 4> output_layer_weights = output_layer.weights_view_;

        array_view<const int, 3> top_layer_active = top_layer.active_view_;

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
            if (top_layer_active[idx] == 0)
            {
                top_layer_expect[idx] = 0.0f;
                top_layer_value[idx] = 0.0f;
            }
            else
            {
                array_view<const float, 3> current_neuron = neuron_weights[idx[0]];// projection
                index<3> base_idx(0, idx[1], idx[2]);

                float result = hbias[idx[0]];

                for (int output_idx = 0; output_idx < output_layer_value.extent[0]; output_idx++)
                {
                    result += output_layer_value[output_idx] * output_layer_weights[output_idx][idx];
                }

                for (int depth_idx = 0; depth_idx < current_neuron.extent[0]; depth_idx++)
                {
                    for (int height_idx = 0; height_idx < current_neuron.extent[1]; height_idx++)
                    {
                        for (int width_idx = 0; width_idx < current_neuron.extent[2]; width_idx++)
                        {
                            index<3> neuron_idx(depth_idx, height_idx, width_idx);
                            result += bottom_layer_value[base_idx + neuron_idx] * current_neuron[neuron_idx];
                        }
                    }
                }

                // Logistic activation function. Maybe more types of activation function later.
                float prob = 1.0f / (1.0f + fast_math::expf(-result));
                top_layer_expect[idx] = prob;
                top_layer_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;
            }
        });

#ifdef DEBUG_SYNC
        top_layer_value.synchronize();
        top_layer_expect.synchronize();
#endif
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

        array_view<const int, 3> bottom_layer_active = bottom_layer.active_view_;

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
            if (bottom_layer_active[idx] == 0)
            {
                bottom_layer_expect[idx] = 0.0f;
                bottom_layer_value[idx] = 0.0f;
            }
            else
            {
                int cur_depth_idx = idx[0];
                int cur_height_idx = idx[1];
                int cur_width_idx = idx[2];

                float result = vbias[cur_depth_idx];

                for (int neuron_idx = 0; neuron_idx < neuron_weights.extent[0]; neuron_idx++)
                {
                    array_view<const float, 3> current_neuron = neuron_weights[neuron_idx];

                    for (int height_idx = 0; height_idx < current_neuron.extent[1]; height_idx++)
                    {
                        for (int width_idx = 0; width_idx < current_neuron.extent[2]; width_idx++)
                        {
                            // make sure the convolve window fits in the bottom layer
                            if (cur_width_idx - width_idx >= 0 && cur_height_idx - height_idx >= 0 &&
                                cur_height_idx - height_idx + current_neuron.extent[1] <= bottom_layer_value.extent[1] &&
                                cur_width_idx - width_idx + current_neuron.extent[2] <= bottom_layer_value.extent[2])
                            {
                                result += current_neuron(cur_depth_idx, height_idx, width_idx) *
                                    top_layer_value(neuron_idx, cur_height_idx - height_idx, cur_width_idx - width_idx);
                            }
                        }
                    }
                }

                // Logistic activation function. Maybe more types of activation function later.
                float prob = 1.0f / (1.0f + fast_math::expf(-result));
                bottom_layer_expect[idx] = prob;
                bottom_layer_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;
            }
        });

#ifdef DEBUG_SYNC
        bottom_layer_value.synchronize();
        bottom_layer_expect.synchronize();
#endif
    }

    void ConvolveLayer::PassDown(const DataLayer& top_layer, bool top_switcher,
        DataLayer& bottom_layer, bool bottom_switcher,
        OutputLayer& output_layer, bool output_switcher) const
    {
        assert(top_layer.depth() == this->neuron_num() && top_layer.depth() == output_layer.input_depth());
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1 && top_layer.width() == output_layer.input_width());
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1 && top_layer.height() == output_layer.input_height());

        // readonly
        array_view<const float, 4> neuron_weights = weights_view_;
        array_view<const float> vbias = vbias_view_;
        array_view<const float, 3> top_layer_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;
        array_view<const float> output_layer_bias = output_layer.bias_view_;
        array_view<const float, 4> output_layer_weights = output_layer.weights_view_;

        array_view<const int, 3> bottom_layer_active = bottom_layer.active_view_;

        // writeonly
        array_view<float, 3> bottom_layer_value = bottom_switcher ? bottom_layer.value_view_ : bottom_layer.next_value_view_;
        array_view<float, 3> bottom_layer_expect = bottom_switcher ? bottom_layer.expect_view_ : bottom_layer.next_expect_view_;
        bottom_layer_value.discard_data();
        bottom_layer_expect.discard_data();
        array_view<float> output_layer_value = output_switcher ? output_layer.outputs_view_ : output_layer.next_outputs_view_;
        output_layer_value.discard_data();

        auto& rand_collection = bottom_layer.rand_collection_;

        // non-tiled version
        parallel_for_each(bottom_layer_value.extent,
            [=](index<3> idx) restrict(amp)
        {
            if (bottom_layer_active[idx] == 0)
            {
                bottom_layer_expect[idx] = 0.0f;
                bottom_layer_value[idx] = 0.0f;
            }
            else
            {
                int cur_depth_idx = idx[0];
                int cur_height_idx = idx[1];
                int cur_width_idx = idx[2];

                float result = vbias[cur_depth_idx];

                for (int neuron_idx = 0; neuron_idx < neuron_weights.extent[0]; neuron_idx++)
                {
                    array_view<const float, 3> current_neuron = neuron_weights[neuron_idx];

                    for (int height_idx = 0; height_idx < current_neuron.extent[1]; height_idx++)
                    {
                        for (int width_idx = 0; width_idx < current_neuron.extent[2]; width_idx++)
                        {
                            // make sure the convolve window fits in the bottom layer
                            if (cur_width_idx - width_idx >= 0 && cur_height_idx - height_idx >= 0 &&
                                cur_height_idx - height_idx + current_neuron.extent[1] <= bottom_layer_value.extent[1] &&
                                cur_width_idx - width_idx + current_neuron.extent[2] <= bottom_layer_value.extent[2])
                            {
                                result += current_neuron(cur_depth_idx, height_idx, width_idx) *
                                    top_layer_value(neuron_idx, cur_height_idx - height_idx, cur_width_idx - width_idx);
                            }
                        }
                    }
                }

                // Logistic activation function. Maybe more types of activation function later.
                float prob = 1.0f / (1.0f + fast_math::expf(-result));
                bottom_layer_expect[idx] = prob;
                bottom_layer_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;
            }
        });

        // non-tiled version
        parallel_for_each(output_layer_value.extent,
            [=](index<1> idx) restrict(amp)
        {
            float result = output_layer_bias[idx];

            const auto& weights = output_layer_weights[idx[0]];

            for (int depth_idx = 0; depth_idx < top_layer_value.extent[0]; depth_idx++)
            {
                for (int height_idx = 0; height_idx < top_layer_value.extent[1]; height_idx++)
                {
                    for (int width_idx = 0; width_idx < top_layer_value.extent[2]; width_idx++)
                    {
                        result += top_layer_value(depth_idx, height_idx, width_idx) 
                            * weights(depth_idx, height_idx, width_idx);
                    }
                }
            }

            output_layer_value[idx] = 1.0f / (1.0f + fast_math::expf(-result));
        });

#ifdef DEBUG_SYNC
        bottom_layer_value.synchronize();
        bottom_layer_expect.synchronize();
        output_layer_value.synchronize();
#endif
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

            for (int top_height_idx = 0; top_height_idx < top_layer_expect.extent[1]; top_height_idx++)
            {
                for (int top_width_idx = 0; top_width_idx < top_layer_expect.extent[2]; top_width_idx++)
                {
                    float top_expect = top_layer_expect(neuron_idx, top_height_idx, top_width_idx);
                    float top_next_expect = top_layer_next_expect(neuron_idx, top_height_idx, top_width_idx);

                    float bottom_value = bottom_layer_value(idx[1], idx[2] + top_height_idx, idx[3] + top_width_idx);
                    float bottom_next_value = bottom_layer_next_value(idx[1], idx[2] + top_height_idx, idx[3] + top_width_idx);

                    delta += bottom_value * top_expect - bottom_next_value * top_next_expect;
                }
            }

            weights[idx] += delta / (top_layer_expect.extent[1] * top_layer_expect.extent[2]) * learning_rate;
        });

        parallel_for_each(vbias.extent, [=](index<1> idx) restrict(amp)
        {
            float delta = 0.0f;

            int depth_idx = idx[0];

            for (int bottom_height_idx = 0; bottom_height_idx < bottom_layer_value.extent[1]; bottom_height_idx++)
            {
                for (int bottom_width_idx = 0; bottom_width_idx < bottom_layer_value.extent[2]; bottom_width_idx++)
                {
                    float bottom_value = bottom_layer_value(depth_idx, bottom_height_idx, bottom_width_idx);
                    float bottom_next_value = bottom_layer_next_value(depth_idx, bottom_height_idx, bottom_width_idx);

                    delta += bottom_value - bottom_next_value;
                }
            }

            vbias[idx] += delta / (bottom_layer_value.extent[1] * bottom_layer_value.extent[2]) * learning_rate;
        });

        parallel_for_each(hbias.extent, [=](index<1> idx) restrict(amp)
        {
            float delta = 0.0f;

            int neuron_idx = idx[0];

            for (int top_height_idx = 0; top_height_idx < top_layer_expect.extent[1]; top_height_idx++)
            {
                for (int top_width_idx = 0; top_width_idx < top_layer_expect.extent[2]; top_width_idx++)
                {
                    float top_expect = top_layer_expect(neuron_idx, top_height_idx, top_width_idx);
                    float top_next_expect = top_layer_next_expect(neuron_idx, top_height_idx, top_width_idx);

                    delta += top_expect - top_next_expect;
                }
            }

            hbias[idx] += delta / (top_layer_expect.extent[1] * top_layer_expect.extent[2]) * learning_rate;
        });

#ifdef DEBUG_SYNC
        weights.synchronize();
        vbias.synchronize();
        hbias.synchronize();
#endif
    }

    void ConvolveLayer::Train(const DataLayer& bottom_layer, OutputLayer& output_layer, const DataLayer& top_layer,
        float learning_rate, bool buffered_update, bool discriminative)
    {
        array_view<float, 4> weights = buffered_update ? weights_delta_ : weights_view_;
        array_view<float> vbias = buffered_update ? vbias_delta_ : vbias_view_;
        array_view<float> hbias = buffered_update ? hbias_delta_ : hbias_view_;
        array_view<float, 4> output_weights = buffered_update ? output_layer.weights_delta_ : output_layer.weights_view_;
        array_view<float> output_bias = buffered_update ? output_layer.bias_delta_ : output_layer.bias_view_;

        array_view<const float, 3> top_layer_expect = top_layer.expect_view_;
        array_view<const float, 3> top_layer_next_expect = top_layer.next_expect_view_;
        array_view<const float, 3> bottom_layer_value = bottom_layer.value_view_;
        array_view<const float, 3> bottom_layer_next_value = discriminative ? bottom_layer.value_view_ : bottom_layer.next_value_view_;
        array_view<const float> output_layer_value = output_layer.outputs_view_;
        array_view<const float> output_layer_next_value = output_layer.next_outputs_view_;

        // non-tiled version
        parallel_for_each(weights.extent, [=](index<4> idx) restrict(amp)
        {
            float delta = 0.0f;

            int neuron_idx = idx[0];

            for (int top_height_idx = 0; top_height_idx < top_layer_expect.extent[1]; top_height_idx++)
            {
                for (int top_width_idx = 0; top_width_idx < top_layer_expect.extent[2]; top_width_idx++)
                {
                    float top_expect = top_layer_expect(neuron_idx, top_height_idx, top_width_idx);
                    float top_next_expect = top_layer_next_expect(neuron_idx, top_height_idx, top_width_idx);

                    float bottom_value = bottom_layer_value(idx[1], idx[2] + top_height_idx, idx[3] + top_width_idx);
                    float bottom_next_value = bottom_layer_next_value(idx[1], idx[2] + top_height_idx, idx[3] + top_width_idx);

                    delta += bottom_value * top_expect - bottom_next_value * top_next_expect;
                }
            }

            weights[idx] += delta / (top_layer_expect.extent[1] * top_layer_expect.extent[2]) * learning_rate;
        });

        parallel_for_each(vbias.extent, [=](index<1> idx) restrict(amp)
        {
            float delta = 0.0f;

            int depth_idx = idx[0];

            for (int bottom_height_idx = 0; bottom_height_idx < bottom_layer_value.extent[1]; bottom_height_idx++)
            {
                for (int bottom_width_idx = 0; bottom_width_idx < bottom_layer_value.extent[2]; bottom_width_idx++)
                {
                    float bottom_value = bottom_layer_value(depth_idx, bottom_height_idx, bottom_width_idx);
                    float bottom_next_value = bottom_layer_next_value(depth_idx, bottom_height_idx, bottom_width_idx);

                    delta += bottom_value - bottom_next_value;
                }
            }

            vbias[idx] += delta / (bottom_layer_value.extent[1] * bottom_layer_value.extent[2]) * learning_rate;
        });

        parallel_for_each(hbias.extent, [=](index<1> idx) restrict(amp)
        {
            float delta = 0.0f;

            int neuron_idx = idx[0];

            for (int top_height_idx = 0; top_height_idx < top_layer_expect.extent[1]; top_height_idx++)
            {
                for (int top_width_idx = 0; top_width_idx < top_layer_expect.extent[2]; top_width_idx++)
                {
                    float top_expect = top_layer_expect(neuron_idx, top_height_idx, top_width_idx);
                    float top_next_expect = top_layer_next_expect(neuron_idx, top_height_idx, top_width_idx);

                    delta += top_expect - top_next_expect;
                }
            }

            hbias[idx] += delta / (top_layer_expect.extent[1] * top_layer_expect.extent[2]) * learning_rate;
        });

        // for output layer
        parallel_for_each(output_weights.extent, [=](index<4> idx) restrict(amp)
        {
            float delta = output_layer_value(idx[0]) * top_layer_expect(idx[1], idx[2], idx[3]) - 
                output_layer_next_value(idx[0]) * top_layer_next_expect(idx[1], idx[2], idx[3]);

            output_weights[idx] += delta * learning_rate;

        });

        parallel_for_each(output_bias.extent, [=](index<1> idx) restrict(amp)
        {
            float delta = output_layer_value[idx] - output_layer_next_value[idx];

            output_bias[idx] += delta * learning_rate;
        });

#ifdef DEBUG_SYNC
        weights.synchronize();
        vbias.synchronize();
        hbias.synchronize();
        output_bias.synchronize();
        output_weights.synchronize();
#endif
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

#ifdef DEBUG_SYNC
        weights.synchronize();
        vbias.synchronize();
        hbias.synchronize();
#endif
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

    bitmap_image ConvolveLayer::GenerateImage() const
    {
        weights_view_.synchronize();
        vbias_view_.synchronize();
        hbias_view_.synchronize();

        bitmap_image image;
        
        const int block_size = 2;

        float max_abs_weight = std::numeric_limits<float>::min();
        for (float weight : weights_)
        {
            max_abs_weight = std::max(max_abs_weight, std::abs(weight));
        }

        float max_abs_vbias = std::numeric_limits<float>::min();
        for (float vbias : vbias_)
        {
            max_abs_vbias = std::max(max_abs_vbias, std::abs(vbias));
        }

        float max_abs_hbias = std::numeric_limits<float>::min();
        for (float hbias : hbias_)
        {
            max_abs_hbias = std::max(max_abs_hbias, std::abs(hbias));
        }

        if (neuron_width() == 1 && neuron_height() == 1)
        {
            image.setwidth_height((2 + neuron_depth()) * (block_size + 1), (2 + neuron_num()) * (block_size + 1), true);

            for (int i = 0; i < vbias_.size(); i++)
            {
                image.set_region((2 + i) * (block_size + 1), 0, block_size, block_size,
                    vbias_[i] >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(std::abs(vbias_[i]) / max_abs_vbias * 255.0));
            }

            for (int i = 0; i < hbias_.size(); i++)
            {
                image.set_region(0, (2 + i) * (block_size + 1), block_size, block_size,
                    hbias_[i] >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(std::abs(hbias_[i]) / max_abs_hbias * 255.0));
            }

            for (int neuron_idx = 0; neuron_idx < neuron_num(); neuron_idx++)
            {
                for (int depth_idx = 0; depth_idx < neuron_depth(); depth_idx++)
                {
                    float value = weights_[neuron_idx * neuron_depth() + depth_idx];
                    image.set_region((2 + depth_idx) * (block_size + 1), (2 + neuron_idx) * (block_size + 1), block_size, block_size,
                        value >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                        static_cast<unsigned char>(std::abs(value) / max_abs_weight * 255.0));
                }
            }
        }
        else
        {
            image.setwidth_height((2 + neuron_depth() * (neuron_width() + 1)) * (block_size + 1),
                (2 + neuron_num() * (neuron_height() + 1)) * (block_size + 1), true);

            for (int i = 0; i < vbias_.size(); i++)
            {
                image.set_region((2 + neuron_width() / 2 + (neuron_width() + 1) * i) * (block_size + 1), 0, block_size, block_size,
                    vbias_[i] >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(std::abs(vbias_[i]) / max_abs_vbias * 255.0));
            }

            for (int i = 0; i < hbias_.size(); i++)
            {
                image.set_region(0, (2 + neuron_height() / 2 + (neuron_height() + 1) * i) * (block_size + 1), block_size, block_size,
                    hbias_[i] >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(std::abs(hbias_[i]) / max_abs_hbias * 255.0));
            }

            for (int neuron_idx = 0; neuron_idx < neuron_num(); neuron_idx++)
            {
                for (int depth_idx = 0; depth_idx < neuron_depth(); depth_idx++)
                {
                    for (int height_idx = 0; height_idx < neuron_height(); height_idx++)
                    {
                        for (int width_idx = 0; width_idx < neuron_width(); width_idx++)
                        {
                            float value = weights_[neuron_idx * neuron_depth() * neuron_height() * neuron_width()
                                + depth_idx * neuron_height() * neuron_width() + height_idx * neuron_width() + width_idx];

                            image.set_region((2 + width_idx + depth_idx * (neuron_width() + 1)) * (block_size + 1),
                                (2 + neuron_idx * (neuron_height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                                value >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane, 
                                static_cast<unsigned char>(std::abs(value) / max_abs_weight * 255.0));
                        }
                    }
                }
            }
        }

        return image;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    PoolingLayer::PoolingLayer(int block_height, int block_width)
        : block_height_(block_height), block_width_(block_width)
    {

    }

    void PoolingLayer::PassUp(const DataLayer& bottom_layer, bool bottom_switcher,
        DataLayer& top_layer, bool top_switcher) const
    {
        assert(top_layer.height() * block_height_ == bottom_layer.height());
        assert(top_layer.width() * block_width_ == bottom_layer.width());

        // readonly
        int block_height = block_height_;
        int block_width = block_width_;
        
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
            
            for (int height_idx = 0; height_idx < block_height; height_idx++)
            {
                for (int width_idx = 0; width_idx < block_width; width_idx++)
                {
                    float value = bottom_layer_value(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx);
                    float expect = bottom_layer_expect(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx);

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
        assert(top_layer.height() * block_height_ == bottom_layer.height());
        assert(top_layer.width() * block_width_ == bottom_layer.width());

        // readonly
        int block_height = block_height_;
        int block_width = block_width_;
        
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

            int height_idx = idx[1] / block_height;// trunc towards zero
            int width_idx = idx[2] / block_width;
            

            bottom_layer_expect[idx] = 1.0f - fast_math::powf(1.0f - top_layer_expect(idx[0], height_idx, width_idx),
                -1.0f * block_width * block_height);
            bottom_layer_value[idx] = 0.0f;// clear the value
        });

        parallel_for_each(top_layer_value.extent, [=](index<3> idx) restrict(amp)
        {
            if (top_layer_value[idx] == 1.0f)
            {
                // randomly select a node in bottom_layer to activate
                int height_idx = rand_collection[idx].next_uint() % block_height;
                int width_idx = rand_collection[idx].next_uint() % block_width;
                
                bottom_layer_value(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx) = 1.0f;
            }
        });
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    DeepModel::DeepModel(unsigned int model_seed) : random_engine_(model_seed)
    {

    }

    void DeepModel::AddDataLayer(int depth, int height, int width, int seed)
    {
        data_layers_.emplace_back(depth, height, width, seed);
    }

    void DeepModel::AddConvolveLayer(int num_neuron, int neuron_depth, int neuron_height, int neuron_width, unsigned int rand_seed)
    {
        convolve_layers_.emplace_back(num_neuron, neuron_depth, neuron_height, neuron_width);
        convolve_layers_.back().RandomizeParams(rand_seed);
    }

    void DeepModel::AddOutputLayer(int data_layer_idx, int output_num, unsigned int seed)
    {
        auto& data_layer = data_layers_[data_layer_idx];

        output_layers_.emplace(std::piecewise_construct, std::forward_as_tuple(data_layer_idx),
            std::forward_as_tuple(output_num, data_layer.depth(), data_layer.height(), data_layer.width()));

        output_layers_.at(data_layer_idx).RandomizeParams(seed);
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

            top_data_layer.Activate();

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

            bottom_data_layer.Activate();

            conv_layer.PassDown(top_data_layer, false, bottom_data_layer, false);
        }
    }

    float DeepModel::TrainLayer(const std::vector<float>& data, int layer_idx, float learning_rate, float dropout_prob)
    {
        auto& bottom_layer = data_layers_[layer_idx];
        auto& top_layer = data_layers_[layer_idx + 1];

        auto& conv_layer = convolve_layers_[layer_idx];

        // train with contrastive divergence (CD) algorithm to maximize likelihood on dataset
        bottom_layer.SetValue(data);
        top_layer.Activate(1.0f - dropout_prob);

        conv_layer.PassUp(bottom_layer, true, top_layer, true);
        conv_layer.PassDown(top_layer, true, bottom_layer, false);
        conv_layer.PassUp(bottom_layer, false, top_layer, false);

        conv_layer.Train(bottom_layer, top_layer, learning_rate, false);

        return bottom_layer.ReconstructionError();
    }

    float DeepModel::TrainLayer(const std::vector<float>& data, const int label, int layer_idx,
        float learning_rate, float dropout_prob, bool discriminative)
    {
        auto& bottom_layer = data_layers_[layer_idx];
        auto& top_layer = data_layers_[layer_idx + 1];
        auto& conv_layer = convolve_layers_[layer_idx];
        auto& output_layer = output_layers_.at(layer_idx + 1);

        // train with contrastive divergence (CD) algorithm to maximize likelihood on dataset
        bottom_layer.SetValue(data);
        top_layer.Activate(1.0f - dropout_prob);
        output_layer.SetLabel(label);

        conv_layer.PassUp(bottom_layer, true, output_layer, true, top_layer, true);

        if (discriminative)
        {
            output_layer.PassDown(top_layer, true, false);
            conv_layer.PassUp(bottom_layer, true, output_layer, false, top_layer, false);
        }
        else
        {
            conv_layer.PassDown(top_layer, true, bottom_layer, false, output_layer, false);
            conv_layer.PassUp(bottom_layer, false, output_layer, false, top_layer, false);
        }
        
        conv_layer.Train(bottom_layer, output_layer, top_layer, learning_rate, false, discriminative);

        return bottom_layer.ReconstructionError();
    }

    void DeepModel::TrainLayer(const std::vector<const std::vector<float>>& dataset,
        int layer_idx, int mini_batch_size, float learning_rate, float dropout_prob, int iter_count)
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
                top_layer.Activate(1.0f - dropout_prob);

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

    void DeepModel::TrainLayer(const std::vector<const std::vector<float>>& dataset, const std::vector<const int>& labels,
        int layer_idx, int mini_batch_size, float learning_rate, float dropout_prob, int iter_count, bool discriminative)
    {
        assert(dataset.size() == labels.size());

        std::uniform_int_distribution<int> dataset_uniform_dist(0, (int)dataset.size() - 1);

        auto& bottom_layer = data_layers_[layer_idx];
        auto& top_layer = data_layers_[layer_idx + 1];
        auto& conv_layer = convolve_layers_[layer_idx];
        auto& output_layer = output_layers_.at(layer_idx + 1);

        for (int iter = 0; iter < iter_count; iter++)
        {
            // sample mini-batch, sample with replacement
            for (int mini_batch_idx = 0; mini_batch_idx < mini_batch_size; mini_batch_idx++)
            {
                top_layer.Activate(1.0f - dropout_prob);

                int data_idx = dataset_uniform_dist(random_engine_);
                auto& data = dataset[data_idx];
                int label = labels[data_idx];

                bottom_layer.SetValue(data);
                output_layer.SetLabel(label);

                conv_layer.PassUp(bottom_layer, true, output_layer, true, top_layer, true);
                if (discriminative)
                {
                    output_layer.PassDown(top_layer, true, false);
                    conv_layer.PassUp(bottom_layer, true, output_layer, false, top_layer, false);
                }
                else
                {
                    conv_layer.PassDown(top_layer, true, bottom_layer, false, output_layer, false);
                    conv_layer.PassUp(bottom_layer, false, output_layer, false, top_layer, false);
                }
                
                conv_layer.Train(bottom_layer, output_layer, top_layer, learning_rate, true, discriminative);
            }

            conv_layer.ApplyBufferedUpdate(mini_batch_size);
            output_layer.ApplyBufferedUpdate(mini_batch_size);

            std::cout << (discriminative ? "[D]" : "[G]") << "iter = " << iter << "\t err = " << bottom_layer.ReconstructionError() << std::endl;
        }
    }

    int DeepModel::PredictLabel(const std::vector<float>& data, const int layer_idx, const float dropout_prob)
    {
        auto& bottom_layer = data_layers_[layer_idx];
        auto& top_layer = data_layers_[layer_idx + 1];
        auto& conv_layer = convolve_layers_[layer_idx];
        auto& output_layer = output_layers_.at(layer_idx + 1);

        bottom_layer.SetValue(data);
        // top layer activation is ignored when predicting labels
        return output_layer.PredictLabel(bottom_layer, true, top_layer, true, conv_layer, dropout_prob);
    }

    float DeepModel::Evaluate(const std::vector<const std::vector<float>>& dataset, const std::vector<const int>& labels,
        int layer_idx, const float dropout_prob)
    {
        assert(dataset.size() == labels.size());

        float correct_count = 0.0f;

        for (int i = 0; i < dataset.size(); i++)
        {
            int predicted_label = PredictLabel(dataset[i], layer_idx, dropout_prob);
            if (predicted_label == labels[i])
            {
                correct_count++;
            }
        }

        return correct_count / labels.size();
    }

    void DeepModel::GenerateImages(const std::string& folder) const
    {
        for (int i = 0; i < data_layers_.size(); i++)
        {
            data_layers_[i].GenerateImage().save_image(folder + "\\layer" + std::to_string(i) + "_data.bmp");
        }
        
        for (int i = 0; i < convolve_layers_.size(); i++)
        {
            convolve_layers_[i].GenerateImage().save_image(folder + "\\layer" + std::to_string(i) + "_conv.bmp");
        }

        for (const auto& pair : output_layers_)
        {
            pair.second.GenerateImage().save_image(folder + "\\layer" + std::to_string(pair.first) + "_output.bmp");
        }
    }
}
