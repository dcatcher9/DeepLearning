#include "DeepModel.h"

#include <assert.h>
#include <random>
#include <amp_math.h>

#include <iostream>

#include "AmpUtility.h"

namespace deep_learning_lib
{
    using namespace concurrency;

#pragma region data layer

    DataLayer::DataLayer(int memory_num, int depth, int height, int width, int seed)
        : memory_num_(memory_num),
        // put memory at the end
        data_array_(make_extent(6 + memory_num, depth, height, width)),
        value_view_(data_array_[0]),
        expect_view_(data_array_[1]),
        next_value_view_(data_array_[2]),
        next_expect_view_(data_array_[3]),
        temp_value_view_(data_array_[4]),
        temp_expect_view_(data_array_[5]),
        active_prob_(1.0f),
        active_(value_view_.extent.size(), 1),
        active_view_(value_view_.extent, active_),
        // there is no empty array_view support in amp now, 
        // so we just map the memory_view_ to the whole data view when the memory_num == 0
        memory_view_(memory_num == 0 ? data_array_ : data_array_.section(make_index(6, 0, 0, 0), make_extent(memory_num, depth, height, width))),
        rand_collection_(value_view_.extent, seed)
    {
        fill(data_array_, 0.0f);
    }

    DataLayer::DataLayer(DataLayer&& other)
        : memory_num_(other.memory_num_),
        data_array_(std::move(other.data_array_)),
        value_view_(other.value_view_),
        expect_view_(other.expect_view_),
        next_value_view_(other.next_value_view_),
        next_expect_view_(other.next_expect_view_),
        temp_value_view_(other.temp_value_view_),
        temp_expect_view_(other.temp_expect_view_),
        active_prob_(other.active_prob_),
        active_(std::move(other.active_)),
        active_view_(other.active_view_),
        memory_view_(other.memory_view_),
        rand_collection_(other.rand_collection_)
    {
    }

    void DataLayer::SetValue(const std::vector<float>& data)
    {
        assert(data.size() == value_view_.extent.size());

        // Copy the data
        concurrency::copy(data, value_view_);

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
        array_view<const float, 3> next_value_view = next_value_view_;

        // TODO: compare with reduce method for performance
        parallel_for_each(value_view.extent,
            [=](index<3> idx) restrict(amp)
        {
            float diff = value_view[idx] - next_value_view[idx];
            atomic_fetch_add(&result(0), diff * diff);
        });

        return std::sqrtf(result(0));
    }

    //bool DataLayer::Memorize()
    //{
    //    array_view<float> diffs_view(memory_pool_size());

    //    array_view<const float, 3> value_view = value_view_;
    //    array_view<const float, 4> memory_pool_view = memory_pool_view_;

    //    parallel_for_each(value_view.extent,
    //        [=](index<3> idx) restrict(amp)
    //    {
    //        for (int i = 0; i < diffs_view.extent[0]; i++)
    //        {
    //            float diff = memory_pool_view[i][idx] - value_view[idx];
    //            atomic_fetch_add(&diffs_view(i), diff * diff);
    //        }
    //    });

    //    float min_diff = std::numeric_limits<float>::max();
    //    int min_idx = -1;

    //    for (int i = 0; i < diffs_view.extent[0]; i++)
    //    {
    //        float diff = std::sqrtf(diffs_view(i));
    //        if (diff < min_diff)
    //        {
    //            min_diff = diff;
    //            min_idx = i;
    //        }
    //    }

    //    float recon_error = ReconstructionError();

    //    // memory refreshment logic, adaboost style
    //    if (recon_error <= min_diff)
    //    {
    //        // current value is too new or already well recognized. 
    //        int min_intensity_idx = -1;
    //        float min_intensity = std::numeric_limits<float>::max();

    //        for (int j = 0; j < memory_intensity_.size(); j++)
    //        {
    //            float& intensity = memory_intensity_[j];

    //            intensity *= kMemoryDecayRate;

    //            if (intensity < min_intensity)
    //            {
    //                min_intensity = intensity;
    //                min_intensity_idx = j;
    //            }
    //        }

    //        if (recon_error > min_intensity)
    //        {
    //            // replace existing min_intensity_idx
    //            value_view_.copy_to(memory_pool_view_[min_intensity_idx]);
    //            memory_intensity_[min_intensity_idx] = recon_error;
    //            return true;
    //        }
    //        // discard current value since the model is already doing well with it.
    //    }
    //    else // recon_error > min_diff
    //    {
    //        // the model is doing worse at current value than just using the closest memory
    //        // so we replace the output with the closest memory
    //        memory_pool_view_[min_idx].copy_to(next_value_view_);

    //        if (recon_error > memory_intensity_[min_idx])
    //        {
    //            // current bad value is a more worthy case to remember, so we replace the closest memory with current value
    //            value_view_.copy_to(memory_pool_view_[min_idx]);
    //            memory_intensity_[min_idx] = recon_error;
    //            return true;
    //        }
    //    }

    //    return false;
    //}

    bitmap_image DataLayer::GenerateImage() const
    {
        value_view_.synchronize();
        expect_view_.synchronize();
        next_value_view_.synchronize();
        next_expect_view_.synchronize();
        memory_view_.synchronize();

        bitmap_image image;

        const int block_size = 2;

        if (width() == 1 && height() == 1)
        {
            image.setwidth_height(depth() * (block_size + 1), (4 + 2 + memory_num()) * (block_size + 1), true);
            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 0, block_size, block_size,
                    value_view_(i, 0, 0) == 0.0f ? 0 : 255);
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), block_size + 1, block_size, block_size,
                    next_value_view_(i, 0, 0) == 0.0f ? 0 : 255);
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 2 * (block_size + 1), block_size, block_size,
                    static_cast<unsigned char>(255.0 * expect_view_(i, 0, 0)));
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 3 * (block_size + 1), block_size, block_size,
                    static_cast<unsigned char>(255.0 * next_expect_view_(i, 0, 0)));
            }

            for (int i = 0; i < memory_num(); i++)
            {
                auto memory_slice_view = memory_view_[i];
                for (int j = 0; j < depth(); j++)
                {
                    image.set_region(j * (block_size + 1), (6 + i) * (block_size + 1), block_size, block_size,
                        static_cast<unsigned char>(255.0 * memory_slice_view(j, 0, 0)));
                }
            }
        }
        else
        {
            image.setwidth_height(depth() * (width() + 1) * (block_size + 1),
                ((4 + memory_num()) * (height() + 1) + 2) * (block_size + 1), true);
            for (int depth_idx = 0; depth_idx < depth(); depth_idx++)
            {
                for (int height_idx = 0; height_idx < height(); height_idx++)
                {
                    for (int width_idx = 0; width_idx < width(); width_idx++)
                    {
                        image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                            height_idx * (block_size + 1), block_size, block_size,
                            value_view_(depth_idx, height_idx, width_idx) == 0.0f ? 0 : 255);
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
                            next_value_view_(depth_idx, height_idx, width_idx) == 0.0f ? 0 : 255);
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
                            static_cast<unsigned char>(255.0 * expect_view_(depth_idx, height_idx, width_idx)));
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
                            static_cast<unsigned char>(255.0 * next_expect_view_(depth_idx, height_idx, width_idx)));
                    }
                }
            }

            for (int memory_idx = 0; memory_idx < memory_num(); memory_idx++)
            {
                auto memory_slice_view = memory_view_[memory_idx];
                for (int depth_idx = 0; depth_idx < depth(); depth_idx++)
                {
                    for (int height_idx = 0; height_idx < height(); height_idx++)
                    {
                        for (int width_idx = 0; width_idx < width(); width_idx++)
                        {
                            image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                                ((4 + memory_idx) * (height() + 1) + height_idx + 2) * (block_size + 1), block_size, block_size,
                                static_cast<unsigned char>(255.0 * memory_slice_view(depth_idx, height_idx, width_idx)));
                        }
                    }
                }
            }
        }

        return image;
    }

#pragma endregion

#pragma region output layer

    OutputLayer::OutputLayer(int output_num, int input_depth, int input_height, int input_width)
        : outputs_(output_num),
        outputs_view_(output_num, outputs_),
        next_outputs_(output_num),
        next_outputs_view_(output_num, next_outputs_),
        bias_(output_num),
        bias_view_(output_num, bias_),
        weights_(output_num * input_depth * input_height * input_width),
        weights_view_(make_extent(output_num, input_depth, input_height, input_width), weights_)
    {
    }

    OutputLayer::OutputLayer(OutputLayer&& other)
        : outputs_(std::move(other.outputs_)),
        outputs_view_(other.outputs_view_),
        next_outputs_(std::move(other.next_outputs_)),
        next_outputs_view_(other.next_outputs_view_),
        bias_(std::move(other.bias_)),
        bias_view_(other.bias_view_),
        weights_(std::move(other.weights_)),
        weights_view_(other.weights_view_)
    {
    }

    void OutputLayer::SetLabel(const int label)
    {
        std::fill(outputs_.begin(), outputs_.end(), 0.0f);
        outputs_[label] = 1.0f;

        outputs_view_.refresh();
    }

    void OutputLayer::RandomizeParams(unsigned int seed)
    {
        std::default_random_engine generator(seed);
        std::normal_distribution<float> distribution(0.0f, 0.05f);

        for (float& w : weights_)
        {
            w = distribution(generator);
        }

        weights_view_.refresh();
    }

    int OutputLayer::PredictLabel(const DataLayer& bottom_layer, bool bottom_switcher,
        DataLayer& top_layer, bool top_switcher, const ConvolveLayer& conv_layer, const float dropout_prob)
    {
        assert(top_layer.depth() == conv_layer.neuron_num() && top_layer.depth() == this->input_depth());
        assert(top_layer.width() == bottom_layer.width() - conv_layer.neuron_width() + 1
            && top_layer.width() == this->input_width());
        assert(top_layer.height() == bottom_layer.height() - conv_layer.neuron_height() + 1
            && top_layer.height() == this->input_height());

        // read only
        const int bottom_value_depth = bottom_layer.depth();
        const int bottom_memory_depth = bottom_layer.memory_num() * bottom_value_depth;
        const int neuron_depth = conv_layer.neuron_depth();
        const int neuron_height = conv_layer.neuron_height();
        const int neuron_width = conv_layer.neuron_width();
        const int top_depth = top_layer.depth();
        const int top_height = top_layer.height();
        const int top_width = top_layer.width();

        array_view<const float, 3> bottom_memories = bottom_layer.memory_flatten_view_;
        array_view<const float, 3> bottom_value = bottom_switcher ? bottom_layer.value_view_ : bottom_layer.next_value_view_;
        array_view<const float, 4> neuron_weights = conv_layer.neurons_view_;
        array_view<const float> hbias = conv_layer.hbias_view_;
        array_view<const float> output_bias = this->bias_view_;
        array_view<const float, 4> output_weights = this->weights_view_;

        // write only
        array_view<float, 3> top_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;
        array_view<float> outputs = this->outputs_view_;
        top_value.discard_data();
        outputs.discard_data();

        // calculate base score, ignore top layer activation, bottom-up
        parallel_for_each(top_value.extent, [=](index<3> idx) restrict(amp)
        {
            int cur_depth_idx = idx[0];
            int cur_height_idx = idx[1];
            int cur_width_idx = idx[2];

            array_view<const float, 3> current_neuron = neuron_weights[cur_depth_idx];// projection

            float result = hbias[cur_depth_idx];

            // score from memory
            for (int depth_idx = 0; depth_idx < bottom_memory_depth; depth_idx++)
            {
                for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                {
                    for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                    {
                        result += bottom_memories(depth_idx, cur_height_idx + height_idx, cur_width_idx + width_idx)
                            * current_neuron(depth_idx, height_idx, width_idx);
                    }
                }
            }

            // from bottom current value
            for (int depth_idx = 0; depth_idx < bottom_value_depth; depth_idx++)
            {
                for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                {
                    for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                    {
                        result += bottom_value(depth_idx, cur_height_idx + height_idx, cur_width_idx + width_idx)
                            * current_neuron(bottom_memory_depth + depth_idx, height_idx, width_idx);
                    }
                }
            }

            top_value[idx] = result;
        });

        parallel_for_each(outputs.extent, [=](index<1> idx) restrict(amp)
        {
            float result = output_bias[idx];

            auto& current_output_weights = output_weights[idx[0]];

            for (int depth_idx = 0; depth_idx < top_depth; depth_idx++)
            {
                for (int height_idx = 0; height_idx < top_height; height_idx++)
                {
                    for (int width_idx = 0; width_idx < top_width; width_idx++)
                    {
                        float score = top_value(depth_idx, height_idx, width_idx)
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
        const int top_depth = top_layer.depth();
        const int top_height = top_layer.height();
        const int top_width = top_layer.width();
        array_view<const float, 3> top_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;
        array_view<const float> output_bias = bias_view_;
        array_view<const float, 4> output_weights = weights_view_;

        // writeonly
        array_view<float> output_value = output_switcher ? outputs_view_ : next_outputs_view_;
        output_value.discard_data();

        // non-tiled version
        parallel_for_each(output_value.extent,
            [=](index<1> idx) restrict(amp)
        {
            float result = output_bias[idx];

            const auto& cur_weights = output_weights[idx[0]];

            for (int depth_idx = 0; depth_idx < top_depth; depth_idx++)
            {
                for (int height_idx = 0; height_idx < top_height; height_idx++)
                {
                    for (int width_idx = 0; width_idx < top_width; width_idx++)
                    {
                        result += top_value(depth_idx, height_idx, width_idx)
                            * cur_weights(depth_idx, height_idx, width_idx);
                    }
                }
            }

            output_value[idx] = 1.0f / (1.0f + fast_math::expf(-result));
        });
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
                auto cur_weights_view = weights_view_[output_idx];

                for (int depth_idx = 0; depth_idx < input_depth(); depth_idx++)
                {
                    for (int height_idx = 0; height_idx < input_height(); height_idx++)
                    {
                        for (int width_idx = 0; width_idx < input_width(); width_idx++)
                        {
                            float value = cur_weights_view(depth_idx, height_idx, width_idx);

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

#pragma endregion

#pragma region convolve layer

    ConvolveLayer::ConvolveLayer(int longterm_memory_num, int neuron_num,
        int shortterm_memory_num, int neuron_depth, int neuron_height, int neuron_width)
        : longterm_memory_num_(longterm_memory_num), shortterm_memory_num_(shortterm_memory_num),
        weights_((neuron_num + longterm_memory_num) * (1 + shortterm_memory_num) * neuron_depth * neuron_height * neuron_width),
        weights_view_(make_extent(neuron_num + longterm_memory_num, 1 + shortterm_memory_num, neuron_depth, neuron_height, neuron_width), weights_),
        // when longterm_memory_num == 0, we just use the neuron weights
        memory_view_(longterm_memory_num == 0 ? weights_view_ : 
            weights_view_.section(make_index(neuron_num, 0, 0, 0, 0), 
                make_extent(longterm_memory_num, 1 + shortterm_memory_num, neuron_depth, neuron_height, neuron_width))),
        neurons_view_(weights_view_.section(make_extent(neuron_num, 1 + shortterm_memory_num, neuron_depth, neuron_height, neuron_width))),
        value_neurons_view_(weights_view_.section(make_extent(neuron_num, 1, neuron_depth, neuron_height, neuron_width))),
        // no vbias for short-term memory because they are not generative
        vbias_(neuron_depth),
        vbias_view_(neuron_depth, vbias_),
        hbias_(neuron_num),
        hbias_view_(neuron_num, hbias_)
    {
    }

    ConvolveLayer::ConvolveLayer(ConvolveLayer&& other)
        : longterm_memory_num_(other.longterm_memory_num_),
        shortterm_memory_num_(other.shortterm_memory_num_),
        weights_(std::move(other.weights_)),
        weights_view_(other.weights_view_),
        memory_view_(other.memory_view_),
        neurons_view_(other.neurons_view_),
        value_neurons_view_(other.value_neurons_view_),
        vbias_(std::move(other.vbias_)),
        vbias_view_(other.vbias_view_),
        hbias_(std::move(other.hbias_)),
        hbias_view_(other.hbias_view_)
    {
    }

    void ConvolveLayer::PassUp(const DataLayer& bottom_layer, DataSlot bottom_slot,
        DataLayer& top_layer, DataSlot top_slot, const OutputLayer* output_layer, DataSlot output_slot) const
    {
        assert(top_layer.depth() == this->neuron_num() + this->longterm_memory_num());
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1);
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1);

        bool output_layer_exist = output_layer != nullptr;

        if (output_layer_exist)
        {
            assert(top_layer.depth() == output_layer->input_depth());
            assert(top_layer.width() == output_layer->input_width());
            assert(top_layer.height() == output_layer->input_height());
        }

        // readonly
        const int neuron_depth = this->neuron_depth();
        const int neuron_height = this->neuron_height();
        const int neuron_width = this->neuron_width();

        array_view<const float, 5> neuron_weights = neurons_view_;
        array_view<const float> hbias = hbias_view_;
        array_view<const float, 3> bottom_value = bottom_layer[bottom_slot].first;
        const int bottom_memory_num = bottom_layer.memory_num();
        array_view<const float, 4> bottom_memories = bottom_layer.memory_view_;

        array_view<const int, 3> top_active = top_layer.active_view_;

        // output layer
        static array_view<float> s_empty_output_value(1);
        array_view<const float> output_value = !output_layer_exist ? s_empty_output_value : (*output_layer)[output_slot];

        static array_view<float, 4> s_empty_output_weights(make_extent(1, 1, 1, 1));
        array_view<const float, 4> output_weights = !output_layer_exist ? s_empty_output_weights
            : output_layer->weights_view_;

        // writeonly
        auto top_data = top_layer[top_slot];
        array_view<float, 3> top_value = top_data.first;
        array_view<float, 3> top_expect = top_data.second;

        top_value.discard_data();
        top_expect.discard_data();

        auto& rand_collection = top_layer.rand_collection_;

        // non-tiled version
        parallel_for_each(top_value.extent,
            [=](index<3> idx) restrict(amp)
        {
            if (top_active[idx] == 0)
            {
                top_expect[idx] = 0.0f;
                top_value[idx] = 0.0f;
            }
            else
            {
                int cur_depth_idx = idx[0];
                int cur_height_idx = idx[1];
                int cur_width_idx = idx[2];

                float result = hbias[cur_depth_idx];

                if (output_layer_exist)
                {
                    for (int output_idx = 0; output_idx < output_value.extent[0]; output_idx++)
                    {
                        result += output_value[output_idx] * output_weights[output_idx][idx];
                    }
                }

                array_view<const float, 4> current_neuron = neuron_weights[cur_depth_idx];// projection

                auto current_value_neuron = current_neuron[0];// the first index is always value neuron weights

                for (int depth_idx = 0; depth_idx < neuron_depth; depth_idx++)
                {
                    for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                    {
                        for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                        {
                            result += bottom_value(depth_idx, cur_height_idx + height_idx, cur_width_idx + width_idx)
                                * current_value_neuron(depth_idx, height_idx, width_idx);
                        }
                    }
                }

                // convolve short-term memory in bottom layer if exists.
                for (int memory_idx = 0; memory_idx < bottom_memory_num; memory_idx++)
                {
                    auto current_memory_neuron = current_neuron[1 + memory_idx];
                    auto current_bottom_memory = bottom_memories[memory_idx];
                    for (int depth_idx = 0; depth_idx < neuron_depth; depth_idx++)
                    {
                        for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                        {
                            for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                            {
                                result += current_bottom_memory(depth_idx, cur_height_idx + height_idx, cur_width_idx + width_idx)
                                    * current_memory_neuron(depth_idx, height_idx, width_idx);
                            }
                        }
                    }
                }

                // Logistic activation function. Maybe more types of activation function later.
                float prob = 1.0f / (1.0f + fast_math::expf(-result));
                top_expect[idx] = prob;
                top_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;
            }
        });
    }

    void ConvolveLayer::PassDown(const DataLayer& top_layer, DataSlot top_slot,
        DataLayer& bottom_layer, DataSlot bottom_slot, OutputLayer* output_layer, DataSlot output_slot) const
    {
        assert(top_layer.depth() == this->neuron_num());
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1);
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1);

        // PassDown will not touch bottom short-term memory for simplicity

        // readonly
        const int neuron_num = this->neuron_num();
        const int neuron_height = this->neuron_height();
        const int neuron_width = this->neuron_width();
        const int bottom_height = bottom_layer.height();
        const int bottom_width = bottom_layer.width();

        array_view<const float, 5> neuron_weights = neurons_view_;
        array_view<const float> vbias = vbias_view_;
        array_view<const float, 3> top_value = top_layer[top_slot].first;

        array_view<const int, 3> bottom_active = bottom_layer.active_view_;

        // writeonly
        auto bottom_data = bottom_layer[bottom_slot];
        array_view<float, 3> bottom_value = bottom_data.first;
        array_view<float, 3> bottom_expect = bottom_data.second;
        bottom_value.discard_data();
        bottom_expect.discard_data();

        auto& rand_collection = bottom_layer.rand_collection_;

        // non-tiled version
        parallel_for_each(bottom_value.extent,
            [=](index<3> idx) restrict(amp)
        {
            if (bottom_active[idx] == 0)
            {
                bottom_expect[idx] = 0.0f;
                bottom_value[idx] = 0.0f;
            }
            else
            {
                int cur_depth_idx = idx[0];
                int cur_height_idx = idx[1];
                int cur_width_idx = idx[2];

                // make sure the convolve window fits in the bottom layer
                int height_idx_min = max(0, cur_height_idx - (bottom_height - neuron_height));
                int height_idx_max = min(neuron_height - 1, cur_height_idx);
                int width_idx_min = max(0, cur_width_idx - (bottom_width - neuron_width));
                int width_idx_max = min(neuron_width - 1, cur_width_idx);

                float result = vbias[cur_depth_idx];

                for (int neuron_idx = 0; neuron_idx < neuron_num; neuron_idx++)
                {
                    array_view<const float, 3> current_neuron = neuron_weights[neuron_idx][0];

                    for (int height_idx = height_idx_min; height_idx <= height_idx_max; height_idx++)
                    {
                        int top_height_idx = cur_height_idx - height_idx;
                        for (int width_idx = width_idx_min; width_idx <= width_idx_max; width_idx++)
                        {
                            int top_width_idx = cur_width_idx - width_idx;
                            result += current_neuron(cur_depth_idx, height_idx, width_idx) *
                                top_value(neuron_idx, top_height_idx, top_width_idx);
                        }
                    }
                }

                // Logistic activation function. Maybe more types of activation function later.
                float prob = 1.0f / (1.0f + fast_math::expf(-result));
                bottom_expect[idx] = prob;
                bottom_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;
            }
        });

        // pass down to output layer
        if (output_layer != nullptr)
        {
            assert(top_layer.depth() == output_layer->input_depth());
            assert(top_layer.width() == output_layer->input_width());
            assert(top_layer.height() == output_layer->input_height());

            array_view<const float> output_bias = output_layer->bias_view_;
            array_view<const float, 4> output_weights = output_layer->weights_view_;

            array_view<float> output_value = (*output_layer)[output_slot];
            output_value.discard_data();

            // non-tiled version
            parallel_for_each(output_value.extent,
                [=](index<1> idx) restrict(amp)
            {
                float result = output_bias[idx];

                const auto& weights = output_weights[idx[0]];

                for (int depth_idx = 0; depth_idx < top_value.extent[0]; depth_idx++)
                {
                    for (int height_idx = 0; height_idx < top_value.extent[1]; height_idx++)
                    {
                        for (int width_idx = 0; width_idx < top_value.extent[2]; width_idx++)
                        {
                            result += top_value(depth_idx, height_idx, width_idx)
                                * weights(depth_idx, height_idx, width_idx);
                        }
                    }
                }

                output_value[idx] = 1.0f / (1.0f + fast_math::expf(-result));
            });
        }

    }

    void ConvolveLayer::SuppressMemory(DataLayer& top_layer, DataSlot top_slot,
        const DataLayer& bottom_layer, DataSlot bottom_slot) const
    {
        // readonly
        auto bottom_data = bottom_layer[bottom_slot];
        array_view<const float, 3> bottom_value = bottom_data.first;
        array_view<const float, 3> bottom_expect = bottom_data.second;

        // read write
        auto top_data = top_layer[top_slot];
        array_view<float, 3> top_value = top_data.first;
        array_view<float, 3> top_expect = top_data.second;
    }

    void ConvolveLayer::Train(const DataLayer& bottom_layer, const DataLayer& top_layer, float learning_rate,
        OutputLayer* output_layer, bool discriminative_training)
    {
        // readonly
        const int top_height = top_layer.height();
        const int top_width = top_layer.width();
        const int bottom_height = bottom_layer.height();
        const int bottom_width = bottom_layer.width();
        const int bottom_memory_num = bottom_layer.memory_num();

        array_view<const float, 3> top_expect = top_layer.expect_view_;
        array_view<const float, 3> top_next_expect = top_layer.next_expect_view_;
        array_view<const float, 3> bottom_value = bottom_layer.value_view_;
        array_view<const float, 3> bottom_next_value = bottom_layer.next_value_view_;
        array_view<const float, 4> bottom_memories = bottom_layer.memory_view_;

        // parameters to train
        array_view<float, 5> value_neuron_weights = this->value_neurons_view_;
        array_view<float, 5> longterm_memory_weights = this->memory_view_;

        array_view<float> vbias = vbias_view_;
        array_view<float> hbias = hbias_view_;

        // non-tiled version
        parallel_for_each(value_neuron_weights.extent, [=](index<5> idx) restrict(amp)
        {
            float delta = 0.0f;

            int neuron_idx = idx[0];
            //int short_memory_idx = idx[1]; should always = 0
            int neuron_depth_idx = idx[2];
            int neuron_height_idx = idx[3];
            int neuron_width_idx = idx[4];

            for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
            {
                for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
                {
                    float cur_top_expect = top_expect(neuron_idx, top_height_idx, top_width_idx);
                    float cur_top_next_expect = top_next_expect(neuron_idx, top_height_idx, top_width_idx);

                    float cur_bottom_value = bottom_value(neuron_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx);
                    float cur_bottom_next_value = discriminative_training ? cur_bottom_value :
                        bottom_next_value(neuron_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx);

                    delta += cur_bottom_value * cur_top_expect - cur_bottom_next_value * cur_top_next_expect;
                }
            }

            //if (neuron_depth_idx < bottom_memory_size)
            //{
            //    // bottom memory, depth index starts from 0.
            //    // we train memory weights in a discriminative way for simplicity.
            //    for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
            //    {
            //        for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
            //        {
            //            float cur_top_expect = top_expect(neuron_idx, top_height_idx, top_width_idx);
            //            float cur_top_next_expect = top_next_expect(neuron_idx, top_height_idx, top_width_idx);

            //            float cur_bottom_value = bottom_short_memory(neuron_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx);

            //            delta += cur_bottom_value * (cur_top_expect - cur_top_next_expect);
            //        }
            //    }
            //}
            //else
            //{
            //    
            //}

            value_neuron_weights[idx] += delta / (top_height * top_width) * learning_rate;
        });

        // update vbias, only for generative training
        if (!discriminative_training)
        {
            parallel_for_each(vbias.extent, [=](index<1> idx) restrict(amp)
            {
                float delta = 0.0f;

                int depth_idx = idx[0];

                for (int bottom_height_idx = 0; bottom_height_idx < bottom_height; bottom_height_idx++)
                {
                    for (int bottom_width_idx = 0; bottom_width_idx < bottom_width; bottom_width_idx++)
                    {
                        float cur_bottom_value = bottom_value(depth_idx, bottom_height_idx, bottom_width_idx);
                        float cur_bottom_next_value = bottom_next_value(depth_idx, bottom_height_idx, bottom_width_idx);

                        delta += cur_bottom_value - cur_bottom_next_value;
                    }
                }

                vbias[idx] += delta / (bottom_height * bottom_width) * learning_rate;
            });
        }

        // update hbias
        parallel_for_each(hbias.extent, [=](index<1> idx) restrict(amp)
        {
            float delta = 0.0f;

            int neuron_idx = idx[0];

            for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
            {
                for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
                {
                    float cur_top_expect = top_expect(neuron_idx, top_height_idx, top_width_idx);
                    float cur_top_next_expect = top_next_expect(neuron_idx, top_height_idx, top_width_idx);

                    delta += cur_top_expect - cur_top_next_expect;
                }
            }

            hbias[idx] += delta / (top_height * top_width) * learning_rate;
        });

        // for output layer
        if (output_layer != nullptr)
        {
            // parameters to train
            array_view<float, 4> output_weights = output_layer->weights_view_;
            array_view<float> output_bias = output_layer->bias_view_;

            // readonly
            array_view<const float> output_value = output_layer->outputs_view_;
            array_view<const float> output_next_value = output_layer->next_outputs_view_;

            parallel_for_each(output_weights.extent, [=](index<4> idx) restrict(amp)
            {
                int output_idx = idx[0];
                int top_depth_idx = idx[1];
                int top_height_idx = idx[2];
                int top_width_idx = idx[3];

                float delta = output_value(output_idx) * top_expect(top_depth_idx, top_height_idx, top_width_idx) -
                    output_next_value(output_idx) * top_next_expect(top_depth_idx, top_height_idx, top_width_idx);

                output_weights[idx] += delta * learning_rate;

            });

            parallel_for_each(output_bias.extent, [=](index<1> idx) restrict(amp)
            {
                float delta = output_value[idx] - output_next_value[idx];

                output_bias[idx] += delta * learning_rate;
            });
        }
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
        memory_view_.synchronize();
        neurons_view_.synchronize();
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
                auto neuron_view = neurons_view_[neuron_idx];
                for (int depth_idx = 0; depth_idx < neuron_depth(); depth_idx++)
                {
                    float value = neuron_view(depth_idx, 0, 0);
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
                auto neuron_view = neurons_view_[neuron_idx];
                for (int depth_idx = 0; depth_idx < neuron_depth(); depth_idx++)
                {
                    for (int height_idx = 0; height_idx < neuron_height(); height_idx++)
                    {
                        for (int width_idx = 0; width_idx < neuron_width(); width_idx++)
                        {
                            float value = neuron_view(depth_idx, height_idx, width_idx);

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
#pragma endregion

#pragma region pooling layer

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

        array_view<const float, 3> bottom_value = bottom_switcher ? bottom_layer.value_view_ : bottom_layer.next_value_view_;
        array_view<const float, 3> bottom_expect = bottom_switcher ? bottom_layer.expect_view_ : bottom_layer.next_expect_view_;

        // writeonly
        array_view<float, 3> top_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;
        array_view<float, 3> top_expect = top_switcher ? top_layer.expect_view_ : top_layer.next_expect_view_;
        top_value.discard_data();
        top_expect.discard_data();

        parallel_for_each(top_value.extent, [=](index<3> idx) restrict(amp)
        {
            float max_value = 0.0f;
            float max_expect = 1.0f;

            for (int height_idx = 0; height_idx < block_height; height_idx++)
            {
                for (int width_idx = 0; width_idx < block_width; width_idx++)
                {
                    float value = bottom_value(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx);
                    float expect = bottom_expect(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx);

                    max_value = fast_math::fmaxf(max_value, value);
                    max_expect *= (1.0f - expect); // the probability that all nodes are 0
                }
            }
            max_expect = 1.0f - max_expect;// the probability that at least one node is 1.

            top_value[idx] = max_value;
            top_expect[idx] = max_expect;
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

        array_view<const float, 3> top_value = top_switcher ? top_layer.value_view_ : top_layer.next_value_view_;
        array_view<const float, 3> top_expect = top_switcher ? top_layer.expect_view_ : top_layer.next_expect_view_;

        // writeonly
        array_view<float, 3> bottom_value = bottom_switcher ? bottom_layer.value_view_ : bottom_layer.next_value_view_;
        array_view<float, 3> bottom_expect = bottom_switcher ? bottom_layer.expect_view_ : bottom_layer.next_expect_view_;
        bottom_value.discard_data();
        bottom_expect.discard_data();

        auto& rand_collection = bottom_layer.rand_collection_;

        parallel_for_each(bottom_value.extent, [=](index<3> idx) restrict(amp)
        {
            // when we have memory, the bottom_layer can activate according to its memory. 
            // But now we just use uniform activation.

            int height_idx = idx[1] / block_height;// truncate towards zero
            int width_idx = idx[2] / block_width;


            bottom_expect[idx] = 1.0f - fast_math::powf(1.0f - top_expect(idx[0], height_idx, width_idx),
                -1.0f * block_width * block_height);
            bottom_value[idx] = 0.0f;// clear the value
        });

        parallel_for_each(top_value.extent, [=](index<3> idx) restrict(amp)
        {
            if (top_value[idx] == 1.0f)
            {
                // randomly select a node in bottom_layer to activate
                int height_idx = rand_collection[idx].next_uint() % block_height;
                int width_idx = rand_collection[idx].next_uint() % block_width;

                bottom_value(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx) = 1.0f;
            }
        });
    }

#pragma endregion

#pragma region deep model

    DeepModel::DeepModel(unsigned int model_seed) : random_engine_(model_seed)
    {

    }

    void DeepModel::AddDataLayer(int memory_num, int depth, int height, int width)
    {
        data_layers_.emplace_back(memory_num, depth, height, width, std::uniform_int_distribution<int>()(random_engine_));
    }

    void DeepModel::AddConvolveLayer(int memory_num, int neuron_num, int neuron_depth, int neuron_height, int neuron_width)
    {
        convolve_layers_.emplace_back(memory_num, neuron_num, neuron_depth, neuron_height, neuron_width);
        convolve_layers_.back().RandomizeParams(std::uniform_int_distribution<int>()(random_engine_));
    }

    void DeepModel::AddOutputLayer(int data_layer_idx, int output_num)
    {
        auto& data_layer = data_layers_[data_layer_idx];

        output_layers_.emplace(std::piecewise_construct, std::forward_as_tuple(data_layer_idx),
            std::forward_as_tuple(output_num, data_layer.depth(), data_layer.height(), data_layer.width()));

        output_layers_.at(data_layer_idx).RandomizeParams(std::uniform_int_distribution<int>()(random_engine_));
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

    float DeepModel::TrainLayer(const std::vector<float>& data, int layer_idx, float learning_rate, float dropout_prob,
        const int label, bool discriminative_training)
    {
        auto& bottom_layer = data_layers_[layer_idx];
        auto& top_layer = data_layers_[layer_idx + 1];

        auto& conv_layer = convolve_layers_[layer_idx];

        // train with contrastive divergence (CD) algorithm to maximize likelihood on dataset
        bottom_layer.SetValue(data);
        top_layer.Activate(1.0f - dropout_prob);

        if (label == -1)
        {
            // purely generative training without label
            conv_layer.PassUp(bottom_layer, true, top_layer, true);
            conv_layer.PassDown(top_layer, true, bottom_layer, false);
            conv_layer.PassUp(bottom_layer, false, top_layer, false);

            conv_layer.Train(bottom_layer, top_layer, learning_rate);
        }
        else
        {
            // training data has label
            auto& output_layer = output_layers_.at(layer_idx + 1);
            output_layer.SetLabel(label);

            conv_layer.PassUp(bottom_layer, true, top_layer, true, &output_layer, true);

            if (discriminative_training)
            {
                output_layer.PassDown(top_layer, true, false);
                conv_layer.PassUp(bottom_layer, true, top_layer, false, &output_layer, false);
            }
            else
            {
                conv_layer.PassDown(top_layer, true, bottom_layer, false, &output_layer, false);
                conv_layer.PassUp(bottom_layer, false, top_layer, false, &output_layer, false);
            }

            conv_layer.Train(bottom_layer, top_layer, learning_rate, &output_layer, discriminative_training);
        }

        return bottom_layer.ReconstructionError();
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

#pragma endregion

}
