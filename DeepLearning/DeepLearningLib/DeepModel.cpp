#include "DeepModel.h"

#include <assert.h>
#include <random>
#include <amp_math.h>
#include <iostream>

#include "AmpUtility.h"

namespace deep_learning_lib
{
    using namespace std;
    using namespace concurrency;
    using namespace concurrency::precise_math;

#pragma region data layer

    DataLayer::DataLayer(int shortterm_memory_num, int depth, int height, int width, int seed)
        : shortterm_memory_num_(shortterm_memory_num),
        value_view_(depth, height, width),
        expect_view_(value_view_.extent),
        next_value_view_(value_view_.extent),
        next_expect_view_(value_view_.extent),
        // there is no empty array_view support in amp now, so we just set the extent to (1,1,1,1) when the shortterm_memory_num == 0
        shortterm_memory_view_(shortterm_memory_num == 0 ? make_extent(1, 1, 1, 1) : make_extent(shortterm_memory_num, depth, height, width)),
        shortterm_memory_index_view_(std::max(1, shortterm_memory_num)),
        longterm_memory_activations_view_(value_view_.extent),
        dropout_prob_(0.0),
        dropout_activations_view_(value_view_.extent),
        raw_weights_view_(value_view_.extent),
        rand_collection_(value_view_.extent, seed)
    {
        fill(value_view_, 0.0);
        fill(expect_view_, 0.0);
        fill(next_value_view_, 0.0);
        fill(next_expect_view_, 0.0);
        fill(shortterm_memory_view_, 0.0);
        for (int time = 0; time < shortterm_memory_num; time++)
        {
            shortterm_memory_index_view_[time] = time;
        }
        fill(dropout_activations_view_, 0);
    }

    DataLayer::DataLayer(DataLayer&& other)
        : shortterm_memory_num_(other.shortterm_memory_num_),
        value_view_(other.value_view_),
        expect_view_(other.expect_view_),
        next_value_view_(other.next_value_view_),
        next_expect_view_(other.next_expect_view_),
        shortterm_memory_view_(other.shortterm_memory_view_),
        shortterm_memory_index_view_(other.shortterm_memory_index_view_),
        longterm_memory_activations_view_(other.longterm_memory_activations_view_),
        dropout_prob_(other.dropout_prob_),
        dropout_activations_view_(other.dropout_activations_view_),
        raw_weights_view_(other.raw_weights_view_),
        rand_collection_(other.rand_collection_)
    {
    }

    void DataLayer::SetValue(const vector<double>& data)
    {
        assert(data.size() == value_view_.extent.size());

        // Copy the data
        copy(data.begin(), data.end(), value_view_);
    }

    void DataLayer::ActivateDropout(double dropout_prob)
    {
        // no need to change activation when the prob = 1.0 or 0.0 again.
        if (dropout_prob == dropout_prob_ && (dropout_prob == 1.0 || dropout_prob == 0.0))
        {
            return;
        }

        array_view<int, 3> dropout_activations = this->dropout_activations_view_;
        auto& rand_collection = this->rand_collection_;

        parallel_for_each(dropout_activations.extent,
            [=](index<3> idx) restrict(amp)
        {
            dropout_activations[idx] = rand_collection[idx].next_single() < dropout_prob ? 1 : 0;
        });

        dropout_prob_ = dropout_prob;
    }

    void DataLayer::ActivateLongtermMemory(const ConvolveLayer& conv_layer)
    {
        array_view<const double> neuron_activation_counts = conv_layer.neuron_activation_counts_view_;
        array_view<const double> neuron_life_counts = conv_layer.neuron_life_counts_view_;
        array_view<int, 3> longterm_memory_activations = this->longterm_memory_activations_view_;
        auto& rand_collection = this->rand_collection_;

        parallel_for_each(longterm_memory_activations.extent,
            [=](index<3> idx) restrict(amp)
        {
            auto neuron_activation_prob = neuron_activation_counts[idx[0]] / fmax(1.0, neuron_life_counts[idx[0]]);
            longterm_memory_activations[idx] = rand_collection[idx].next_single() < neuron_activation_prob ? 0 : 1;
        });
    }

    void DataLayer::Memorize()
    {
        if (shortterm_memory_num() <= 0)
        {
            return;
        }

        int last_memory_index = shortterm_memory_index_view_(shortterm_memory_num() - 1);

        copy(value_view_, shortterm_memory_view_[last_memory_index]);

        // right shift the shortterm memory index
        for (int time = shortterm_memory_num() - 1; time > 0; time--)
        {
            shortterm_memory_index_view_(time) = shortterm_memory_index_view_(time - 1);
        }

        shortterm_memory_index_view_(0) = last_memory_index;
    }

    float DataLayer::ReconstructionError(DataSlot slot) const
    {
        array_view<float> result(1);
        result(0) = 0.0f;

        array_view<const double, 3> value_view = this->value_view_;
        array_view<const double, 3> recon_expect_view = (*this)[slot].second;

        // TODO: compare with reduce method for performance
        parallel_for_each(value_view.extent,
            [=](index<3> idx) restrict(amp)
        {
            auto diff = value_view[idx] - recon_expect_view[idx];
            atomic_fetch_add(&result(0), static_cast<float>(diff * diff));
        });

        return sqrtf(result(0));
    }

    bitmap_image DataLayer::GenerateImage() const
    {
        bitmap_image image;

        const int block_size = 2;

        if (width() == 1 && height() == 1)
        {
            image.setwidth_height(depth() * (block_size + 1), (4 + 2 + shortterm_memory_num()) * (block_size + 1), true);
            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 0, block_size, block_size,
                    value_view_(i, 0, 0) == 0.0 ? 0 : 255);
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), block_size + 1, block_size, block_size,
                    next_value_view_(i, 0, 0) == 0.0 ? 0 : 255);
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

            for (int i = 0; i < shortterm_memory_num(); i++)
            {
                auto memory_slice_view = shortterm_memory_view_[shortterm_memory_index_view_[i]];
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
                ((4 + shortterm_memory_num()) * (height() + 1) + 2) * (block_size + 1), true);
            for (int depth_idx = 0; depth_idx < depth(); depth_idx++)
            {
                for (int height_idx = 0; height_idx < height(); height_idx++)
                {
                    for (int width_idx = 0; width_idx < width(); width_idx++)
                    {
                        image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                            height_idx * (block_size + 1), block_size, block_size,
                            value_view_(depth_idx, height_idx, width_idx) == 0.0 ? 0 : 255);
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
                            next_value_view_(depth_idx, height_idx, width_idx) == 0.0 ? 0 : 255);
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

            for (int memory_idx = 0; memory_idx < shortterm_memory_num(); memory_idx++)
            {
                auto memory_slice_view = shortterm_memory_view_[shortterm_memory_index_view_[memory_idx]];
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
        : outputs_view_(output_num),
        next_outputs_view_(output_num),
        bias_(output_num),
        bias_view_(output_num, bias_),
        weights_(output_num * input_depth * input_height * input_width),
        weights_view_(make_extent(output_num, input_depth, input_height, input_width), weights_)
    {
        fill(outputs_view_, 0.0);
        fill(next_outputs_view_, 0.0);
    }

    OutputLayer::OutputLayer(OutputLayer&& other)
        : outputs_view_(other.outputs_view_),
        next_outputs_view_(other.next_outputs_view_),
        bias_(move(other.bias_)),
        bias_view_(other.bias_view_),
        weights_(move(other.weights_)),
        weights_view_(other.weights_view_)
    {
    }

    void OutputLayer::SetLabel(const int label)
    {
        assert(label >= 0 && label < this->output_num());
        fill(outputs_view_, 0.0);
        outputs_view_[label] = 1.0;
    }

    void OutputLayer::RandomizeParams(unsigned int seed)
    {
        default_random_engine generator(seed);
        normal_distribution<double> distribution(0.0, 0.1);

        for (auto& w : weights_)
        {
            w = distribution(generator);
        }

        weights_view_.refresh();
    }

    int OutputLayer::PredictLabel(const DataLayer& bottom_layer, DataSlot bottom_slot, DataLayer& top_layer, DataSlot top_slot,
        const ConvolveLayer& conv_layer, const double dropout_prob)
    {
        assert(top_layer.depth() == conv_layer.neuron_num() && top_layer.depth() == this->input_depth());
        assert(top_layer.width() == bottom_layer.width() - conv_layer.neuron_width() + 1 && top_layer.width() == this->input_width());
        assert(top_layer.height() == bottom_layer.height() - conv_layer.neuron_height() + 1 && top_layer.height() == this->input_height());

        // calculate base score, ignore top layer activation. 
        // longterm memory activation is ignored, since we use raw weight directly.
        // pass up with full activation in top layers
        top_layer.ActivateDropout();
        conv_layer.PassUp(bottom_layer, bottom_slot, top_layer, top_slot);

        // read only
        const int top_depth = top_layer.depth();
        const int top_height = top_layer.height();
        const int top_width = top_layer.width();

        array_view<const double> neuron_activation_counts = conv_layer.neuron_activation_counts_view_;
        array_view<const double> neuron_life_counts = conv_layer.neuron_life_counts_view_;
        array_view<const double, 3> top_raw_weights = top_layer.raw_weights_view_;
        array_view<const double> output_bias = this->bias_view_;
        array_view<const double, 4> output_weights = this->weights_view_;

        // write only
        array_view<double> outputs = this->outputs_view_;
        outputs.discard_data();

        parallel_for_each(outputs.extent, [=](index<1> idx) restrict(amp)
        {
            auto energe = output_bias[idx];

            const auto& current_output_weights = output_weights[idx[0]];

            for (int depth_idx = 0; depth_idx < top_depth; depth_idx++)
            {
                auto longterm_memory_activation_prob = 1.0 - neuron_activation_counts[depth_idx] / fmax(1.0, neuron_life_counts[depth_idx]);
                for (int height_idx = 0; height_idx < top_height; height_idx++)
                {
                    for (int width_idx = 0; width_idx < top_width; width_idx++)
                    {
                        auto raw_weight = top_raw_weights(depth_idx, height_idx, width_idx) + current_output_weights(depth_idx, height_idx, width_idx);
                        if (raw_weight > 0)// neuron activated, even if it is a longterm memory neuron
                        {
                            energe += log((exp(raw_weight) + 1.0) * (1.0 - dropout_prob) + 2.0 * dropout_prob);
                        }
                        else
                        {
                            energe += log(((exp(raw_weight) + 1.0) * (1.0 - longterm_memory_activation_prob)
                                + 2.0 * longterm_memory_activation_prob) * (1.0 - dropout_prob) + 2.0 * dropout_prob);
                        }
                    }
                }
            }

            outputs[idx] = energe;
        });

        int max_idx = 0;
        double max_value = outputs[max_idx];

        for (int i = 1; i < this->output_num(); i++)
        {
            if (outputs[i] > max_value)
            {
                max_value = outputs[i];
                max_idx = i;
            }
        }

        return max_idx;
    }

    void OutputLayer::PassDown(const DataLayer& top_layer, DataSlot top_slot, DataSlot output_slot)
    {
        assert(top_layer.depth() == this->input_depth());
        assert(top_layer.width() == this->input_width());
        assert(top_layer.height() == this->input_height());

        // readonly
        const int top_depth = top_layer.depth();
        const int top_height = top_layer.height();
        const int top_width = top_layer.width();
        array_view<const double, 3> top_value = top_layer[top_slot].first;
        array_view<const double> output_bias = this->bias_view_;
        array_view<const double, 4> output_weights = this->weights_view_;

        // writeonly
        array_view<double> output_value = (*this)[output_slot];
        output_value.discard_data();

        // non-tiled version
        parallel_for_each(output_value.extent,
            [=](index<1> idx) restrict(amp)
        {
            auto result = output_bias[idx];

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

            output_value[idx] = 1.0 / (1.0 + exp(-result));
        });
    }

    bitmap_image OutputLayer::GenerateImage() const
    {
        weights_view_.synchronize();
        bias_view_.synchronize();

        bitmap_image image;

        const int block_size = 2;

        auto max_abs_weight = numeric_limits<double>::min();
        for (auto weight : weights_)
        {
            max_abs_weight = fmax(max_abs_weight, abs(weight));
        }

        auto max_abs_bias = numeric_limits<double>::min();
        for (auto bias : bias_)
        {
            max_abs_bias = fmax(max_abs_bias, abs(bias));
        }

        if (input_width() == 1 && input_height() == 1)
        {
            image.setwidth_height((2 + input_depth()) * (block_size + 1), (2 + output_num()) * (block_size + 1), true);

            for (int i = 0; i < bias_.size(); i++)
            {
                image.set_region((2 + i) * (block_size + 1), 0, block_size, block_size,
                    bias_[i] >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(abs(bias_[i]) / max_abs_bias * 255.0));
            }

            for (int output_idx = 0; output_idx < output_num(); output_idx++)
            {
                for (int depth_idx = 0; depth_idx < input_depth(); depth_idx++)
                {
                    auto value = weights_[output_idx * input_depth() + depth_idx];
                    image.set_region((2 + depth_idx) * (block_size + 1), (2 + output_idx) * (block_size + 1), block_size, block_size,
                        value >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                        static_cast<unsigned char>(abs(value) / max_abs_weight * 255.0));
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
                    static_cast<unsigned char>(abs(bias_[i]) / max_abs_bias * 255.0));
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
                            auto value = cur_weights_view(depth_idx, height_idx, width_idx);

                            image.set_region((2 + width_idx + depth_idx * (input_width() + 1)) * (block_size + 1),
                                (2 + output_idx * (input_height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                                value >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                                static_cast<unsigned char>(abs(value) / max_abs_weight * 255.0));
                        }
                    }
                }
            }
        }

        return image;
    }

#pragma endregion

#pragma region convolve layer

    ConvolveLayer::ConvolveLayer(int neuron_num, int neuron_depth, int neuron_height, int neuron_width)
        : neuron_weights_(neuron_num * neuron_depth * neuron_height * neuron_width),
        neuron_activation_counts_(neuron_num, 1.0),
        neuron_life_counts_(neuron_num, 1.0),
        neuron_activation_counts_view_(neuron_num, neuron_activation_counts_),
        neuron_life_counts_view_(neuron_num, neuron_life_counts_),
        neuron_weights_view_(make_extent(neuron_num, neuron_depth, neuron_height, neuron_width), neuron_weights_),
        vbias_(neuron_depth),
        vbias_view_(neuron_depth, vbias_),
        hbias_(neuron_num),
        hbias_view_(neuron_num, hbias_)
    {
    }

    ConvolveLayer::ConvolveLayer(ConvolveLayer&& other)
        : neuron_weights_(move(other.neuron_weights_)),
        neuron_activation_counts_(move(other.neuron_activation_counts_)),
        neuron_life_counts_(move(other.neuron_life_counts_)),
        neuron_activation_counts_view_(other.neuron_activation_counts_view_),
        neuron_life_counts_view_(other.neuron_life_counts_view_),
        neuron_weights_view_(other.neuron_weights_view_),
        vbias_(move(other.vbias_)),
        vbias_view_(other.vbias_view_),
        hbias_(move(other.hbias_)),
        hbias_view_(other.hbias_view_)
    {
    }

    void ConvolveLayer::PassUp(const DataLayer& bottom_layer, DataSlot bottom_slot,
        DataLayer& top_layer, DataSlot top_slot,
        const OutputLayer* output_layer, DataSlot output_slot) const
    {
        assert(top_layer.depth() == this->neuron_num());
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1);
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1);
        assert(this->neuron_depth() == (bottom_layer.shortterm_memory_num() + 1) * bottom_layer.depth());

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
        const int bottom_depth = bottom_layer.depth();
        const int shortterm_memory_num = bottom_layer.shortterm_memory_num();

        array_view<const double, 4> neuron_weights = this->neuron_weights_view_;
        array_view<const double> hbias = this->hbias_view_;
        array_view<const double, 3> bottom_value = bottom_layer[bottom_slot].first;
        array_view<const double, 4> bottom_shortterm_memory = bottom_layer.shortterm_memory_view_;
        array_view<const int, 1> bottom_shortterm_memory_index = bottom_layer.shortterm_memory_index_view_;

        array_view<const int, 3> dropout_activation = top_layer.dropout_activations_view_;
        array_view<const int, 3> longterm_memory_activation = top_layer.longterm_memory_activations_view_;

        // output layer
        static array_view<double> s_empty_output_value(1);
        array_view<const double> output_value = output_layer_exist ? (*output_layer)[output_slot] : s_empty_output_value;

        static array_view<double, 4> s_empty_output_weights(make_extent(1, 1, 1, 1));
        array_view<const double, 4> output_weights = output_layer_exist ? output_layer->weights_view_ : s_empty_output_weights;

        // writeonly
        const auto& top_data = top_layer[top_slot];
        array_view<double, 3> top_value = top_data.first;
        array_view<double, 3> top_expect = top_data.second;
        array_view<double, 3> top_raw_weights = top_layer.raw_weights_view_;

        top_value.discard_data();
        top_expect.discard_data();
        top_raw_weights.discard_data();

        auto& rand_collection = top_layer.rand_collection_;

        // non-tiled version
        parallel_for_each(top_value.extent,
            [=](index<3> idx) restrict(amp)
        {
            int top_depth_idx = idx[0];
            int top_height_idx = idx[1];
            int top_width_idx = idx[2];

            if (dropout_activation[idx] == 1)
            {
                top_expect[idx] = 0.0;
                top_value[idx] = 0.0;
                top_raw_weights[idx] = 0.0;
            }
            else
            {
                auto raw_weight = hbias[top_depth_idx];

                if (output_layer_exist)
                {
                    for (int output_idx = 0; output_idx < output_value.extent[0]; output_idx++)
                    {
                        raw_weight += output_value[output_idx] * output_weights[output_idx][idx];
                    }
                }

                array_view<const double, 3> current_neuron = neuron_weights[top_depth_idx];

                for (int depth_idx = 0; depth_idx < bottom_depth; depth_idx++)
                {
                    for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                    {
                        for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                        {
                            auto value = bottom_value(depth_idx, top_height_idx + height_idx, top_width_idx + width_idx);
                            auto weight = current_neuron(depth_idx, height_idx, width_idx);
                            raw_weight += value * weight;
                        }
                    }
                }

                // convolve short-term memory in bottom layer if exists.
                for (int memory_idx = 0; memory_idx < shortterm_memory_num; memory_idx++)
                {
                    auto current_bottom_memory = bottom_shortterm_memory[bottom_shortterm_memory_index[memory_idx]];

                    for (int depth_idx = 0; depth_idx < bottom_depth; depth_idx++)
                    {
                        for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                        {
                            for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                            {
                                raw_weight += current_bottom_memory(depth_idx, top_height_idx + height_idx, top_width_idx + width_idx)
                                    * current_neuron(bottom_depth * (1 + memory_idx) + depth_idx, height_idx, width_idx);
                            }
                        }
                    }
                }

                top_raw_weights[idx] = raw_weight;

                if (longterm_memory_activation[idx] == 1 && raw_weight <= 0)
                {
                    top_expect[idx] = 0.0;
                    top_value[idx] = 0.0;
                }
                else
                {
                    auto prob = 1.0 / (1.0 + exp(-raw_weight));
                    top_expect[idx] = prob;
                    top_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0 : 0.0;
                }
            }
        });
    }

    void ConvolveLayer::PassDown(const DataLayer& top_layer, DataSlot top_slot,
        DataLayer& bottom_layer, DataSlot bottom_slot, OutputLayer* output_layer, DataSlot output_slot) const
    {
        assert(top_layer.depth() == this->neuron_num());
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1);
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1);

        // readonly
        const int neuron_num = this->neuron_num();
        const int neuron_height = this->neuron_height();
        const int neuron_width = this->neuron_width();
        const int bottom_height = bottom_layer.height();
        const int bottom_width = bottom_layer.width();

        array_view<const double, 4> neuron_weights = this->neuron_weights_view_;
        array_view<const double> vbias = this->vbias_view_;
        array_view<const double, 3> top_value = top_layer[top_slot].first;

        // writeonly
        const auto& bottom_data = bottom_layer[bottom_slot];
        array_view<double, 3> bottom_value = bottom_data.first;
        array_view<double, 3> bottom_expect = bottom_data.second;
        bottom_value.discard_data();
        bottom_expect.discard_data();

        auto& rand_collection = bottom_layer.rand_collection_;

        // non-tiled version
        // PassDown will not touch bottom short-term memory for simplicity
        // so here only update bottom_value
        parallel_for_each(bottom_value.extent,
            [=](index<3> idx) restrict(amp)
        {
            int cur_depth_idx = idx[0];
            int cur_height_idx = idx[1];
            int cur_width_idx = idx[2];

            // make sure the convolve window fits in the bottom layer
            int height_idx_min = max(0, cur_height_idx - (bottom_height - neuron_height));
            int height_idx_max = min(neuron_height - 1, cur_height_idx);
            int width_idx_min = max(0, cur_width_idx - (bottom_width - neuron_width));
            int width_idx_max = min(neuron_width - 1, cur_width_idx);

            auto raw_weight = vbias[cur_depth_idx];

            for (int neuron_idx = 0; neuron_idx < neuron_num; neuron_idx++)
            {
                array_view<const double, 3> current_neuron = neuron_weights[neuron_idx];

                for (int height_idx = height_idx_min; height_idx <= height_idx_max; height_idx++)
                {
                    int top_height_idx = cur_height_idx - height_idx;
                    for (int width_idx = width_idx_min; width_idx <= width_idx_max; width_idx++)
                    {
                        int top_width_idx = cur_width_idx - width_idx;
                        raw_weight += current_neuron(cur_depth_idx, height_idx, width_idx) *
                            top_value(neuron_idx, top_height_idx, top_width_idx);
                    }
                }
            }

            // Logistic activation function. Maybe more types of activation function later.
            auto prob = 1.0 / (1.0 + exp(-raw_weight));

            bottom_expect[idx] = prob;
            bottom_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0 : 0.0;
        });

        // pass down to output layer
        if (output_layer != nullptr)
        {
            assert(top_layer.depth() == output_layer->input_depth());
            assert(top_layer.width() == output_layer->input_width());
            assert(top_layer.height() == output_layer->input_height());

            const int top_depth = top_layer.depth();
            const int top_width = top_layer.width();
            const int top_height = top_layer.height();

            array_view<const double> output_bias = output_layer->bias_view_;
            array_view<const double, 4> output_weights = output_layer->weights_view_;

            array_view<double> output_value = (*output_layer)[output_slot];
            output_value.discard_data();

            // non-tiled version
            parallel_for_each(output_value.extent,
                [=](index<1> idx) restrict(amp)
            {
                auto raw_weight = output_bias[idx];

                const auto& weights = output_weights[idx[0]];

                for (int depth_idx = 0; depth_idx < top_depth; depth_idx++)
                {
                    for (int height_idx = 0; height_idx < top_height; height_idx++)
                    {
                        for (int width_idx = 0; width_idx < top_width; width_idx++)
                        {
                            raw_weight += top_value(depth_idx, height_idx, width_idx)
                                * weights(depth_idx, height_idx, width_idx);
                        }
                    }
                }

                output_value[idx] = 1.0 / (1.0 + exp(-raw_weight));
            });
        }
    }

    void ConvolveLayer::Train(const DataLayer& bottom_layer, const DataLayer& top_layer, double learning_rate,
        OutputLayer* output_layer, bool discriminative_training)
    {
        // readonly
        const int top_height = top_layer.height();
        const int top_width = top_layer.width();
        const int bottom_depth = bottom_layer.depth();
        const int bottom_height = bottom_layer.height();
        const int bottom_width = bottom_layer.width();
        const int shortterm_memory_num = bottom_layer.shortterm_memory_num();

        array_view<const double, 3> top_expect = top_layer.expect_view_;
        array_view<const double, 3> top_next_expect = top_layer.next_expect_view_;
        array_view<const double, 3> bottom_value = bottom_layer.value_view_;
        array_view<const double, 3> bottom_next_value = bottom_layer.next_value_view_;
        array_view<const double, 3> bottom_next_expect = bottom_layer.next_expect_view_;
        array_view<const double, 4> bottom_shortterm_memories = bottom_layer.shortterm_memory_view_;

        // parameters to train
        array_view<double, 4> neuron_weights = this->neuron_weights_view_;
        array_view<double> neuron_activation_counts = this->neuron_activation_counts_view_;
        array_view<double> neuron_life_counts = this->neuron_life_counts_view_;

        array_view<double> vbias = this->vbias_view_;
        array_view<double> hbias = this->hbias_view_;

        // neuron activation
        const double kNeuronDecay = this->kNeuronDecay;
        array_view<const int, 3> dropout_activation = top_layer.dropout_activations_view_;
        array_view<const int, 3> longterm_memory_activation = top_layer.longterm_memory_activations_view_;

        //parallel_for_each(neuron_activation_counts.extent, [=](index<1> idx) restrict(amp)
        //{
        //    int neuron_idx = idx[0];

        //    for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
        //    {
        //        for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
        //        {
        //            // not dropped out
        //            if (dropout_activation(neuron_idx, top_height_idx, top_width_idx) == 0)
        //            {
        //                if (longterm_memory_activation(neuron_idx, top_height_idx, top_width_idx) == 0)
        //                {
        //                    auto& life_count = neuron_life_counts(neuron_idx);
        //                    auto& activation_count = neuron_activation_counts(neuron_idx);

        //                    life_count = life_count * kNeuronDecay + 1;
        //                    activation_count = activation_count * kNeuronDecay;

        //                    if (top_expect(neuron_idx, top_height_idx, top_width_idx) >= 0.5)
        //                    {
        //                        activation_count++;
        //                    }
        //                }
        //                else if (top_expect(neuron_idx, top_height_idx, top_width_idx) >= 0.5)
        //                {
        //                    auto& life_count = neuron_life_counts(neuron_idx);
        //                    auto& activation_count = neuron_activation_counts(neuron_idx);

        //                    life_count = life_count * kNeuronDecay + 1;
        //                    activation_count = activation_count * kNeuronDecay + 1;
        //                }
        //            }
        //        }
        //    }
        //});

        // neuron weights
        // non-tiled version
        parallel_for_each(neuron_weights.extent, [=](index<4> idx) restrict(amp)
        {
            auto delta = 0.0;

            int neuron_idx = idx[0];
            int neuron_depth_idx = idx[1];
            int neuron_height_idx = idx[2];
            int neuron_width_idx = idx[3];

            int shortterm_memory_depth_idx = neuron_depth_idx % bottom_depth;
            int shortterm_memory_idx = (neuron_depth_idx - shortterm_memory_depth_idx) / bottom_depth - 1;

            for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
            {
                for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
                {
                    auto cur_top_expect = top_expect(neuron_idx, top_height_idx, top_width_idx);
                    auto cur_top_next_expect = top_next_expect(neuron_idx, top_height_idx, top_width_idx);

                    auto cur_bottom_value = shortterm_memory_idx < 0 ?
                        bottom_value(neuron_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx) :
                        bottom_shortterm_memories[shortterm_memory_idx](shortterm_memory_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx);
                    auto cur_bottom_next_value = (discriminative_training || shortterm_memory_idx >= 0) ? cur_bottom_value :
                        bottom_next_value(neuron_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx);

                    delta += cur_bottom_value * cur_top_expect - cur_bottom_next_value * cur_top_next_expect;
                }
            }

            neuron_weights[idx] += delta / (top_height * top_width) * learning_rate;
        });

        // update vbias, only for generative training and only for value not shortterm memory
        if (!discriminative_training)
        {
            // vbias does not cover shortterm memory part
            parallel_for_each(concurrency::extent<1>(bottom_depth), [=](index<1> idx) restrict(amp)
            {
                auto delta = 0.0;

                int depth_idx = idx[0];

                for (int bottom_height_idx = 0; bottom_height_idx < bottom_height; bottom_height_idx++)
                {
                    for (int bottom_width_idx = 0; bottom_width_idx < bottom_width; bottom_width_idx++)
                    {
                        auto cur_bottom_value = bottom_value(depth_idx, bottom_height_idx, bottom_width_idx);
                        auto cur_bottom_next_value = bottom_next_value(depth_idx, bottom_height_idx, bottom_width_idx);

                        delta += cur_bottom_value - cur_bottom_next_value;
                    }
                }

                vbias[idx] += delta / (bottom_height * bottom_width) * learning_rate;
            });
        }

        // update hbias
        parallel_for_each(hbias.extent, [=](index<1> idx) restrict(amp)
        {
            auto delta = 0.0;

            int neuron_idx = idx[0];

            for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
            {
                for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
                {
                    auto cur_top_expect = top_expect(neuron_idx, top_height_idx, top_width_idx);
                    auto cur_top_next_expect = top_next_expect(neuron_idx, top_height_idx, top_width_idx);

                    delta += cur_top_expect - cur_top_next_expect;
                }
            }

            hbias[idx] += delta / (top_height * top_width) * learning_rate;
        });

        // for output layer
        if (output_layer != nullptr)
        {
            // parameters to train
            array_view<double, 4> output_weights = output_layer->weights_view_;
            array_view<double> output_bias = output_layer->bias_view_;

            // readonly
            array_view<const double> output_value = output_layer->outputs_view_;
            array_view<const double> output_next_value = output_layer->next_outputs_view_;

            parallel_for_each(output_weights.extent, [=](index<4> idx) restrict(amp)
            {
                int output_idx = idx[0];
                int top_depth_idx = idx[1];
                int top_height_idx = idx[2];
                int top_width_idx = idx[3];

                auto delta = output_value(output_idx) * top_expect(top_depth_idx, top_height_idx, top_width_idx) -
                    output_next_value(output_idx) * top_next_expect(top_depth_idx, top_height_idx, top_width_idx);

                output_weights[idx] += delta * learning_rate;
            });

            parallel_for_each(output_bias.extent, [=](index<1> idx) restrict(amp)
            {
                auto delta = output_value[idx] - output_next_value[idx];

                output_bias[idx] += delta * learning_rate;
            });
        }
    }

    void ConvolveLayer::RandomizeParams(unsigned int seed)
    {
        default_random_engine generator(seed);
        normal_distribution<double> distribution(0.0, 0.1);

        for (auto& w : neuron_weights_)
        {
            w = distribution(generator);
        }

        neuron_weights_view_.refresh();
    }

    bitmap_image ConvolveLayer::GenerateImage() const
    {
        neuron_weights_view_.synchronize();
        neuron_activation_counts_view_.synchronize();
        neuron_life_counts_view_.synchronize();
        vbias_view_.synchronize();
        hbias_view_.synchronize();

        for (int i = 0; i < neuron_activation_counts_.size(); i++)
        {
            cout << neuron_activation_counts_[i] / fmax(1.0, neuron_life_counts_[i]) << endl;
        }

        bitmap_image image;

        const int block_size = 2;

        auto max_abs_weight = numeric_limits<double>::min();
        for (auto weight : neuron_weights_)
        {
            max_abs_weight = fmax(max_abs_weight, abs(weight));
        }

        auto max_abs_vbias = numeric_limits<double>::min();
        for (auto vbias : vbias_)
        {
            max_abs_vbias = fmax(max_abs_vbias, abs(vbias));
        }

        auto max_abs_hbias = numeric_limits<double>::min();
        for (auto hbias : hbias_)
        {
            max_abs_hbias = fmax(max_abs_hbias, abs(hbias));
        }

        if (neuron_width() == 1 && neuron_height() == 1)
        {
            image.setwidth_height((2 + neuron_depth()) * (block_size + 1), (2 + neuron_num()) * (block_size + 1), true);

            for (int i = 0; i < vbias_.size(); i++)
            {
                image.set_region((2 + i) * (block_size + 1), 0, block_size, block_size,
                    vbias_[i] >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(abs(vbias_[i]) / max_abs_vbias * 255.0));
            }

            for (int i = 0; i < hbias_.size(); i++)
            {
                image.set_region(0, (2 + i) * (block_size + 1), block_size, block_size,
                    hbias_[i] >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(abs(hbias_[i]) / max_abs_hbias * 255.0));
            }

            for (int neuron_idx = 0; neuron_idx < neuron_num(); neuron_idx++)
            {
                auto neuron_view = neuron_weights_view_[neuron_idx];
                for (int depth_idx = 0; depth_idx < neuron_depth(); depth_idx++)
                {
                    auto value = neuron_view(depth_idx, 0, 0);
                    image.set_region((2 + depth_idx) * (block_size + 1), (2 + neuron_idx) * (block_size + 1), block_size, block_size,
                        value >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                        static_cast<unsigned char>(abs(value) / max_abs_weight * 255.0));
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
                    static_cast<unsigned char>(abs(vbias_[i]) / max_abs_vbias * 255.0));
            }

            for (int i = 0; i < hbias_.size(); i++)
            {
                image.set_region(0, (2 + neuron_height() / 2 + (neuron_height() + 1) * i) * (block_size + 1), block_size, block_size,
                    hbias_[i] >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(abs(hbias_[i]) / max_abs_hbias * 255.0));
            }

            for (int neuron_idx = 0; neuron_idx < neuron_num(); neuron_idx++)
            {
                auto neuron_view = neuron_weights_view_[neuron_idx];
                for (int depth_idx = 0; depth_idx < neuron_depth(); depth_idx++)
                {
                    for (int height_idx = 0; height_idx < neuron_height(); height_idx++)
                    {
                        for (int width_idx = 0; width_idx < neuron_width(); width_idx++)
                        {
                            auto value = neuron_view(depth_idx, height_idx, width_idx);

                            image.set_region((2 + width_idx + depth_idx * (neuron_width() + 1)) * (block_size + 1),
                                (2 + neuron_idx * (neuron_height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                                value >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                                static_cast<unsigned char>(abs(value) / max_abs_weight * 255.0));
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

    PoolingLayer::PoolingLayer(PoolingLayer&& other)
        : block_height_(other.block_height_),
        block_width_(other.block_width_)
    {
    }

    void PoolingLayer::PassUp(const DataLayer& bottom_layer, DataSlot bottom_slot,
        DataLayer& top_layer, DataSlot top_slot) const
    {
        assert(top_layer.height() * block_height_ == bottom_layer.height());
        assert(top_layer.width() * block_width_ == bottom_layer.width());

        // readonly
        int block_height = block_height_;
        int block_width = block_width_;

        const auto& bottom_data = bottom_layer[bottom_slot];
        array_view<const double, 3> bottom_value = bottom_data.first;
        array_view<const double, 3> bottom_expect = bottom_data.second;

        // writeonly
        const auto& top_data = top_layer[top_slot];
        array_view<double, 3> top_value = top_data.first;
        array_view<double, 3> top_expect = top_data.second;
        top_value.discard_data();
        top_expect.discard_data();

        parallel_for_each(top_value.extent, [=](index<3> idx) restrict(amp)
        {
            auto max_value = 0.0;
            auto max_expect = 1.0;

            for (int height_idx = 0; height_idx < block_height; height_idx++)
            {
                for (int width_idx = 0; width_idx < block_width; width_idx++)
                {
                    auto value = bottom_value(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx);
                    auto expect = bottom_expect(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx);

                    max_value = fmax(max_value, value);
                    max_expect *= (1.0 - expect); // the probability that all nodes are 0
                }
            }
            max_expect = 1.0 - max_expect;// the probability that at least one node is 1.

            top_value[idx] = max_value;
            top_expect[idx] = max_expect;
        });
    }

    void PoolingLayer::PassDown(const DataLayer& top_layer, DataSlot top_slot,
        DataLayer& bottom_layer, DataSlot bottom_slot) const
    {
        assert(top_layer.height() * block_height_ == bottom_layer.height());
        assert(top_layer.width() * block_width_ == bottom_layer.width());

        // readonly
        int block_height = block_height_;
        int block_width = block_width_;

        const auto& top_data = top_layer[top_slot];
        array_view<const double, 3> top_value = top_data.first;
        array_view<const double, 3> top_expect = top_data.second;

        // writeonly
        const auto& bottom_data = bottom_layer[bottom_slot];
        array_view<double, 3> bottom_value = bottom_data.first;
        array_view<double, 3> bottom_expect = bottom_data.second;
        bottom_value.discard_data();
        bottom_expect.discard_data();

        auto& rand_collection = bottom_layer.rand_collection_;

        parallel_for_each(bottom_value.extent, [=](index<3> idx) restrict(amp)
        {
            // when we have memory, the bottom_layer can activate according to its memory. 
            // But now we just use uniform activation.

            int height_idx = idx[1] / block_height;// truncate towards zero
            int width_idx = idx[2] / block_width;


            bottom_expect[idx] = 1.0 - pow(1.0 - top_expect(idx[0], height_idx, width_idx), -1.0 * block_width * block_height);
            bottom_value[idx] = 0.0;// clear the value
        });

        parallel_for_each(top_value.extent, [=](index<3> idx) restrict(amp)
        {
            if (top_value[idx] == 1.0)
            {
                // randomly select a node in bottom_layer to activate
                int height_idx = rand_collection[idx].next_uint() % block_height;
                int width_idx = rand_collection[idx].next_uint() % block_width;

                bottom_value(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx) = 1.0;
            }
        });
    }

#pragma endregion

#pragma region deep model

    DeepModel::DeepModel(unsigned int model_seed) : random_engine_(model_seed)
    {
    }

    void DeepModel::AddDataLayer(int depth, int height, int width, int shortterm_memory_num)
    {
        assert(layer_stack_.empty());
        data_layers_.emplace_back(shortterm_memory_num, depth, height, width, uniform_int_distribution<int>()(random_engine_));
        layer_stack_.emplace_back(LayerType::kDataLayer, data_layers_.size() - 1);
    }

    void DeepModel::AddDataLayer(int shortterm_memory_num)
    {
        assert(layer_stack_.size() >= 2);
        assert(layer_stack_.back().first == LayerType::kConvolveLayer || layer_stack_.back().first == LayerType::kPoolingLayer);
        assert(layer_stack_[layer_stack_.size() - 2].first == LayerType::kDataLayer);

        const auto& last_data_layer = data_layers_[layer_stack_[layer_stack_.size() - 2].second];
        if (layer_stack_.back().first == LayerType::kConvolveLayer)
        {
            auto& conv_layer = convolve_layers_[layer_stack_.back().second];
            data_layers_.emplace_back(shortterm_memory_num,
                conv_layer.neuron_num(),
                last_data_layer.height() - conv_layer.neuron_height() + 1,
                last_data_layer.width() - conv_layer.neuron_width() + 1,
                uniform_int_distribution<int>()(random_engine_));
        }
        else
        {
            const auto& pooling_layer = pooling_layers[layer_stack_.back().second];
            assert(last_data_layer.height() % pooling_layer.block_height() == 0);
            assert(last_data_layer.width() % pooling_layer.block_width() == 0);
            data_layers_.emplace_back(shortterm_memory_num,
                last_data_layer.depth(),
                last_data_layer.height() / pooling_layer.block_height(),
                last_data_layer.width() / pooling_layer.block_width(),
                uniform_int_distribution<int>()(random_engine_));
        }
        layer_stack_.emplace_back(LayerType::kDataLayer, data_layers_.size() - 1);
    }

    void DeepModel::AddConvolveLayer(int neuron_num, int neuron_height, int neuron_width)
    {
        assert(!layer_stack_.empty() && layer_stack_.back().first == LayerType::kDataLayer);
        const auto& last_data_layer = data_layers_[layer_stack_.back().second];
        convolve_layers_.emplace_back(neuron_num, last_data_layer.depth() * (1 + last_data_layer.shortterm_memory_num()), neuron_height, neuron_width);
        convolve_layers_.back().RandomizeParams(uniform_int_distribution<int>()(random_engine_));
        layer_stack_.emplace_back(LayerType::kConvolveLayer, convolve_layers_.size() - 1);
    }

    void DeepModel::AddOutputLayer(int output_num)
    {
        assert(!layer_stack_.empty() && layer_stack_.back().first == LayerType::kDataLayer);
        auto last_data_layer_idx = layer_stack_.back().second;
        const auto& last_data_layer = data_layers_[last_data_layer_idx];

        if (!output_layers_.count(last_data_layer_idx))
        {
            output_layers_.emplace(piecewise_construct, forward_as_tuple(last_data_layer_idx),
                forward_as_tuple(output_num, last_data_layer.depth(), last_data_layer.height(), last_data_layer.width()));

            output_layers_.at(last_data_layer_idx).RandomizeParams(uniform_int_distribution<int>()(random_engine_));
        }
    }

    void DeepModel::PassUp(const vector<double>& data)
    {
        assert(!layer_stack_.empty() && layer_stack_.front().first == LayerType::kDataLayer);

        // set the bottom data layer
        data_layers_[layer_stack_.front().second].SetValue(data);

        for (int layer_idx = 1; layer_idx + 1 < layer_stack_.size(); layer_idx += 2)
        {
            assert(layer_stack_[layer_idx - 1].first == LayerType::kDataLayer);
            assert(layer_stack_[layer_idx + 1].first == LayerType::kDataLayer);
            auto& bottom_data_layer = data_layers_[layer_stack_[layer_idx - 1].second];
            auto& top_data_layer = data_layers_[layer_stack_[layer_idx + 1].second];
            top_data_layer.ActivateDropout();

            assert(layer_stack_[layer_idx].first == LayerType::kConvolveLayer || layer_stack_[layer_idx].first == LayerType::kPoolingLayer);
            if (layer_stack_[layer_idx].first == LayerType::kConvolveLayer)
            {
                const auto& conv_layer = convolve_layers_[layer_stack_[layer_idx].second];
                top_data_layer.ActivateLongtermMemory(conv_layer);
                conv_layer.PassUp(bottom_data_layer, DataSlot::kCurrent, top_data_layer, DataSlot::kCurrent);
            }
            else
            {
                const auto& pooling_layer = pooling_layers[layer_stack_[layer_idx].second];
                pooling_layer.PassUp(bottom_data_layer, DataSlot::kCurrent, top_data_layer, DataSlot::kCurrent);
            }
        }
    }

    void DeepModel::PassDown()
    {
        assert(!layer_stack_.empty() && layer_stack_.back().first == LayerType::kDataLayer);

        // prepare top layer for passing down
        auto& roof_data_layer = data_layers_[layer_stack_.back().second];
        roof_data_layer.value_view_.copy_to(roof_data_layer.next_value_view_);
        roof_data_layer.expect_view_.copy_to(roof_data_layer.next_expect_view_);

        for (int layer_idx = static_cast<int>(convolve_layers_.size()) - 2; layer_idx >= 1; layer_idx -= 2)
        {
            assert(layer_stack_[layer_idx - 1].first == LayerType::kDataLayer);
            assert(layer_stack_[layer_idx + 1].first == LayerType::kDataLayer);
            auto& bottom_data_layer = data_layers_[layer_stack_[layer_idx - 1].second];
            auto& top_data_layer = data_layers_[layer_stack_[layer_idx + 1].second];
            bottom_data_layer.ActivateDropout();

            assert(layer_stack_[layer_idx].first == LayerType::kConvolveLayer || layer_stack_[layer_idx].first == LayerType::kPoolingLayer);
            if (layer_stack_[layer_idx].first == LayerType::kConvolveLayer)
            {
                const auto& conv_layer = convolve_layers_[layer_stack_[layer_idx].second];
                conv_layer.PassDown(top_data_layer, DataSlot::kNext, bottom_data_layer, DataSlot::kNext);
            }
            else
            {
                const auto& pooling_layer = pooling_layers[layer_stack_[layer_idx].second];
                pooling_layer.PassDown(top_data_layer, DataSlot::kNext, bottom_data_layer, DataSlot::kNext);
            }
        }
    }

    double DeepModel::TrainLayer(const vector<double>& data, int layer_idx, double learning_rate,
        double dropout_prob, const int label, bool discriminative_training)
    {
        assert(layer_stack_[layer_idx].first == LayerType::kConvolveLayer);
        assert(layer_idx >= 1 && layer_stack_[layer_idx - 1].first == LayerType::kDataLayer);
        assert(layer_stack_[layer_idx + 1].first == LayerType::kDataLayer);

        auto& bottom_data_layer = data_layers_[layer_stack_[layer_idx - 1].second];
        auto& top_data_layer = data_layers_[layer_stack_[layer_idx + 1].second];

        auto& conv_layer = convolve_layers_[layer_stack_[layer_idx].second];

        // train with contrastive divergence (CD) algorithm to maximize likelihood on dataset
        bottom_data_layer.SetValue(data);
        top_data_layer.ActivateDropout(dropout_prob);
        top_data_layer.ActivateLongtermMemory(conv_layer);

        if (label == -1)
        {
            // purely generative training without label
            conv_layer.PassUp(bottom_data_layer, DataSlot::kCurrent, top_data_layer, DataSlot::kCurrent);
            conv_layer.PassDown(top_data_layer, DataSlot::kCurrent, bottom_data_layer, DataSlot::kNext);
            conv_layer.PassUp(bottom_data_layer, DataSlot::kNext, top_data_layer, DataSlot::kNext);

            conv_layer.Train(bottom_data_layer, top_data_layer, learning_rate);
        }
        else
        {
            // training data has label
            auto& output_layer = output_layers_.at(layer_stack_[layer_idx + 1].second);
            output_layer.SetLabel(label);

            conv_layer.PassUp(bottom_data_layer, DataSlot::kCurrent,
                top_data_layer, DataSlot::kCurrent, &output_layer, DataSlot::kCurrent);

            if (discriminative_training)
            {
                output_layer.PassDown(top_data_layer, DataSlot::kCurrent, DataSlot::kNext);
                conv_layer.PassUp(bottom_data_layer, DataSlot::kCurrent,
                    top_data_layer, DataSlot::kNext, &output_layer, DataSlot::kNext);
            }
            else
            {
                conv_layer.PassDown(top_data_layer, DataSlot::kCurrent,
                    bottom_data_layer, DataSlot::kNext, &output_layer, DataSlot::kNext);

                conv_layer.PassUp(bottom_data_layer, DataSlot::kNext,
                    top_data_layer, DataSlot::kNext, &output_layer, DataSlot::kNext);
            }

            conv_layer.Train(bottom_data_layer, top_data_layer, learning_rate, &output_layer, discriminative_training);
        }

        // update shortterm memory
        bottom_data_layer.Memorize();

        return bottom_data_layer.ReconstructionError(DataSlot::kNext);
    }

    int DeepModel::PredictLabel(const vector<double>& data, const int layer_idx, const double dropout_prob)
    {
        assert(layer_idx >= 0 && layer_idx + 2 < layer_stack_.size()
            && layer_stack_[layer_idx].first == LayerType::kDataLayer
            && layer_stack_[layer_idx + 1].first == LayerType::kConvolveLayer
            && layer_stack_[layer_idx + 2].first == LayerType::kDataLayer);

        auto& bottom_data_layer = data_layers_[layer_stack_[layer_idx].second];
        const auto& conv_layer = convolve_layers_[layer_stack_[layer_idx + 1].second];
        auto& top_data_layer = data_layers_[layer_stack_[layer_idx + 2].second];

        auto& output_layer = output_layers_.at(layer_stack_[layer_idx + 2].second);

        bottom_data_layer.SetValue(data);
        // top layer activation is ignored when predicting labels
        return output_layer.PredictLabel(bottom_data_layer, DataSlot::kCurrent,
            top_data_layer, DataSlot::kCurrent, conv_layer, dropout_prob);
    }

    double DeepModel::Evaluate(const vector<const vector<double>>& dataset, const vector<const int>& labels,
        int layer_idx, const double dropout_prob)
    {
        assert(dataset.size() == labels.size());

        auto correct_count = 0.0;

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

    void DeepModel::GenerateImages(const string& folder) const
    {
        for (int i = 0; i < data_layers_.size(); i++)
        {
            data_layers_[i].GenerateImage().save_image(folder + "\\layer" + to_string(i) + "_data.bmp");
        }

        for (int i = 0; i < convolve_layers_.size(); i++)
        {
            convolve_layers_[i].GenerateImage().save_image(folder + "\\layer" + to_string(i) + "_conv.bmp");
        }

        for (const auto& pair : output_layers_)
        {
            pair.second.GenerateImage().save_image(folder + "\\layer" + to_string(pair.first) + "_output.bmp");
        }
    }

#pragma endregion
}