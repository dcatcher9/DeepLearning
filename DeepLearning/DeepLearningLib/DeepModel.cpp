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
        temp_value_view_(value_view_.extent),
        temp_expect_view_(value_view_.extent),
        // there is no empty array_view support in amp now, so we just set the extent to (1,1,1,1) when the shortterm_memory_num == 0
        shortterm_memory_view_(shortterm_memory_num == 0 ? make_extent(1, 1, 1, 1) : make_extent(shortterm_memory_num, depth, height, width)),
        shortterm_memory_index_view_(std::max(1, shortterm_memory_num)),
        rand_collection_(value_view_.extent, seed)
    {
        fill(value_view_, 0.0f);
        fill(expect_view_, 0.0f);
        fill(next_value_view_, 0.0f);
        fill(next_expect_view_, 0.0f);
        fill(temp_value_view_, 0.0f);
        fill(temp_expect_view_, 0.0f);
        fill(shortterm_memory_view_, 0.0f);
        for (int time = 0; time < shortterm_memory_num; time++)
        {
            shortterm_memory_index_view_[time] = time;
        }
    }

    DataLayer::DataLayer(DataLayer&& other)
        : shortterm_memory_num_(other.shortterm_memory_num_),
        value_view_(other.value_view_),
        expect_view_(other.expect_view_),
        next_value_view_(other.next_value_view_),
        next_expect_view_(other.next_expect_view_),
        temp_value_view_(other.temp_value_view_),
        temp_expect_view_(other.temp_expect_view_),
        shortterm_memory_view_(other.shortterm_memory_view_),
        shortterm_memory_index_view_(other.shortterm_memory_index_view_),
        rand_collection_(other.rand_collection_)
    {
    }

    void DataLayer::SetValue(const vector<float>& data)
    {
        assert(data.size() == value_view_.extent.size());

        // Copy the data
        copy(data.begin(), data.end(), value_view_);
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

        array_view<const float, 3> value_view = this->value_view_;
        array_view<const float, 3> recon_expect_view = (*this)[slot].second;

        // TODO: compare with reduce method for performance
        parallel_for_each(value_view.extent,
            [=](index<3> idx) restrict(amp)
        {
            float diff = value_view[idx] - recon_expect_view[idx];
            atomic_fetch_add(&result(0), diff * diff);
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
        temp_outputs_view_(output_num),
        bias_(output_num),
        bias_view_(output_num, bias_),
        weights_(output_num * input_depth * input_height * input_width),
        weights_view_(make_extent(output_num, input_depth, input_height, input_width), weights_)
    {
        fill(outputs_view_, 0.0f);
        fill(next_outputs_view_, 0.0f);
        fill(temp_outputs_view_, 0.0f);
    }

    OutputLayer::OutputLayer(OutputLayer&& other)
        : outputs_view_(other.outputs_view_),
        next_outputs_view_(other.next_outputs_view_),
        temp_outputs_view_(other.temp_outputs_view_),
        bias_(move(other.bias_)),
        bias_view_(other.bias_view_),
        weights_(move(other.weights_)),
        weights_view_(other.weights_view_)
    {
    }

    void OutputLayer::SetLabel(const int label)
    {
        assert(label >= 0 && label < this->output_num());
        fill(outputs_view_, 0.0f);
        outputs_view_[label] = 1.0f;
    }

    void OutputLayer::RandomizeParams(unsigned int seed)
    {
        default_random_engine generator(seed);
        normal_distribution<float> distribution(0.0f, 0.05f);

        for (float& w : weights_)
        {
            w = distribution(generator);
        }

        weights_view_.refresh();
    }

    int OutputLayer::PredictLabel(DataLayer& bottom_layer, DataSlot bottom_slot, DataLayer& top_layer, DataSlot top_slot,
        const ConvolveLayer& conv_layer, const float dropout_prob)
    {
        assert(bottom_slot != DataSlot::kTemp);// we will use DataSlot::kTemp in this function.
        assert(top_layer.depth() == conv_layer.longterm_memory_num() + conv_layer.neuron_num() && top_layer.depth() == this->input_depth());
        assert(top_layer.width() == bottom_layer.width() - conv_layer.neuron_width() + 1 && top_layer.width() == this->input_width());
        assert(top_layer.height() == bottom_layer.height() - conv_layer.neuron_height() + 1 && top_layer.height() == this->input_height());

        // calculate base score, ignore top layer activation
        // pass up with full activation in top layers
        top_layer.Activate();
        conv_layer.PassUp(bottom_layer, bottom_slot, top_layer, top_slot);
        if (conv_layer.longterm_memory_num() > 0)
        {
            conv_layer.PassDown(top_layer, top_slot, bottom_layer, DataSlot::kTemp);
            conv_layer.ActivateMemory(top_layer, top_slot, bottom_layer, bottom_slot, DataSlot::kTemp);
        }

        // read only
        const int top_depth = top_layer.depth();
        const int top_height = top_layer.height();
        const int top_width = top_layer.width();

        array_view<float, 3> top_raw_weight = top_layer.raw_weight_view_;
        array_view<const float> output_bias = this->bias_view_;
        array_view<const float, 4> output_weights = this->weights_view_;

        // write only
        array_view<float> outputs = this->outputs_view_;
        outputs.discard_data();

        parallel_for_each(outputs.extent, [=](index<1> idx) restrict(amp)
        {
            float result = output_bias[idx];

            const auto& current_output_weights = output_weights[idx[0]];

            for (int depth_idx = 0; depth_idx < top_depth; depth_idx++)
            {
                for (int height_idx = 0; height_idx < top_height; height_idx++)
                {
                    for (int width_idx = 0; width_idx < top_width; width_idx++)
                    {
                        float score = top_raw_weight(depth_idx, height_idx, width_idx) + current_output_weights(depth_idx, height_idx, width_idx);
                        result += logf((expf(score) + 1.0f) * (1.0f - dropout_prob) + 2.0f * dropout_prob);
                    }
                }
            }

            outputs[idx] = result;
        });

        int max_idx = 0;
        float max_value = outputs[max_idx];

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
        array_view<const float, 3> top_value = top_layer[top_slot].first;
        array_view<const float> output_bias = this->bias_view_;
        array_view<const float, 4> output_weights = this->weights_view_;

        // writeonly
        array_view<float> output_value = (*this)[output_slot];
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

            output_value[idx] = 1.0f / (1.0f + expf(-result));
        });
    }

    bitmap_image OutputLayer::GenerateImage() const
    {
        weights_view_.synchronize();
        bias_view_.synchronize();

        bitmap_image image;

        const int block_size = 2;

        float max_abs_weight = numeric_limits<float>::min();
        for (float weight : weights_)
        {
            max_abs_weight = fmax(max_abs_weight, abs(weight));
        }

        float max_abs_bias = numeric_limits<float>::min();
        for (float bias : bias_)
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
                    float value = weights_[output_idx * input_depth() + depth_idx];
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
                            float value = cur_weights_view(depth_idx, height_idx, width_idx);

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
        neuron_activation_probs_(neuron_num),
        model_likelihood_view_(1, 1),
        neuron_likelihood_view_(1, 1, 1),
        neuron_activation_probs_view_(neuron_num, neuron_activation_probs_),
        neuron_activations_view_(1, 1, 1),
        neuron_weights_view_(make_extent(neuron_num, neuron_depth, neuron_height, neuron_width), neuron_weights_),
        vbias_(neuron_depth),
        vbias_view_(neuron_depth, vbias_),
        hbias_(neuron_num),
        hbias_view_(neuron_num, hbias_),
        dropout_prob_(0.0f),
        dropout_activations_view_(1, 1, 1),
        top_raw_weights_view_(1, 1, 1)
    {
    }

    ConvolveLayer::ConvolveLayer(ConvolveLayer&& other)
        : neuron_weights_(move(other.neuron_weights_)),
        neuron_activation_probs_(move(other.neuron_activation_probs_)),
        model_likelihood_view_(other.model_likelihood_view_),
        neuron_likelihood_view_(other.neuron_likelihood_view_),
        neuron_activation_probs_view_(other.neuron_activation_probs_view_),
        neuron_activations_view_(other.neuron_activations_view_),
        neuron_weights_view_(other.neuron_weights_view_),
        vbias_(move(other.vbias_)),
        vbias_view_(other.vbias_view_),
        hbias_(move(other.hbias_)),
        hbias_view_(other.hbias_view_),
        dropout_prob_(other.dropout_prob_),
        dropout_activations_view_(other.dropout_activations_view_),
        top_raw_weights_view_(other.top_raw_weights_view_)
    {
    }

    void ConvolveLayer::ActivateDropout(tinymt_collection<3>& rand_collection, float dropout_prob)
    {
        assert(rand_collection.extent() == this->dropout_activations_view_.extent);
        // no need to change activation when the prob = 1.0 or 0.0 again.
        if (dropout_prob == dropout_prob_ &&
            (dropout_prob == 1.0f || dropout_prob == 0.0f))
        {
            return;
        }

        array_view<int, 3> dropout_activations = this->dropout_activations_view_;

        parallel_for_each(dropout_activations.extent,
            [=](index<3> idx) restrict(amp)
        {
            dropout_activations[idx] = rand_collection[idx].next_single() < dropout_prob ? 1 : 0;
        });

        dropout_prob_ = dropout_prob;
    }

    void ConvolveLayer::ActivateRegularNeurons(tinymt_collection<3>& rand_collection)
    {
        assert(rand_collection.extent() == this->neuron_activations_view_.extent);

        array_view<const float> neuron_activation_probs = this->neuron_activation_probs_view_;
        array_view<int, 3> neuron_activations = this->neuron_activations_view_;

        parallel_for_each(neuron_activations.extent,
            [=](index<3> idx) restrict(amp)
        {
            neuron_activations[idx] = rand_collection[idx].next_single() < neuron_activation_probs[idx[0]] ? 1 : 0;
        });
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
        const float min_float = numeric_limits<float>::lowest();
        const int neuron_depth = this->neuron_depth();
        const int neuron_height = this->neuron_height();
        const int neuron_width = this->neuron_width();
        const int bottom_depth = bottom_layer.depth();
        const int shortterm_memory_num = bottom_layer.shortterm_memory_num();

        array_view<const float, 4> neuron_weights = this->neuron_weights_view_;
        array_view<const float> hbias = this->hbias_view_;
        array_view<const float, 3> bottom_value = bottom_layer[bottom_slot].first;
        array_view<const float, 4> bottom_shortterm_memory = bottom_layer.shortterm_memory_view_;
        array_view<const int, 1> bottom_shortterm_memory_index = bottom_layer.shortterm_memory_index_view_;

        array_view<const int, 3> dropout_activation = this->dropout_activations_view_;
        array_view<const int, 3> neuron_activation = this->neuron_activations_view_;

        // output layer
        static array_view<float> s_empty_output_value(1);
        array_view<const float> output_value = output_layer_exist ? (*output_layer)[output_slot] : s_empty_output_value;

        static array_view<float, 4> s_empty_output_weights(make_extent(1, 1, 1, 1));
        array_view<const float, 4> output_weights = output_layer_exist ? output_layer->weights_view_ : s_empty_output_weights;

        // writeonly
        const auto& top_data = top_layer[top_slot];
        array_view<float, 3> top_value = top_data.first;
        array_view<float, 3> top_expect = top_data.second;
        array_view<float, 3> top_raw_weights = this->top_raw_weights_view_;
        array_view<float, 3> neuron_likelihood = this->neuron_likelihood_view_;
        
        top_value.discard_data();
        top_expect.discard_data();
        top_raw_weights.discard_data();
        neuron_likelihood.discard_data();
        
        auto& rand_collection = top_layer.rand_collection_;

        // non-tiled version
        // top_value = longterm_memory + neuron 
        parallel_for_each(top_value.extent,
            [=](index<3> idx) restrict(amp)
        {
            int top_depth_idx = idx[0];
            int top_height_idx = idx[1];
            int top_width_idx = idx[2];

            if (dropout_activation[idx] == 1)
            {
                top_expect[idx] = 0.0f;
                top_value[idx] = 0.0f;
                top_raw_weights[idx] = min_float;
                neuron_likelihood[idx] = min_float;
            }
            else
            {
                float raw_weight = hbias[top_depth_idx];
                float likelihood = 0.0f;

                if (output_layer_exist)
                {
                    for (int output_idx = 0; output_idx < output_value.extent[0]; output_idx++)
                    {
                        raw_weight += output_value[output_idx] * output_weights[output_idx][idx];
                    }
                }

                array_view<const float, 3> current_neuron = neuron_weights[top_depth_idx];

                for (int depth_idx = 0; depth_idx < bottom_depth; depth_idx++)
                {
                    for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                    {
                        for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                        {
                            float value = bottom_value(depth_idx, top_height_idx + height_idx, top_width_idx + width_idx);
                            float weight = current_neuron(depth_idx, height_idx, width_idx);
                            raw_weight += value * weight;
                            likelihood += -logf(1.0f + expf((1.0f - 2 * value) * weight));
                        }
                    }
                }

                // convolve short-term memory in bottom layer if exists. likelihood does not consider shortterm memory
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
                neuron_likelihood[idx] = likelihood;

                // only activate the regular neuron
                if (neuron_activation[idx] == 1)
                {
                    float prob = 1.0f / (1.0f + expf(-raw_weight));
                    top_expect[idx] = prob;
                    top_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;
                }
                else
                {
                    top_expect[idx] = 0.0f;
                    top_value[idx] = 0.0f;
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

        array_view<const float, 4> neuron_weights = this->neuron_weights_view_;
        array_view<const float> vbias = this->vbias_view_;
        array_view<const float, 3> top_value = top_layer[top_slot].first;

        // writeonly
        const auto& bottom_data = bottom_layer[bottom_slot];
        array_view<float, 3> bottom_value = bottom_data.first;
        array_view<float, 3> bottom_expect = bottom_data.second;
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

            float raw_weight = vbias[cur_depth_idx];

            for (int neuron_idx = 0; neuron_idx < neuron_num; neuron_idx++)
            {
                array_view<const float, 3> current_neuron = neuron_weights[neuron_idx];

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
            float prob = 1.0f / (1.0f + expf(-raw_weight));

            bottom_expect[idx] = prob;
            bottom_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;
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

            array_view<const float> output_bias = output_layer->bias_view_;
            array_view<const float, 4> output_weights = output_layer->weights_view_;

            array_view<float> output_value = (*output_layer)[output_slot];
            output_value.discard_data();

            // non-tiled version
            parallel_for_each(output_value.extent,
                [=](index<1> idx) restrict(amp)
            {
                float raw_weight = output_bias[idx];

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

                output_value[idx] = 1.0f / (1.0f + expf(-raw_weight));
            });
        }
    }

    void ConvolveLayer::ActivateMemoryNeuron(DataLayer& top_layer, DataSlot top_slot,
        const DataLayer& bottom_layer, DataSlot bottom_data_slot, DataSlot bottom_model_slot) const
    {
        // readonly
        const int bottom_depth = bottom_layer.depth();
        const int neuron_height = this->neuron_height();
        const int neuron_width = this->neuron_width();
        array_view<const float, 3> bottom_data_value = bottom_layer[bottom_data_slot].first;
        array_view<const float, 3> bottom_model_expect = bottom_layer[bottom_model_slot].second;

        // calculate data and model similarity, based on logistic neural activation.
        // there are many ways to measure similarity, we use likelihood to align with maximum likelihood training
        // does not consider shortterm memory
        array_view<float, 2> model_likelihood_view = this->model_likelihood_view_;// write only
        model_likelihood_view.discard_data();

        parallel_for_each(model_likelihood_view.extent,
            [=](index<2> idx) restrict(amp)
        {
            int cur_height_idx = idx[0];
            int cur_width_idx = idx[1];

            float likelihood = 0.0f;
            for (int depth_idx = 0; depth_idx < bottom_depth; depth_idx++)
            {
                for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                {
                    for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                    {
                        float value = bottom_data_value(depth_idx, height_idx + cur_height_idx, width_idx + cur_width_idx);
                        float model_expect = bottom_model_expect(depth_idx, height_idx + cur_height_idx, width_idx + cur_width_idx);
                        likelihood += logf(value * model_expect + (1.0f - value) * (1.0f - model_expect));
                    }
                }
            }

            model_likelihood_view[idx] = likelihood;
        });

        // read only
        array_view<const float, 3> neuron_likelihood_view = this->neuron_likelihood_view_;
        array_view<const float, 3> top_raw_weights = this->top_raw_weights_view_;
        array_view<const int, 3> dropout_activation = this->dropout_activations_view_;
        array_view<const int, 3> neuron_activation = this->neuron_activations_view_;

        const auto& top_data = top_layer[top_slot];
        array_view<float, 3> top_value = top_data.first;
        array_view<float, 3> top_expect = top_data.second;

        auto& rand_collection = top_layer.rand_collection_;

        parallel_for_each(neuron_likelihood_view.extent,
            [=](index<3> idx) restrict(amp)
        {
            int cur_depth_idx = idx[0];
            int cur_height_idx = idx[1];
            int cur_width_idx = idx[2];

            // not dropped out but inactivated neuron
            if (dropout_activation[idx] == 0 && neuron_activation[idx] == 0)
            {
                if (neuron_likelihood_view[idx] >= model_likelihood_view(cur_height_idx, cur_width_idx))
                {
                    float prob = 1.0f / (1.0f + expf(-top_raw_weights[idx]));
                    top_expect[idx] = prob;
                    top_value[idx] = rand_collection[idx].next_single() <= prob ? 1.0f : 0.0f;;
                }
            }
        });
    }

    void ConvolveLayer::Train(const DataLayer& bottom_layer, const DataLayer& top_layer, float learning_rate,
        OutputLayer* output_layer, bool discriminative_training)
    {
        // readonly
        const int top_height = top_layer.height();
        const int top_width = top_layer.width();
        const int bottom_depth = bottom_layer.depth();
        const int bottom_height = bottom_layer.height();
        const int bottom_width = bottom_layer.width();
        const int shortterm_memory_num = bottom_layer.shortterm_memory_num();

        array_view<const float, 3> top_expect = top_layer.expect_view_;
        array_view<const float, 3> top_next_expect = top_layer.next_expect_view_;
        array_view<const float, 3> bottom_value = bottom_layer.value_view_;
        array_view<const float, 3> bottom_next_value = bottom_layer.next_value_view_;
        array_view<const float, 3> bottom_next_expect = bottom_layer.next_expect_view_;
        array_view<const float, 4> bottom_shortterm_memories = bottom_layer.shortterm_memory_view_;

        // parameters to train
        array_view<float, 4> neuron_weights = this->neuron_weights_view_;
        array_view<float, 4> longterm_memory_weights = this->longterm_memory_weights_view_;

        array_view<float> vbias = this->vbias_view_;
        array_view<float> hbias = this->hbias_view_;

        // non-tiled version
        parallel_for_each(neuron_weights.extent, [=](index<4> idx) restrict(amp)
        {
            float delta = 0.0f;

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
                    // top = longterm memory + neuron
                    float cur_top_expect = top_expect(neuron_idx + longterm_memory_num, top_height_idx, top_width_idx);
                    float cur_top_next_expect = top_next_expect(neuron_idx + longterm_memory_num, top_height_idx, top_width_idx);

                    float cur_bottom_value = shortterm_memory_idx < 0 ?
                        bottom_value(neuron_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx) :
                        bottom_shortterm_memories[shortterm_memory_idx](shortterm_memory_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx);
                    float cur_bottom_next_value = (discriminative_training || shortterm_memory_idx >= 0) ? cur_bottom_value :
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
                    float cur_top_expect = top_expect(neuron_idx + longterm_memory_num, top_height_idx, top_width_idx);
                    float cur_top_next_expect = top_next_expect(neuron_idx + longterm_memory_num, top_height_idx, top_width_idx);

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

                float delta = 0;
                if (top_depth_idx < longterm_memory_num)
                {
                    delta = (output_value(output_idx) - output_next_value(output_idx)) * top_expect(top_depth_idx, top_height_idx, top_width_idx);
                }
                else
                {
                    delta = output_value(output_idx) * top_expect(top_depth_idx, top_height_idx, top_width_idx) -
                        output_next_value(output_idx) * top_next_expect(top_depth_idx, top_height_idx, top_width_idx);
                }

                output_weights[idx] += delta * learning_rate;
            });

            parallel_for_each(output_bias.extent, [=](index<1> idx) restrict(amp)
            {
                float delta = output_value[idx] - output_next_value[idx];

                output_bias[idx] += delta * learning_rate;
            });
        }

        // update longterm memory weights, the key idea is weighted k-mean clustering
        if (longterm_memory_num > 0)
        {
            const int longterm_memory_depth = this->longterm_memory_depth_;
            const int neuron_height = this->neuron_height();
            const int neuron_width = this->neuron_width();

            array_view<const float, 2> longterm_memory_affinity_prior = this->longterm_memory_affinity_prior_view_;
            array_view<const int, 2> longterm_memory_max_affinity_index = this->longterm_memory_max_affinity_index_view_;
            array_view<const float, 3> longterm_memory_affinity = this->longterm_memory_affinity_view_;
            array_view<float> longterm_memory_gain = this->longterm_memory_gain_view_;

            // enhanced existing longterm memories, expire old memories if necessary
            const float kLongtermMemoryDecay = this->kLongtermMemoryDecay;
            parallel_for_each(longterm_memory_gain.extent, [=](index<1> idx) restrict(amp)
            {
                int longterm_memory_idx = idx[0];

                auto cur_longterm_memory_weights = longterm_memory_weights[longterm_memory_idx];

                for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
                {
                    for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
                    {
                        // only update the maximum activated memory at this position
                        if (longterm_memory_idx != longterm_memory_max_affinity_index(top_height_idx, top_width_idx))
                        {
                            continue;
                        }

                        float memory_affinity = longterm_memory_affinity(longterm_memory_idx, top_height_idx, top_width_idx);
                        float model_affinity = longterm_memory_affinity_prior(top_height_idx, top_width_idx);

                        // enhance these activated longterm memories
                        if (memory_affinity >= model_affinity)
                        {
                            for (int depth_idx = 0; depth_idx < longterm_memory_depth; ++depth_idx)
                            {
                                for (int height_idx = 0; height_idx < neuron_height; ++height_idx)
                                {
                                    for (int width_idx = 0; width_idx < neuron_width; ++width_idx)
                                    {
                                        float& cur_memory_weight = cur_longterm_memory_weights(depth_idx, height_idx, width_idx);
                                        float cur_memory_expect = 1.0f / (1.0f + expf(-cur_memory_weight));

                                        float cur_bottom_value = bottom_value(depth_idx, height_idx + top_height_idx, width_idx + top_width_idx);
                                        cur_memory_weight += -2 * (cur_memory_expect - cur_bottom_value) * cur_memory_expect * (1 - cur_memory_expect) * learning_rate;
                                    }
                                }
                            }

                            longterm_memory_gain[idx] += memory_affinity - model_affinity;
                        }
                    }
                }

                longterm_memory_gain[idx] *= kLongtermMemoryDecay;
            });

            const auto& min_gain = min(longterm_memory_gain);
            const auto& min_affinity = min(longterm_memory_affinity_prior);

            // the affinity of input data to itself as weight
            const float self_affinity = static_cast<float>(bottom_depth * this->neuron_height() * this->neuron_width() * (1.0 - pow(1 + exp(1), -2)));

            if (min_gain.second < self_affinity - min_affinity.second)
            {
                longterm_memory_gain[min_gain.first] = self_affinity - min_affinity.second;

                const int min_affinity_height_idx = min_affinity.first[0];
                const int min_affinity_width_idx = min_affinity.first[1];
                array_view<float, 3> min_gain_weights = longterm_memory_weights[min_gain.first[0]];
                min_gain_weights.discard_data();

                parallel_for_each(min_gain_weights.extent, [=](index<3> idx) restrict(amp)
                {
                    int neuron_depth_idx = idx[0];
                    int neuron_height_idx = idx[1];
                    int neuron_width_idx = idx[2];

                    float cur_bottom_value = bottom_value(neuron_depth_idx, neuron_height_idx + min_affinity_height_idx, neuron_width_idx + min_affinity_width_idx);
                    min_gain_weights[idx] = (cur_bottom_value * 2 - 1.0f);// map value 1 to weight 1, value 0 to weight -1
                });

                if (output_layer != nullptr)
                {
                    for (int i = 0; i < output_layer->output_num(); ++i)
                    {
                        output_layer->weights_view_[i](min_gain.first[0], min_affinity_height_idx, min_affinity_width_idx) = (output_layer->outputs_view_[i] * 2 - 1.0f);
                    }
                }
            }
        }
    }

    bool ConvolveLayer::FitTopLayer(const DataLayer& top_layer)
    {
        assert(this->neuron_num() == top_layer.depth());
        if (model_likelihood_view_.extent[0] == top_layer.height()
            && model_likelihood_view_.extent[1] == top_layer.width())
        {
            return false;
        }

        model_likelihood_view_ = array_view<float, 2>(top_layer.height(), top_layer.width());
        neuron_likelihood_view_ = array_view<float, 3>(top_layer.depth(), top_layer.height(), top_layer.width());
        neuron_activations_view_ = array_view<int, 3>(neuron_likelihood_view_.extent);
        dropout_activations_view_ = array_view<int, 3>(neuron_likelihood_view_.extent);
        top_raw_weights_view_ = array_view<float, 3>(neuron_likelihood_view_.extent);

        dropout_prob_ = 0.0f;
        fill(dropout_activations_view_, 0);

        return true;
    }

    void ConvolveLayer::RandomizeParams(unsigned int seed)
    {
        default_random_engine generator(seed);
        normal_distribution<float> distribution(0.0f, 0.05f);

        for (float& w : neuron_weights_)
        {
            w = distribution(generator);
        }

        neuron_weights_view_.refresh();
    }

    bitmap_image ConvolveLayer::GenerateImage() const
    {
        neuron_weights_view_.synchronize();
        neuron_activation_probs_view_.synchronize();
        vbias_view_.synchronize();
        hbias_view_.synchronize();

        for (int i = 0; i < neuron_activation_probs_.size(); i++)
        {
            cout << neuron_activation_probs_[i] << endl;
        }

        bitmap_image image;

        const int block_size = 2;

        float max_abs_weight = numeric_limits<float>::min();
        for (float weight : neuron_weights_)
        {
            max_abs_weight = fmax(max_abs_weight, abs(weight));
        }

        float max_abs_vbias = numeric_limits<float>::min();
        for (float vbias : vbias_)
        {
            max_abs_vbias = fmax(max_abs_vbias, abs(vbias));
        }

        float max_abs_hbias = numeric_limits<float>::min();
        for (float hbias : hbias_)
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
                    float value = neuron_view(depth_idx, 0, 0);
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
                            float value = neuron_view(depth_idx, height_idx, width_idx);

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
        array_view<const float, 3> bottom_value = bottom_data.first;
        array_view<const float, 3> bottom_expect = bottom_data.second;

        // writeonly
        const auto& top_data = top_layer[top_slot];
        array_view<float, 3> top_value = top_data.first;
        array_view<float, 3> top_expect = top_data.second;
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

                    max_value = fmaxf(max_value, value);
                    max_expect *= (1.0f - expect); // the probability that all nodes are 0
                }
            }
            max_expect = 1.0f - max_expect;// the probability that at least one node is 1.

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
        array_view<const float, 3> top_value = top_data.first;
        array_view<const float, 3> top_expect = top_data.second;

        // writeonly
        const auto& bottom_data = bottom_layer[bottom_slot];
        array_view<float, 3> bottom_value = bottom_data.first;
        array_view<float, 3> bottom_expect = bottom_data.second;
        bottom_value.discard_data();
        bottom_expect.discard_data();

        auto& rand_collection = bottom_layer.rand_collection_;

        parallel_for_each(bottom_value.extent, [=](index<3> idx) restrict(amp)
        {
            // when we have memory, the bottom_layer can activate according to its memory. 
            // But now we just use uniform activation.

            int height_idx = idx[1] / block_height;// truncate towards zero
            int width_idx = idx[2] / block_width;


            bottom_expect[idx] = 1.0f - powf(1.0f - top_expect(idx[0], height_idx, width_idx), -1.0f * block_width * block_height);
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
                conv_layer.longterm_memory_num() + conv_layer.neuron_num(),
                last_data_layer.height() - conv_layer.neuron_height() + 1,
                last_data_layer.width() - conv_layer.neuron_width() + 1,
                uniform_int_distribution<int>()(random_engine_));
            if (conv_layer.longterm_memory_num() > 0)
            {
                conv_layer.FitLongtermMemory(data_layers_.back());
            }
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

    void DeepModel::AddConvolveLayer(int neuron_num, int neuron_height, int neuron_width, int longterm_memory_num)
    {
        assert(!layer_stack_.empty() && layer_stack_.back().first == LayerType::kDataLayer);
        const auto& last_data_layer = data_layers_[layer_stack_.back().second];
        convolve_layers_.emplace_back(longterm_memory_num, last_data_layer.depth(),
            neuron_num, last_data_layer.depth() * (1 + last_data_layer.shortterm_memory_num()), neuron_height, neuron_width);
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

    void DeepModel::PassUp(const vector<float>& data)
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
            top_data_layer.Activate();

            assert(layer_stack_[layer_idx].first == LayerType::kConvolveLayer || layer_stack_[layer_idx].first == LayerType::kPoolingLayer);
            if (layer_stack_[layer_idx].first == LayerType::kConvolveLayer)
            {
                const auto& conv_layer = convolve_layers_[layer_stack_[layer_idx].second];
                conv_layer.PassUp(bottom_data_layer, DataSlot::kCurrent, top_data_layer, DataSlot::kCurrent);
                if (conv_layer.longterm_memory_num() > 0)
                {
                    conv_layer.PassDown(top_data_layer, DataSlot::kCurrent, bottom_data_layer, DataSlot::kTemp);
                    conv_layer.ActivateMemory(top_data_layer, DataSlot::kCurrent, bottom_data_layer, DataSlot::kCurrent, DataSlot::kTemp);
                }
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
            bottom_data_layer.Activate();

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

    float DeepModel::TrainLayer(const vector<float>& data, int layer_idx, float learning_rate,
        float dropout_prob, const int label, bool discriminative_training)
    {
        assert(layer_stack_[layer_idx].first == LayerType::kConvolveLayer);
        assert(layer_idx >= 1 && layer_stack_[layer_idx - 1].first == LayerType::kDataLayer);
        assert(layer_stack_[layer_idx + 1].first == LayerType::kDataLayer);

        auto& bottom_data_layer = data_layers_[layer_stack_[layer_idx - 1].second];
        auto& top_data_layer = data_layers_[layer_stack_[layer_idx + 1].second];

        auto& conv_layer = convolve_layers_[layer_stack_[layer_idx].second];

        // train with contrastive divergence (CD) algorithm to maximize likelihood on dataset
        bottom_data_layer.SetValue(data);
        top_data_layer.Activate(1.0f - dropout_prob);

        if (label == -1)
        {
            // purely generative training without label
            conv_layer.PassUp(bottom_data_layer, DataSlot::kCurrent, top_data_layer, DataSlot::kCurrent);
            if (conv_layer.longterm_memory_num() > 0)
            {
                conv_layer.PassDown(top_data_layer, DataSlot::kCurrent, bottom_data_layer, DataSlot::kTemp);
                conv_layer.ActivateMemory(top_data_layer, DataSlot::kCurrent, bottom_data_layer, DataSlot::kCurrent, DataSlot::kTemp);
            }
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

            if (conv_layer.longterm_memory_num() > 0)
            {
                conv_layer.PassDown(top_data_layer, DataSlot::kCurrent,
                    bottom_data_layer, DataSlot::kTemp, &output_layer, DataSlot::kTemp);

                float orgin_err = bottom_data_layer.ReconstructionError(DataSlot::kTemp);
                // TODO: support memory suppression based on both data and label.
                // currently only data is considered.
                conv_layer.ActivateMemory(top_data_layer, DataSlot::kCurrent, bottom_data_layer, DataSlot::kCurrent, DataSlot::kTemp);

                conv_layer.PassDown(top_data_layer, DataSlot::kCurrent,
                    bottom_data_layer, DataSlot::kTemp, &output_layer, DataSlot::kTemp);
                float new_err = bottom_data_layer.ReconstructionError(DataSlot::kTemp);
                cout << "orgin err = " << orgin_err << "\tnew err = " << new_err << "\tgain = " << orgin_err - new_err << endl;
            }

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

    int DeepModel::PredictLabel(const vector<float>& data, const int layer_idx, const float dropout_prob)
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

    float DeepModel::Evaluate(const vector<const vector<float>>& dataset, const vector<const int>& labels,
        int layer_idx, const float dropout_prob)
    {

        convolve_layers_.front().GenerateImage().save_image("model_dump\\layer_conv.bmp");

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