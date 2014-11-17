#include "DeepModel.h"

#include <cassert>
#include <random>
#include <amp_math.h>
#include <iostream>

#include "AmpUtility.h"

namespace deep_learning_lib
{
    using std::pair;
    using std::make_pair;
    using std::tuple;
    using std::string;
    using std::to_string;
    using std::vector;
    using std::default_random_engine;
    using std::normal_distribution;
    using std::uniform_int_distribution;
    using std::numeric_limits;
    using std::fmax;
    using std::abs;

    using concurrency::array_view;
    using concurrency::index;
    using concurrency::extent;
    using concurrency::parallel_for_each;

    using concurrency::precise_math::log;
    using concurrency::precise_math::exp;
    using concurrency::precise_math::pow;
    using concurrency::precise_math::fabs;

#pragma region data layer

    DataLayer::DataSlot::DataSlot(int depth, int height, int width)
        : values_view_(depth, height, width),
        expects_view_(values_view_.extent),
        raw_weights_view_(values_view_.extent)
    {
        fill(values_view_, 0.0);
        fill(expects_view_, 0.0);
        fill(raw_weights_view_, 0.0);
    }

    DataLayer::DataLayer(int shortterm_memory_num, int depth, int height, int width, int seed)
        : shortterm_memory_num_(shortterm_memory_num),
        cur_data_slot_(depth, height, width),
        next_data_slot_(depth, height, width),
        tmp_data_slot_(depth, height, width),
        // there is no empty array_view support in amp now, so we just set the extent to (1,1,1,1) when the shortterm_memory_num == 0
        shortterm_memories_view_(shortterm_memory_num == 0 ? make_extent(1, 1, 1, 1) : make_extent(shortterm_memory_num, depth, height, width)),
        shortterm_memory_index_view_(std::max(1, shortterm_memory_num)),
        rand_collection_(extent<3>(depth, height, width), seed)
    {
        fill(shortterm_memories_view_, 0.0);
        for (int time = 0; time < shortterm_memory_num; time++)
        {
            shortterm_memory_index_view_[time] = time;
        }
    }

    void DataLayer::SetValue(const vector<double>& data)
    {
        assert(data.size() == cur_data_slot_.values_view_.extent.size());

        // Copy the data
        copy(data.begin(), data.end(), cur_data_slot_.values_view_);
        cur_data_slot_.values_view_.copy_to(cur_data_slot_.expects_view_);
        fill(cur_data_slot_.raw_weights_view_, 0.0);
    }

    void DataLayer::Clear(DataSlotType slot_type)
    {
        const auto& data_slot = (*this)[slot_type];
        fill(data_slot.values_view_, 0.0);
        fill(data_slot.expects_view_, 0.0);
        fill(data_slot.raw_weights_view_, 0.0);
    }

    void DataLayer::Memorize()
    {
        if (shortterm_memory_num() <= 0)
        {
            return;
        }

        int last_memory_index = shortterm_memory_index_view_(shortterm_memory_num() - 1);

        copy(cur_data_slot_.values_view_, shortterm_memories_view_[last_memory_index]);

        // right shift the shortterm memory index
        for (int time = shortterm_memory_num() - 1; time > 0; time--)
        {
            shortterm_memory_index_view_(time) = shortterm_memory_index_view_(time - 1);
        }

        shortterm_memory_index_view_(0) = last_memory_index;
    }

    float DataLayer::ReconstructionError(DataSlotType slot_type) const
    {
        array_view<float> result(1);
        result(0) = 0.0f;

        array_view<const double, 3> values_view = cur_data_slot_.values_view_;
        array_view<const double, 3> recon_expects_view = (*this)[slot_type].expects_view_;

        // TODO: compare with reduce method for performance
        parallel_for_each(values_view.extent,
            [=](index<3> idx) restrict(amp)
        {
            auto diff = values_view[idx] - recon_expects_view[idx];
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
            image.setwidth_height(depth() * (block_size + 1), (6 + 2 + shortterm_memory_num()) * (block_size + 1), true);
            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 0, block_size, block_size,
                    static_cast<unsigned char>(255.0 * cur_data_slot_.expects_view_(i, 0, 0)));
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), block_size + 1, block_size, block_size,
                    cur_data_slot_.values_view_(i, 0, 0) == 0.0 ? 0 : 255);
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 2 * (block_size + 1), block_size, block_size,
                    static_cast<unsigned char>(255.0 * next_data_slot_.expects_view_(i, 0, 0)));
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 3 * (block_size + 1), block_size, block_size,
                    next_data_slot_.values_view_(i, 0, 0) == 0.0 ? 0 : 255);
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 4 * (block_size + 1), block_size, block_size,
                    static_cast<unsigned char>(255.0 * tmp_data_slot_.expects_view_(i, 0, 0)));
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 5 * (block_size + 1), block_size, block_size,
                    tmp_data_slot_.values_view_(i, 0, 0) == 0.0 ? 0 : 255);
            }

            for (int i = 0; i < shortterm_memory_num(); i++)
            {
                auto memory_slice_view = shortterm_memories_view_[shortterm_memory_index_view_[i]];
                for (int j = 0; j < depth(); j++)
                {
                    image.set_region(j * (block_size + 1), (8 + i) * (block_size + 1), block_size, block_size,
                        static_cast<unsigned char>(255.0 * memory_slice_view(j, 0, 0)));
                }
            }
        }
        else
        {
            image.setwidth_height(depth() * (width() + 1) * (block_size + 1),
                ((6 + shortterm_memory_num()) * (height() + 1) + 2) * (block_size + 1), true);
            for (int depth_idx = 0; depth_idx < depth(); depth_idx++)
            {
                for (int height_idx = 0; height_idx < height(); height_idx++)
                {
                    for (int width_idx = 0; width_idx < width(); width_idx++)
                    {
                        image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                            height_idx * (block_size + 1), block_size, block_size,
                            static_cast<unsigned char>(255.0 * cur_data_slot_.expects_view_(depth_idx, height_idx, width_idx)));
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
                            cur_data_slot_.values_view_(depth_idx, height_idx, width_idx) == 0.0 ? 0 : 255);
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
                            static_cast<unsigned char>(255.0 * next_data_slot_.expects_view_(depth_idx, height_idx, width_idx)));
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
                            next_data_slot_.values_view_(depth_idx, height_idx, width_idx) == 0.0 ? 0 : 255);
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
                            (4 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                            static_cast<unsigned char>(255.0 * tmp_data_slot_.expects_view_(depth_idx, height_idx, width_idx)));
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
                            (5 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                            tmp_data_slot_.values_view_(depth_idx, height_idx, width_idx) == 0.0 ? 0 : 255);
                    }
                }
            }

            for (int memory_idx = 0; memory_idx < shortterm_memory_num(); memory_idx++)
            {
                auto memory_slice_view = shortterm_memories_view_[shortterm_memory_index_view_[memory_idx]];
                for (int depth_idx = 0; depth_idx < depth(); depth_idx++)
                {
                    for (int height_idx = 0; height_idx < height(); height_idx++)
                    {
                        for (int width_idx = 0; width_idx < width(); width_idx++)
                        {
                            image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                                ((6 + memory_idx) * (height() + 1) + height_idx + 2) * (block_size + 1), block_size, block_size,
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

    OutputLayer::DataSlot::DataSlot(int output_num)
        : outputs_view_(output_num),
        raw_weights_view_(output_num)
    {
        fill(outputs_view_, 0.0);
        fill(raw_weights_view_, 0.0);
    }

    OutputLayer::OutputLayer(int output_num, int input_depth, int input_height, int input_width)
        : cur_data_slot_(output_num),
        next_data_slot_(output_num),
        tmp_data_slot_(output_num),
        bias_(output_num),
        bias_view_(output_num, bias_),
        neuron_weights_(output_num * input_depth * input_height * input_width),
        neuron_weights_view_(make_extent(output_num, input_depth, input_height, input_width), neuron_weights_)
    {
    }

    OutputLayer::OutputLayer(OutputLayer&& other)
        : cur_data_slot_(other.cur_data_slot_),
        next_data_slot_(other.next_data_slot_),
        tmp_data_slot_(other.tmp_data_slot_),
        bias_(move(other.bias_)),
        bias_view_(other.bias_view_),
        neuron_weights_(move(other.neuron_weights_)),
        neuron_weights_view_(other.neuron_weights_view_)
    {
    }

    void OutputLayer::SetLabel(const int label)
    {
        assert(label >= 0 && label < this->output_num());
        fill(cur_data_slot_.outputs_view_, 0.0);
        fill(cur_data_slot_.raw_weights_view_, 0.0);
        cur_data_slot_.outputs_view_[label] = 1.0;
    }

    void OutputLayer::PassDown(const DataLayer& top_layer, DataSlotType top_slot_type, DataSlotType output_slot_type)
    {
        assert(top_layer.depth() == this->input_depth());
        assert(top_layer.width() == this->input_width());
        assert(top_layer.height() == this->input_height());

        const int top_depth = top_layer.depth();
        const int top_width = top_layer.width();
        const int top_height = top_layer.height();

        array_view<const double, 3> top_expects = top_layer[top_slot_type].expects_view_;

        array_view<const double> output_bias = this->bias_view_;
        array_view<const double, 4> output_neuron_weights = this->neuron_weights_view_;

        const auto& output_slot = (*this)[output_slot_type];
        auto output_value = output_slot.outputs_view_;
        auto output_raw_weights = output_slot.raw_weights_view_;
        output_value.discard_data();
        output_raw_weights.discard_data();

        // non-tiled version
        parallel_for_each(output_value.extent,
            [=](index<1> idx) restrict(amp)
        {
            auto raw_weight = output_bias[idx];

            const auto& cur_output_neuron = output_neuron_weights[idx[0]];

            for (int depth_idx = 0; depth_idx < top_depth; depth_idx++)
            {
                for (int height_idx = 0; height_idx < top_height; height_idx++)
                {
                    for (int width_idx = 0; width_idx < top_width; width_idx++)
                    {
                        raw_weight += top_expects(depth_idx, height_idx, width_idx)
                            * cur_output_neuron(depth_idx, height_idx, width_idx);
                    }
                }
            }

            output_raw_weights[idx] = raw_weight;
            output_value[idx] = 1.0 / (1.0 + exp(-raw_weight));
        });
    }

    void OutputLayer::Train(const DataLayer& top_layer, double learning_rate)
    {
        array_view<const double, 3> top_expects = top_layer.cur_data_slot_.expects_view_;

        array_view<const double> outputs = this->cur_data_slot_.outputs_view_;
        array_view<const double> next_outputs = this->next_data_slot_.outputs_view_;

        // parameters to train
        array_view<double, 4> output_neuron_weights = this->neuron_weights_view_;
        array_view<double> output_bias = this->bias_view_;

        parallel_for_each(output_neuron_weights.extent, [=](index<4> idx) restrict(amp)
        {
            int output_idx = idx[0];
            int top_depth_idx = idx[1];
            int top_height_idx = idx[2];
            int top_width_idx = idx[3];

            auto top_expect = top_expects(top_depth_idx, top_height_idx, top_width_idx);

            auto delta = top_expect * (outputs(output_idx) - next_outputs(output_idx));

            output_neuron_weights[idx] += delta * learning_rate;
        });

        parallel_for_each(output_bias.extent, [=](index<1> idx) restrict(amp)
        {
            auto delta = outputs[idx] - next_outputs[idx];

            output_bias[idx] += delta * learning_rate;
        });
    }

    void OutputLayer::RandomizeParams(unsigned int seed)
    {
        default_random_engine generator(seed);
        normal_distribution<double> distribution(0.0, 0.1);

        for (auto& w : neuron_weights_)
        {
            w = distribution(generator);
        }

        neuron_weights_view_.refresh();
    }

    int OutputLayer::PredictLabel(DataLayer& bottom_layer, DataLayer& top_layer, const ConvolveLayer& conv_layer)
    {
        assert(top_layer.depth() == conv_layer.neuron_num() && top_layer.depth() == this->input_depth());
        assert(top_layer.width() == bottom_layer.width() - conv_layer.neuron_width() + 1 && top_layer.width() == this->input_width());
        assert(top_layer.height() == bottom_layer.height() - conv_layer.neuron_height() + 1 && top_layer.height() == this->input_height());

        conv_layer.InferUp(bottom_layer, DataSlotType::kCurrent, top_layer, DataSlotType::kCurrent);

        conv_layer.PassDown(top_layer, DataSlotType::kCurrent, bottom_layer, DataSlotType::kNext);
        std::cout << "Test : " << bottom_layer.ReconstructionError(DataSlotType::kNext) << std::endl;

        this->PassDown(top_layer, DataSlotType::kCurrent, DataSlotType::kCurrent);
        array_view<const double> outputs = cur_data_slot_.outputs_view_;

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

    bitmap_image OutputLayer::GenerateImage() const
    {
        neuron_weights_view_.synchronize();
        bias_view_.synchronize();

        bitmap_image image;

        const int block_size = 2;

        auto max_abs_weight = numeric_limits<double>::min();
        for (auto weight : neuron_weights_)
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
                    auto value = neuron_weights_[output_idx * input_depth() + depth_idx];
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
                auto cur_weights_view = neuron_weights_view_[output_idx];

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
        neuron_weights_view_(make_extent(neuron_num, neuron_depth, neuron_height, neuron_width), neuron_weights_),
        vbias_(neuron_depth),
        vbias_view_(neuron_depth, vbias_),
        hbias_(neuron_num),
        hbias_view_(neuron_num, hbias_)
    {
    }

    ConvolveLayer::ConvolveLayer(ConvolveLayer&& other)
        : neuron_weights_(move(other.neuron_weights_)),
        neuron_weights_view_(other.neuron_weights_view_),
        vbias_(move(other.vbias_)),
        vbias_view_(other.vbias_view_),
        hbias_(move(other.hbias_)),
        hbias_view_(other.hbias_view_)
    {
    }

    void ConvolveLayer::InitContext(const DataLayer& bottom_layer, DataSlotType bottom_slot_type,
        DataLayer& top_layer, DataSlotType top_slot_type) const
    {
        // neuron layer
        const int neuron_height = this->neuron_height();
        const int neuron_width = this->neuron_width();

        array_view<const double, 4> conv_neuron_weights = this->neuron_weights_view_;
        array_view<const double> conv_hbias = this->hbias_view_;

        // top layer
        const auto& top_slot = top_layer[top_slot_type];
        array_view<double, 3> top_values = top_slot.values_view_;
        array_view<double, 3> top_expects = top_slot.expects_view_;
        array_view<double, 3> top_raw_weights = top_slot.raw_weights_view_;// don't discard_data

        top_values.discard_data();
        top_expects.discard_data();

        // bottom layer
        const int bottom_depth = bottom_layer.depth();
        const int shortterm_memory_num = bottom_layer.shortterm_memory_num();

        array_view<const double, 4> bottom_shortterm_memories = bottom_layer.shortterm_memories_view_;
        array_view<const int> bottom_shortterm_memory_index = bottom_layer.shortterm_memory_index_view_;

        auto& rand_collection = top_layer.rand_collection_;

        // Optimization for discriminative input, e.g. short-term memory and hidden layer bias
        // Discriminative inputs only affect the initial value of top_layer weights.
        // Given enough iterations, they do not affect the final value of top_layer weights.
        // so they serve as the accelerator of thinking process. Amazing!
        parallel_for_each(top_values.extent,
            [=](index<3> idx) restrict(amp)
        {
            int top_depth_idx = idx[0];
            int top_height_idx = idx[1];
            int top_width_idx = idx[2];

            // Note that, the value of raw_weight is not set to zero here.
            // So you can assign custom prior weights from other sources, e.g. upper layers
            auto& raw_weight = top_raw_weights[idx];

            raw_weight += conv_hbias[top_depth_idx];

            if (shortterm_memory_num > 0)
            {
                const auto& current_neuron = conv_neuron_weights[top_depth_idx];

                // convolve short-term memory in bottom layer if exists.
                for (int memory_idx = 0; memory_idx < shortterm_memory_num; memory_idx++)
                {
                    const auto& cur_bottom_memory = bottom_shortterm_memories[bottom_shortterm_memory_index[memory_idx]];

                    for (int depth_idx = 0; depth_idx < bottom_depth; depth_idx++)
                    {
                        for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                        {
                            for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                            {
                                raw_weight += cur_bottom_memory(depth_idx, top_height_idx + height_idx, top_width_idx + width_idx)
                                    * current_neuron(bottom_depth * (1 + memory_idx) + depth_idx, height_idx, width_idx);
                            }
                        }
                    }
                }
            }

            auto expect = 1.0 / (1.0 + exp(-raw_weight));
            top_expects[idx] = expect;
            top_values[idx] = rand_collection[idx].next_single() <= expect ? 1.0 : 0.0;
        });
    }

    void ConvolveLayer::PassUp(DataLayer& bottom_layer, DataSlotType bottom_data_slot_type, DataSlotType bottom_model_slot_type,
        DataLayer& top_layer, DataSlotType top_slot_type) const
    {
        // neuron layer
        const int neuron_height = this->neuron_height();
        const int neuron_width = this->neuron_width();

        array_view<const double, 4> conv_neuron_weights = this->neuron_weights_view_;

        // top layer
        const auto& top_slot = top_layer[top_slot_type];
        array_view<double, 3> top_values = top_slot.values_view_;
        array_view<double, 3> top_expects = top_slot.expects_view_;
        array_view<double, 3> top_raw_weights = top_slot.raw_weights_view_;// don't discard_data

        top_values.discard_data();
        top_expects.discard_data();

        // bottom layer
        const int bottom_depth = bottom_layer.depth();

        array_view<const double, 3> bottom_data_values = bottom_layer[bottom_data_slot_type].values_view_;
        array_view<const double, 3> bottom_model_expects = bottom_layer[bottom_model_slot_type].expects_view_;

        auto& rand_collection = top_layer.rand_collection_;

        // pass up the DIFFERENCE between model and data
        // non-tiled version
        parallel_for_each(top_values.extent,
            [=](index<3> idx) restrict(amp)
        {
            int top_depth_idx = idx[0];
            int top_height_idx = idx[1];
            int top_width_idx = idx[2];

            auto& raw_weight = top_raw_weights[idx];
            auto weight_delta = 0.0;

            const auto& current_neuron = conv_neuron_weights[top_depth_idx];

            for (int depth_idx = 0; depth_idx < bottom_depth; depth_idx++)
            {
                for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                {
                    for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                    {
                        auto data_value = bottom_data_values(depth_idx, top_height_idx + height_idx, top_width_idx + width_idx);
                        auto model_value = bottom_model_expects(depth_idx, top_height_idx + height_idx, top_width_idx + width_idx);
                        auto weight = current_neuron(depth_idx, height_idx, width_idx);
                        weight_delta += (data_value - model_value) * weight;
                    }
                }
            }

            //raw_weight = raw_weight * 0.9 + weight_delta;
            raw_weight += weight_delta;

            auto expect = 1.0 / (1.0 + exp(-raw_weight));
            top_expects[idx] = expect;
            top_values[idx] = rand_collection[idx].next_single() <= expect ? 1.0 : 0.0;
        });
    }

    void ConvolveLayer::InferUp(DataLayer& bottom_layer, DataSlotType bottom_slot_type,
        DataLayer& top_layer, DataSlotType top_slot_type) const
    {
        assert(top_layer.depth() == this->neuron_num());
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1);
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1);
        assert(this->neuron_depth() == (bottom_layer.shortterm_memory_num() + 1) * bottom_layer.depth());

        InitContext(bottom_layer, bottom_slot_type, top_layer, top_slot_type);

        // this two-stage pass-up process seeks the optimal balance between PoE and MoE.
        // i.e. minimum number of bits used to store the information.
        for (int iter = 0; iter < kInferIteration; iter++)
        {
            // first evaluate how well the bottom layer is modeled by now.
            PassDown(top_layer, top_slot_type, bottom_layer, DataSlotType::kTemp);

            PassUp(bottom_layer, bottom_slot_type, DataSlotType::kTemp, top_layer, top_slot_type);

            //bottom_layer.GenerateImage().save_image("model_dump\\debug_bottom_data.bmp");
        }
    }

    void ConvolveLayer::PassDown(const DataLayer& top_layer, DataSlotType top_slot_type,
        DataLayer& bottom_layer, DataSlotType bottom_slot_type) const
    {
        assert(top_layer.depth() == this->neuron_num());
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1);
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1);

        // neuron layer
        const int neuron_num = this->neuron_num();
        const int neuron_height = this->neuron_height();
        const int neuron_width = this->neuron_width();

        array_view<const double, 4> conv_neuron_weights = this->neuron_weights_view_;
        array_view<const double> conv_vbias = this->vbias_view_;

        // top layer
        //array_view<const double, 3> top_values = top_layer[top_slot_type].values_view_;
        array_view<const double, 3> top_values = top_layer[top_slot_type].expects_view_;

        // bottom layer
        const int bottom_height = bottom_layer.height();
        const int bottom_width = bottom_layer.width();

        const auto& bottom_slot = bottom_layer[bottom_slot_type];
        auto bottom_values = bottom_slot.values_view_;
        auto bottom_expects = bottom_slot.expects_view_;
        auto bottom_raw_weights = bottom_slot.raw_weights_view_;
        bottom_values.discard_data();
        bottom_expects.discard_data();
        bottom_raw_weights.discard_data();


        auto& rand_collection = bottom_layer.rand_collection_;

        // non-tiled version
        // PassDown will not touch bottom short-term memory for simplicity
        // so here only update bottom_value
        parallel_for_each(bottom_values.extent,
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

            auto raw_weight = conv_vbias[cur_depth_idx];

            for (int neuron_idx = 0; neuron_idx < neuron_num; neuron_idx++)
            {
                const auto& current_neuron = conv_neuron_weights[neuron_idx];

                for (int height_idx = height_idx_min; height_idx <= height_idx_max; height_idx++)
                {
                    int top_height_idx = cur_height_idx - height_idx;
                    for (int width_idx = width_idx_min; width_idx <= width_idx_max; width_idx++)
                    {
                        int top_width_idx = cur_width_idx - width_idx;
                        raw_weight += current_neuron(cur_depth_idx, height_idx, width_idx) *
                            top_values(neuron_idx, top_height_idx, top_width_idx);
                    }
                }
            }

            // Logistic activation function. Maybe more types of activation function later.
            bottom_raw_weights[idx] = raw_weight;
            auto prob = 1.0 / (1.0 + exp(-raw_weight));

            bottom_expects[idx] = prob;
            bottom_values[idx] = rand_collection[idx].next_single() <= prob ? 1.0 : 0.0;
        });
    }

    void ConvolveLayer::Train(DataLayer& bottom_layer, DataLayer& top_layer, double learning_rate)
    {
        assert(bottom_layer.depth() * (1 + bottom_layer.shortterm_memory_num()) == this->neuron_depth());

        // top layer
        const int top_height = top_layer.height();
        const int top_width = top_layer.width();
        array_view<const double, 3> top_expects = top_layer.cur_data_slot_.expects_view_;
        array_view<const double, 3> top_next_expects = top_layer.next_data_slot_.expects_view_;
        array_view<const double, 3> top_tmp_expects = top_layer.tmp_data_slot_.expects_view_;

        // bottom layer
        const int bottom_depth = bottom_layer.depth();
        const int bottom_height = bottom_layer.height();
        const int bottom_width = bottom_layer.width();
        const int shortterm_memory_num = bottom_layer.shortterm_memory_num();

        array_view<const double, 3> bottom_expects = bottom_layer.cur_data_slot_.expects_view_;
        array_view<const double, 3> bottom_next_expects = bottom_layer.next_data_slot_.expects_view_;
        array_view<const double, 3> bottom_tmp_expects = bottom_layer.tmp_data_slot_.expects_view_;
        array_view<const double, 4> bottom_shortterm_memories = bottom_layer.shortterm_memories_view_;
        array_view<const int> bottom_shortterm_memory_index = bottom_layer.shortterm_memory_index_view_;

        // neuron layer
        // parameters to train
        array_view<double, 4> conv_neuron_weights = this->neuron_weights_view_;

        array_view<double> conv_vbias = this->vbias_view_;
        array_view<double> conv_hbias = this->hbias_view_;

        InitContext(bottom_layer, DataSlotType::kCurrent, top_layer, DataSlotType::kTemp);

        /*auto top_values_tmp = CopyToVector(top_layer.cur_data_slot_.values_view_);
        auto top_expects_tmp = CopyToVector(top_layer.cur_data_slot_.expects_view_);
        auto top_weights_tmp = CopyToVector(top_layer.cur_data_slot_.raw_weights_view_);

        auto top_values_tmp2 = CopyToVector(top_layer.tmp_data_slot_.values_view_);
        auto top_expects_tmp2 = CopyToVector(top_layer.tmp_data_slot_.expects_view_);
        auto top_weights_tmp2 = CopyToVector(top_layer.tmp_data_slot_.raw_weights_view_);

        auto conv_hbias_tmp = CopyToVector(this->hbias_view_);
        auto conv_vbias_tmp = CopyToVector(this->vbias_view_);*/

        for (int iter = 0; iter < kInferIteration; iter++)
        {
            PassDown(top_layer, DataSlotType::kTemp, bottom_layer, DataSlotType::kTemp);

            /*auto bottom_values_tmp = CopyToVector(bottom_layer.tmp_data_slot_.values_view_);
            auto bottom_expects_tmp = CopyToVector(bottom_layer.tmp_data_slot_.expects_view_);
            auto bottom_weights_tmp = CopyToVector(bottom_layer.tmp_data_slot_.raw_weights_view_);*/

            top_layer.tmp_data_slot_.raw_weights_view_.copy_to(top_layer.cur_data_slot_.raw_weights_view_);
            PassUp(bottom_layer, DataSlotType::kCurrent, DataSlotType::kTemp, top_layer, DataSlotType::kCurrent);

           /* auto top_values_tmp3 = CopyToVector(top_layer.cur_data_slot_.values_view_);
            auto top_expects_tmp3 = CopyToVector(top_layer.cur_data_slot_.expects_view_);
            auto top_weights_tmp3 = CopyToVector(top_layer.cur_data_slot_.raw_weights_view_);*/

            PassDown(top_layer, DataSlotType::kCurrent, bottom_layer, DataSlotType::kNext);

            /*auto bottom_values_tmp2 = CopyToVector(bottom_layer.next_data_slot_.values_view_);
            auto bottom_expects_tmp2 = CopyToVector(bottom_layer.next_data_slot_.expects_view_);
            auto bottom_weights_tmp2 = CopyToVector(bottom_layer.next_data_slot_.raw_weights_view_);*/

            top_layer.tmp_data_slot_.raw_weights_view_.copy_to(top_layer.next_data_slot_.raw_weights_view_);
            PassUp(bottom_layer, DataSlotType::kNext, DataSlotType::kTemp, top_layer, DataSlotType::kNext);

            /*auto top_values_tmp4 = CopyToVector(top_layer.next_data_slot_.values_view_);
            auto top_expects_tmp4 = CopyToVector(top_layer.next_data_slot_.expects_view_);
            auto top_weights_tmp4 = CopyToVector(top_layer.next_data_slot_.raw_weights_view_);

            bottom_layer.GenerateImage().save_image("model_dump\\debug_bottom_data.bmp");*/

            // non-tiled version
            parallel_for_each(make_extent(neuron_num(), bottom_depth, neuron_height(), neuron_width()), [=](index<4> idx) restrict(amp)
            {
                auto delta = 0.0;

                int neuron_idx = idx[0];
                int bottom_depth_idx = idx[1];
                int neuron_height_idx = idx[2];
                int neuron_width_idx = idx[3];

                for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
                {
                    for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
                    {
                        index<3> top_idx(neuron_idx, top_height_idx, top_width_idx);
                        index<3> bottom_idx(bottom_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx);

                        auto top_expect = top_expects[top_idx];
                        auto top_next_expect = top_next_expects[top_idx];
                        auto top_tmp_expect = top_tmp_expects[top_idx];

                        auto bottom_expect = bottom_expects[bottom_idx];
                        auto bottom_next_expect = bottom_next_expects[bottom_idx];
                        auto bottom_tmp_expect = bottom_tmp_expects[bottom_idx];

                        delta += (bottom_expect - bottom_tmp_expect) * (top_expect - top_tmp_expect) -
                            (bottom_next_expect - bottom_tmp_expect) * (top_next_expect - top_tmp_expect);
                    }
                }

                conv_neuron_weights[idx] += delta / (top_height * top_width) * learning_rate;
            });

            if (shortterm_memory_num > 0)
            {
                // short-term memory is treated as discriminative input like hbias
                parallel_for_each(make_extent(neuron_num(), bottom_depth * shortterm_memory_num, neuron_height(), neuron_width()),
                    [=](index<4> idx) restrict(amp)
                {
                    auto delta = 0.0;

                    int neuron_idx = idx[0];
                    int neuron_depth_idx = idx[1];
                    int neuron_height_idx = idx[2];
                    int neuron_width_idx = idx[3];

                    int bottom_depth_idx = neuron_depth_idx % bottom_depth;
                    int bottom_memory_idx = (neuron_depth_idx - bottom_depth_idx) / bottom_depth;

                    const auto& bottom_memory_values = bottom_shortterm_memories[bottom_shortterm_memory_index[bottom_memory_idx]];

                    for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
                    {
                        for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
                        {
                            index<3> top_idx(neuron_idx, top_height_idx, top_width_idx);
                            index<3> bottom_idx(bottom_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx);

                            auto top_expect = top_expects[top_idx];
                            auto top_next_expect = top_next_expects[top_idx];

                            auto bottom_value = bottom_memory_values[bottom_idx];

                            delta += bottom_value * (top_expect - top_next_expect);
                        }
                    }

                    conv_neuron_weights[idx] += delta / (top_height * top_width) * learning_rate;
                });
            }

            // update vbias, only for generative training and only for bottom value
            // vbias does not cover shortterm memory part
            parallel_for_each(extent<1>(bottom_depth), [=](index<1> idx) restrict(amp)
            {
                auto delta = 0.0;

                int depth_idx = idx[0];

                for (int bottom_height_idx = 0; bottom_height_idx < bottom_height; bottom_height_idx++)
                {
                    for (int bottom_width_idx = 0; bottom_width_idx < bottom_width; bottom_width_idx++)
                    {
                        index<3> bottom_idx(depth_idx, bottom_height_idx, bottom_width_idx);
                        auto bottom_expect = bottom_expects[bottom_idx];
                        auto bottom_next_expect = bottom_next_expects[bottom_idx];

                        delta += bottom_expect - bottom_next_expect;
                    }
                }

                conv_vbias[idx] += delta / (bottom_height * bottom_width) * learning_rate;
            });

            // update hbias
            parallel_for_each(conv_hbias.extent, [=](index<1> idx) restrict(amp)
            {
                auto delta = 0.0;

                int neuron_idx = idx[0];

                for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
                {
                    for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
                    {
                        index<3> top_idx(neuron_idx, top_height_idx, top_width_idx);
                        auto top_expect = top_expects[top_idx];
                        auto top_next_expect = top_next_expects[top_idx];

                        delta += top_expect - top_next_expect;
                    }
                }

                conv_hbias[idx] += delta / (top_height * top_width) * learning_rate;
            });

            PassUp(bottom_layer, DataSlotType::kCurrent, DataSlotType::kTemp, top_layer, DataSlotType::kTemp);
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
        vbias_view_.synchronize();
        hbias_view_.synchronize();

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
            image.setwidth_height((3 + neuron_depth()) * (block_size + 1), (2 + neuron_num()) * (block_size + 1), true);

            for (int i = 0; i < vbias_.size(); i++)
            {
                image.set_region((3 + i) * (block_size + 1), 0, block_size, block_size,
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
                    image.set_region((3 + depth_idx) * (block_size + 1), (2 + neuron_idx) * (block_size + 1), block_size, block_size,
                        value >= 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                        static_cast<unsigned char>(abs(value) / max_abs_weight * 255.0));
                }
            }
        }
        else
        {
            image.setwidth_height((3 + neuron_depth() * (neuron_width() + 1)) * (block_size + 1),
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

                            image.set_region((3 + width_idx + depth_idx * (neuron_width() + 1)) * (block_size + 1),
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

    void PoolingLayer::PassUp(const DataLayer& bottom_layer, DataSlotType bottom_slot_type,
        DataLayer& top_layer, DataSlotType top_slot_type) const
    {
        assert(top_layer.height() * block_height_ == bottom_layer.height());
        assert(top_layer.width() * block_width_ == bottom_layer.width());

        // readonly
        int block_height = block_height_;
        int block_width = block_width_;

        const auto& bottom_slot = bottom_layer[bottom_slot_type];
        array_view<const double, 3> bottom_values = bottom_slot.values_view_;
        array_view<const double, 3> bottom_expects = bottom_slot.expects_view_;

        // writeonly
        const auto& top_slot = top_layer[top_slot_type];
        array_view<double, 3> top_values = top_slot.values_view_;
        array_view<double, 3> top_expects = top_slot.expects_view_;
        top_values.discard_data();
        top_expects.discard_data();

        parallel_for_each(top_values.extent, [=](index<3> idx) restrict(amp)
        {
            auto max_value = 0.0;
            auto max_expect = 1.0;

            for (int height_idx = 0; height_idx < block_height; height_idx++)
            {
                for (int width_idx = 0; width_idx < block_width; width_idx++)
                {
                    auto value = bottom_values(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx);
                    auto expect = bottom_expects(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx);

                    max_value = concurrency::precise_math::fmax(max_value, value);
                    max_expect *= (1.0 - expect); // the probability that all nodes are 0
                }
            }
            max_expect = 1.0 - max_expect;// the probability that at least one node is 1.

            top_values[idx] = max_value;
            top_expects[idx] = max_expect;
        });
    }

    void PoolingLayer::PassDown(const DataLayer& top_layer, DataSlotType top_slot_type,
        DataLayer& bottom_layer, DataSlotType bottom_slot_type) const
    {
        assert(top_layer.height() * block_height_ == bottom_layer.height());
        assert(top_layer.width() * block_width_ == bottom_layer.width());

        // readonly
        int block_height = block_height_;
        int block_width = block_width_;

        const auto& top_slot = top_layer[top_slot_type];
        array_view<const double, 3> top_values = top_slot.values_view_;
        array_view<const double, 3> top_expects = top_slot.expects_view_;

        // writeonly
        const auto& bottom_slot = bottom_layer[bottom_slot_type];
        array_view<double, 3> bottom_values = bottom_slot.values_view_;
        array_view<double, 3> bottom_expects = bottom_slot.expects_view_;
        bottom_values.discard_data();
        bottom_expects.discard_data();

        auto& rand_collection = bottom_layer.rand_collection_;

        parallel_for_each(bottom_values.extent, [=](index<3> idx) restrict(amp)
        {
            // when we have memory, the bottom_layer can activate according to its memory. 
            // But now we just use uniform activation.

            int height_idx = idx[1] / block_height;// truncate towards zero
            int width_idx = idx[2] / block_width;


            bottom_expects[idx] = 1.0 - pow(1.0 - top_expects(idx[0], height_idx, width_idx), -1.0 * block_width * block_height);
            bottom_values[idx] = 0.0;// clear the value
        });

        parallel_for_each(top_values.extent, [=](index<3> idx) restrict(amp)
        {
            if (top_values[idx] == 1.0)
            {
                // randomly select a node in bottom_layer to activate
                int height_idx = rand_collection[idx].next_uint() % block_height;
                int width_idx = rand_collection[idx].next_uint() % block_width;

                bottom_values(idx[0], idx[1] * block_height + height_idx, idx[2] * block_width + width_idx) = 1.0;
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
        convolve_layers_.emplace_back(neuron_num,
            last_data_layer.depth() * (1 + last_data_layer.shortterm_memory_num()),
            neuron_height, neuron_width);
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
            output_layers_.emplace(std::piecewise_construct, std::forward_as_tuple(last_data_layer_idx),
                std::forward_as_tuple(output_num, last_data_layer.depth(), last_data_layer.height(), last_data_layer.width()));

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

            top_data_layer.Clear(DataSlotType::kCurrent);

            assert(layer_stack_[layer_idx].first == LayerType::kConvolveLayer || layer_stack_[layer_idx].first == LayerType::kPoolingLayer);
            if (layer_stack_[layer_idx].first == LayerType::kConvolveLayer)
            {
                const auto& conv_layer = convolve_layers_[layer_stack_[layer_idx].second];
                conv_layer.InferUp(bottom_data_layer, DataSlotType::kCurrent, top_data_layer, DataSlotType::kCurrent);
            }
            else
            {
                const auto& pooling_layer = pooling_layers[layer_stack_[layer_idx].second];
                pooling_layer.PassUp(bottom_data_layer, DataSlotType::kCurrent, top_data_layer, DataSlotType::kCurrent);
            }
        }
    }

    void DeepModel::PassDown()
    {
        assert(!layer_stack_.empty() && layer_stack_.back().first == LayerType::kDataLayer);

        // prepare top layer for passing down
        auto& roof_data_layer = data_layers_[layer_stack_.back().second];
        roof_data_layer.cur_data_slot_.values_view_.copy_to(roof_data_layer.next_data_slot_.values_view_);
        roof_data_layer.cur_data_slot_.expects_view_.copy_to(roof_data_layer.next_data_slot_.expects_view_);

        for (int layer_idx = static_cast<int>(convolve_layers_.size()) - 2; layer_idx >= 1; layer_idx -= 2)
        {
            assert(layer_stack_[layer_idx - 1].first == LayerType::kDataLayer);
            assert(layer_stack_[layer_idx + 1].first == LayerType::kDataLayer);
            auto& bottom_data_layer = data_layers_[layer_stack_[layer_idx - 1].second];
            auto& top_data_layer = data_layers_[layer_stack_[layer_idx + 1].second];

            assert(layer_stack_[layer_idx].first == LayerType::kConvolveLayer || layer_stack_[layer_idx].first == LayerType::kPoolingLayer);
            if (layer_stack_[layer_idx].first == LayerType::kConvolveLayer)
            {
                const auto& conv_layer = convolve_layers_[layer_stack_[layer_idx].second];
                conv_layer.PassDown(top_data_layer, DataSlotType::kNext, bottom_data_layer, DataSlotType::kNext);
            }
            else
            {
                const auto& pooling_layer = pooling_layers[layer_stack_[layer_idx].second];
                pooling_layer.PassDown(top_data_layer, DataSlotType::kNext, bottom_data_layer, DataSlotType::kNext);
            }
        }
    }

    double DeepModel::TrainLayer(const vector<double>& data, int layer_idx, double learning_rate, const int label)
    {
        assert(layer_stack_[layer_idx].first == LayerType::kConvolveLayer);
        assert(layer_idx >= 1 && layer_stack_[layer_idx - 1].first == LayerType::kDataLayer);
        assert(layer_stack_[layer_idx + 1].first == LayerType::kDataLayer);

        auto& bottom_data_layer = data_layers_[layer_stack_[layer_idx - 1].second];
        auto& top_data_layer = data_layers_[layer_stack_[layer_idx + 1].second];

        auto& conv_layer = convolve_layers_[layer_stack_[layer_idx].second];

        // train with contrastive divergence (CD) algorithm to maximize likelihood on dataset
        top_data_layer.Clear(DataSlotType::kCurrent);
        top_data_layer.Clear(DataSlotType::kTemp);
        bottom_data_layer.SetValue(data);

        conv_layer.Train(bottom_data_layer, top_data_layer, learning_rate);

        if (label >= 0)
        {
            // training data has label
            auto& output_layer = output_layers_.at(layer_stack_[layer_idx + 1].second);
            output_layer.SetLabel(label);

            output_layer.PassDown(top_data_layer, DataSlotType::kCurrent, DataSlotType::kNext);
            output_layer.Train(top_data_layer, learning_rate);
        }

        // update short-term memory
        bottom_data_layer.Memorize();

        return bottom_data_layer.ReconstructionError(DataSlotType::kNext);
    }

    int DeepModel::PredictLabel(const vector<double>& data, const int layer_idx)
    {
        assert(layer_idx >= 0 && layer_idx + 2 < layer_stack_.size()
            && layer_stack_[layer_idx].first == LayerType::kDataLayer
            && layer_stack_[layer_idx + 1].first == LayerType::kConvolveLayer
            && layer_stack_[layer_idx + 2].first == LayerType::kDataLayer);

        auto& bottom_data_layer = data_layers_[layer_stack_[layer_idx].second];
        const auto& conv_layer = convolve_layers_[layer_stack_[layer_idx + 1].second];
        auto& top_data_layer = data_layers_[layer_stack_[layer_idx + 2].second];

        auto& output_layer = output_layers_.at(layer_stack_[layer_idx + 2].second);

        top_data_layer.Clear(DataSlotType::kCurrent);
        bottom_data_layer.SetValue(data);
        // top layer activation is ignored when predicting labels
        return output_layer.PredictLabel(bottom_data_layer, top_data_layer, conv_layer);
    }

    pair<double, vector<tuple<int, int, int>>> DeepModel::Evaluate(const vector<const vector<double>>& dataset, const vector<const int>& labels, int layer_idx)
    {
        assert(dataset.size() == labels.size());

        auto correct_count = 0.0;

        vector<tuple<int, int, int>> wrong_cases;

        for (int i = 0; i < dataset.size(); i++)
        {
            int predicted_label = PredictLabel(dataset[i], layer_idx);
            if (predicted_label == labels[i])
            {
                correct_count++;
            }
            else
            {
                wrong_cases.emplace_back(i, predicted_label, labels[i]);

                //auto& bottom_data_layer = data_layers_[layer_stack_[layer_idx].second];
                //bottom_data_layer.GenerateImage().save_image("model_dump\\debug_bottom_data.bmp");
            }
        }

        return make_pair(correct_count / labels.size(), move(wrong_cases));
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