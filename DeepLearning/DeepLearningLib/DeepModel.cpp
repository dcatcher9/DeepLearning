#include "DeepModel.h"

#include <cassert>
#include <random>
#include <amp_math.h>
#include <iostream>
#include <iomanip>

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
    using std::ofstream;
    using std::cout;
    using std::endl;

    using concurrency::array_view;
    using concurrency::index;
    using concurrency::extent;
    using concurrency::parallel_for_each;

    using concurrency::precise_math::log;
    using concurrency::precise_math::exp;
    using concurrency::precise_math::pow;
    using concurrency::precise_math::fabs;
    using concurrency::precise_math::fmin;

#pragma region data layer

    DataLayer::DataSlot::DataSlot(int depth, int height, int width)
        : values_view_(depth, height, width),
        expects_view_(values_view_.extent),
        raw_weights_view_(values_view_.extent),
        delta_view_(values_view_.extent)
    {
        fill(values_view_, 0.0);
        fill(expects_view_, 0.0);
        fill(raw_weights_view_, 0.0);
        fill(delta_view_, 0.0);
    }

    void DataLayer::DataSlot::CopyTo(DataLayer::DataSlot& other) const
    {
        values_view_.copy_to(other.values_view_);
        expects_view_.copy_to(other.expects_view_);
        raw_weights_view_.copy_to(other.raw_weights_view_);
        delta_view_.copy_to(other.delta_view_);
    }

    void DataLayer::DataSlot::Dump(ofstream& ofs) const
    {
        ofs << std::defaultfloat;

        ofs << "[value]" << endl;
        if (values_view_.extent[1] == 1 && values_view_.extent[2] == 1)
        {
            for (int i = 0; i < values_view_.extent[0]; i++)
            {
                ofs << values_view_(i, 0, 0) << "\t";
            }
            ofs << endl;
        }
        else
        {
            for (int i = 0; i < values_view_.extent[0]; i++)
            {
                ofs << "--> depth " << i << " <--" << endl;
                for (int j = 0; j < values_view_.extent[1]; j++)
                {
                    for (int k = 0; k < values_view_.extent[2]; k++)
                    {
                        ofs << values_view_(i, j, k) << "\t";
                    }
                    ofs << endl;
                }
            }
        }


        ofs << std::fixed;
        ofs.precision(6);

        ofs << "[expect]" << endl;
        if (expects_view_.extent[1] == 1 && expects_view_.extent[2] == 1)
        {
            for (int i = 0; i < expects_view_.extent[0]; i++)
            {
                ofs << expects_view_(i, 0, 0) << "\t";
            }
            ofs << endl;
        }
        else
        {
            for (int i = 0; i < expects_view_.extent[0]; i++)
            {
                ofs << "--> depth " << i << " <--" << endl;
                for (int j = 0; j < expects_view_.extent[1]; j++)
                {
                    for (int k = 0; k < expects_view_.extent[2]; k++)
                    {
                        ofs << expects_view_(i, j, k) << "\t";
                    }
                    ofs << endl;
                }
            }
        }


        ofs << "[weight]" << endl;
        if (raw_weights_view_.extent[1] == 1 && raw_weights_view_.extent[2] == 1)
        {
            for (int i = 0; i < raw_weights_view_.extent[0]; i++)
            {
                ofs << raw_weights_view_(i, 0, 0) << "\t";
            }
            ofs << endl;
        }
        else
        {
            for (int i = 0; i < raw_weights_view_.extent[0]; i++)
            {
                ofs << "--> depth " << i << " <--" << endl;
                for (int j = 0; j < raw_weights_view_.extent[1]; j++)
                {
                    for (int k = 0; k < raw_weights_view_.extent[2]; k++)
                    {
                        ofs << raw_weights_view_(i, j, k) << "\t";
                    }
                    ofs << endl;
                }
            }
        }

        ofs << "[delta]" << endl;
        if (delta_view_.extent[1] == 1 && delta_view_.extent[2] == 1)
        {
            for (int i = 0; i < delta_view_.extent[0]; i++)
            {
                ofs << delta_view_(i, 0, 0) << "\t";
            }
            ofs << endl;
        }
        else
        {
            for (int i = 0; i < delta_view_.extent[0]; i++)
            {
                ofs << "--> depth " << i << " <--" << endl;
                for (int j = 0; j < delta_view_.extent[1]; j++)
                {
                    for (int k = 0; k < delta_view_.extent[2]; k++)
                    {
                        ofs << delta_view_(i, j, k) << "\t";
                    }
                    ofs << endl;
                }
            }
        }


        ofs << endl;
    }

    DataLayer::DataLayer(int depth, int height, int width, int seed)
        : cur_data_slot_(depth, height, width),
        next_data_slot_(depth, height, width),
        context_data_slot_(depth, height, width),
        last_data_slot_(depth, height, width),
        rand_collection_(extent<3>(depth, height, width), seed)
    {

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
            image.setwidth_height(depth() * (block_size + 1), (10) * (block_size + 1), true);
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
                    cur_data_slot_.delta_view_(i, 0, 0) > 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(255.0 * abs(cur_data_slot_.delta_view_(i, 0, 0))));
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 3 * (block_size + 1), block_size, block_size,
                    static_cast<unsigned char>(255.0 * next_data_slot_.expects_view_(i, 0, 0)));
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 4 * (block_size + 1), block_size, block_size,
                    next_data_slot_.values_view_(i, 0, 0) == 0.0 ? 0 : 255);
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 5 * (block_size + 1), block_size, block_size,
                    next_data_slot_.delta_view_(i, 0, 0) > 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(255.0 * abs(next_data_slot_.delta_view_(i, 0, 0))));
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 6 * (block_size + 1), block_size, block_size,
                    static_cast<unsigned char>(255.0 * context_data_slot_.expects_view_(i, 0, 0)));
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 7 * (block_size + 1), block_size, block_size,
                    context_data_slot_.values_view_(i, 0, 0) == 0.0 ? 0 : 255);
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 8 * (block_size + 1), block_size, block_size,
                    static_cast<unsigned char>(255.0 * last_data_slot_.expects_view_(i, 0, 0)));
            }

            for (int i = 0; i < depth(); i++)
            {
                image.set_region(i * (block_size + 1), 9 * (block_size + 1), block_size, block_size,
                    last_data_slot_.values_view_(i, 0, 0) == 0.0 ? 0 : 255);
            }
        }
        else
        {
            image.setwidth_height(depth() * (width() + 1) * (block_size + 1),
                ((10) * (height() + 1) + 2) * (block_size + 1), true);
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
                        auto delta = cur_data_slot_.delta_view_(depth_idx, height_idx, width_idx);
                        image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                            (2 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                            delta > 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                            static_cast<unsigned char>(255.0 * abs(delta)));
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
                            (4 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
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
                        auto delta = next_data_slot_.delta_view_(depth_idx, height_idx, width_idx);
                        image.set_region((depth_idx * (width() + 1) + width_idx) * (block_size + 1),
                            (5 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                            delta > 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                            static_cast<unsigned char>(255.0 * abs(delta)));
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
                            (6 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                            static_cast<unsigned char>(255.0 * context_data_slot_.expects_view_(depth_idx, height_idx, width_idx)));
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
                            (7 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                            context_data_slot_.values_view_(depth_idx, height_idx, width_idx) == 0.0 ? 0 : 255);
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
                            (8 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                            static_cast<unsigned char>(255.0 * last_data_slot_.expects_view_(depth_idx, height_idx, width_idx)));
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
                            (9 * (height() + 1) + height_idx) * (block_size + 1), block_size, block_size,
                            last_data_slot_.values_view_(depth_idx, height_idx, width_idx) == 0.0 ? 0 : 255);
                    }
                }
            }
        }

        return image;
    }

    void DataLayer::Dump(const string& filename) const
    {
        ofstream ofs;
        ofs.open(filename);

        ofs << "[cur_slot]" << endl;
        cur_data_slot_.Dump(ofs);

        ofs << "[next_slot]" << endl;
        next_data_slot_.Dump(ofs);

        ofs << "[context_slot]" << endl;
        context_data_slot_.Dump(ofs);

        ofs << "[last_slot]" << endl;
        last_data_slot_.Dump(ofs);

        ofs.close();
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
        context_data_slot_(output_num),
        bias_(output_num),
        bias_view_(output_num, bias_),
        neuron_weights_(output_num * input_depth * input_height * input_width),
        neuron_weights_view_(make_extent(output_num, input_depth, input_height, input_width), neuron_weights_)
    {
    }

    OutputLayer::OutputLayer(OutputLayer&& other)
        : cur_data_slot_(other.cur_data_slot_),
        next_data_slot_(other.next_data_slot_),
        context_data_slot_(other.context_data_slot_),
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
            output_value[idx] = CalcActivationProb(raw_weight);
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

        conv_layer.InitContext(bottom_layer, top_layer);
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

    void OutputLayer::Dump(const string& filename) const
    {

    }

#pragma endregion

#pragma region convolve layer

    ConvolveLayer::ConvolveLayer(int neuron_num, int neuron_depth, int neuron_height, int neuron_width)
        : neuron_weights_(neuron_num * neuron_depth * neuron_height * neuron_width),
        neuron_weights_view_(make_extent(neuron_num, neuron_depth, neuron_height, neuron_width), neuron_weights_),
        neuron_weights_delta_view_(neuron_weights_view_.extent),
        vbias_(neuron_depth),
        vbias_view_(neuron_depth, vbias_),
        vbias_delta_view_(vbias_view_.extent),
        hbias_(neuron_num),
        hbias_view_(neuron_num, hbias_),
        hbias_delta_view_(hbias_view_.extent),
        activation_view_(neuron_num)
    {
        fill(neuron_weights_delta_view_, 0.0);
        fill(vbias_delta_view_, 0.0);
        fill(hbias_delta_view_, 0.0);

        fill(activation_view_, 50.0);
        total_activation_count_ = 100.0;
    }

    ConvolveLayer::ConvolveLayer(ConvolveLayer&& other)
        : batch_size_(other.batch_size_),
        neuron_weights_(move(other.neuron_weights_)),
        neuron_weights_view_(other.neuron_weights_view_),
        neuron_weights_delta_view_(other.neuron_weights_delta_view_),
        vbias_(move(other.vbias_)),
        vbias_view_(other.vbias_view_),
        vbias_delta_view_(other.vbias_delta_view_),
        hbias_(move(other.hbias_)),
        hbias_view_(other.hbias_view_),
        hbias_delta_view_(other.hbias_delta_view_),
        activation_view_(other.activation_view_),
        total_activation_count_(other.total_activation_count_)
    {
    }

    void ConvolveLayer::InitContext(DataLayer& bottom_layer, DataLayer& top_layer) const
    {
        // neuron layer
        array_view<const double> conv_hbias = this->hbias_view_;

        // top layer
        array_view<double, 3> top_context_values = top_layer.context_data_slot_.values_view_;
        array_view<double, 3> top_context_expects = top_layer.context_data_slot_.expects_view_;
        array_view<double, 3> top_context_raw_weights = top_layer.context_data_slot_.raw_weights_view_;

        top_context_values.discard_data();
        top_context_expects.discard_data();
        top_context_raw_weights.discard_data();

        auto& rand_collection = top_layer.rand_collection_;

        // Discriminative inputs only affect the initial value of top_layer weights.
        // Given enough iterations, they do not affect the final value of top_layer weights.
        // so they serve as the accelerator of thinking process. Amazing!
        parallel_for_each(top_context_values.extent,
            [=](index<3> idx) restrict(amp)
        {
            int top_depth_idx = idx[0];

            // Note that, the value of raw_weight is not set to zero here.
            // So you can assign custom prior weights from other sources, e.g. upper layers
            auto& top_context_raw_weight = top_context_raw_weights[idx];

            top_context_raw_weight = conv_hbias[top_depth_idx];

            auto expect = CalcActivationProb(top_context_raw_weight);
            top_context_expects[idx] = expect;
            top_context_values[idx] = rand_collection[idx].next_single() <= expect ? 1.0 : 0.0;
        });

        PassDown(top_layer, DataSlotType::kContext, bottom_layer, DataSlotType::kContext);
    }

    void ConvolveLayer::PassUp(DataLayer& bottom_layer, DataSlotType bottom_slot_type,
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
        array_view<double, 3> top_raw_weights = top_slot.raw_weights_view_;
        array_view<double, 3> top_deltas = top_slot.delta_view_;

        top_values.discard_data();
        top_expects.discard_data();
        top_raw_weights.discard_data();
        top_deltas.discard_data();

        array_view<const double, 3> top_context_expects = top_layer.context_data_slot_.expects_view_;
        array_view<const double, 3> top_context_raw_weights = top_layer.context_data_slot_.raw_weights_view_;

        // bottom layer
        const int bottom_depth = bottom_layer.depth();

        //array_view<const double, 3> bottom_values = bottom_layer[bottom_slot_type].values_view_;
        array_view<const double, 3> bottom_expects = bottom_layer[bottom_slot_type].expects_view_;
        array_view<const double, 3> bottom_context_expects = bottom_layer.context_data_slot_.expects_view_;
        array_view<const double, 3> bottom_context_raw_weights = bottom_layer.context_data_slot_.raw_weights_view_;

        array_view<double, 3> bottom_deltas = bottom_layer[bottom_slot_type].delta_view_;
        bottom_deltas.discard_data();

        auto& rand_collection = top_layer.rand_collection_;
        const double rawWeightDecay = this->kRawWeightDecay;

        parallel_for_each(bottom_deltas.extent,
            [=](index<3> idx) restrict(amp)
        {
            //bottom_deltas[idx] = bottom_values[idx] - bottom_context_expects[idx];
            bottom_deltas[idx] = bottom_expects[idx] - bottom_context_expects[idx];
        });


        // pass up the DIFFERENCE between context and data
        // non-tiled version
        parallel_for_each(top_values.extent,
            [=](index<3> idx) restrict(amp)
        {
            int top_depth_idx = idx[0];
            int top_height_idx = idx[1];
            int top_width_idx = idx[2];

            auto& top_raw_weight = top_raw_weights[idx];
            auto top_context_expect = top_context_expects[idx];
            auto top_context_raw_weight = top_context_raw_weights[idx];

            auto weight_delta = 0.0;

            auto model_delta_norm = 0.0;

            const auto& current_neuron = conv_neuron_weights[top_depth_idx];

            for (int depth_idx = 0; depth_idx < bottom_depth; depth_idx++)
            {
                for (int height_idx = 0; height_idx < neuron_height; height_idx++)
                {
                    for (int width_idx = 0; width_idx < neuron_width; width_idx++)
                    {
                        index<3> bottom_idx(depth_idx, top_height_idx + height_idx, top_width_idx + width_idx);

                        auto data_expect = bottom_expects[bottom_idx];
                        auto bottom_context_expect = bottom_context_expects[bottom_idx];
                        auto bottom_context_raw_weight = bottom_context_raw_weights[bottom_idx];

                        auto weight = current_neuron(depth_idx, height_idx, width_idx);

                        auto top_inactive_bottom_context_expect = CalcActivationProb(bottom_context_raw_weight - top_context_expect * weight);
                        auto top_active_bottom_context_expect = CalcActivationProb(bottom_context_raw_weight + (1.0 - top_context_expect) * weight);

                        auto data_delta = data_expect - bottom_context_expect;
                        auto model_delta_abs = fmin(fabs(top_active_bottom_context_expect - bottom_context_expect),
                            fabs(bottom_context_expect - top_inactive_bottom_context_expect));
                        auto model_delat_dir = data_delta * weight > 0 ? 1.0 : -1.0;

                        //if (data_delta > 0.0)
                        //{
                        //    if (weight > 0.0)
                        //    {
                        //        model_delta = top_active_bottom_context_expect - bottom_context_expect;// > 0
                        //    }
                        //    else
                        //    {
                        //        model_delta = bottom_context_expect - top_inactive_bottom_context_expect;// < 0
                        //    }
                        //}
                        //else
                        //{
                        //    if (weight > 0.0)
                        //    {
                        //        model_delta = bottom_context_expect - top_inactive_bottom_context_expect;// > 0
                        //    }
                        //    else
                        //    {
                        //        model_delta = top_active_bottom_context_expect - bottom_context_expect;// < 0
                        //    }
                        //}

                        weight_delta += data_delta * model_delta_abs * model_delat_dir;
                        //model_delta_norm += model_delta_abs;
                    }
                }
            }

            top_raw_weight = (top_context_raw_weight + weight_delta);

            auto expect = CalcActivationProb(top_raw_weight);
            top_expects[idx] = expect;
            top_values[idx] = rand_collection[idx].next_single() <= expect ? 1.0 : 0.0;

            //top_deltas[idx] = expect - top_context_expects[idx];
            top_deltas[idx] = CalcActivationProb(weight_delta);
        });
    }

    void ConvolveLayer::InferUp(DataLayer& bottom_layer, DataSlotType bottom_slot_type,
        DataLayer& top_layer, DataSlotType top_slot_type) const
    {
        assert(top_layer.depth() == this->neuron_num());
        assert(top_layer.height() == bottom_layer.height() - this->neuron_height() + 1);
        assert(top_layer.width() == bottom_layer.width() - this->neuron_width() + 1);

        // Suppose we have already setup context
        //InitContext(bottom_layer, top_layer);

        // this two-stage pass-up process seeks the optimal balance between PoE and MoE.
        // i.e. minimum number of bits used to store the information.
        for (int iter = 0; iter < kInferIteration; iter++)
        {
            PassUp(bottom_layer, bottom_slot_type, top_layer, top_slot_type);

            // update context for top layer
            top_layer[top_slot_type].CopyTo(top_layer.context_data_slot_);

            PassDown(top_layer, top_slot_type, bottom_layer, DataSlotType::kContext);

            /* bottom_layer.GenerateImage().save_image("model_dump\\debug_bottom_data_" + std::to_string(iter) + ".bmp");
             top_layer.GenerateImage().save_image("model_dump\\debug_top_data_" + std::to_string(iter) + ".bmp");*/
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
        array_view<const double, 3> top_expects = top_layer[top_slot_type].expects_view_;

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
                            //top_values(neuron_idx, top_height_idx, top_width_idx);
                            top_expects(neuron_idx, top_height_idx, top_width_idx);
                    }
                }
            }

            // Logistic activation function. Maybe more types of activation function later.
            bottom_raw_weights[idx] = raw_weight;
            auto prob = CalcActivationProb(raw_weight);

            bottom_expects[idx] = prob;
            bottom_values[idx] = rand_collection[idx].next_single() <= prob ? 1.0 : 0.0;
        });
    }

    void ConvolveLayer::Train(DataLayer& bottom_layer, DataLayer& top_layer, double learning_rate)
    {
        // top layer
        const int top_height = top_layer.height();
        const int top_width = top_layer.width();
        array_view<const double, 3> top_context_expects = top_layer.context_data_slot_.expects_view_;
        array_view<const double, 3> top_expects = top_layer.cur_data_slot_.expects_view_;
        array_view<const double, 3> top_next_expects = top_layer.next_data_slot_.expects_view_;
        array_view<const double, 3> top_last_expects = top_layer.last_data_slot_.expects_view_;

        // bottom layer
        const int bottom_height = bottom_layer.height();
        const int bottom_width = bottom_layer.width();


        array_view<const double, 3> bottom_expects = bottom_layer.cur_data_slot_.expects_view_;
        array_view<const double, 3> bottom_next_expects = bottom_layer.next_data_slot_.expects_view_;
        array_view<const double, 3> bottom_last_expects = bottom_layer.last_data_slot_.expects_view_;
        array_view<const double, 3> bottom_next_raw_weights = bottom_layer.next_data_slot_.raw_weights_view_;
        array_view<const double, 3> bottom_context_expects = bottom_layer.context_data_slot_.expects_view_;
        array_view<const double, 3> bottom_context_raw_weights = bottom_layer.context_data_slot_.raw_weights_view_;

        // neuron layer
        // parameters to train
        array_view<double, 4> conv_neuron_weights_delta = this->neuron_weights_delta_view_;

        array_view<double> conv_vbias_delta = this->vbias_delta_view_;
        array_view<double> conv_hbias_delta = this->hbias_delta_view_;
        /*array_view<double, 4> conv_neuron_weights_delta = this->neuron_weights_view_;

        array_view<double> conv_vbias_delta = this->vbias_view_;
        array_view<double> conv_hbias_delta = this->hbias_view_;*/

        //array_view<const double, 4> conv_neuron_weights = this->neuron_weights_view_;
        array_view<double> activation_view = this->activation_view_;

        /*InitContext(bottom_layer, top_layer);

        InferUp(bottom_layer, DataSlotType::kCurrent, top_layer, DataSlotType::kCurrent);

        bottom_layer.context_data_slot_.CopyTo(bottom_layer.last_data_slot_);*/

        /*bottom_layer.GenerateImage().save_image("model_dump\\debug_bottom_data_init.bmp");
        top_layer.GenerateImage().save_image("model_dump\\debug_top_data_init.bmp");
        bottom_layer.Dump("model_dump\\debug_bottom_data_init.txt");
        top_layer.Dump("model_dump\\debug_top_data_init.txt");*/

        //InitContext(bottom_layer, top_layer);

        //for (int iter = 0; iter < kInferIteration; iter++)
        {
            InitContext(bottom_layer, top_layer);

            InferUp(bottom_layer, DataSlotType::kCurrent, top_layer, DataSlotType::kCurrent);

            PassDown(top_layer, DataSlotType::kCurrent, bottom_layer, DataSlotType::kNext);

            InitContext(bottom_layer, top_layer);

            InferUp(bottom_layer, DataSlotType::kNext, top_layer, DataSlotType::kNext);

            /*bottom_layer.GenerateImage().save_image("model_dump\\debug_1_bottom_data_" + std::to_string(iter) + ".bmp");
            top_layer.GenerateImage().save_image("model_dump\\debug_1_top_data_" + std::to_string(iter) + ".bmp");
            bottom_layer.Dump("model_dump\\debug_1_bottom_data_" + std::to_string(iter) + ".txt");
            top_layer.Dump("model_dump\\debug_1_top_data_" + std::to_string(iter) + ".txt");*/

            // non-tiled version
            parallel_for_each(conv_neuron_weights_delta.extent, [=](index<4> idx) restrict(amp)
            {
                auto delta = 0.0;

                int neuron_idx = idx[0];
                int bottom_depth_idx = idx[1];
                int neuron_height_idx = idx[2];
                int neuron_width_idx = idx[3];

                //auto neuron_weight = conv_neuron_weights[idx];

                for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
                {
                    for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
                    {
                        index<3> top_idx(neuron_idx, top_height_idx, top_width_idx);
                        index<3> bottom_idx(bottom_depth_idx, neuron_height_idx + top_height_idx, neuron_width_idx + top_width_idx);

                        auto top_expect = top_expects[top_idx];
                        auto top_next_expect = top_next_expects[top_idx];
                        /*auto top_last_expect = top_last_expects[top_idx];
                        auto top_context_expect = top_context_expects[top_idx];*/

                        auto bottom_expect = bottom_expects[bottom_idx];
                        auto bottom_next_expect = bottom_next_expects[bottom_idx];
                        //auto bottom_last_expect = bottom_last_expects[bottom_idx];

                        delta += bottom_expect * top_expect - bottom_next_expect * top_next_expect;
                        //delta += bottom_expect * top_expect - bottom_last_expect * top_last_expect;
                    }
                }

                conv_neuron_weights_delta[idx] += delta / (top_height * top_width) * learning_rate;
            });

            /*top_layer.cur_data_slot_.CopyTo(top_layer.context_data_slot_);
            bottom_layer.next_data_slot_.CopyTo(bottom_layer.context_data_slot_);*/

            std::cout << "debug = " << bottom_layer.ReconstructionError(DataSlotType::kNext) << std::endl;
        }

        // neuron activation
        const double act_decay = this->kActivationDecay;
        parallel_for_each(activation_view.extent, [=](index<1> idx) restrict(amp)
        {
            int neuron_idx = idx[0];
            double top_activation_prob = 0.0;

            for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
            {
                for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
                {
                    index<3> top_idx(neuron_idx, top_height_idx, top_width_idx);
                    auto top_expect = top_expects[top_idx];

                    top_activation_prob += top_expect;
                }
            }
            activation_view[idx] = activation_view[idx] * act_decay + top_activation_prob / top_height / top_width;
        });

        total_activation_count_ = total_activation_count_ * act_decay + 1;

        ////
        //bottom_layer.context_data_slot_.CopyTo(bottom_layer.next_data_slot_);
        //InitContext(bottom_layer, top_layer);

        //for (int iter = 0; iter < kInferIteration; iter++)
        //{
        //    PassUp(bottom_layer, DataSlotType::kNext, top_layer, DataSlotType::kNext);

        //    PassDown(top_layer, DataSlotType::kNext, bottom_layer, DataSlotType::kContext);

        //    /*bottom_layer.GenerateImage().save_image("model_dump\\debug_2_bottom_data_" + std::to_string(iter) + ".bmp");
        //    top_layer.GenerateImage().save_image("model_dump\\debug_2_top_data_" + std::to_string(iter) + ".bmp");
        //    bottom_layer.Dump("model_dump\\debug_2_bottom_data_" + std::to_string(iter) + ".txt");
        //    top_layer.Dump("model_dump\\debug_2_top_data_" + std::to_string(iter) + ".txt");*/

        //    top_layer.next_data_slot_.CopyTo(top_layer.context_data_slot_);

        //    std::cout << "debug 2: " << iter << " = " << bottom_layer.ReconstructionError(DataSlotType::kContext) << std::endl;
        //}



        //// update vbias
        //parallel_for_each(conv_vbias_delta.extent, [=](index<1> idx) restrict(amp)
        //{
        //    auto delta = 0.0;

        //    int depth_idx = idx[0];

        //    for (int bottom_height_idx = 0; bottom_height_idx < bottom_height; bottom_height_idx++)
        //    {
        //        for (int bottom_width_idx = 0; bottom_width_idx < bottom_width; bottom_width_idx++)
        //        {
        //            index<3> bottom_idx(depth_idx, bottom_height_idx, bottom_width_idx);
        //            auto bottom_expect = bottom_expects[bottom_idx];
        //            auto bottom_next_expect = bottom_next_expects[bottom_idx];

        //            delta += bottom_expect - bottom_next_expect;
        //        }
        //    }

        //    conv_vbias_delta[idx] += delta / (bottom_height * bottom_width) * learning_rate;
        //});

        //// update hbias
        //parallel_for_each(conv_hbias_delta.extent, [=](index<1> idx) restrict(amp)
        //{
        //    auto delta = 0.0;

        //    int neuron_idx = idx[0];

        //    for (int top_height_idx = 0; top_height_idx < top_height; top_height_idx++)
        //    {
        //        for (int top_width_idx = 0; top_width_idx < top_width; top_width_idx++)
        //        {
        //            index<3> top_idx(neuron_idx, top_height_idx, top_width_idx);
        //            auto top_expect = top_expects[top_idx];
        //            auto top_next_expect = top_next_expects[top_idx];

        //            delta += top_expect - top_next_expect;
        //        }
        //    }

        //    conv_hbias_delta[idx] += delta / (top_height * top_width) * learning_rate;
        //});

        this->batch_size_++;
    }

    void ConvolveLayer::ApplyGradients()
    {
        int batch_size = this->batch_size_;

        if (batch_size <= 0)
        {
            return;
        }

        array_view<double, 4> conv_neuron_weights_delta = this->neuron_weights_delta_view_;

        array_view<double> conv_vbias_delta = this->vbias_delta_view_;
        array_view<double> conv_hbias_delta = this->hbias_delta_view_;

        array_view<double, 4> conv_neuron_weights = this->neuron_weights_view_;

        array_view<double> conv_vbias = this->vbias_view_;
        array_view<double> conv_hbias = this->hbias_view_;

        int inferIter = 1;// kInferIteration;

        parallel_for_each(conv_neuron_weights.extent, [=](index<4> idx) restrict(amp)
        {
            conv_neuron_weights[idx] += conv_neuron_weights_delta[idx] / batch_size / inferIter;
            conv_neuron_weights_delta[idx] = 0.0;

            /*auto& weight = conv_neuron_weights[idx];

            const double decay = 0.001;

            if (weight > decay)
            {
                weight -= decay;
            }
            else if (weight < -decay)
            {
                weight += decay;
            }
            else
            {
                weight = 0;
            }*/
        });

        parallel_for_each(conv_hbias.extent, [=](index<1> idx) restrict(amp)
        {
            conv_hbias[idx] += conv_hbias_delta[idx] / batch_size / inferIter;
            conv_hbias_delta[idx] = 0.0;
        });

        parallel_for_each(conv_vbias.extent, [=](index<1> idx) restrict(amp)
        {
            conv_vbias[idx] += conv_vbias_delta[idx] / batch_size / inferIter;
            conv_vbias_delta[idx] = 0.0;
        });

        this->batch_size_ = 0;
    }

    void ConvolveLayer::RandomizeParams(unsigned int seed)
    {
        default_random_engine generator(seed);
        normal_distribution<double> distribution(0.0, 0.1);

        for (auto& w : neuron_weights_)
        {
            w = (distribution(generator));
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

            for (int i = 0; i < hbias_.size(); i++)
            {
                double activation_prob = activation_view_[i] / total_activation_count_;

                image.set_region(1 * (block_size + 1), (2 + i) * (block_size + 1), block_size, block_size,
                    activation_prob >= 0.5 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(abs(2 * activation_prob - 1.0) * 255.0));
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

            for (int i = 0; i < hbias_.size(); i++)
            {
                double activation_prob = activation_view_[i] / total_activation_count_;

                image.set_region(1 * (block_size + 1), (2 + neuron_height() / 2 + (neuron_height() + 1) * i) * (block_size + 1), block_size, block_size,
                    activation_prob >= 0.5 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(abs(2 * activation_prob - 1.0) * 255.0));
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

    void ConvolveLayer::Dump(const string& filename) const
    {
        ofstream ofs;
        ofs.open(filename);

        ofs << std::fixed;
        ofs.precision(6);

        ofs << "[vbias]" << endl;
        for (int i = 0; i < vbias_.size(); i++)
        {
            ofs << vbias_view_[i] << "\t";
        }
        ofs << endl;

        ofs << "[hbias]" << endl;
        for (int i = 0; i < hbias_.size(); i++)
        {
            ofs << hbias_view_[i] << "\t";
        }
        ofs << endl;

        ofs << "[activation]:" << total_activation_count_ << endl;
        for (int i = 0; i < hbias_.size(); i++)
        {
            ofs << activation_view_[i] / total_activation_count_ << "\t";
        }
        ofs << endl;

        ofs << "[neurons]" << endl;
        for (int i = 0; i < neuron_num(); i++)
        {
            ofs << "--> neuron " << i << " <--" << endl;
            auto single_neuron_weights_view = neuron_weights_view_[i];
            for (int j = 0; j < neuron_depth(); j++)
            {
                ofs << "---> depth " << j << " <---" << endl;
                for (int k = 0; k < neuron_height(); k++)
                {
                    for (int m = 0; m < neuron_width(); m++)
                    {
                        ofs << single_neuron_weights_view(j, k, m) << "\t";
                    }
                    ofs << endl;
                }
                ofs << endl;
            }
            ofs << endl;
        }
        ofs << endl;

        ofs.close();
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

    void DeepModel::AddDataLayer(int depth, int height, int width)
    {
        assert(layer_stack_.empty());
        data_layers_.emplace_back(depth, height, width, uniform_int_distribution<int>()(random_engine_));
        layer_stack_.emplace_back(LayerType::kDataLayer, data_layers_.size() - 1);
    }

    void DeepModel::AddDataLayer()
    {
        assert(layer_stack_.size() >= 2);
        assert(layer_stack_.back().first == LayerType::kConvolveLayer || layer_stack_.back().first == LayerType::kPoolingLayer);
        assert(layer_stack_[layer_stack_.size() - 2].first == LayerType::kDataLayer);

        const auto& last_data_layer = data_layers_[layer_stack_[layer_stack_.size() - 2].second];
        if (layer_stack_.back().first == LayerType::kConvolveLayer)
        {
            auto& conv_layer = convolve_layers_[layer_stack_.back().second];
            data_layers_.emplace_back(conv_layer.neuron_num(),
                last_data_layer.height() - conv_layer.neuron_height() + 1,
                last_data_layer.width() - conv_layer.neuron_width() + 1,
                uniform_int_distribution<int>()(random_engine_));
        }
        else
        {
            const auto& pooling_layer = pooling_layers[layer_stack_.back().second];
            assert(last_data_layer.height() % pooling_layer.block_height() == 0);
            assert(last_data_layer.width() % pooling_layer.block_width() == 0);
            data_layers_.emplace_back(last_data_layer.depth(),
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
            last_data_layer.depth() * (1),
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

        if (conv_layer.batch_size() >= 1)
        {
            conv_layer.ApplyGradients();
        }

        /*conv_layer.InitContext(bottom_data_layer, top_data_layer);
        conv_layer.InferUp(bottom_data_layer, DataSlotType::kCurrent, top_data_layer, DataSlotType::kCurrent);*/

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

    void DeepModel::Dump(const string& folder) const
    {
        for (int i = 0; i < data_layers_.size(); i++)
        {
            data_layers_[i].Dump(folder + "\\layer" + to_string(i) + "_data.txt");
        }

        for (int i = 0; i < convolve_layers_.size(); i++)
        {
            convolve_layers_[i].Dump(folder + "\\layer" + to_string(i) + "_conv.txt");
        }
    }

#pragma endregion
}