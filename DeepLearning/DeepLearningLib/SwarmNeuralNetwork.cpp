#include "SwarmNeuralNetwork.h"

#include <random>
#include <amp_math.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "AmpUtility.h"
// for bitmap generation
#include "bitmap_image.hpp"

namespace deep_learning_lib
{
    using std::string;
    using std::to_string;
    using std::vector;
    using std::default_random_engine;
    using std::normal_distribution;
    using std::uniform_int_distribution;
    using std::numeric_limits;
    using std::ofstream;
    using std::cout;
    using std::endl;

    using concurrency::array_view;
    using concurrency::index;
    using concurrency::extent;
    using concurrency::parallel_for_each;
    using concurrency::atomic_fetch_add;

    using concurrency::precise_math::log;
    using concurrency::precise_math::exp;
    using concurrency::precise_math::pow;
    using concurrency::precise_math::fabs;
    using concurrency::precise_math::fmin;

    SwarmNN::SwarmNN(int bottom_length, int top_length, unsigned int seed)
        : bottom_length_(bottom_length),
        top_length_(top_length),
        neuron_weights_(top_length, bottom_length),
        bottom_biases_(bottom_length),
        bottom_values_(bottom_length),
        top_values_(top_length),
        bottom_recon_values_(bottom_length),
        bottom_rand_(extent<1>(bottom_length), seed),
        top_rand_(extent<1>(top_length), seed + 1)
    {
        fill(bottom_biases_, 0.0);

        fill(bottom_values_, 0);
        fill(top_values_, 0);
        fill(bottom_recon_values_, 0);

        RandomizeParams(seed);
    }

    void SwarmNN::RandomizeParams(unsigned int seed)
    {
        default_random_engine generator(seed);
        normal_distribution<double> distribution(0.0, 0.1);

        vector<double> init_weights(bottom_length_ * top_length_);

        for (auto& w : init_weights)
        {
            w = distribution(generator);
        }

        concurrency::copy(init_weights.begin(), neuron_weights_);
    }

    double SwarmNN::Feed(const vector<int>& input_data, double data_weight)
    {
        concurrency::copy(input_data.begin(), bottom_values_);

        PassUp(data_weight);
        PassDown(data_weight);

        return CalcReconError();
    }

    void SwarmNN::PassUp(double data_weight)
    {
        int top_length = top_length_;
        int bottom_length = bottom_length_;

        array_view<int> top_values = top_values_;

        top_values.discard_data();

        array_view<const int> bottom_values = bottom_values_;
        array_view<const int> bottom_recon_values = bottom_recon_values_;

        array_view<const double> bottom_biases = bottom_biases_;
        array_view<double, 2> neuron_weights = neuron_weights_;

        auto& top_rand = top_rand_;

        parallel_for_each(extent<1>(top_length), [=](index<1> idx) restrict(amp)
        {
            // for each top neuron
            int top_idx = idx[0];

            double messsage = 0.0;
            for (int bottom_idx = 0; bottom_idx < bottom_length; bottom_idx++)
            {
                double bottom_bias = bottom_biases[bottom_idx];
                double neuron_weight = neuron_weights(top_idx, bottom_idx);

                messsage += log(exp(bottom_bias) + 1.0) + neuron_weight * bottom_values[bottom_idx]
                    - log(exp(neuron_weight + bottom_bias) + 1.0);
            }

            double top_expect = CalcActivationProb(messsage);

            int top_value = top_rand[idx].next_single() <= top_expect ? 1 : 0;
            top_values[idx] = top_value;

            if (data_weight > 0)
            {
                // learning along with inferring
                if (top_value == 1)// dropout effect for 0
                {
                    for (int bottom_idx = 0; bottom_idx < bottom_length; bottom_idx++)
                    {
                        int bottom_value = bottom_values[bottom_idx];
                        double& neuron_weight = neuron_weights(top_idx, bottom_idx);

                        if (neuron_weight > 0 && bottom_value == 0)
                        {
                            neuron_weight -= fmin(neuron_weight, data_weight);
                        }
                        else if (neuron_weight < 0 && bottom_value == 1)
                        {
                            neuron_weight += fmin(-neuron_weight, data_weight);
                        }
                    }
                }
            }
        });
    }

    void SwarmNN::PassDown(double data_weight)
    {
        int top_length = top_length_;
        int bottom_length = bottom_length_;

        array_view<const int> top_values = top_values_;

        array_view<const int> bottom_values = bottom_values_;
        array_view<int> bottom_recon_values = bottom_recon_values_;

        bottom_recon_values.discard_data();

        array_view<double> bottom_biases = bottom_biases_;
        array_view<double, 2> neuron_weights = neuron_weights_;

        auto& bottom_rand = bottom_rand_;

        parallel_for_each(extent<1>(bottom_length), [=](index<1> idx) restrict(amp)
        {
            // for each bottom neuron
            int bottom_idx = idx[0];

            double& bottom_bias = bottom_biases[bottom_idx];
            double bottom_innate_expect = CalcActivationProb(bottom_bias);

            double message = bottom_bias;
            for (int top_idx = 0; top_idx < top_length; top_idx++)
            {
                message += neuron_weights(top_idx, bottom_idx) * top_values[top_idx];
            }

            double bottom_recon_expect = CalcActivationProb(message);

            int bottom_recon_value = bottom_rand[idx].next_single() <= bottom_recon_expect ? 1 : 0;
            bottom_recon_values[idx] = bottom_recon_value;

            if (data_weight > 0)
            {
                // learning along with inferring
                int bottom_value = bottom_values[bottom_idx];

                for (int top_idx = 0; top_idx < top_length; top_idx++)
                {
                    // only activated top neuron can influence bottom neuron, dropout effect.
                    if (top_values[top_idx] == 1)
                    {
                        neuron_weights(top_idx, bottom_idx) += (bottom_value - bottom_recon_expect) * data_weight;
                    }
                }

                bottom_bias += (bottom_value - bottom_innate_expect) * data_weight;
            }
        });
    }

    double SwarmNN::CalcReconError() const
    {
        array_view<int> recon_error(1);
        recon_error(0) = 0;

        array_view<const int> bottom_values = bottom_values_;
        array_view<const int> bottom_recon_values = bottom_recon_values_;

        parallel_for_each(bottom_values.extent, [=](index<1> idx) restrict(amp)
        {
            int diff = bottom_values[idx] - bottom_recon_values[idx];
            atomic_fetch_add(&recon_error(0), diff * diff);
        });

        return sqrt(recon_error(0));
    }

    void SwarmNN::Dump(const string& folder, const string& tag) const
    {
        bitmap_image image;

        const int block_size = 2;

        ofstream ofs;

        ofs.open(folder + "\\neuron_weights_" + tag + ".txt");

        double max_value = numeric_limits<double>::min();

        ofs << "[bottom biases]" << endl;
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                auto bias = bottom_biases_[i * 16 + j];
                ofs << bias << "\t";

                max_value = std::max(max_value, abs(bias));
            }
            ofs << endl;
        }

        ofs << endl;

        for (int top_idx = 0; top_idx < top_length_; top_idx++)
        {
            auto single_neuron = neuron_weights_[top_idx];

            ofs << "[neuron]:" << top_idx << endl;

            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    auto weight = single_neuron[i * 16 + j];
                    ofs << weight << "\t";

                    max_value = std::max(max_value, abs(weight));
                }

                ofs << endl;
            }

            ofs << endl;
        }

        ofs.close();

        image.setwidth_height(16 * (block_size + 1), (top_length_ + 1) * (16 + 1) * (block_size + 1), true);
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                auto bias = bottom_biases_[i * 16 + j];

                image.set_region(j * (block_size + 1), i * (block_size + 1), block_size, block_size,
                    bias > 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(abs(bias) / max_value * 255.0));
            }
        }

        for (int top_idx = 0; top_idx < top_length_; top_idx++)
        {
            auto single_neuron = neuron_weights_[top_idx];

            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    auto weight = single_neuron[i * 16 + j];

                    image.set_region(j * (block_size + 1), ((top_idx + 1) * (16 + 1) + i) * (block_size + 1), block_size, block_size,
                        weight > 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                        static_cast<unsigned char>(abs(weight) / max_value * 255.0));
                }
            }
        }

        image.save_image(folder + "\\neuron_weights_" + tag + ".bmp");

        ofs.open(folder + "\\bottom_" + tag + ".txt");

        image.setwidth_height(16 * (block_size + 1), (16 + 16 + 1)*(block_size + 1), true);

        ofs << "[bottom values]" << endl;
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                auto value = bottom_values_[i * 16 + j];
                ofs << value << "\t";

                image.set_region(j * (block_size + 1), i * (block_size + 1), block_size, block_size,
                    static_cast<unsigned char>(value * 255.0));
            }
            ofs << endl;
        }

        ofs << endl;

        ofs << "[bottom recon values]" << endl;
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                auto value = bottom_recon_values_[i * 16 + j];
                ofs << value << "\t";

                image.set_region(j * (block_size + 1), (16 + 1 + i) * (block_size + 1), block_size, block_size,
                    static_cast<unsigned char>(value * 255.0));
            }
            ofs << endl;
        }

        ofs.close();
        image.save_image(folder + "\\bottom_" + tag + ".bmp");

        ofs.open(folder + "\\top_" + tag + ".txt");

        ofs << "[top values]" << endl;
        for (int top_idx = 0; top_idx < top_length_; top_idx++)
        {
            ofs << top_values_[top_idx] << "\t";
        }
        ofs.close();
    }
}