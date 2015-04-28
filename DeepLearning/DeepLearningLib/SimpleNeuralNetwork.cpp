#include "SimpleNeuralNetwork.h"

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
    using concurrency::precise_math::fmax;

    SimpleNN::SimpleNN(int bottom_length, int top_length, unsigned int seed)
        : bottom_length_(bottom_length),
        top_length_(top_length),
        neuron_weights_(top_length, bottom_length),
        bottom_biases_(bottom_length),
        top_biases_(top_length),
        top_expects_(top_length),
        top_values_(top_length),
        bottom_values_(bottom_length),
        bottom_recon_expects_(bottom_length),
        bottom_clusters_(bottom_length),
        bottom_up_messages_(bottom_length, top_length),
        top_energies_(top_length),
        top_rand_(extent<1>(top_length), seed)
    {
        fill(bottom_biases_, 0.0);
        fill(top_biases_, 0.0);


        fill(top_values_, 0);
        fill(top_expects_, 0.0);

        fill(bottom_values_, 0);
        fill(bottom_recon_expects_, 0.0);
        fill(bottom_clusters_, -1);

        fill(bottom_up_messages_, 0.0);
        fill(top_energies_, 0.0);

        RandomizeParams(seed);
    }

    void SimpleNN::RandomizeParams(unsigned int seed)
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

    double SimpleNN::Feed(const vector<int>& input_data, double data_weight)
    {
        concurrency::copy(input_data.begin(), bottom_values_);

        // temporal and space correlation should be handled by convolving
        fill(top_values_, 0);
        fill(bottom_up_messages_, 0.0);
        fill(top_energies_, 0.0);

        PassDown();

        for (int i = 0; i < kInferenceCount; i++)
        {
            PassUp();
            PassDown();

            /*Observe();
            double err = CalcReconError();
            cout << "\t iter = " << i << "\terr = " << err << endl;

            Dump("model_dump", "iter" + to_string(i));*/
        }

        Learn(data_weight);

        return CalcReconError();
    }

    void SimpleNN::PassUp()
    {
        int top_length = top_length_;
        int bottom_length = bottom_length_;

        array_view<const double> top_biases = top_biases_;
        array_view<const int> bottom_values = bottom_values_;
        array_view<const double, 2> neuron_weights = neuron_weights_;
        array_view<const double, 2> bottom_up_messages = bottom_up_messages_;

        array_view<double> top_energies = top_energies_;
        top_energies.discard_data();

        parallel_for_each(extent<1>(top_length), [=](index<1> idx) restrict(amp)
        {
            // for each top neuron
            int top_idx = idx[0];

            double top_bias = top_biases[top_idx];

            double top_energy = top_bias;
            // each bottom neuron is independent in the top neuron factor context
            for (int bottom_idx = 0; bottom_idx < bottom_length; bottom_idx++)
            {
                int bottom_value = bottom_values[bottom_idx];
                double neuron_weight = neuron_weights(top_idx, bottom_idx);
                double bottom_up_message = bottom_up_messages(bottom_idx, top_idx);

                // bottom_acitive : this bottom neuron is explained by this top neuron
                double top_energy_bottom_active =
                    neuron_weight * bottom_value + fmin(0.0, bottom_up_message);

                // bottom_inactive : this bottom neuron is explained by other top neuron
                double top_energy_bottom_inactive =
                    -fabs(neuron_weight) + fmin(0.0, -bottom_up_message);

                top_energy += fmax(top_energy_bottom_active, top_energy_bottom_inactive);
            }

            // each top neuron has the same message for bottom neuron
            top_energies[idx] = top_energy;
        });
    }

    void SimpleNN::PassDown()
    {
        int top_length = top_length_;
        int bottom_length = bottom_length_;

        array_view<const int> bottom_values = bottom_values_;

        array_view<const double> bottom_biases = bottom_biases_;
        array_view<const double, 2> neuron_weights = neuron_weights_;
        array_view<const double> top_energies = top_energies_;

        array_view<double, 2> bottom_up_messages = bottom_up_messages_;
        bottom_up_messages.discard_data();

        array_view<int> bottom_clusters = bottom_clusters_;
        bottom_clusters.discard_data();

        double kDoubleLowest = numeric_limits<double>::lowest();

        parallel_for_each(extent<1>(bottom_length), [=](index<1> idx) restrict(amp)
        {
            // for each bottom neuron
            int bottom_idx = idx[0];
            int bottom_value = bottom_values[bottom_idx];

            double bottom_bias = bottom_biases[bottom_idx];
            double bottom_bias_energy = bottom_bias * bottom_value;

            double max_bottom_energy = bottom_bias_energy;
            int max_bottom_energy_top_idx = -1;
            double second_max_bottom_energy = kDoubleLowest;

            for (int top_idx = 0; top_idx < top_length; top_idx++)
            {
                double neuron_weight = neuron_weights(top_idx, bottom_idx);
                double top_energy = top_energies(top_idx);

                double bottom_energy_top_active = neuron_weight * bottom_value
                    + fabs(neuron_weight) + fmin(0.0, -fabs(neuron_weight) + top_energy);
                double bottom_energy_top_inactive = fmin(0.0, fabs(neuron_weight) - top_energy);

                double bottom_energy = fmax(bottom_energy_top_active, bottom_energy_top_inactive);

                bottom_up_messages(bottom_idx, top_idx) = bottom_energy;

                if (bottom_energy > max_bottom_energy)
                {
                    second_max_bottom_energy = max_bottom_energy;
                    max_bottom_energy = bottom_energy;
                    max_bottom_energy_top_idx = top_idx;
                }
                else if (bottom_energy > second_max_bottom_energy)
                {
                    second_max_bottom_energy = bottom_energy;
                }
            }

            for (int top_idx = 0; top_idx < top_length; top_idx++)
            {
                if (top_idx == max_bottom_energy_top_idx)
                {
                    bottom_up_messages(bottom_idx, top_idx) -= second_max_bottom_energy;
                }
                else
                {
                    bottom_up_messages(bottom_idx, top_idx) -= max_bottom_energy;
                }
            }

            bottom_clusters[idx] = max_bottom_energy_top_idx;
        });
    }

    void SimpleNN::Learn(double data_weight)
    {
        if (data_weight <= 0.0)
        {
            return;
        }

        Observe();

        int top_length = top_length_;
        int bottom_length = bottom_length_;

        array_view<const double> top_expects = top_expects_;
        array_view<const int> bottom_values = bottom_values_;
        array_view<const double> bottom_recon_expects = bottom_recon_expects_;
        array_view<const int> bottom_clusters = bottom_clusters_;

        array_view<double> top_biases = top_biases_;
        array_view<double> bottom_biases = bottom_biases_;
        array_view<double, 2> neuron_weights = neuron_weights_;

        parallel_for_each(top_expects.extent, [=](index<1> idx) restrict(amp)
        {
            int top_idx = idx[0];

            double top_expect = top_expects[top_idx];

            for (int bottom_idx = 0; bottom_idx < bottom_length; bottom_idx++)
            {
                int bottom_value = bottom_values[bottom_idx];
                double bottom_recon_expect = bottom_recon_expects[bottom_idx];

                double gradient = top_expect * (bottom_value - bottom_recon_expect);

                neuron_weights(top_idx, bottom_idx) += gradient * data_weight;
            }

            // special logic for top bias
            double& top_bias = top_biases[top_idx];
            double top_bias_expect = CalcActivationProb(top_bias);

            top_bias += (top_expect - top_bias_expect) * data_weight;
        });

        parallel_for_each(bottom_clusters.extent, [=](index<1> idx) restrict(amp)
        {
            int bottom_idx = idx[0];

            int bottom_value = bottom_values[bottom_idx];
            double bottom_recon_expect = bottom_recon_expects[bottom_idx];
            int bottom_cluster = bottom_clusters[bottom_idx];

            for (int top_idx = 0; top_idx < top_length; top_idx++)
            {
                double top_expect = top_expects[top_idx];
                double& neuron_weight = neuron_weights(top_idx, bottom_idx);

                if (top_idx == bottom_cluster)
                {
                    double gradient = top_expect * (bottom_value - bottom_recon_expect);
                    neuron_weight += gradient * data_weight;
                }
                else
                {
                    double gradient = top_expect * -neuron_weight;
                    neuron_weight += gradient * data_weight;
                }
            }

            // special logic for bottom bias
            double& bottom_bias = bottom_biases[bottom_idx];
            double bottom_bias_expect = CalcActivationProb(bottom_bias);

            bottom_bias += (bottom_value - bottom_bias_expect) * data_weight;
        });
    }

    double SimpleNN::CalcReconError() const
    {
        array_view<float> recon_error(1);
        recon_error(0) = 0.0f;

        array_view<const int> bottom_values = bottom_values_;
        array_view<const double> bottom_recon_expects = bottom_recon_expects_;

        parallel_for_each(bottom_values.extent, [=](index<1> idx) restrict(amp)
        {
            float diff = (float)(bottom_values[idx] - bottom_recon_expects[idx]);
            atomic_fetch_add(&recon_error(0), diff * diff);
        });

        return sqrt(recon_error(0));
    }

    void SimpleNN::Observe()
    {
        array_view<double> top_expects = top_expects_;
        array_view<int> top_values = top_values_;
        top_expects.discard_data();
        top_values.discard_data();

        array_view<double> bottom_recon_expects = bottom_recon_expects_;
        bottom_recon_expects.discard_data();

        array_view<const double> top_energies = top_energies_;
        array_view<const int> bottom_clusters = bottom_clusters_;
        array_view<const double> bottom_biases = bottom_biases_;
        array_view<const double, 2> neuron_weights = neuron_weights_;

        auto& top_rand = top_rand_;

        parallel_for_each(top_values.extent, [=](index<1> idx) restrict(amp)
        {
            double top_expect = CalcActivationProb(top_energies[idx]);
            top_expects[idx] = top_expect;
            top_values[idx] = top_rand[idx].next_single() <= top_expect ? 1 : 0;
        });

        parallel_for_each(bottom_recon_expects.extent, [=](index<1> idx) restrict(amp)
        {
            int bottom_idx = idx[0];
            int bottom_cluster = bottom_clusters[idx];

            if (bottom_cluster == -1)
            {
                // this bottom neuron is explained by its own bias
                bottom_recon_expects[idx] = CalcActivationProb(bottom_biases[idx]);
            }
            else
            {
                // this bottom neuron is explained by top neuron indexed by bottom_cluster
                double top_expect = top_expects[bottom_cluster];
                bottom_recon_expects[idx] =
                    top_expect * CalcActivationProb(neuron_weights(bottom_cluster, bottom_idx)) +
                    (1.0 - top_expect) * 0.5;
            }
        });
    }

    void SimpleNN::Dump(const string& folder, const string& tag) const
    {
        bitmap_image image;

        const int block_size = 2;

        ofstream ofs;

        ofs.open(folder + "\\neuron_weights_" + tag + ".txt");
        ofs << std::fixed;
        ofs.precision(6);

        double max_value = numeric_limits<double>::lowest();

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

        ofs << "[top biases]" << endl;
        for (int top_idx = 0; top_idx < top_length_; top_idx++)
        {
            ofs << top_biases_[top_idx] << "\t";
        }

        ofs << endl << endl;

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
            image.set_region(0, ((top_idx + 1) * (16 + 1) + 16) * (block_size + 1),
                16 * (block_size + 1), 2, static_cast<unsigned char>(127));
        }

        image.save_image(folder + "\\recoginzation_neuron_weights_" + tag + ".bmp");

        image.setwidth_height(top_length_ * (block_size + 1), bottom_length_ * (block_size + 1), true);

        for (int bottom_idx = 0; bottom_idx < bottom_length_; bottom_idx++)
        {
            for (int top_idx = 0; top_idx < top_length_; top_idx++)
            {
                auto weight = neuron_weights_(top_idx, bottom_idx);

                image.set_region(top_idx * (block_size + 1), bottom_idx * (block_size + 1), block_size, block_size,
                    weight > 0 ? bitmap_image::color_plane::green_plane : bitmap_image::color_plane::red_plane,
                    static_cast<unsigned char>(abs(weight) / max_value * 255.0));
            }
        }

        image.save_image(folder + "\\generation_neuron_weights_" + tag + ".bmp");

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
        ofs << "[bottom clusters]" << endl;
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                auto value = bottom_clusters_[i * 16 + j];
                ofs << value << "\t";
            }
            ofs << endl;
        }

        ofs << endl;
        ofs << std::fixed;
        ofs.precision(6);

        ofs << "[bottom recon expects]" << endl;
        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                auto value = bottom_recon_expects_[i * 16 + j];
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

        ofs << endl;
        ofs << std::fixed;
        ofs.precision(6);

        ofs << "[top expects]" << endl;
        for (int top_idx = 0; top_idx < top_length_; top_idx++)
        {
            ofs << top_expects_[top_idx] << "\t";
        }

        ofs << endl;

        ofs << "[top energies]" << endl;
        for (int top_idx = 0; top_idx < top_length_; top_idx++)
        {
            ofs << top_energies_[top_idx] << "\t";
        }
        ofs.close();
    }
}