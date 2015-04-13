#include "SwarmNeuralNetwork.h"

#include <random>
#include <amp_math.h>

#include "AmpUtility.h"

namespace deep_learning_lib
{
    using std::to_string;
    using std::vector;
    using std::default_random_engine;
    using std::normal_distribution;
    using std::uniform_int_distribution;
    using std::numeric_limits;

    using concurrency::array_view;
    using concurrency::index;
    using concurrency::extent;
    using concurrency::parallel_for_each;

    using concurrency::precise_math::log;
    using concurrency::precise_math::exp;
    using concurrency::precise_math::pow;
    using concurrency::precise_math::fabs;
    using concurrency::precise_math::fmin;

    SwarmNN::SwarmNN(int bottom_length, int top_length, unsigned int seed)
        : bottom_length_(bottom_length),
        top_length_(top_length),
        neuron_weights_(bottom_length, top_length),
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
                double neuron_weight = neuron_weights(bottom_idx, top_idx);

                messsage += log(exp(bottom_bias) + 1.0) + neuron_weight * bottom_values[bottom_idx] 
                    - log(exp(neuron_weight + bottom_bias) + 1.0);
            }

            double top_expect = CalcActivationProb(messsage);

            double top_value = top_rand[idx].next_single() <= top_expect ? 1 : 0;
            top_values[idx] = top_value;

            if (data_weight > 0)
            {

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
                message += neuron_weights(bottom_idx, top_idx) * top_values[top_idx];
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
                    // only activated top neuron can influence bottom neuron
                    if (top_values[top_idx] == 1)
                    {
                        neuron_weights(bottom_idx, top_idx) += (bottom_value - bottom_recon_expect) * data_weight;
                    }
                }

                bottom_bias += (bottom_value - bottom_innate_expect) * data_weight;
            }
        });
    }
}