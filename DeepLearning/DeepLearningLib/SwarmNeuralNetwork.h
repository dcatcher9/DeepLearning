#pragma once
#include <amp.h>
#include <vector>

// for random number generator on GPU
#include "amp_tinymt_rng.h"

namespace deep_learning_lib
{
    class SwarmNN
    {
    public:
        explicit SwarmNN(int bottom_length, int top_length, unsigned int seed = 0);
        SwarmNN(const SwarmNN&) = delete;

        double Feed(const std::vector<int>& input_data, double data_weight = 0);

    private:
        void RandomizeParams(unsigned int seed);

        void PassUp(double data_weight = 0);
        void PassDown(double data_weight = 0);

    private:
        int bottom_length_;
        int top_length_;

        // model parameters to learn
        // [bottom, top]
        concurrency::array_view<double, 2> neuron_weights_;
        // indicates bottom neuron innate activation expect
        concurrency::array_view<double> bottom_biases_;

        // current data
        concurrency::array_view<int> bottom_values_;
        concurrency::array_view<int> top_values_;
        concurrency::array_view<int> bottom_recon_values_;

        //
        tinymt_collection<1> bottom_rand_;
        tinymt_collection<1> top_rand_;
    };
}