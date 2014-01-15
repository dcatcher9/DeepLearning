/*----------------------------------------------------------------------------
 * Copyright (c) Microsoft Corp. 
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not 
 * use this file except in compliance with the License.  You may obtain a copy 
 * of the License at http://www.apache.org/licenses/LICENSE-2.0  
 *
 * THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY 
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED 
 * WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE, 
 * MERCHANTABLITY OR NON-INFRINGEMENT. 
 *
 * See the Apache Version 2.0 License for specific language governing 
 * permissions and limitations under the License.
 *
 * -----------------------------------------------------------------------------------
 * 
 * File: main.cpp
 * 
 * This file is a sample demonstrating usage of C++ AMP tinyMT RNG and Sobol Quasi RNG
 *------------------------------------------------------------------------------------- */

#include <amp.h>
#include <iostream>
#include "amp_tinymt_rng.h"
#include "amp_sobol_rng.h"

using namespace concurrency;

void tinymt_test1(FILE *ofile, int seed)
{
    std::cout << "TinyMT   Usage 1: state initialized OUTSIDE kernel (seed=" << seed << ")" << std::endl;

    const int rank = 2;
    extent<rank> e_size(100, 100);
    tinymt_collection<rank> myrand(e_size, seed);
    array<float, rank> rand_out_data(e_size);

    parallel_for_each(e_size, [=, &rand_out_data] (index<2> idx) restrict(amp)
    {
        auto t = myrand[idx];

        // calling below function in loop will give more numbers
        rand_out_data[idx] =  t.next_single();
    });

    std::vector<float> ref_data(e_size.size());
    copy(rand_out_data, ref_data.begin());

    for(unsigned i = 0; i < ref_data.size(); i++)
        fprintf_s(ofile, "%lf \n", ref_data[i]);
}

void tinymt_test2(FILE *ofile)
{
    std::cout << "TinyMT   Usage 2: state initialized INSIDE kernel" << std::endl;

    const int seed = 5489;
    const int rank = 2;
    extent<rank> e_size(10, 20);
    tinymt_collection<rank> myrand(e_size);

    array<unsigned, rank> rand_out_data(e_size);

    parallel_for_each(e_size, [=, &rand_out_data] (index<2> idx) restrict(amp)
    {
        // for tiled implementation use global index
        auto t = myrand[idx];

        // Set seed
        t.initialize(seed);

        // calling below function in loop will give more numbers
        rand_out_data[idx] = t.next_uint();
    });

    std::vector<unsigned> ref_data(e_size.size());
    copy(rand_out_data, ref_data.begin());

    for(unsigned i = 0; i < ref_data.size(); i++)
        fprintf_s(ofile, "%li \n", ref_data[i]);
}

void tinymt_test3(FILE *ofile)
{
    std::cout << "TinyMT   Usage 3: using tiled p_f_e" << std::endl;

    static const int rank = 3;
    extent<rank> e_size(12, 12, 3);
    tinymt_collection<rank> myrand(e_size);

    array<float, rank> rand_out_data(e_size);

    parallel_for_each(e_size.tile<3, 3, 3>(), [=, &rand_out_data] (tiled_index<3, 3, 3> idx) restrict(amp)
    {
        // for tiled implementation use global index
        auto t = myrand[idx.global];

        // calling below function in loop will give more numbers
        rand_out_data[idx] = t.next_single12();
    });

    std::vector<float> ref_data(e_size.size());
    copy(rand_out_data, ref_data.begin());

    for(unsigned i = 0; i < ref_data.size(); i++)
        fprintf_s(ofile, "%i \n", ref_data[i]);
}  

void sobol_rng_test1(FILE *ofile, unsigned skipahead)
{
    std::cout << "SobolRNG Usage 1: generating a 2-dimension Sobol sequence" << std::endl;

    static const int rank = 1;
    static const unsigned dimensions = 2;

    extent<rank> e_size(10000);
    sobol_rng_collection<sobol_rng<dimensions>, rank> sc_rng(e_size, skipahead);

    typedef sobol_rng<dimensions>::sobol_number<float> sobol_float_number;
    array<sobol_float_number, rank> rand_out_data(e_size);

    // Each thread generates one multi-dimension sobol_float_number 
    parallel_for_each(e_size, [=, &rand_out_data] (index<rank> idx) restrict(amp)
    {
        // Get the sobol RNG 
        auto rng = sc_rng[idx];

        // Skip ahead to the right position
        rng.skip(sc_rng.direction_numbers(), idx[0]);

        // Get the sobol number 
        sobol_float_number& sf_num = rand_out_data[idx];
        for (int i=1; i<=dimensions; i++)
        {
            sf_num[i-1] = rng.get_single(i);
        }
    });

    // Read the sobol sequence back to host
    std::vector<sobol_float_number> ref_data(e_size.size());
    copy(rand_out_data, ref_data.begin());

    // Write to a data file
    for(unsigned i=0; i<ref_data.size(); i++)
    {
        sobol_float_number& sf_num = ref_data[i]; 

        for (unsigned j=0; j<dimensions; j++)
        {
            fprintf_s(ofile, "%lf ", sf_num[j]);
        }
        fprintf_s(ofile, "\n");
    }
}

void sobol_rng_test2(FILE *ofile)
{
    std::cout << "SobolRNG Usage 2: generating a 100-dimension Sobol sequence using tiled p_f_e" << std::endl;

    static const int rank = 2;
    static const unsigned dimensions = 100;

    extent<rank> e_size(100,100);
    sobol_rng_collection<sobol_rng<dimensions>, rank> sc_rng(e_size);

    typedef sobol_rng<dimensions>::sobol_number<unsigned> sobol_uint_number;
    array<sobol_uint_number, rank> rand_out_data(e_size);

    // Each thread generates one multi-dimension sobol_uint_number 
    parallel_for_each(e_size.tile<10,10>(), [=, &rand_out_data] (tiled_index<10,10> tidx) restrict(amp)
    {
        // Get the sobol RNG. For tiled implementation use global index
        auto rng = sc_rng[tidx.global];

        // Skip ahead to the right position
        rng.skip(sc_rng.direction_numbers(), tidx.global[0]*e_size[1]+tidx.global[1]);

        // Get the sobol number 
        sobol_uint_number& sui_num = rand_out_data[tidx];
        for (int i=1; i<=dimensions; i++)
        {
            sui_num[i-1] = rng.get_uint(i);
        }
    });

    // Read the sobol sequence back to host
    std::vector<sobol_uint_number> ref_data(e_size.size());
    copy(rand_out_data, ref_data.begin());

    // Write to a data file
    for(unsigned i=0; i<ref_data.size(); i++)
    {
        sobol_uint_number& sui_num = ref_data[i]; 

        for (unsigned j=0; j<dimensions; j++)
        {
            fprintf_s(ofile, "%li ", sui_num[j]);
        }
        fprintf_s(ofile, "\n");
    }
}

void tinymt_test()
{
    FILE *ofile = NULL;
    if (fopen_s(&ofile, "tinymt_data.txt", "wt") != 0)
    {
        printf("open file tinymt_data.txt file\n");
    }

    int seed = 5489;
    tinymt_test1(ofile, seed);
    seed = 7859;
    tinymt_test1(ofile, seed);
    tinymt_test2(ofile);
    tinymt_test3(ofile);

    fclose(ofile);
}

void sobol_rng_test()
{
    FILE *ofile = NULL;
    if (fopen_s(&ofile, "sobol_data.txt", "wt") != 0)
    {
        printf("open file sobol_data.txt file\n");
    }

    const unsigned skipahead = 5489;
    sobol_rng_test1(ofile, skipahead);
    sobol_rng_test2(ofile);

    fclose(ofile);
}

int main()
{
    accelerator default_device;
    std::wcout << L"Using device : " << default_device.get_description() << std::endl;
    if (default_device == accelerator(accelerator::direct3d_ref))
        std::cout << "WARNING!! Running on very slow emulator! Only use this accelerator for debugging." << std::endl;

    tinymt_test();
    sobol_rng_test();

    return 0;
}
