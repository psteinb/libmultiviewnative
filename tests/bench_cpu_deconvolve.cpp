#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE CPU_DECONVOLVE
#include "boost/test/unit_test.hpp"
#include "tiff_fixtures.hpp"
#include "multiviewnative.h"
#include "convert_tiff_fixtures.hpp"

#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"
#include "cpu_kernels.h"
#include "test_algorithms.hpp"

#include <boost/timer/timer.hpp>
#include "boost/thread.hpp"

using namespace multiviewnative;
using boost::timer::cpu_timer;
using boost::timer::cpu_times;
using boost::timer::nanosecond_type;

static const ReferenceData reference;
static const all_iterations guesses;

BOOST_AUTO_TEST_SUITE( bernchmark  )

BOOST_AUTO_TEST_CASE( deconvolve_all_cpus )
{
  //setup
  ReferenceData local_ref(reference);
  all_iterations local_guesses(guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input, local_guesses.lambda_, local_guesses.minValue_);
  input.num_iterations_ = 9;

  image_stack input_psi = *local_guesses.psi(0);
  cpu_times durations[input.num_iterations_];
  float l2norms[input.num_iterations_];
  float sums_received[input.num_iterations_];
  float sums_expected[input.num_iterations_];
  //test
  
  for(int i = 0;i < input.num_iterations_;++i){
    std::cout << i << "/" << input.num_iterations_ << " on "<< boost::thread::hardware_concurrency() << " threads\n";
    cpu_timer timer;
    inplace_cpu_deconvolve_iteration(input_psi.data(), input, boost::thread::hardware_concurrency());
    durations[i] = timer.elapsed();
    l2norms[i] = multiviewnative::l2norm(input_psi.data(), local_guesses.psi(1+i)->data(), input_psi.num_elements());
    sums_received[i] = std::accumulate(input_psi.data(),input_psi.data() + input_psi.num_elements(),0.f);
    sums_expected[i] = std::accumulate(local_guesses.psi(1+i)->data(),local_guesses.psi(1+i)->data() + local_guesses.psi(1+i)->num_elements(),0.f);
    std::cout << boost::thread::hardware_concurrency() << " threads: l2norm " << l2norms[i] << ", rel difference of sums: " << std::fabs(sums_received[i]-sums_expected[i])*100./sums_expected[i] << " % ("<< double(durations[i].system + durations[i].user)/1e6 <<" ms )\n";
  }
  

  BOOST_REQUIRE_CLOSE(sums_expected[9], sums_received[9], 0.001);
  //check norms
  
  double time_ms = 0.f;

  for(int i = 0;i<input.num_iterations_;++i){
    time_ms += double(durations[i].system + durations[i].user)/1e6;
  }
  double mega_pixels_per_sec = input_psi.num_elements()/(time_ms);
  
  std::cout << boost::thread::hardware_concurrency() << " threads: total time " << time_ms << " ms, throughput " << mega_pixels_per_sec << " Mpixel/s\n";
  //tear-down
  delete [] input.data_;
}

BOOST_AUTO_TEST_SUITE_END()













