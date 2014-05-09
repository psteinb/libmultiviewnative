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
  fill_workspace(local_ref, input);
  image_stack input_psi = *local_guesses.psi(0);

  //test
  cpu_timer timer;
  for(int i = 0;i < 10;++i){
    inplace_cpu_deconvolve_iteration(input_psi.data(), input, boost::thread::hardware_concurrency(), local_guesses.lambda_, local_guesses.minValue_);
  }
  cpu_times duration = timer.elapsed();

  float sum_received = std::accumulate(input_psi.data(),input_psi.data() + input_psi.num_elements(),0.f);
  float sum_expected = std::accumulate(local_guesses.psi(9)->data(),local_guesses.psi(9)->data() + local_guesses.psi(9)->num_elements(),0.f);

  BOOST_REQUIRE_CLOSE(sum_expected, sum_received, 0.001);

  //check norms
  float l2norm = multiviewnative::l2norm(input_psi.data(), local_guesses.psi(9)->data(), input_psi.num_elements());
  
  std::cout << boost::thread::hardware_concurrency() << " threads: l2norm " << l2norm << ", rel difference of sums: " << std::fabs(sum_received-sum_expected)*100./sum_expected << " %\n";
  //tear-down
  delete [] input.data_;
}

BOOST_AUTO_TEST_SUITE_END()













