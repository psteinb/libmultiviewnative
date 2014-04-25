#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE CPU_DECONVOLVE
#include "boost/test/unit_test.hpp"
#include "tiff_fixtures.hpp"
#include "multiviewnative.h"
#include "convert_tiff_fixtures.hpp"

using namespace multiviewnative;

static const ReferenceData reference;
static const first_2_iterations guesses;

BOOST_AUTO_TEST_SUITE( deconvolve_psi0  )
   
BOOST_AUTO_TEST_CASE( reconstruct_anything )
{
  //setup
  ReferenceData local_ref(reference);
  first_2_iterations local_guesses(guesses);
  workspace input;
  input.data_ = 0;
  fill_workspace(local_ref, input);

  image_stack psi = *(reference.views_[0].image());
  double sum_initial = std::accumulate(psi.data(),psi.data() + psi.num_elements(),0.f);
  double avg = sum_initial/psi.num_elements();
  std::fill(psi.data(), psi.data() + psi.num_elements(),float(avg));
  std::cout << "filling psi with " << avg << "\n";

  //test
  inplace_cpu_deconvolve_iteration(psi.data(), input, 1, local_guesses.lambda_, local_guesses.minValue_);
  float sum_received = std::accumulate(psi.data(),psi.data() + psi.num_elements(),0.f);
  float sum_expected = std::accumulate(local_guesses.psi(0)->data(),local_guesses.psi(0)->data() + local_guesses.psi(0)->num_elements(),0.f);

  BOOST_REQUIRE_NE(sum_initial, sum_received);
  try{
    BOOST_REQUIRE_CLOSE(sum_expected, sum_received, 0.001);
  }
  catch (...){
    write_image_stack(psi,"./reconstruct_anything_psi0.tif");
    std::cout << "max(received)["<< psi.num_elements()<<"] " << *std::max_element(psi.data(),psi.data() + psi.num_elements()) 
	      << ", max(expected)["<< local_guesses.psi(0)->num_elements()<<"]: " << *std::max_element(local_guesses.psi(0)->data(),local_guesses.psi(0)->data() + local_guesses.psi(0)->num_elements()) 
	      << "\n";
  }
  //tear-down
  delete [] input.data_;
}

BOOST_AUTO_TEST_SUITE_END()
