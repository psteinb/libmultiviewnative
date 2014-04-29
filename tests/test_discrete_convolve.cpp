#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE INDEPENDENT_CPU_CONVOLVE
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include "test_algorithms.hpp"

template <unsigned divisor = 2>
struct divide_by_{

  float operator()(const float& _in){
    return _in/divisor;
  }

};

BOOST_FIXTURE_TEST_SUITE( convolution_works, multiviewnative::default_3D_fixture )

BOOST_AUTO_TEST_CASE( identity_convolve )
{

  using namespace multiviewnative;

  multiviewnative::image_stack result = image_;
  std::fill(result.data(), result.data() + result.num_elements(), 0.f);
  
  std::vector<int> offsets = kernel_dims_;
  std::transform(offsets.begin(),offsets.end(),offsets.begin(), divide_by_<2>());

  multiviewnative::convolve(image_, identity_kernel_, 
			    result,
			    offsets);

  float sum_expected = multiviewnative::sum_from_offset(image_,offsets);
  float sum_received = multiviewnative::sum_from_offset(result,offsets);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .0001f);

}

BOOST_AUTO_TEST_CASE( horizont_convolve )
{

  using namespace multiviewnative;

  multiviewnative::image_stack result = image_;
  std::fill(result.data(), result.data() + result.num_elements(), 0.f);
  
  std::vector<int> offsets = kernel_dims_;
  std::transform(offsets.begin(),offsets.end(),offsets.begin(), divide_by_<2>());

  multiviewnative::convolve(image_, horizont_kernel_, 
			    result,
			    offsets);

  float sum_expected = multiviewnative::sum_from_offset(image_folded_by_horizontal_,offsets);
  float sum_received = multiviewnative::sum_from_offset(result,offsets);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .0001f);

}

BOOST_AUTO_TEST_CASE( vertical_convolve )
{

  using namespace multiviewnative;

  multiviewnative::image_stack result = image_;
  std::fill(result.data(), result.data() + result.num_elements(), 0.f);
  
  std::vector<int> offsets = kernel_dims_;
  std::transform(offsets.begin(),offsets.end(),offsets.begin(), divide_by_<2>());

  multiviewnative::convolve(image_, vertical_kernel_, 
			    result,
			    offsets);

  float sum_expected = multiviewnative::sum_from_offset(image_folded_by_vertical_,offsets);
  float sum_received = multiviewnative::sum_from_offset(result,offsets);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .0001f);

}

BOOST_AUTO_TEST_CASE( all1_convolve )
{

  using namespace multiviewnative;

  multiviewnative::image_stack result = image_;
  std::fill(result.data(), result.data() + result.num_elements(), 0.f);
  
  std::vector<int> offsets = kernel_dims_;
  std::transform(offsets.begin(),offsets.end(),offsets.begin(), divide_by_<2>());

  multiviewnative::convolve(image_, all1_kernel_, 
			    result,
			    offsets);

  float sum_expected = multiviewnative::sum_from_offset(image_folded_by_all1_,offsets);
  float sum_received = multiviewnative::sum_from_offset(result,offsets);

  BOOST_CHECK_CLOSE(sum_expected, sum_received, .0001f);

}
BOOST_AUTO_TEST_SUITE_END()
