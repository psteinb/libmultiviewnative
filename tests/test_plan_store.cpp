#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE PLAN_STORE
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iterator>

#include "plan_store.h"

typedef boost::multi_array<float, 3, fftw_allocator<float> > fftw_image_stack;
namespace mvn = multiviewnative;

BOOST_FIXTURE_TEST_SUITE(store_minimal_api, mvn::default_3D_fixture)
BOOST_AUTO_TEST_CASE(default_constructs) {

  BOOST_CHECK(mvn::plan_store<float>::get()->empty() == true);
}

BOOST_AUTO_TEST_CASE(add_item) {

  mvn::shape_t any(image_dims_.begin(), image_dims_.end());

  fftw_image_stack output = image_;
  unsigned last_dim = 2 * ((image_.shape()[2] / 2) + 1);

  output.resize(boost::extents[image_.shape()[0]][image_.shape()[1]][last_dim]);

  mvn::plan_store<float>::get()->add(any);

  BOOST_CHECK(mvn::plan_store<float>::get()->empty() != true);
}

BOOST_AUTO_TEST_CASE(add_correct_item) {

  mvn::shape_t cube(image_dims_.begin(), image_dims_.end());

  mvn::plan_store<float>::get()->clear();
  BOOST_CHECK_MESSAGE(
      mvn::plan_store<float>::get()->empty() == true,
      "not empty ! size = " << mvn::plan_store<float>::get()->size());

  fftw_image_stack output = image_;
  unsigned last_dim = 2 * ((image_.shape()[2] / 2) + 1);
  output.resize(boost::extents[image_.shape()[0]][image_.shape()[1]][last_dim]);

  mvn::plan_store<float>::get()->add(cube);

  mvn::plan_store<float>::plan_t* result = 0;

  result = mvn::plan_store<float>::get()->get_forward(cube);
  BOOST_CHECK(result != 0);
  result = 0;
  result = mvn::plan_store<float>::get()->get_backward(cube);
  BOOST_CHECK(result != 0);
}

BOOST_AUTO_TEST_CASE(add_correct_item_through_boolean) {

  mvn::shape_t cube(3, 8);
  mvn::shape_t big(3, 9);
  mvn::shape_t arb(3, 42);

  fftw_image_stack output = image_;
  unsigned last_dim = 2 * ((image_.shape()[2] / 2) + 1);
  output.resize(boost::extents[image_.shape()[0]][image_.shape()[1]][last_dim]);
  mvn::plan_store<float>::get()->clear();
  mvn::plan_store<float>::get()->add(cube);

  last_dim = 2 * ((big[2] / 2) + 1);
  output.resize(boost::extents[big[0]][big[1]][last_dim]);
  mvn::plan_store<float>::get()->add(big);
  BOOST_CHECK(mvn::plan_store<float>::get()->has_key(cube));
  BOOST_CHECK(mvn::plan_store<float>::get()->has_key(big));
  BOOST_CHECK(!mvn::plan_store<float>::get()->has_key(arb));
}

using namespace mvn;

BOOST_AUTO_TEST_CASE(fft_ifft_unaligned) {
  shape_t real_dims(image_dims_.begin(), image_dims_.end());
  shape_t padded_dims = real_dims;
  padded_dims[2] = (padded_dims[2] / 2 + 1) * 2;

  fftw_image_stack padded_for_fft(padded_dims);
  padded_for_fft[boost::indices[range(0, image_dims_[0])][range(
      0, image_dims_[1])][range(0, image_dims_[2])]] = image_;

  plan_store<float>::get()->clear();
  plan_store<float>::get()->add(real_dims);

  fftw_api_definitions<float>::reuse_plan_r2c(
      *plan_store<float>::get()->get_forward(real_dims), padded_for_fft.data(),
      (fftw_api_definitions<float>::complex_type*)padded_for_fft.data());

  fftw_api_definitions<float>::reuse_plan_c2r(
      *plan_store<float>::get()->get_backward(real_dims),
      (fftw_api_definitions<float>::complex_type*)padded_for_fft.data(),
      padded_for_fft.data());

  fftw_image_stack result = padded_for_fft[boost::indices[range(
      0, image_dims_[0])][range(0, image_dims_[1])][range(0, image_dims_[2])]];
  float fft_size = std::accumulate(real_dims.begin(), real_dims.end(), 1,
                                   std::multiplies<unsigned>());
  std::for_each(result.data(), result.data() + result.num_elements(),
                [&](float& pixel) { pixel /= fft_size; });
  BOOST_CHECK(result == image_);
}

BOOST_AUTO_TEST_CASE(fft_ifft_aligned) {
  typedef plan_store<float, aligned<float> > aligned_plan_store;

  shape_t real_dims(image_dims_.begin(), image_dims_.end());
  shape_t padded_dims = real_dims;
  padded_dims[2] = (padded_dims[2] / 2 + 1) * 2;

  fftw_image_stack padded_for_fft(padded_dims);
  padded_for_fft[boost::indices[range(0, image_dims_[0])][range(
      0, image_dims_[1])][range(0, image_dims_[2])]] = image_;

  aligned_plan_store::get()->clear();
  aligned_plan_store::get()->add(real_dims);

  fftw_api_definitions<float>::reuse_plan_r2c(
      *aligned_plan_store::get()->get_forward(real_dims), padded_for_fft.data(),
      (fftw_api_definitions<float>::complex_type*)padded_for_fft.data());

  fftw_api_definitions<float>::reuse_plan_c2r(
      *aligned_plan_store::get()->get_backward(real_dims),
      (fftw_api_definitions<float>::complex_type*)padded_for_fft.data(),
      padded_for_fft.data());

  fftw_image_stack result = padded_for_fft[boost::indices[range(
      0, image_dims_[0])][range(0, image_dims_[1])][range(0, image_dims_[2])]];
  float fft_size = std::accumulate(real_dims.begin(), real_dims.end(), 1,
                                   std::multiplies<unsigned>());
  std::for_each(result.data(), result.data() + result.num_elements(),
                [&](float& pixel) { pixel /= fft_size; });
  BOOST_CHECK(result == image_);
}
//   mvn::shape_t cube(3,128);
//   mvn::shape_t big(3,512);
//   mvn::shape_t arb(3,42);

//   std::stringstream local("");
//   std::copy(cube.begin(), cube.end(), std::ostream_iterator<unsigned>(local,"
// "));
//   std::string cube_str = local.str();

//   local.str("");
//   std::copy(big.begin(), big.end(), std::ostream_iterator<unsigned>(local,"
// "));
//   std::string big_str = local.str();

//   local.str("");
//   std::copy(arb.begin(), arb.end(), std::ostream_iterator<unsigned>(local,"
// "));
//   std::string arb_str = local.str();

//   mvn::plan_store<std::string> foo;
//   foo.add(cube, cube_str);
//   foo.add(big, big_str);

//   mvn::plan_store<std::string>::map_iter_t begin = foo.begin();
//   for(; begin!=foo.end();++begin){
//     if((begin->first) == cube)
//       begin->second += " foobar ";
//   }

//   BOOST_CHECK(foo.has_key(cube));
//   BOOST_CHECK(foo.get(cube).find("foobar")!=std::string::npos);
//   BOOST_CHECK_NE(foo.get(cube), cube_str);

// }

// // BOOST_AUTO_TEST_CASE( mutable_iterators_to_pointers  )
// // {

// //   mvn::shape_t cube(3,128);
// //   mvn::shape_t big(3,512);
// //   mvn::shape_t arb(3,42);

// //   std::stringstream local("");
// //   std::copy(cube.begin(), cube.end(),
// std::ostream_iterator<unsigned>(local," "));
// //   std::string cube_str = local.str();

// //   local.str("");
// //   std::copy(big.begin(), big.end(),
// std::ostream_iterator<unsigned>(local," "));
// //   std::string big_str = local.str();

// //   local.str("");
// //   std::copy(arb.begin(), arb.end(),
// std::ostream_iterator<unsigned>(local," "));
// //   std::string arb_str = local.str();

// //   mvn::plan_store<std::string*> foo;
// //   foo.add(cube, new std::string(cube_str));
// //   foo.add(big, new std::string(big_str));

// //   mvn::plan_store<std::string>::map_iter_t begin = foo.begin();
// //   for(; begin!=foo.end();++begin){
// //     delete begin->second;
// //     begin->second = 0;
// //   }

// //   BOOST_CHECK(foo.has_key(cube));

// // }
BOOST_AUTO_TEST_SUITE_END()
