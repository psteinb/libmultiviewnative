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

#include "plan_store.cuh"


namespace mvn = multiviewnative;

BOOST_FIXTURE_TEST_SUITE(store_minimal_api, mvn::default_3D_fixture)
BOOST_AUTO_TEST_CASE(default_constructs) {

  BOOST_CHECK(mvn::gpu::plan_store<float>::get()->empty() == true);
}

BOOST_AUTO_TEST_CASE(add_item) {

  mvn::shape_t any(image_dims_.begin(), image_dims_.end());

  
  unsigned last_dim = 2 * ((image_.shape()[2] / 2) + 1);

  image_.resize(boost::extents[image_.shape()[0]][image_.shape()[1]][last_dim]);

  mvn::gpu::plan_store<float>::get()->add(any);

  BOOST_CHECK(mvn::gpu::plan_store<float>::get()->empty() != true);
}

BOOST_AUTO_TEST_CASE(add_correct_item) {

  mvn::shape_t cube(image_dims_.begin(), image_dims_.end());

  mvn::gpu::plan_store<float>::get()->clear();
  BOOST_CHECK_MESSAGE(
      mvn::gpu::plan_store<float>::get()->empty() == true,
      "not empty ! size = " << mvn::gpu::plan_store<float>::get()->size());


  unsigned last_dim = 2 * ((image_.shape()[2] / 2) + 1);
  image_.resize(boost::extents[image_.shape()[0]][image_.shape()[1]][last_dim]);

  mvn::gpu::plan_store<float>::get()->add(cube);

  mvn::gpu::plan_store<float>::plan_t* result = 0;

  result = mvn::gpu::plan_store<float>::get()->get_forward(cube);
  BOOST_CHECK(result != 0);
  result = 0;
  result = mvn::gpu::plan_store<float>::get()->get_backward(cube);
  BOOST_CHECK(result != 0);
}

BOOST_AUTO_TEST_CASE(add_correct_item_through_boolean) {

  mvn::shape_t cube(3, 8);
  mvn::shape_t big(3, 9);
  mvn::shape_t arb(3, 42);


  unsigned last_dim = 2 * ((image_.shape()[2] / 2) + 1);
  image_.resize(boost::extents[image_.shape()[0]][image_.shape()[1]][last_dim]);
  mvn::gpu::plan_store<float>::get()->clear();
  mvn::gpu::plan_store<float>::get()->add(cube);

  last_dim = 2 * ((big[2] / 2) + 1);
  image_.resize(boost::extents[big[0]][big[1]][last_dim]);
  mvn::gpu::plan_store<float>::get()->add(big);
  BOOST_CHECK(mvn::gpu::plan_store<float>::get()->has_key(cube));
  BOOST_CHECK(mvn::gpu::plan_store<float>::get()->has_key(big));
  BOOST_CHECK(!mvn::gpu::plan_store<float>::get()->has_key(arb));
}



BOOST_AUTO_TEST_SUITE_END()
