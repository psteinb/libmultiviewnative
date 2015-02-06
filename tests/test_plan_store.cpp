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

typedef boost::multi_array<float,3, fftw_allocator<float> >    fftw_image_stack;

BOOST_FIXTURE_TEST_SUITE( store_minimal_api , multiviewnative::default_3D_fixture )
BOOST_AUTO_TEST_CASE( default_constructs  )
{


  BOOST_CHECK(multiviewnative::plan_store<float>::get()->empty()==true);
  
}

BOOST_AUTO_TEST_CASE( add_item  )
{

  multiviewnative::shape_t any(image_dims_.begin(), image_dims_.end());
  
  
  fftw_image_stack output = image_;
  unsigned last_dim = 2*((image_.shape()[2]/2) + 1);
  
  output.resize(boost::extents[image_.shape()[0]][image_.shape()[1]][last_dim]);
  
  multiviewnative::plan_store<float>::get()->add(any, 
						 output.data(), 
						 reinterpret_cast<multiviewnative::plan_store<float>::complex_t*>(output.data()));
  
  BOOST_CHECK(multiviewnative::plan_store<float>::get()->empty()!=true);
}

BOOST_AUTO_TEST_CASE( add_correct_item  )
{

  
  multiviewnative::shape_t cube(image_dims_.begin(), image_dims_.end());
  
  multiviewnative::plan_store<float>::get()->clear();
  BOOST_CHECK_MESSAGE(multiviewnative::plan_store<float>::get()->empty()==true, "not empty ! size = " << multiviewnative::plan_store<float>::get()->size());
  

  fftw_image_stack output = image_;
  unsigned last_dim = 2*((image_.shape()[2]/2) + 1);
  output.resize(boost::extents[image_.shape()[0]][image_.shape()[1]][last_dim]);

  multiviewnative::plan_store<float>::get()->add(cube, 
						 output.data(), 
						 reinterpret_cast<multiviewnative::plan_store<float>::complex_t*>(output.data()));
  
  multiviewnative::plan_store<float>::plan_t* result = 0;
  
  result = multiviewnative::plan_store<float>::get()->get_forward(cube);
  BOOST_CHECK(result != 0);  
  result = 0;
  result = multiviewnative::plan_store<float>::get()->get_backward(cube);
  BOOST_CHECK(result != 0);  
}

BOOST_AUTO_TEST_CASE( add_correct_item_through_boolean  )
{
  
  multiviewnative::shape_t cube(3,8);
  multiviewnative::shape_t big(3,9);
  multiviewnative::shape_t arb(3,42);
  
  fftw_image_stack output = image_;
  unsigned last_dim = 2*((image_.shape()[2]/2) + 1);
  output.resize(boost::extents[image_.shape()[0]][image_.shape()[1]][last_dim]);
  multiviewnative::plan_store<float>::get()->clear();
  multiviewnative::plan_store<float>::get()->add(cube, output.data(), 
						 reinterpret_cast<multiviewnative::plan_store<float>::complex_t*>(output.data()));

  last_dim = 2*((big[2]/2) + 1);
  output.resize(boost::extents[big[0]][big[1]][last_dim]);
  multiviewnative::plan_store<float>::get()->add(big, output.data(), 
						 reinterpret_cast<multiviewnative::plan_store<float>::complex_t*>(output.data()));  
  BOOST_CHECK(  multiviewnative::plan_store<float>::get()->has_key(cube));  
  BOOST_CHECK(  multiviewnative::plan_store<float>::get()->has_key(big));  
  BOOST_CHECK( !multiviewnative::plan_store<float>::get()->has_key(arb));  
  
}

// BOOST_AUTO_TEST_CASE( mutable_iterators  )
// {

//   multiviewnative::shape_t cube(3,128);
//   multiviewnative::shape_t big(3,512);
//   multiviewnative::shape_t arb(3,42);

//   std::stringstream local("");
//   std::copy(cube.begin(), cube.end(), std::ostream_iterator<unsigned>(local," "));
//   std::string cube_str = local.str();

//   local.str("");
//   std::copy(big.begin(), big.end(), std::ostream_iterator<unsigned>(local," "));
//   std::string big_str = local.str();

//   local.str("");
//   std::copy(arb.begin(), arb.end(), std::ostream_iterator<unsigned>(local," "));
//   std::string arb_str = local.str();

//   multiviewnative::plan_store<std::string> foo;
//   foo.add(cube, cube_str);
//   foo.add(big, big_str);

//   multiviewnative::plan_store<std::string>::map_iter_t begin = foo.begin();
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

// //   multiviewnative::shape_t cube(3,128);
// //   multiviewnative::shape_t big(3,512);
// //   multiviewnative::shape_t arb(3,42);

// //   std::stringstream local("");
// //   std::copy(cube.begin(), cube.end(), std::ostream_iterator<unsigned>(local," "));
// //   std::string cube_str = local.str();

// //   local.str("");
// //   std::copy(big.begin(), big.end(), std::ostream_iterator<unsigned>(local," "));
// //   std::string big_str = local.str();

// //   local.str("");
// //   std::copy(arb.begin(), arb.end(), std::ostream_iterator<unsigned>(local," "));
// //   std::string arb_str = local.str();

// //   multiviewnative::plan_store<std::string*> foo;
// //   foo.add(cube, new std::string(cube_str));
// //   foo.add(big, new std::string(big_str));

// //   multiviewnative::plan_store<std::string>::map_iter_t begin = foo.begin();
// //   for(; begin!=foo.end();++begin){
// //     delete begin->second;
// //     begin->second = 0;
// //   }
  
// //   BOOST_CHECK(foo.has_key(cube));  


// // }
BOOST_AUTO_TEST_SUITE_END()
