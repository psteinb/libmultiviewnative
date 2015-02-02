#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE PLAN_STORE_OF_STRING
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iterator>

#include "plan_store.h"


BOOST_FIXTURE_TEST_SUITE( store_minimal_api , multiviewnative::default_3D_fixture )
BOOST_AUTO_TEST_CASE( default_constructs  )
{

  multiviewnative::plan_store<std::string> foo;
  BOOST_CHECK(foo.empty()==true);
  
}

BOOST_AUTO_TEST_CASE( add_item  )
{

  multiviewnative::shape_t any(3,128);
  std::stringstream local("");

  std::copy(any.begin(), any.end(), std::ostream_iterator<unsigned>(local,"x"));
  std::string any_str = local.str();
  
  multiviewnative::plan_store<std::string> foo;
  foo.add(any, any_str);
  BOOST_CHECK(foo.empty()==false);  

}

BOOST_AUTO_TEST_CASE( add_correct_item  )
{

  multiviewnative::shape_t cube(3,128);
  std::stringstream local("");
  std::copy(cube.begin(), cube.end(), std::ostream_iterator<unsigned>(local," "));
  
  std::string cube_str = local.str();
  
  multiviewnative::plan_store<std::string> foo;
  foo.add(cube, cube_str);
  BOOST_CHECK(foo.empty()==false);  
  
  std::string value = foo.get(cube);
  BOOST_CHECK_MESSAGE(std::count(value.begin(), value.end(), '1')==3, "didn't find 1 three times in " << value);  
  BOOST_CHECK_MESSAGE(std::count(value.begin(), value.end(), '2')==3, "didn't find 2 three times in " << value);  
  BOOST_CHECK_MESSAGE(std::count(value.begin(), value.end(), '8')==3, "didn't find 8 three times in " << value);  
}

BOOST_AUTO_TEST_CASE( add_correct_item_through_boolean  )
{

  multiviewnative::shape_t cube(3,128);
  multiviewnative::shape_t big(3,512);
  multiviewnative::shape_t arb(3,42);
  std::stringstream local("");
  std::copy(cube.begin(), cube.end(), std::ostream_iterator<unsigned>(local," "));
  std::string cube_str = local.str();
  local.str("");
  std::copy(big.begin(), big.end(), std::ostream_iterator<unsigned>(local," "));
  std::string big_str = local.str();
  
  multiviewnative::plan_store<std::string> foo;
  foo.add(cube, cube_str);
  foo.add(big, big_str);  
  BOOST_CHECK(foo.has_key(cube));  
  BOOST_CHECK(foo.has_key(big));  
  BOOST_CHECK(!foo.has_key(arb));  
  
}

BOOST_AUTO_TEST_CASE( mutable_iterators  )
{

  multiviewnative::shape_t cube(3,128);
  multiviewnative::shape_t big(3,512);
  multiviewnative::shape_t arb(3,42);

  std::stringstream local("");
  std::copy(cube.begin(), cube.end(), std::ostream_iterator<unsigned>(local," "));
  std::string cube_str = local.str();

  local.str("");
  std::copy(big.begin(), big.end(), std::ostream_iterator<unsigned>(local," "));
  std::string big_str = local.str();

  local.str("");
  std::copy(arb.begin(), arb.end(), std::ostream_iterator<unsigned>(local," "));
  std::string arb_str = local.str();

  multiviewnative::plan_store<std::string> foo;
  foo.add(cube, cube_str);
  foo.add(big, big_str);

  multiviewnative::plan_store<std::string>::map_iter_t begin = foo.begin();
  for(; begin!=foo.end();++begin){
    if((begin->first) == cube)
      begin->second += " foobar ";
  }
  
  BOOST_CHECK(foo.has_key(cube));  
  BOOST_CHECK(foo.get(cube).find("foobar")!=std::string::npos);  
  BOOST_CHECK_NE(foo.get(cube), cube_str);  

}

// BOOST_AUTO_TEST_CASE( mutable_iterators_to_pointers  )
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

//   multiviewnative::plan_store<std::string*> foo;
//   foo.add(cube, new std::string(cube_str));
//   foo.add(big, new std::string(big_str));

//   multiviewnative::plan_store<std::string>::map_iter_t begin = foo.begin();
//   for(; begin!=foo.end();++begin){
//     delete begin->second;
//     begin->second = 0;
//   }
  
//   BOOST_CHECK(foo.has_key(cube));  


// }
BOOST_AUTO_TEST_SUITE_END()
