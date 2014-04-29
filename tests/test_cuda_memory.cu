#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE CUDA_MEMORY_SUITE
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include "cuda_memory.cuh"

template <typename T>
__global__ void add_1(T* _container, unsigned _size){

  unsigned global = threadIdx.x + blockIdx.x*blockDim.x;

  if(global < _size)
    _container[global] += 1.f;
  
}

 
BOOST_FIXTURE_TEST_SUITE( constructor_suite, multiviewnative::default_3D_fixture )
   
BOOST_AUTO_TEST_CASE( instantiate )
{
  multiviewnative::stack_on_device<multiviewnative::image_stack> nullary;
  BOOST_CHECK_MESSAGE(nullary.host_stack_ == 0, "strange, stack_on_device has host memory loaded although it was constructed without any argument");
}

BOOST_AUTO_TEST_CASE( by_copy )
{
  multiviewnative::stack_on_device<multiviewnative::image_stack> simple(image_);
  BOOST_CHECK_MESSAGE(simple.host_stack_ != 0, "stack_on_device has no host memory loaded");
  BOOST_CHECK_MESSAGE(simple.device_stack_ptr_ != 0, "stack_on_device has no device memory loaded");
}


BOOST_AUTO_TEST_CASE( by_operator_equal )
{
  multiviewnative::stack_on_device<multiviewnative::image_stack> simple = (image_);
  BOOST_CHECK_MESSAGE(simple.host_stack_ != 0, "stack_on_device has no host memory loaded");
  BOOST_CHECK_MESSAGE(simple.device_stack_ptr_ != 0, "stack_on_device has no device memory loaded");
}
BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE( offload_suite, multiviewnative::default_3D_fixture )
   
BOOST_AUTO_TEST_CASE( instantiate_add_1 )
{
  using namespace multiviewnative;

  unsigned sum_original = std::accumulate(image_.data(), image_.data() + image_.num_elements(), 0.);
  stack_on_device<image_stack> simple = image_;
  
  simple.push_to_device<synch<image_stack> >();

  dim3 blocks = image_.shape()[2];
  dim3 threads = image_.shape()[1]*image_.shape()[0];
  std::cout << "launching add1: " << blocks.x << "-" << threads.x << "\n";
  add_1<<<blocks,threads>>>(simple.device_stack_ptr_, image_.num_elements());

  simple.pull_from_device<synch<image_stack> >();

  unsigned sum = std::accumulate(image_.data(), image_.data() + image_.num_elements(), 0.);
  
  BOOST_CHECK_NE(sum_original, sum);
  BOOST_CHECK_EQUAL(sum_original+image_.num_elements(), sum);
}
BOOST_AUTO_TEST_SUITE_END()


















