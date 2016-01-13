#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE CUDA_MEMORY_SUITE
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include "cuda_memory.cuh"

template <typename T>
__global__ void add_1(T* _container, unsigned _size) {

  unsigned global = threadIdx.x + blockIdx.x * blockDim.x;

  if (global < _size) _container[global] += 1.f;
}

typedef multiviewnative::stack_on_device<multiviewnative::image_stack>
    default_stack_on_device;
typedef multiviewnative::stack_on_device<multiviewnative::image_stack,
                                         multiviewnative::asynch>
    asynch_stack_on_device;
typedef multiviewnative::stack_on_device<
    multiviewnative::image_stack, multiviewnative::synch> synch_stack_on_device;

BOOST_FIXTURE_TEST_SUITE(constructor_suite, multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(instantiate) {
  synch_stack_on_device nullary;
  cudaError_t err = cudaGetLastError();
  BOOST_CHECK_EQUAL(err, cudaSuccess);
}

BOOST_AUTO_TEST_CASE(instantiate_from_number) {
  synch_stack_on_device any_size(256);
  BOOST_CHECK(any_size.device_stack_ptr_ != 0);
  cudaError_t err = cudaGetLastError();
  BOOST_CHECK_EQUAL(err, cudaSuccess);
}

BOOST_AUTO_TEST_CASE(by_from_stack) {
  synch_stack_on_device simple(image_);
  BOOST_CHECK_MESSAGE(simple.device_stack_ptr_ != 0,
                      "stack_on_device has no device memory loaded");
  BOOST_CHECK_EQUAL(
      simple.size_in_byte_,
      image_.num_elements() * sizeof(synch_stack_on_device::value_type));
  cudaError_t err = cudaGetLastError();
  BOOST_CHECK_EQUAL(err, cudaSuccess);
}

BOOST_AUTO_TEST_CASE(by_assigment) {

  default_stack_on_device simple(image_);
  cudaError_t err = cudaGetLastError();
  BOOST_CHECK_EQUAL(err, cudaSuccess);

  { default_stack_on_device simple2 = simple; }

  err = cudaGetLastError();
  BOOST_CHECK_EQUAL(err, cudaSuccess);
}

BOOST_AUTO_TEST_CASE(copy_by_assigment) {

  default_stack_on_device simple(image_);
  default_stack_on_device other(one_);
  cudaError_t err = cudaGetLastError();
  BOOST_CHECK_EQUAL(err, cudaSuccess);

  dim3 blocks = image_.shape()[2];
  dim3 threads = image_.shape()[1] * image_.shape()[0];
  add_1 << <blocks, threads>>>
    (other.device_stack_ptr_, image_.num_elements());

  simple = other;
  simple.pull_from_device(image_folded_by_horizontal_);

  
  
  std::vector<float> all_ones(one_.data(),one_.data()+32);
  for(float & i : all_ones)
    i+=1;
  
  BOOST_CHECK_EQUAL_COLLECTIONS(all_ones.begin(), all_ones.end(),
				image_folded_by_horizontal_.data(),image_folded_by_horizontal_.data()+32);

  other.pull_from_device(image_folded_by_vertical_);
  BOOST_CHECK_EQUAL_COLLECTIONS(all_ones.begin(), all_ones.end(),
				image_folded_by_vertical_.data(),image_folded_by_vertical_.data()+32);
}


BOOST_AUTO_TEST_CASE(by_assigment_from_stack) {
  default_stack_on_device simple = image_;
  BOOST_CHECK_MESSAGE(simple.device_stack_ptr_ != 0,
                      "stack_on_device has no device memory loaded");
  BOOST_CHECK_EQUAL(
      simple.size_in_byte_,
      image_.num_elements() * sizeof(default_stack_on_device::value_type));
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(offload_suite, multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(instantiate_add_1_synched) {
  using namespace multiviewnative;

  unsigned sum_original =
      std::accumulate(image_.data(), image_.data() + image_.num_elements(), 0.);
  synch_stack_on_device simple(image_);

  simple.push_to_device(image_);

  dim3 blocks = image_.shape()[2];
  dim3 threads = image_.shape()[1] * image_.shape()[0];

  add_1 << <blocks, threads>>>
      (simple.device_stack_ptr_, image_.num_elements());

  simple.pull_from_device(image_);

  unsigned sum =
      std::accumulate(image_.data(), image_.data() + image_.num_elements(), 0.);

  BOOST_CHECK_NE(sum_original, sum);
  BOOST_CHECK_EQUAL(sum_original + image_.num_elements(), sum);
}

BOOST_AUTO_TEST_CASE(instantiate_add_1_asynched) {
  using namespace multiviewnative;
  cudaStream_t tstream;
  cudaStreamCreate(&tstream);

  unsigned sum_original =
      std::accumulate(image_.data(), image_.data() + image_.num_elements(), 0.);
  asynch_stack_on_device simple(image_, image_.num_elements(), &tstream);

  dim3 blocks = image_.shape()[2];
  dim3 threads = image_.shape()[1] * image_.shape()[0];

  add_1 << <blocks, threads, 0, tstream>>>
      (simple.device_stack_ptr_, image_.num_elements());

  simple.pull_from_device(image_, &tstream);

  cudaStreamSynchronize(tstream);
  unsigned sum =
      std::accumulate(image_.data(), image_.data() + image_.num_elements(), 0.);

  BOOST_CHECK_NE(sum_original, sum);
  BOOST_CHECK_EQUAL(sum_original + image_.num_elements(), sum);
  cudaStreamDestroy(tstream);
}

BOOST_AUTO_TEST_SUITE_END()

typedef boost::multi_array<float, 3, multiviewnative::pinned_allocator<float> >
    pinned_image_stack;

BOOST_FIXTURE_TEST_SUITE(allocator_suite, multiviewnative::default_3D_fixture)
BOOST_AUTO_TEST_CASE(pinned_allocator_works) {

  std::vector<unsigned> shapes(image_.shape(), image_.shape() + 3);
  pinned_image_stack test(shapes);
  BOOST_CHECK_EQUAL_COLLECTIONS(test.shape(), test.shape() + 3, image_.shape(),
                                image_.shape() + 3);
}

BOOST_AUTO_TEST_CASE(pinned_allocator_copies_non_pinned) {

  pinned_image_stack test = image_;
  BOOST_CHECK_EQUAL_COLLECTIONS(test.shape(), test.shape() + 3, image_.shape(),
                                image_.shape() + 3);
  BOOST_CHECK(test == image_);

  multiviewnative::image_stack returned = test;
  BOOST_CHECK(returned == image_);
}
BOOST_AUTO_TEST_SUITE_END()
