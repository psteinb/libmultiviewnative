#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE GPU_KERNELS_IMPL
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <random>
#include <numeric>

#include "multiviewnative.h"
#include "padd_utils.h"

#include "cuda_memory.cuh"
#include "gpu_convolve.cuh"
#include "cpu_kernels.h"

using device_stack = multiviewnative::stack_on_device<multiviewnative::image_stack> ;
using image_stack = multiviewnative::image_stack;
using wrap_around_padding = multiviewnative::zero_padd<image_stack>;
using device_transform = multiviewnative::inplace_3d_transform_on_device<float> ;


BOOST_FIXTURE_TEST_SUITE(divide,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(fixture_divide) {
  
  std::fill(padded_one_.data(), padded_one_.data()+padded_one_.num_elements(),10);
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),5);
  
  device_stack d_padded_one_	(padded_one_);
  device_stack d_padded_image_	(padded_image_);

  dim3 threads(128);
  dim3 blocks(
      largestDivisor(padded_one_.num_elements(), size_t(threads.x)));

  device_divide <<<blocks, threads>>>
    (d_padded_one_.data(), d_padded_image_.data(), padded_one_.num_elements());
  HANDLE_LAST_ERROR();

  d_padded_image_.pull_from_device(padded_image_);
  float sum = std::accumulate(padded_image_.data(),
			      padded_image_.data()+padded_image_.num_elements(),
			      0.f);
  try {
    BOOST_REQUIRE_CLOSE(sum, 2.f*padded_image_.num_elements(), .15);
  }
  catch (...) {
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
              << "result:\n";
    multiviewnative::print_stack(padded_image_);
    std::cout << "\n"
	      << "expexted: all 2.f\n";
  }

}

BOOST_AUTO_TEST_CASE(cube256_divide) {

  std::vector<unsigned> shape(3,256);
  padded_one_.resize(shape);
  padded_image_.resize(shape);
  
  std::fill(padded_one_.data(), padded_one_.data()+padded_one_.num_elements(),1.f);
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),5.f);

  static const float expected_value = 1.f/5.f;
  
  device_stack d_padded_one_	(padded_one_);
  device_stack d_padded_image_	(padded_image_);

  dim3 threads(128);
  dim3 blocks(
      largestDivisor(padded_one_.num_elements(), size_t(threads.x)));

  // padded_image = padded_one / padded_image
  device_divide <<<blocks, threads>>>
    (d_padded_one_.data(), d_padded_image_.data(), padded_one_.num_elements());
  HANDLE_LAST_ERROR();

  image_stack gpu_result = padded_image_;
  d_padded_image_.pull_from_device(gpu_result);
  double sum = std::accumulate(gpu_result.data(),
			       gpu_result.data()+gpu_result.num_elements(),
			       0.);

  for(std::size_t i = 0;i<padded_image_.num_elements();++i)
    BOOST_REQUIRE_EQUAL(expected_value,gpu_result.data()[i]);

  BOOST_REQUIRE_GT(sum,0.);
  
  try {
    BOOST_REQUIRE_CLOSE(sum, .2*(padded_image_.num_elements()), .15);
  }
  catch (...) {
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
              << expected_value << " * " << padded_image_.num_elements() << "\n";
  }


  image_stack cpu_result = padded_image_;
  multiviewnative::cpu::par::compute_quotient(padded_one_.data(),
					      cpu_result.data(),
					      padded_one_.num_elements());
  
  for(std::size_t i = 0;i<padded_image_.num_elements();++i)
    BOOST_REQUIRE_EQUAL(cpu_result.data()[i],gpu_result.data()[i]);
  
}

BOOST_AUTO_TEST_CASE(oddcube256_divide) {

  std::vector<unsigned> shape(3,256);
  shape[1]=255;
  shape[2]=257;
  
  padded_one_.resize(shape);
  padded_image_.resize(shape);
  
  std::fill(padded_one_.data(), padded_one_.data()+padded_one_.num_elements(),1.f);
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),5.f);

  static const float expected_value = 1.f/5.f;
  
  device_stack d_padded_one_	(padded_one_);
  device_stack d_padded_image_	(padded_image_);

  dim3 threads(128);
  dim3 blocks(
      largestDivisor(padded_one_.num_elements(), size_t(threads.x)));

  // padded_image = padded_one / padded_image
  device_divide <<<blocks, threads>>>
    (d_padded_one_.data(), d_padded_image_.data(), padded_one_.num_elements());
  HANDLE_LAST_ERROR();

  image_stack gpu_result = padded_image_;
  d_padded_image_.pull_from_device(gpu_result);
  double sum = std::accumulate(gpu_result.data(),
			       gpu_result.data()+gpu_result.num_elements(),
			       0.);

  for(std::size_t i = 0;i<padded_image_.num_elements();++i)
    BOOST_REQUIRE_EQUAL(expected_value,gpu_result.data()[i]);

  BOOST_REQUIRE_GT(sum,0.);
  
  try {
    BOOST_REQUIRE_CLOSE(sum, .2*(padded_image_.num_elements()), .15);
  }
  catch (...) {
    std::cout << boost::unit_test::framework::current_test_case().p_name << "\n"
              << expected_value << " * " << padded_image_.num_elements() << "\n";
  }


  image_stack cpu_result = padded_image_;
  multiviewnative::cpu::par::compute_quotient(padded_one_.data(),
					      cpu_result.data(),
					      padded_one_.num_elements());
  
  for(std::size_t i = 0;i<padded_image_.num_elements();++i)
    BOOST_REQUIRE_EQUAL(cpu_result.data()[i],gpu_result.data()[i]);
  
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_FIXTURE_TEST_SUITE(final_values,
                         multiviewnative::default_3D_fixture)

BOOST_AUTO_TEST_CASE(cube32_const) {

  std::vector<unsigned> shape(3,32);
  one_.resize(shape);
  padded_one_.resize(shape);
  padded_image_.resize(shape);

  std::fill(one_.data(), one_.data()+one_.num_elements(),5.f);
  std::fill(padded_one_.data(), padded_one_.data()+padded_one_.num_elements(),42.f);
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),.1f);

  device_stack d_input_psi	(one_);
  device_stack d_integral	(padded_one_);
  device_stack d_weights	(padded_image_);

  dim3 threads(128);
  dim3 blocks(
      largestDivisor(padded_one_.num_elements(), size_t(threads.x)));

  device_final_values<<<blocks, threads>>>(
            d_input_psi.data(), d_integral.data(), d_weights.data(),
            0.0001f, one_.num_elements());

  image_stack gpu_result = one_;
  d_input_psi.pull_from_device(gpu_result);
  
  for( unsigned d = 0;d<3;++d )
    BOOST_CHECK_EQUAL(shape[d],gpu_result.shape()[d]); 

  image_stack cpu_result = one_;
  multiviewnative::cpu::par::final_values(cpu_result.data(),
					  padded_one_.data(),
					  padded_image_.data(),
					  cpu_result.num_elements()
					  );

  BOOST_CHECK_EQUAL_COLLECTIONS(cpu_result.data(), cpu_result.data() + cpu_result.num_elements(),
				gpu_result.data(), gpu_result.data() + gpu_result.num_elements());

  
}

BOOST_AUTO_TEST_CASE(odd_cube265_const) {

  std::vector<unsigned> shape(3,256);
  shape[1]+=1;
  shape[2]-=1;
  
  one_.resize(shape);
  padded_one_.resize(shape);
  padded_image_.resize(shape);

  std::fill(one_.data(), one_.data()+one_.num_elements(),5.f);
  std::fill(padded_one_.data(), padded_one_.data()+padded_one_.num_elements(),42.f);
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),.1f);

  device_stack d_input_psi	(one_);
  device_stack d_integral	(padded_one_);
  device_stack d_weights	(padded_image_);

  dim3 threads(128);
  dim3 blocks(
      largestDivisor(padded_one_.num_elements(), size_t(threads.x)));

  device_final_values<<<blocks, threads>>>(
            d_input_psi.data(), d_integral.data(), d_weights.data(),
            0.0001f, one_.num_elements());

  image_stack gpu_result = one_;
  d_input_psi.pull_from_device(gpu_result);
  
  for( unsigned d = 0;d<3;++d )
    BOOST_CHECK_EQUAL(shape[d],gpu_result.shape()[d]); 

  image_stack cpu_result = one_;
  multiviewnative::cpu::par::final_values(cpu_result.data(),
					  padded_one_.data(),
					  padded_image_.data(),
					  cpu_result.num_elements()
					  );

  BOOST_REQUIRE_GT(cpu_result.num_elements(),0);
  BOOST_CHECK_EQUAL_COLLECTIONS(cpu_result.data(), cpu_result.data() + cpu_result.num_elements(),
				gpu_result.data(), gpu_result.data() + gpu_result.num_elements());

  
}

BOOST_AUTO_TEST_CASE(cube32_regularize_const) {

  std::vector<unsigned> shape(3,32);
  one_.resize(shape);
  padded_one_.resize(shape);
  padded_image_.resize(shape);

  std::fill(one_.data(), one_.data()+one_.num_elements(),5.f);
  std::fill(padded_one_.data(), padded_one_.data()+padded_one_.num_elements(),42.f);
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),.1f);

  device_stack d_input_psi	(one_);
  device_stack d_integral	(padded_one_);
  device_stack d_weights	(padded_image_);

  dim3 threads(128);
  dim3 blocks(
      largestDivisor(padded_one_.num_elements(), size_t(threads.x)));

  device_regularized_final_values<<<blocks, threads>>>(
            d_input_psi.data(), d_integral.data(), d_weights.data(),
            .006,0.0001f, one_.num_elements());

  image_stack gpu_result = one_;
  d_input_psi.pull_from_device(gpu_result);
  
  for( unsigned d = 0;d<3;++d )
    BOOST_CHECK_EQUAL(shape[d],gpu_result.shape()[d]); 

  image_stack cpu_result = one_;
  multiviewnative::cpu::par::regularized_final_values(cpu_result.data(),
					  padded_one_.data(),
					  padded_image_.data(),
					  cpu_result.num_elements()
					  );

  BOOST_CHECK_EQUAL_COLLECTIONS(cpu_result.data(), cpu_result.data() + cpu_result.num_elements(),
				gpu_result.data(), gpu_result.data() + gpu_result.num_elements());

  
}

BOOST_AUTO_TEST_CASE(odd_cube265_regularize_const) {

  std::vector<unsigned> shape(3,256);
  shape[1]+=1;
  shape[2]-=1;
  
  one_.resize(shape);
  padded_one_.resize(shape);
  padded_image_.resize(shape);

  std::fill(one_.data(), one_.data()+one_.num_elements(),5.f);
  std::fill(padded_one_.data(), padded_one_.data()+padded_one_.num_elements(),42.f);
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),.1f);

  device_stack d_input_psi	(one_);
  device_stack d_integral	(padded_one_);
  device_stack d_weights	(padded_image_);

  dim3 threads(128);
  dim3 blocks(
      largestDivisor(padded_one_.num_elements(), size_t(threads.x)));

  device_regularized_final_values<<<blocks, threads>>>(
            d_input_psi.data(), d_integral.data(), d_weights.data(),
            .006, 0.0001f, one_.num_elements());

  image_stack gpu_result = one_;
  d_input_psi.pull_from_device(gpu_result);
  
  for( unsigned d = 0;d<3;++d )
    BOOST_CHECK_EQUAL(shape[d],gpu_result.shape()[d]); 

  std::fill(one_.data(), one_.data()+one_.num_elements(),5.f);
  std::fill(padded_one_.data(), padded_one_.data()+padded_one_.num_elements(),42.f);
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),.1f);

  image_stack cpu_result = one_;
  multiviewnative::cpu::par::regularized_final_values(cpu_result.data(),
						      padded_one_.data(),
						      padded_image_.data(),
						      cpu_result.num_elements()
						      );

  BOOST_REQUIRE_GT(cpu_result.num_elements(),0);
  BOOST_REQUIRE_EQUAL(cpu_result.num_elements(),gpu_result.num_elements());

  for( std::size_t p = 0;p<cpu_result.num_elements();++p )
    BOOST_REQUIRE_MESSAGE(cpu_result.data()[p]==gpu_result.data()[p], "cpu version differs from gpu at " << p << " as cpu = " << cpu_result.data()[p] << " vs. gpu = " << gpu_result.data()[p]); 

  
}

BOOST_AUTO_TEST_CASE(odd_cube265_regularize_rndm) {

  std::vector<unsigned> shape(3,256);
  shape[1]+=1;
  shape[2]-=1;
  
  one_.resize(shape);
  padded_one_.resize(shape);
  padded_image_.resize(shape);

  std::fill(one_.data(), one_.data()+one_.num_elements(),5.f);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-.1, 1.);
  for( std::size_t p = 0;p<padded_one_.num_elements();++p )
    padded_one_.data()[p] = dis(gen);
    
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),.1f);

  device_stack d_input_psi	(one_);
  device_stack d_integral	(padded_one_);
  device_stack d_weights	(padded_image_);

  dim3 threads(128);
  dim3 blocks(
      largestDivisor(padded_one_.num_elements(), size_t(threads.x)));

  device_regularized_final_values<<<blocks, threads>>>(
            d_input_psi.data(), d_integral.data(), d_weights.data(),
            .006, 0.0001f, one_.num_elements());

  image_stack gpu_result = one_;
  d_input_psi.pull_from_device(gpu_result);
  
  for( unsigned d = 0;d<3;++d )
    BOOST_CHECK_EQUAL(shape[d],gpu_result.shape()[d]); 

  std::fill(one_.data(), one_.data()+one_.num_elements(),5.f);
  
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),.1f);

  image_stack cpu_result = one_;
  multiviewnative::cpu::par::regularized_final_values(cpu_result.data(),
						      padded_one_.data(),
						      padded_image_.data(),
						      cpu_result.num_elements()
						      );

  BOOST_REQUIRE_GT(cpu_result.num_elements(),0);
  BOOST_REQUIRE_EQUAL(cpu_result.num_elements(),gpu_result.num_elements());

  for( std::size_t p = 0;p<cpu_result.num_elements();++p ){
    try{
      BOOST_REQUIRE_CLOSE(cpu_result.data()[p],gpu_result.data()[p],0.001);
    }
    catch(...){
      std::cout <<"cpu version differs from gpu at " << p << " as cpu = " << cpu_result.data()[p] << " vs. gpu = " << gpu_result.data()[p] << "\n";
    }
  }

  
}

BOOST_AUTO_TEST_CASE(odd_cube16_regularize_rndm) {

  std::vector<unsigned> shape(3,16);
  shape[1]+=1;
  shape[2]-=1;
  
  one_.resize(shape);
  padded_one_.resize(shape);
  padded_image_.resize(shape);

  std::fill(one_.data(), one_.data()+one_.num_elements(),5.f);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-.1, 1.);
  for( std::size_t p = 0;p<padded_one_.num_elements();++p )
    padded_one_.data()[p] = dis(gen);

  std::vector<float> aside(padded_one_.data(),padded_one_.data()+padded_one_.num_elements());
  
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),.1f);

  device_stack d_input_psi	(one_);
  device_stack d_integral	(padded_one_);
  device_stack d_weights	(padded_image_);

  dim3 threads(128);
  dim3 blocks(
      largestDivisor(padded_one_.num_elements(), size_t(threads.x)));

  device_regularized_final_values<<<blocks, threads>>>(
            d_input_psi.data(), d_integral.data(), d_weights.data(),
            .006,
	    0.0001f,
	    one_.num_elements());

  image_stack gpu_result = one_;
  d_input_psi.pull_from_device(gpu_result);
  
  for( unsigned d = 0;d<3;++d )
    BOOST_CHECK_EQUAL(shape[d],gpu_result.shape()[d]); 

  std::fill(one_.data(), one_.data()+one_.num_elements(),5.f);
  std::copy(aside.begin(), aside.end(),padded_one_.data());
  std::fill(padded_image_.data(), padded_image_.data()+padded_image_.num_elements(),.1f);

  image_stack cpu_result = one_;
  multiviewnative::cpu::par::regularized_final_values(cpu_result.data(),
						      padded_one_.data(),
						      padded_image_.data(),
						      cpu_result.num_elements(),
						      .006,
						      -1,
						      0.0001f
						      );

  BOOST_REQUIRE_GT(cpu_result.num_elements(),0);
  BOOST_REQUIRE_EQUAL(cpu_result.num_elements(),gpu_result.num_elements());

  for( std::size_t p = 0;p<cpu_result.num_elements();++p ){
    try{
      BOOST_REQUIRE_CLOSE(cpu_result.data()[p],gpu_result.data()[p],0.001);
    }
    catch(...){
      std::cout <<"cpu version differs from gpu at " << p << " as cpu = " << cpu_result.data()[p] << " vs. gpu = " << gpu_result.data()[p] << "\n";
    }
  }

  
}
BOOST_AUTO_TEST_SUITE_END()
