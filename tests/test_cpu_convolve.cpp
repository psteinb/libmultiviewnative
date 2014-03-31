#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Independent
#include "boost/test/unit_test.hpp"
#include "test_fixtures.hpp"
#include <numeric>
#include "multiviewnative.h"
#include "rfftw.h"

BOOST_FIXTURE_TEST_SUITE( cpu_convolution_sandbox, multiviewnative::default_3D_fixture )

BOOST_AUTO_TEST_CASE( trivial_convolve )
{
  
  float* image = new float[image_size_];
  float* kernel = new float[kernel_size_];
  std::fill(kernel, kernel+kernel_size_,0.f);
  std::copy(padded_image_.origin(), padded_image_.origin() + image_size_,image);
  
  ///////////////////////////////////////////////////////////////////////////
  //adapted from 
  //http://www.fftw.org/fftw2_doc/fftw_2.html
  unsigned M = N = K = multiviewnative::default_3D_fixture::image_axis_size;

  fftw_real a[M][N][2*(K/2+1)], b[M][N][2*(K/2+1)], c[M][N][K];
  fftw_complex *A, *B, C[M][N][K/2+1];
  rfftwnd_plan p, pinv;
  fftw_real scale = 1.0 / (M * N * K);
  unsigned i, j, k;
  // ...
  p    = rfftw3d_create_plan(M, N, K, FFTW_REAL_TO_COMPLEX,
			     FFTW_ESTIMATE | FFTW_IN_PLACE);
  pinv = rfftw3d_create_plan(M, N, K, FFTW_COMPLEX_TO_REAL,
			     FFTW_ESTIMATE);

     /* aliases for accessing complex transform outputs: */
     A = (fftw_complex*) &a[0][0][0];
     B = (fftw_complex*) &b[0][0][0];
     // ...
     // for (i = 0; i < M; ++i)
     //      for (j = 0; j < N; ++j) {
     //           a[i][j] = ... ;
     //           b[i][j] = ... ;
     //      }
     // ...
     rfftwnd_one_real_to_complex(p, &a[0][0], NULL);
     rfftwnd_one_real_to_complex(p, &b[0][0], NULL);

     for (i = 0; i < M; ++i)
       for (j = 0; j < N; ++j)
	 for (k = 0; k < K/2+1; ++k) {
	   //int ij = i*(N/2+1) + j;
	   unsigned index = i*N*(K/2+1) + j*(K/2+1) + k;
               C[i][j][k].re = (A[index].re * B[index].re
                             - A[index].im * B[index].im) * scale;
               C[i][j][k].im = (A[index].re * B[index].im
                             + A[index].im * B[index].re) * scale;
          }

     /* inverse transform to get c, the convolution of a and b;
        this has the side effect of overwriting C */
     rfftwnd_one_complex_to_real(pinv, &C[0][0][0], &c[0][0][0]);
     // ...
     rfftwnd_destroy_plan(p);
     rfftwnd_destroy_plan(pinv);

     ///////////////////
  float sum = std::accumulate(image, image + image_size_,0.f);
  BOOST_CHECK_CLOSE(sum, 0.f, .00001);

  delete [] image;
  delete [] kernel;
}



BOOST_AUTO_TEST_CASE( convolve_by_identity )
{
  
  float* image = new float[image_size_];

  std::copy(padded_image_.origin(), padded_image_.origin() + image_size_,image);
  
  print_kernel();
  print_image(image);

  convolution3DfftCUDAInPlace(image, &image_dims_[0], 
			      identity_kernel_.data(),&kernel_dims_[0],
			      selectDeviceWithHighestComputeCapability());

  float * reference = padded_image_.data();
  BOOST_CHECK_EQUAL_COLLECTIONS( image, image+image_size_/2, reference, reference + image_size_/2);
 
  delete [] image;
}


BOOST_AUTO_TEST_SUITE_END()
