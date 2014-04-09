#define __MULTIVIEWNATIVE_CPP__

#include "multiviewnative.h"
#include "cpu_convolve.h"
#include "padd_utils.h"
#include "fft_utils.h"

void inplace_cpu_convolution(imageType* im,
			     int* imDim,
			     imageType* kernel,
			     int* kernelDim,
			       int nthreads){

typedef multiviewnative::zero_padd<multiviewnative::image_stack> padding;
typedef multiviewnative::inplace_3d_transform<multiviewnative::image_stack> transform;
unsigned image_dim[3];
unsigned kernel_dim[3];

std::copy(imDim, imDim + 3, &image_dim[0]);
std::copy(kernelDim, kernelDim + 3, &kernel_dim[0]);

multiviewnative::cpu_convolve<padding,imageType, unsigned> convolver(im, image_dim, kernel, kernel_dim);
convolver.inplace<transform>();


}
