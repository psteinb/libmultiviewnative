#ifndef __MULTIVIEWNATIVE_H__
#define __MULTIVIEWNATIVE_H__

typedef float imageType;
#include <cstddef> 

#ifdef _WIN32
#define FUNCTION_PREFIX extern "C" __declspec(dllexport)
#else
#define FUNCTION_PREFIX extern "C"
#endif

//helper structs because Java clients are using JNA
struct view_data {

  imageType  *  image_    ;
  imageType  *  kernel1_  ;
  imageType  *  kernel2_  ;
  imageType  *  weights_  ;

  int        *  image_dims_    ;
  int        *  kernel1_dims_  ;
  int        *  kernel2_dims_  ;
  int        *  weights_dims_  ;

};


struct workspace {

  view_data  *      data_      ;
  unsigned   short  num_views  ;

};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CPU convolution
FUNCTION_PREFIX void inplace_cpu_convolution(imageType* im,int* imDim,
					     imageType* kernel,int* kernelDim,
					     int nthreads);

FUNCTION_PREFIX void inplace_cpu_deconvolve_iteration(imageType* psi,
						      workspace input,
						      int nthreads, 
						      double lambda, 
						      imageType minValue);

// GPU convolution
FUNCTION_PREFIX void inplace_gpu_convolution(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int device);
FUNCTION_PREFIX void inplace_gpu_deconvolve_iteration(imageType* psi,
						      workspace input,
						      int device, 
						      double lambda, 
						      imageType minValue);


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Legacy GPU convolution(s)
FUNCTION_PREFIX void inplace_gpu_convolution(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int device);

FUNCTION_PREFIX void convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA);

FUNCTION_PREFIX void convolution3DfftCUDAInPlace_core(imageType* _d_imCUDA,int* imDim,
							  imageType* _d_kernelCUDA,int* kernelDim,
							  int devCUDA);

FUNCTION_PREFIX void compute_quotient(imageType* _input,imageType* _output,size_t _size, int _device);
FUNCTION_PREFIX void compute_final_values(imageType* _image,imageType* _integral,imageType* _weight,size_t _size, float _minValue, double _lambda, int _device);
FUNCTION_PREFIX void iterate_fft_plain(imageType* _input,
				     imageType* _kernel,
				     imageType* _output,
				     int* _input_dims,
				     int* _kernel_dims, 
				     int _device);

FUNCTION_PREFIX void iterate_fft_tikhonov(imageType* _input,
					imageType* _kernel,
					imageType* _output,
					int* _input_dims,
					int* _kernel_dims,
					size_t _size, 
					float _minValue, 
					double _lambda, 
					int _device);

FUNCTION_PREFIX int selectDeviceWithHighestComputeCapability();
FUNCTION_PREFIX int getCUDAcomputeCapabilityMinorVersion(int devCUDA);
FUNCTION_PREFIX int getCUDAcomputeCapabilityMajorVersion(int devCUDA);
FUNCTION_PREFIX int getNumDevicesCUDA();
FUNCTION_PREFIX void getNameDeviceCUDA(int devCUDA, char *name);
FUNCTION_PREFIX long long int getMemDeviceCUDA(int devCUDA);


#endif //__CONVOLUTION_3D_FFT_MINE_H__
