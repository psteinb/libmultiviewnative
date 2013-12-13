#ifndef __MULTIVIEWNATIVE_H__
#define __MULTIVIEWNATIVE_H__

typedef float imageType;

#ifdef _WIN32
#define FUNCTION_PREFIX extern "C" __declspec(dllexport)
#else
#define FUNCTION_PREFIX extern "C"
#endif

//--------------------------------------------- Win -------------------------------------------
//#ifdef _WIN32
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

// #ifdef __CUDA_RUNTIME_H__
// FUNCTION_PREFIX void fit_2Dblocks_to_threads_for_device(dim3& _blocks, dim3& _threads, const int& _device);
// #endif

// //--------------------------------------------- Linux/OSX -------------------------------------------
// #else
// void convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA);
// void convolution3DfftCUDAInPlace_core(imageType* _d_imCUDA,int* imDim,
// 							  imageType* _d_kernelCUDA,int* kernelDim,
// 							  int devCUDA);

// void compute_quotient(imageType* _input,imageType* _output,size_t _size, int _device);
// void compute_final_values(imageType* _image,imageType* _integral,imageType* _weight,size_t _size, float _minValue, double _lambda, int _device);

// void iterate_fft_plain(imageType* _input,imageType* _output,int* _input_dims,int* _kernel_dims, int _device);
// void iterate_fft_tikhonov(imageType* _image,imageType* _integral,imageType* _weight,size_t _size, float _minValue, double _lambda, int _device);


// int selectDeviceWithHighestComputeCapability();
// int getCUDAcomputeCapabilityMinorVersion(int devCUDA);
// int getCUDAcomputeCapabilityMajorVersion(int devCUDA);
// int getNumDevicesCUDA();
// void getNameDeviceCUDA(int devCUDA, char *name);
// long long int getMemDeviceCUDA(int devCUDA);

// #ifdef __CUDA_RUNTIME_H__
// void fit_2Dblocks_to_threads_for_device(dim3& _blocks, dim3& _threads, const int& _device);
// #endif

// #endif



#endif //__CONVOLUTION_3D_FFT_MINE_H__
