#ifndef __CONVOLUTION_3D_FFT_MINE_H__
#define __CONVOLUTION_3D_FFT_MINE_H__

typedef float imageType;



#ifdef _WIN32
//extern "C" __declspec(dllexport) void convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA);
__declspec(dllexport) void my_convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA);
__declspec(dllexport) void my_convolution3DfftCUDAInPlace_core(imageType* _d_imCUDA,int* imDim,
							  imageType* _d_kernelCUDA,int* kernelDim,
							  int devCUDA);
__declspec(dllexport) void compute_quotient(imageType* _input,imageType* _output,size_t _size,size_t _offset, int _device);
__declspec(dllexport) void compute_final_values(imageType* _image,imageType* _integral,imageType* _weight,size_t _size,size_t _offset,float _minValue, double _lambda, int _device);

__declspec(dllexport) int selectDeviceWithHighestComputeCapability();
#ifdef __CUDA_RUNTIME_H__
__declspec(dllexport) void fit_2Dblocks_to_threads_for_device(dim3& _blocks, dim3& _threads, const int& _device);
#endif
#else
void my_convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA);
void my_convolution3DfftCUDAInPlace_core(imageType* _d_imCUDA,int* imDim,
							  imageType* _d_kernelCUDA,int* kernelDim,
							  int devCUDA);

void compute_spatial_convolution(imageType* _image,imageType* _output,int* _image_dims,imageType* _kernel,int* _kernel_dims,int _device);
void compute_spatial_convolution_inplace(imageType* _image,int* _image_dims,imageType* _kernel,int* _kernel_dims,int _device);
void compute_quotient(imageType* _input,imageType* _output,size_t _size,size_t _offset, int _device);
void compute_final_values(imageType* _image,imageType* _integral,imageType* _weight,size_t _size,size_t _offset, float _minValue, double _lambda, int _device);


int selectDeviceWithHighestComputeCapability();
#ifdef __CUDA_RUNTIME_H__
void fit_2Dblocks_to_threads_for_device(dim3& _blocks, dim3& _threads, const int& _device);
#endif

#endif



#endif //__CONVOLUTION_3D_FFT_MINE_H__
