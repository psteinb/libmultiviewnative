#ifndef __MULTIVIEWNATIVE_H__
#define __MULTIVIEWNATIVE_H__

typedef float imageType;

//--------------------------------------------- Win -------------------------------------------
#ifdef _WIN32
__declspec(dllexport) void convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA);
__declspec(dllexport) void convolution3DfftCUDAInPlace_core(imageType* _d_imCUDA,int* imDim,
							  imageType* _d_kernelCUDA,int* kernelDim,
							  int devCUDA);
__declspec(dllexport) void compute_quotient(imageType* _input,imageType* _output,size_t _size,size_t _offset, int _device);
__declspec(dllexport) void compute_final_values(imageType* _image,imageType* _integral,imageType* _weight,size_t _size,size_t _offset,float _minValue, double _lambda, int _device);

__declspec(dllexport) int selectDeviceWithHighestComputeCapability();
__declspec(dllexport) int getCUDAcomputeCapabilityMinorVersion(int devCUDA);
__declspec(dllexport) int getCUDAcomputeCapabilityMajorVersion(int devCUDA);
__declspec(dllexport) int getNumDevicesCUDA();
__declspec(dllexport) void getNameDeviceCUDA(int devCUDA, char *name);
__declspec(dllexport) long long int getMemDeviceCUDA(int devCUDA);

#ifdef __CUDA_RUNTIME_H__
__declspec(dllexport) void fit_2Dblocks_to_threads_for_device(dim3& _blocks, dim3& _threads, const int& _device);
#endif

//--------------------------------------------- Linux/OSX -------------------------------------------
#else
void convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA);
void convolution3DfftCUDAInPlace_core(imageType* _d_imCUDA,int* imDim,
							  imageType* _d_kernelCUDA,int* kernelDim,
							  int devCUDA);

void compute_spatial_convolution(imageType* _image,imageType* _output,int* _image_dims,imageType* _kernel,int* _kernel_dims,int _device);
void compute_spatial_convolution_inplace(imageType* _image,int* _image_dims,imageType* _kernel,int* _kernel_dims,int _device);
void compute_quotient(imageType* _input,imageType* _output,size_t _size,size_t _offset, int _device);
void compute_final_values(imageType* _image,imageType* _integral,imageType* _weight,size_t _size,size_t _offset, float _minValue, double _lambda, int _device);

int selectDeviceWithHighestComputeCapability();
int getCUDAcomputeCapabilityMinorVersion(int devCUDA);
int getCUDAcomputeCapabilityMajorVersion(int devCUDA);
int getNumDevicesCUDA();
void getNameDeviceCUDA(int devCUDA, char *name);
long long int getMemDeviceCUDA(int devCUDA);

#ifdef __CUDA_RUNTIME_H__
void fit_2Dblocks_to_threads_for_device(dim3& _blocks, dim3& _threads, const int& _device);
#endif

#endif



#endif //__CONVOLUTION_3D_FFT_MINE_H__
