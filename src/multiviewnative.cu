#define __CONVOLUTION_3D_FFT_MINE_CU__

#include "book.h"
#include "cuda.h"
#include "cufft.h"
#include <iostream>
#include <cmath>
#include <algorithm>

//#include "convolution3Dfft_mine.h"
#include "convolution3Dfft_mine.h"
#include "compute_kernels_gpu.cuh"

template <typename T>
T largestDivisor(const T& _to_divide, const T& _divided_by){
  return (_to_divide + _divided_by -1)/(_divided_by);
};



size_t getMaxNThreadsOfDevice(short _devId){
  cudaDeviceProp prop;
  HANDLE_ERROR( cudaGetDeviceProperties(&prop, _devId));
  return prop.maxThreadsPerBlock;
}

size_t getMaxNBlocksOfDevice(short _devId,short _dim){
  cudaDeviceProp prop;
  HANDLE_ERROR( cudaGetDeviceProperties(&prop, _devId));
  return prop.maxGridSize[_dim];
}

void fit_2Dblocks_to_threads_for_device(dim3& _blocks, dim3& _threads, const int& _device){
  
const size_t max_blocks =  getMaxNBlocksOfDevice(_device,0);
// const size_t size = _blocks.x*_blocks.y*_threads.x*_threads.y;

  if(_blocks.x > max_blocks){
    _blocks.y = (_blocks.x+max_blocks-1)/max_blocks;
    _blocks.x = max_blocks;
  }
  
  
}

//the definition of the following symbol is correct fo sm_20, sm_30 and above
#define LB_MAX_THREADS 1024 

// #ifndef __CONVOLUTION_3D_FFT_H__
// static const int MAX_THREADS_CUDA = 1024; //adjust it for your GPU. This is correct for a 2.0 architecture
// static const int MAX_BLOCKS_CUDA = 65535;
static const int dimsImage = 3;//so thing can be set at co0mpile time

int getCUDAcomputeCapabilityMajorVersion(int devCUDA)
{
	int major = 0, minor = 0;
	cuDeviceComputeCapability 	( 	&major, &minor,devCUDA);

	return major;
}

int getCUDAcomputeCapabilityMinorVersion(int devCUDA)
{
	int major = 0, minor = 0;
	cuDeviceComputeCapability 	( 	&major, &minor,devCUDA);

	return minor;
}

int getNumDevicesCUDA()
{
	int count = 0;
	HANDLE_ERROR(cudaGetDeviceCount ( &count ));
	return count;
}

void getNameDeviceCUDA(int devCUDA, char* name)
{	
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties(&prop, devCUDA));

	memcpy(name,prop.name,sizeof(char)*256);
}

long long int getMemDeviceCUDA(int devCUDA)
{
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties(&prop, devCUDA));
	return ((long long int)prop.totalGlobalMem);
}



void convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA)
{
  my_convolution3DfftCUDAInPlace(im,imDim,kernel,kernelDim,devCUDA);
  return;
}
// #endif

//http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds
// __global__ void __launch_bounds__(MAX_THREADS_CUDA) modulateAndNormalize_kernel(cufftComplex *d_Dst, 
// 										 cufftComplex *d_Src, 
// 										 unsigned int dataSize,
// 			
__device__ float scale_subtracted(const float& _ax, const float& _bx, 
				  const float& _ay, const float& _by, 
				  const float& _c){
  float result = __fmaf_rn(_ax,_bx,
			   __fmul_rn(-1.,
				     __fmul_rn(_ay,_by)
				     )
			   );
  return __fmul_rn(_c,result);
}

__device__ float scale_added(const float& _ax, const float& _bx, 
			     const float& _ay, const float& _by,
			     const float& _c){
  float result = __fmaf_rn(_ax,_bx,__fmul_rn(_ay,_by));
  return __fmul_rn(_c,result);
}


__global__ void  my_modulateAndNormalize_kernel(cufftComplex *d_Dst, 
					     cufftComplex *d_Src, 
					     unsigned int dataSize,
					     float c)
{
  unsigned int globalIdx  = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int kernelSize = blockDim.x * gridDim.x;
  
  cufftComplex result, a, b;
  
  
  while(globalIdx < dataSize)    {		

    a = d_Src[globalIdx];
    b = d_Dst[globalIdx];
    // result.x = c * (a.x * b.x - a.y * b.y);
    // result.y = c * (a.x * b.x + a.y * b.y);

    result.x = scale_subtracted(a.x,b.x,a.y,b.y,c);
    result.y = scale_added(a.x,b.x,a.y,b.y,c);

    d_Dst[globalIdx] = result;
  
    
    globalIdx += kernelSize;
  }
};


//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
// __global__ void __launch_bounds__(MAX_THREADS_CUDA) fftShiftKernel(imageType* kernelCUDA,
// 								   imageType* kernelPaddedCUDA,
// 								   int kernelDim_0,
// 								   int kernelDim_1,
// 								   int kernelDim_2,
// 								   int imDim_0,
// 								   int imDim_1,
// 								   int imDim_2)
__global__ void  __launch_bounds__(LB_MAX_THREADS) my_fftShiftKernel(imageType* kernelCUDA,
								       imageType* kernelPaddedCUDA,
								       unsigned int kernelDim_0,
								       unsigned int kernelDim_1,
								       unsigned int kernelDim_2,
								       unsigned int imDim_0,
								       unsigned int imDim_1,
								       unsigned int imDim_2)
{
	int kernelSize = kernelDim_0 * kernelDim_1 * kernelDim_2;

	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int x,y,z;
	unsigned aux;
	if(tid<kernelSize)
	  {
	    //find coordinates
	    z = tid - (tid / kernelDim_2);
	    aux = (tid - z)/kernelDim_2;
	    y = aux - (aux / kernelDim_1);
	    x = (aux - y)/kernelDim_1;

	    //center coordinates
	    x -= (int)kernelDim_0/2;
	    y -= (int)kernelDim_1/2;
	    z -= (int)kernelDim_2/2;

	    //circular shift if necessary
	    if(x<0) x += imDim_0;
	    if(y<0) y += imDim_1;
	    if(z<0) z += imDim_2;

	    //WOW! this is a depth-major format
	    //calculate position in padded kernel
	    aux = (unsigned int)z + imDim_2 * ((unsigned int)y + imDim_1 * (unsigned int)x);

	    //copy value
	    if(aux<(imDim_0 * imDim_1 * imDim_2))
	      kernelPaddedCUDA[aux] = kernelCUDA[tid];//for the most part it should be a coalescent access in oth places
	  }
}


//=====================================================================
//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
//NOTE: to avoid transferring a large padded kernel, since memcpy is a limiting factor 
void my_convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA)
{
	imageType* imCUDA = NULL;
	imageType* kernelCUDA = NULL;
	

	

	HANDLE_ERROR( cudaSetDevice( devCUDA ) );

	size_t imSize = 1;
	size_t kernelSize = 1;
	for(int ii=0;ii<dimsImage;ii++)
	{
		imSize *= (imDim[ii]);
		kernelSize *= (kernelDim[ii]);
	}

	size_t imSizeFFT = imSize;
	imSizeFFT += 2*imDim[0]*imDim[1]; //size of the R2C transform in cuFFTComplex
	size_t imSizeFFTInByte = imSizeFFT*sizeof(imageType);
	size_t imSizeInByte = imSize*sizeof(imageType);
	size_t kernelSizeInByte = (kernelSize)*sizeof(imageType);
	//allocat ememory in GPU
	HANDLE_ERROR( cudaMalloc( (void**)&(imCUDA), imSizeFFTInByte ) );//a little bit larger to allow in-place FFT
	HANDLE_ERROR( cudaMalloc( (void**)&(kernelCUDA), kernelSizeInByte ) );

	//printf("Copying memory (kernel and image) to GPU\n");
	HANDLE_ERROR( cudaMemcpy( kernelCUDA, kernel, kernelSizeInByte , cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( imCUDA, im, imSizeInByte , cudaMemcpyHostToDevice ) );

	///////////////////////////////////////////////////////////////////////
	// DO THE WORK
	my_convolution3DfftCUDAInPlace_core(imCUDA, imDim,
					    kernelCUDA,kernelDim,
					    // &fftPlanFwd, &fftPlanInv,
					    devCUDA);
	///////////////////////////////////////////////////////////////////////

	//we destroy memory first so we can perform larger block convolutions

	

	//copy result to host and overwrite image
	HANDLE_ERROR(cudaMemcpy(im,imCUDA,sizeof(imageType)*imSize,cudaMemcpyDeviceToHost));

	
    //( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;
	HANDLE_ERROR( cudaFree( imCUDA));
	HANDLE_ERROR( cudaFree( kernelCUDA));
	//HANDLE_ERROR( cudaFree( kernelCUDA));
	//HANDLE_ERROR( cudaFree( kernelPaddedCUDA));

}

//=====================================================================
//WARNING: for cuFFT the fastest running index is z direction!!! so pos = z + imDim[2] * (y + imDim[1] * x)
//NOTE: to avoid transferring a large padded kernel, since memcpy is a limiting factor 
void my_convolution3DfftCUDAInPlace_core(imageType* _d_imCUDA,int* imDim,
					 imageType* _d_kernelCUDA,int* kernelDim,
					
					 int devCUDA)
{
  cufftHandle fftPlanFwd, fftPlanInv;
  imageType* kernelPaddedCUDA = NULL;
  //apply ffshift to kernel and pad it with zeros so we can calculate convolution with FFT

  size_t imSize = 1;
  size_t kernelSize = 1;
  for(int ii=0;ii<dimsImage;ii++)
    {
      imSize *= (imDim[ii]);
      kernelSize *= (kernelDim[ii]);
    }

  size_t imSizeFFT = imSize;
  imSizeFFT += 2*imDim[0]*imDim[1]; //size of the R2C transform in cuFFTComplex
  size_t imSizeFFTInByte = imSizeFFT*sizeof(imageType);
  // size_t imSizeInByte = imSize*sizeof(imageType);
  // size_t kernelSizeInByte = (kernelSize)*sizeof(imageType);


  HANDLE_ERROR( cudaMalloc( (void**)&(kernelPaddedCUDA), imSizeFFTInByte ) );
  HANDLE_ERROR( cudaMemset( kernelPaddedCUDA, 0, imSizeFFTInByte ));

  size_t max_threads_on_device = getMaxNThreadsOfDevice(devCUDA);
  size_t max_blocks_in_x = getMaxNBlocksOfDevice(devCUDA,0); //we are using dim1 blocks only

  int numThreads=std::min( max_threads_on_device , kernelSize);
  size_t numBlocksFromImage = (kernelSize+numThreads-1)/(numThreads);
  int numBlocks=std::min(max_blocks_in_x,numBlocksFromImage);

  my_fftShiftKernel<<<numBlocks,numThreads>>>(_d_kernelCUDA,kernelPaddedCUDA,kernelDim[0],kernelDim[1],kernelDim[2],imDim[0],imDim[1],imDim[2]);HANDLE_ERROR_KERNEL;

	
  //make sure GPU finishes 
  HANDLE_ERROR(cudaDeviceSynchronize());	

  //printf("Creating R2C & C2R FFT plans for size %i x %i x %i\n",imDim[0],imDim[1],imDim[2]);
  cufftPlan3d(&fftPlanFwd, imDim[0], imDim[1], imDim[2], CUFFT_R2C);HANDLE_ERROR_KERNEL;
  cufftSetCompatibilityMode(fftPlanFwd,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL; //for highest performance since we do not need FFTW compatibility
  
  cufftExecR2C(fftPlanFwd, _d_imCUDA, (cufftComplex *)_d_imCUDA);HANDLE_ERROR_KERNEL;
  //transforming image
  cufftExecR2C(fftPlanFwd, kernelPaddedCUDA, (cufftComplex *)kernelPaddedCUDA);HANDLE_ERROR_KERNEL;

	
  size_t halfImSizeFFT = imSizeFFT/2;
  numThreads=std::min( max_threads_on_device , halfImSizeFFT);
  numBlocksFromImage = (halfImSizeFFT+numThreads-1)/(numThreads);
  numBlocks=std::min(max_blocks_in_x,numBlocksFromImage);

  
  my_modulateAndNormalize_kernel<<<numBlocks,numThreads>>>((cufftComplex *)_d_imCUDA, 
							   (cufftComplex *)kernelPaddedCUDA, 
							   halfImSizeFFT,
							   1.0f/(float)(imSize));HANDLE_ERROR_KERNEL;//last parameter is the size of the FFT



  //inverse FFT 

  cufftPlan3d(&fftPlanInv, imDim[0], imDim[1], imDim[2], CUFFT_C2R);HANDLE_ERROR_KERNEL;
  cufftSetCompatibilityMode(fftPlanInv,CUFFT_COMPATIBILITY_NATIVE);HANDLE_ERROR_KERNEL;

  cufftExecC2R(fftPlanInv, (cufftComplex *)_d_imCUDA, _d_imCUDA);HANDLE_ERROR_KERNEL;
	
	
  HANDLE_ERROR( cudaFree( kernelPaddedCUDA));
  //release memory
  ( cufftDestroy(fftPlanFwd) );HANDLE_ERROR_KERNEL;
  ( cufftDestroy(fftPlanInv) );HANDLE_ERROR_KERNEL;
}

int selectDeviceWithHighestComputeCapability(){

  int numDevices=0;
  HANDLE_ERROR(cudaGetDeviceCount ( &numDevices ));
  int computeCapability = 0;
  int meta = 0;
  int value = -1;
  int major = 0; int minor=0;

  for(short devIdx = 0;devIdx < numDevices;++devIdx){
    cuDeviceComputeCapability 	( 	&major, &minor,devIdx);
    meta = 10*major + minor;
    if(meta>computeCapability){
      computeCapability = meta;
      value = devIdx;
    }
  }

  return value;
}


void compute_spatial_convolution(imageType* _image,imageType* _output,int* _image_dims,
				 imageType* _kernel,int* _kernel_dims,
				 int _device){

  HANDLE_ERROR(cudaSetDevice(_device));

  const size_t width = _image_dims[0];
  const size_t height = _image_dims[1];
  const size_t depth = _image_dims[2];

  const size_t width_to_parse = width - _kernel_dims[0] + 1;
  const size_t height_to_parse = height - _kernel_dims[1] + 1;
  const size_t depth_to_parse = depth - _kernel_dims[2] + 1;

  dim3 threads(8,4,4);//
  dim3 blocks(largestDivisor(width_to_parse,size_t(threads.x)),
	      largestDivisor(height_to_parse,size_t(threads.y)),
	      largestDivisor(depth_to_parse,size_t(threads.z))
	      ); 
  uint3 imageDims = {_image_dims[0],_image_dims[1],_image_dims[2]};
  uint3 kernelDims = {_kernel_dims[0],_kernel_dims[1],_kernel_dims[2]};

  float* kernelBegin = &_kernel[0];
  float* kernelEnd = kernelBegin + (_kernel_dims[0]*_kernel_dims[1]*_kernel_dims[2]);
  float* kernelReversed = new float[kernelDims.x*kernelDims.y*kernelDims.z];
  std::reverse_copy(kernelBegin,kernelEnd,&kernelReversed[0]);
  
  cudaExtent image_extent = make_cudaExtent(width * sizeof(float),
				      height, depth);

  cudaExtent kernel_extent = make_cudaExtent(_kernel_dims[0] * sizeof(float),
					     _kernel_dims[1],_kernel_dims[2] );

  cudaPitchedPtr d_image_ ;
  cudaPitchedPtr d_kernel_ ;
  cudaPitchedPtr d_output_ ;

  cudaPitchedPtr h_image_  = make_cudaPitchedPtr((void *)_image, 
						 width*sizeof(float), width, height);
  cudaPitchedPtr h_kernel_ = make_cudaPitchedPtr((void *)_kernel, 
						 _kernel_dims[0] * sizeof(float),
						 _kernel_dims[0] , 
						 _kernel_dims[1]);
  cudaPitchedPtr h_output_ = make_cudaPitchedPtr((void *)_output, width*sizeof(float), width, height);

  HANDLE_ERROR(cudaMalloc3D(&d_image_, image_extent));
  cudaMemcpy3DParms inputParams = { 0 };
  inputParams.srcPtr   = h_image_;
  inputParams.dstPtr = d_image_;
  inputParams.extent   = image_extent;
  inputParams.kind     = cudaMemcpyHostToDevice;
  HANDLE_ERROR(cudaMemcpy3D(&inputParams));

  cudaMemcpy3DParms outputParams = { 0 };
  HANDLE_ERROR(cudaMalloc3D(&d_output_, image_extent));
  outputParams.srcPtr   = d_image_;
  outputParams.dstPtr = d_output_;
  outputParams.extent   = image_extent;
  outputParams.kind     = cudaMemcpyDeviceToDevice;
  HANDLE_ERROR(cudaMemcpy3D(&outputParams));

  HANDLE_ERROR(cudaMalloc3D(&d_kernel_, kernel_extent));
  cudaMemcpy3DParms kernelParams = { 0 };
  kernelParams.srcPtr   = h_kernel_;
  kernelParams.dstPtr = d_kernel_;
  kernelParams.extent   = kernel_extent;
  kernelParams.kind     = cudaMemcpyHostToDevice;
  HANDLE_ERROR(cudaMemcpy3D(&kernelParams));

  kernel3D_naive<<<blocks,threads>>>(d_image_,
				     d_kernel_,
				     d_output_,
				     imageDims,
				     kernelDims);


  outputParams.srcPtr   = d_output_;
  outputParams.dstPtr = h_output_;
  outputParams.extent   = image_extent;
  outputParams.kind     = cudaMemcpyDeviceToHost;
  HANDLE_ERROR(cudaMemcpy3D(&outputParams));

delete [] kernelReversed;
  HANDLE_ERROR(cudaFree(d_image_.ptr));
  HANDLE_ERROR(cudaFree(d_kernel_.ptr));
  HANDLE_ERROR(cudaFree(d_output_.ptr));

}

void compute_spatial_convolution_inplace(imageType* _image,int* _image_dims,
				 imageType* _kernel,int* _kernel_dims,
				 int _device){

  compute_spatial_convolution(_image, _image, _image_dims,
			      _kernel, _kernel_dims,
			      _device);

}

  
void compute_quotient(imageType* _input,imageType* _output,size_t _size,size_t _offset, int _device){

  imageType* d_input = 0;
  imageType* d_output = 0;

  const size_t sizeInByte = _size*sizeof(imageType);
  HANDLE_ERROR(cudaSetDevice(_device));

  HANDLE_ERROR(cudaMalloc((void**)&d_input, sizeInByte));
  HANDLE_ERROR(cudaMalloc((void**)&d_output, sizeInByte));
  
  HANDLE_ERROR(cudaMemcpy(d_input , _input , sizeInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_output , _output , sizeInByte, cudaMemcpyHostToDevice));

 
  size_t items_per_block = 128;
  size_t items_per_grid = _size/items_per_block;
     
  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid); 

  fit_2Dblocks_to_threads_for_device(threads,blocks,_device);
  device_divide<<<blocks,threads>>>(d_input, d_output, (unsigned int)_size);
  
  HANDLE_ERROR(cudaMemcpy(_output , d_output , sizeInByte, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaFree(d_input));
  HANDLE_ERROR(cudaFree(d_output));
  
}

void compute_final_values(imageType* _image,
			  imageType* _integral,
			  imageType* _weight,
			  size_t _size,
			  size_t _offset, 
			  float _minValue,
			  double _lambda, 
			  int _device){

  imageType* d_integral= 0;
  imageType* d_weight  = 0;
  imageType* d_image   = 0;

  const size_t sizeInByte = _size*sizeof(imageType);

  HANDLE_ERROR(cudaMalloc((void**)&d_integral, sizeInByte));
  HANDLE_ERROR(cudaMalloc((void**)&d_weight  , sizeInByte));
  HANDLE_ERROR(cudaMalloc((void**)&d_image   , sizeInByte));

  
  HANDLE_ERROR(cudaMemcpy(d_integral , _integral , sizeInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_weight , _weight , sizeInByte, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(d_image , _image , sizeInByte, cudaMemcpyHostToDevice));

  
  size_t items_per_block = 128;
  size_t items_per_grid = _size/items_per_block;
    
    
  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid); 
  fit_2Dblocks_to_threads_for_device(threads,blocks,_device);


  if(_lambda>0.)
    device_finalValues_tikhonov<<<blocks,threads>>>(d_image, d_integral, d_weight, _minValue, (float)_lambda, _size);
  else
    device_finalValues_plain<<<blocks,threads>>>(d_image, d_integral, d_weight, _minValue, _size);
  
  HANDLE_ERROR(cudaMemcpy(_image , d_image , sizeInByte, cudaMemcpyDeviceToHost));

  HANDLE_ERROR(cudaFree(d_integral));
  HANDLE_ERROR(cudaFree(d_weight));
  HANDLE_ERROR(cudaFree(d_image));
  
}
