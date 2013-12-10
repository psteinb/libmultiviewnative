#define __MULTIVIEWNATIVE_CU__
// ------- C++ ----------
#include <iostream>
#include <cmath>
#include <algorithm>

// ------- CUDA ----------
#include "cuda.h"
#include "cufft.h"

// ------- Project ----------
#include "multiviewnative.h"
#include "cuda_helpers.h"
#include "cuda_kernels.cuh"

#ifndef LB_MAX_THREADS
#define LB_MAX_THREADS 1024 
#endif

#ifndef DIMSIMAGE
static const int dimsImage = 3;
#else
static const int dimsImage = DIMSIMAGE;
#endif

typedef float imageType;//the kind sof images we are working with (you will need to recompile to work with other types)

__global__ void  __launch_bounds__(LB_MAX_THREADS) fftShiftKernel(imageType* kernelCUDA,
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
void convolution3DfftCUDAInPlace(imageType* im,int* imDim,imageType* kernel,int* kernelDim,int devCUDA)
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
	convolution3DfftCUDAInPlace_core(imCUDA, imDim,
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
void convolution3DfftCUDAInPlace_core(imageType* _d_imCUDA,int* imDim,
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

  fftShiftKernel<<<numBlocks,numThreads>>>(_d_kernelCUDA,kernelPaddedCUDA,kernelDim[0],kernelDim[1],kernelDim[2],imDim[0],imDim[1],imDim[2]);HANDLE_ERROR_KERNEL;

	
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

  
  modulateAndNormalize_kernel<<<numBlocks,numThreads>>>((cufftComplex *)_d_imCUDA, 
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
