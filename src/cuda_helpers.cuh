#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Benchmark_GPU_emulate_iteration
#include <sstream> 

#include "../aux/benchmark_fixtures.hpp"
#include "boost/test/unit_test.hpp"
#include <boost/timer/timer.hpp>

#include "../spatial/cuda_helpers.cuh"
#include "../spatial/test_utilities.hpp"
#include "../spatial/cuda_memory.cuh"

#include "convolution3Dfft_mine.h"
#include "compute_kernels_gpu.cuh"

using boost::timer::cpu_timer;            
using boost::timer::cpu_times;
using boost::timer::nanosecond_type;
static const unsigned num_items_to_compare = 1<<9;//512

template < typename T>
bool arrays_differ(T* _first, T* _second, const unsigned& _limit = num_items_to_compare){
  unsigned int _differ = 0;
  for(unsigned item=0;item<num_items_to_compare;++item){
    if(_first[item]!=_second[item])
      _differ++;
  }
  
  return _differ!=0;
}

BOOST_FIXTURE_TEST_CASE( warm_up_gpu_test , largeImage_normalKernel_fixture )
{ 
  std::cout << ">> STARTING GPU WARM UP ...";
  
  float* input = createArrayPtr(*image_);
  float* output = new float[image_->size()];
  float* kernel = createArrayPtr(*kernel_);
  int input_dims[3];
  int kernel_dims[3];
  for(short i = 0;i<3;i++){
    if(i<2){
      input_dims[i] = image_dims_[i];
      kernel_dims[i] = kernel_dims_[i];
    }
    else
      {
	input_dims[i] =1;
	kernel_dims[i] = 1;
      }
      
  }

  int gpu_device = selectDeviceWithHighestComputeCapability();
  

  my_convolution3DfftCUDAInPlace(input,input_dims,kernel,kernel_dims,gpu_device);


  BOOST_CHECK(true);//we know that any check of input/output is useless

  delete [] output;
  std::cout << "... Done.\n";
}


typedef  image_kernel_fixture<2,unsigned(1<<13),1> eightKpixel_noKernel_fixture ; 
typedef  image_kernel_fixture<3,unsigned(512),91> threeD_512image_91kernel_fixture ; //512MB for the image, 2MB for kernel


BOOST_FIXTURE_TEST_SUITE( run_iteration, threeD_512image_91kernel_fixture )
BOOST_AUTO_TEST_CASE( emulate_iteration_plain  )
{ 
  int input_dims[3];
  int kernel_dims[3];
  size_t inputInByte = image_->size()*sizeof(float);
  size_t kernelInByte = kernel_->size()*sizeof(float);

  for(short i = 0;i<3;i++){
      input_dims[i] = image_dims_[i];
      kernel_dims[i] = kernel_dims_[i];
  }

  std::cout << std::setw(35) << boost::unit_test::framework::current_test_case().p_name << ": ";print();

  float* input = createArrayPtr(*image_);
  float* kernel1 = createArrayPtr(*kernel_);
  std::vector<float>* kernel2_ = new std::vector<float>(kernel_->size());
  std::vector<float>* weights_ = new std::vector<float>(image_->size());
  std::fill(kernel2_->begin(),kernel2_->end(),.1f);
  std::fill(weights_->begin(),weights_->end(),1.f);
  float* weights = createArrayPtr(*weights_);
  float* kernel2 = createArrayPtr(*kernel2_);
  cudaEvent_t start1_, stop1_, start2_, stop2_ ;
  cudaEventCreate(&start1_);
  cudaEventCreate(&stop1_);
  cudaEventCreate(&start2_);
  cudaEventCreate(&stop2_); 
  //-> memory consumption to here: 2*image + 2*kernel

  //////////////////////////////////////////////////////////////////////////////////////////
  //
  // Entering Loop here
  //
  float* d_image_ = 0;
  float* d_initial_ = 0;
  float* d_kernel_ = 0;
  float* d_weights_ = 0;

  size_t imSizeFFT = image_->size() + 2*input_dims[0]*input_dims[1]; //size of the R2C transform in cuFFTComplex
  size_t imSizeFFTInByte = imSizeFFT*sizeof(float);
  int gpu_device = selectDeviceWithHighestComputeCapability();

  checkCudaErrors( cudaMalloc( (void**)&(d_image_), imSizeFFTInByte ) );//a little bit larger to allow in-place FFT
  checkCudaErrors( cudaMalloc( (void**)&(d_initial_), inputInByte ) );
  checkCudaErrors( cudaMalloc( (void**)&(d_weights_), inputInByte ) );
  checkCudaErrors( cudaMalloc( (void**)&(d_kernel_), kernelInByte ) );
  checkCudaErrors( cudaMemcpy( d_weights_, weights, inputInByte , cudaMemcpyHostToDevice ) );

  cudaEventRecord(start1_, 0);

  checkCudaErrors( cudaMemcpy( d_kernel_, kernel1, kernelInByte , cudaMemcpyHostToDevice ) );
  checkCudaErrors( cudaMemcpy( d_image_, input, inputInByte , cudaMemcpyHostToDevice ) );
  checkCudaErrors( cudaMemcpy( d_initial_, d_image_, inputInByte , cudaMemcpyDeviceToDevice ) );

  cudaEventRecord(start2_, 0);
  //convolve(input) with kernel1 -> psiBlurred
  my_convolution3DfftCUDAInPlace_core(d_image_, input_dims,
				      d_kernel_,kernel_dims,
				      gpu_device);

  //computeQuotient(psiBlurred,input)
  size_t items_per_block = 128;
  size_t items_per_grid = image_->size()/items_per_block;
     
  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid); 
  fit_2Dblocks_to_threads_for_device(threads,blocks,gpu_device);

  device_divide<<<blocks,threads>>>(d_initial_,d_image_,image_->size());
  //convolve(psiBlurred) with kernel2 -> integral
  checkCudaErrors( cudaMemcpy( d_kernel_, kernel2, kernelInByte , cudaMemcpyHostToDevice ) );
  my_convolution3DfftCUDAInPlace_core(d_image_, input_dims,
				      d_kernel_,kernel_dims,
				      gpu_device);
  //computeFinalValues(input,integral,weights)
  device_finalValues_plain<<<blocks,threads>>>(d_initial_,d_image_,d_weights_,.0001f,image_->size());
  cudaEventRecord(stop2_, 0);
  cudaEventSynchronize(stop2_);
  
  checkCudaErrors(cudaMemcpy(input , d_initial_ , inputInByte , cudaMemcpyDeviceToHost));
  cudaEventRecord(stop1_, 0); 
  cudaEventSynchronize(stop1_);
  checkCudaErrors( cudaFree( d_image_));
  checkCudaErrors( cudaFree( d_initial_));
  checkCudaErrors( cudaFree( d_kernel_));
  checkCudaErrors( cudaFree( d_weights_));

  float elapsedTimeCudaSeconds_noTransfer =0.f;
  float elapsedTimeCudaSeconds_inclTransfer =0.f;
  cudaEventElapsedTime(&elapsedTimeCudaSeconds_noTransfer, start2_, stop2_);
  cudaEventElapsedTime(&elapsedTimeCudaSeconds_inclTransfer, start1_, stop1_);
  elapsedTimeCudaSeconds_noTransfer /=1e3;  
  elapsedTimeCudaSeconds_inclTransfer /=1e3;

  std::cout << "\n\t\t\t\t  w/o memory transfer time/s: " << std::setw(8) << elapsedTimeCudaSeconds_noTransfer
	    << ",\t throughput/Mpixel/s: " << produceMPixelThroughput(image_->size(),elapsedTimeCudaSeconds_noTransfer) << "\n"
	    << "\t\t\t\t with memory transfer time/s: " << std::setw(8) << elapsedTimeCudaSeconds_inclTransfer
	    << ",\t\t throughput/Mpixel/s: " << produceMPixelThroughput(image_->size(),elapsedTimeCudaSeconds_inclTransfer) << "\n";
  

  delete kernel2_;
  delete weights_;
  
}

BOOST_AUTO_TEST_CASE( emulate_iteration_plain_noFFT  )
{ 
  size_t inputInByte = image_->size()*sizeof(float);
  size_t kernelInByte = kernel_->size()*sizeof(float);


  std::cout << std::setw(35) << boost::unit_test::framework::current_test_case().p_name << ": ";print();
  


  float* input = createArrayPtr(*image_);
  float* kernel1 = createArrayPtr(*kernel_);
  std::vector<float>* kernel2_ = new std::vector<float>(kernel_->size());
  std::vector<float>* weights_ = new std::vector<float>(image_->size());
  std::fill(kernel2_->begin(),kernel2_->end(),.1f);
  std::fill(weights_->begin(),weights_->end(),1.f);
  float* weights = createArrayPtr(*weights_);
  float* kernel2 = createArrayPtr(*kernel2_);
  
  cudaEvent_t start1_, stop1_, start2_, stop2_ ;
  cudaEventCreate(&start1_);
  cudaEventCreate(&stop1_);
  cudaEventCreate(&start2_);
  cudaEventCreate(&stop2_); 
  //-> memory consumption to here: 2*image + 2*kernel

  //////////////////////////////////////////////////////////////////////////////////////////
  //
  // Entering Loop here
  //

  int gpu_device = selectDeviceWithHighestComputeCapability();

 const size_t width = image_dims_[0];
  const unsigned short half_kernel_x = kernel_dims_[0]/2;
  const size_t height = image_dims_[1];
  const size_t depth = image_dims_[2];

  const size_t width_to_parse = width - kernel_dims_[0] + 1;
  const size_t height_to_parse = height - kernel_dims_[1] + 1;
  const size_t depth_to_parse = depth - kernel_dims_[2] + 1;

  dim3 threads(8,8,8);//128
  dim3 blocks(largestDivisor(width_to_parse,size_t(threads.x)),
	      largestDivisor(height_to_parse,size_t(threads.y)),
	      largestDivisor(depth_to_parse,size_t(threads.z))
	      ); 
  uint3 imageDims = {image_dims_[0],image_dims_[1],image_dims_[2]};
  uint3 kernelDims = {kernel_dims_[0],kernel_dims_[1],kernel_dims_[2]};


  float* kernelReversed = new float[kernelDims.x*kernelDims.y*kernelDims.z];
  std::reverse_copy(kernel_->begin(),kernel_->end(),&kernelReversed[0]);
  
  cudaExtent image_extent = make_cudaExtent(width * sizeof(float),
				      height, depth);

  cudaExtent weights_extent = image_extent;
  cudaExtent output_extent = image_extent;
  cudaExtent initial_extent = image_extent;


  cudaExtent kernel1_extent = make_cudaExtent(kernel_dims_[0] * sizeof(float),
					     kernel_dims_[1],kernel_dims_[2] );

  cudaExtent kernel2_extent = kernel1_extent;

  cudaPitchedPtr d_image_ ;
  cudaPitchedPtr d_weights_ ;
  cudaPitchedPtr d_initial_ ;
  cudaPitchedPtr d_kernel1_ ;
  cudaPitchedPtr d_kernel2_ ;
  cudaPitchedPtr d_output_ ;

  cudaPitchedPtr h_image_  = make_cudaPitchedPtr((void *)input, width*sizeof(float), width, height);
  cudaPitchedPtr h_weights_  = make_cudaPitchedPtr((void *)weights, width*sizeof(float), width, height);
  cudaPitchedPtr h_kernel1_ = make_cudaPitchedPtr((void *)kernel1, 
						 kernel_dims_[0] * sizeof(float),
						 kernel_dims_[0] , 
						 kernel_dims_[1]);
  cudaPitchedPtr h_kernel2_ = make_cudaPitchedPtr((void *)kernel2, 
						  kernel_dims_[0] * sizeof(float),
						  kernel_dims_[0] , 
						  kernel_dims_[1]);

  float* output_ = new float[image_->size()];
  cudaPitchedPtr h_output_ = make_cudaPitchedPtr((void *)output_, width*sizeof(float), width, height);


  cudaEventRecord(start1_, 0);


    cudaMalloc3D(&d_image_, image_extent);
  cudaMemcpy3DParms inputParams = { 0 };
  inputParams.srcPtr   = h_image_;
  inputParams.dstPtr = d_image_;
  inputParams.extent   = image_extent;
  inputParams.kind     = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&inputParams);

    cudaMalloc3D(&d_initial_, image_extent);
  cudaMemcpy3DParms initialParams = { 0 };
  initialParams.srcPtr   = h_image_;
  initialParams.dstPtr = d_initial_;
  initialParams.extent   = initial_extent;
  initialParams.kind     = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&initialParams);


  cudaMalloc3D(&d_weights_, weights_extent);
  cudaMemcpy3DParms weightsParams = { 0 };
  inputParams.srcPtr   = h_weights_;
  inputParams.dstPtr = d_weights_;
  inputParams.extent   = weights_extent;
  inputParams.kind     = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&weightsParams);


  cudaMemcpy3DParms outputParams = { 0 };
  cudaMalloc3D(&d_output_, image_extent);
  outputParams.srcPtr   = d_initial_;
  outputParams.dstPtr = d_output_;
  outputParams.extent   = image_extent;
  outputParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&outputParams);

  cudaMalloc3D(&d_kernel1_, kernel1_extent);
  cudaMemcpy3DParms kernel1Params = { 0 };
  kernel1Params.srcPtr   = h_kernel1_;
  kernel1Params.dstPtr = d_kernel1_;
  kernel1Params.extent   = kernel1_extent;
  kernel1Params.kind     = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&kernel1Params);


  cudaMalloc3D(&d_kernel2_, kernel2_extent);
  cudaMemcpy3DParms kernel2Params = { 0 };
  kernel2Params.srcPtr   = h_kernel2_;
  kernel2Params.dstPtr = d_kernel2_;
  kernel2Params.extent   = kernel2_extent;
  kernel2Params.kind     = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&kernel2Params);

  cudaEventRecord(start2_, 0);
  //convolve(input) with kernel1 -> psiBlurred
  kernel3D_naive<<<blocks,threads>>>(d_initial_,
				     d_kernel1_,
				     d_output_,
				     imageDims,
				     kernelDims);


  //computeQuotient(psiBlurred,input)
  size_t items_per_block = 128;
  size_t items_per_grid = image_->size()/items_per_block;
     
  dim3 threads_pixel2pixel(items_per_block);
  dim3 blocks_pixel2pixel(items_per_grid); 
  fit_2Dblocks_to_threads_for_device(threads,blocks,gpu_device);

  device_divide_3D<<<blocks_pixel2pixel,threads_pixel2pixel>>>(d_initial_,d_output_,imageDims);

  //convolve(psiBlurred) with kernel2 -> integral
  kernel3D_naive<<<blocks,threads>>>(d_output_, 
				     d_kernel2_,
				     d_image_,
				     imageDims,
				     kernelDims);
  //computeFinalValues(input,integral,weights)
  device_finalValues_plain_3D<<<blocks_pixel2pixel,threads_pixel2pixel>>>(d_initial_,d_image_,d_weights_,.0001f,imageDims);
  cudaEventRecord(stop2_, 0);
  cudaEventSynchronize(stop2_);
  
  // checkCudaErrors(cudaMemcpy(input , d_initial_ , inputInByte , cudaMemcpyDeviceToHost));
  outputParams.srcPtr   = d_initial_;
  outputParams.dstPtr = h_output_;
  outputParams.extent   = output_extent;
  outputParams.kind     = cudaMemcpyDeviceToHost;
  cudaMemcpy3D(&outputParams);


  cudaEventRecord(stop1_, 0); 
  cudaEventSynchronize(stop1_);
  checkCudaErrors( cudaFree( d_image_.ptr  ));
  checkCudaErrors( cudaFree( d_initial_.ptr));
  checkCudaErrors( cudaFree( d_kernel1_.ptr));
  checkCudaErrors( cudaFree( d_kernel2_.ptr));
  checkCudaErrors( cudaFree( d_weights_.ptr));
  checkCudaErrors( cudaFree( d_output_.ptr));

  float elapsedTimeCudaSeconds_noTransfer =0.f;
  float elapsedTimeCudaSeconds_inclTransfer =0.f;
  cudaEventElapsedTime(&elapsedTimeCudaSeconds_noTransfer, start2_, stop2_);
  cudaEventElapsedTime(&elapsedTimeCudaSeconds_inclTransfer, start1_, stop1_);
  elapsedTimeCudaSeconds_noTransfer /=1e3;  
  elapsedTimeCudaSeconds_inclTransfer /=1e3;

  std::cout << "\n\t\t\t\t  w/o memory transfer time/s: " << std::setw(8) << elapsedTimeCudaSeconds_noTransfer
	    << ",\t throughput/Mpixel/s: " << produceMPixelThroughput(image_->size(),elapsedTimeCudaSeconds_noTransfer) << "\n"
	    << "\t\t\t\t with memory transfer time/s: " << std::setw(8) << elapsedTimeCudaSeconds_inclTransfer
	    << ",\t\t throughput/Mpixel/s: " << produceMPixelThroughput(image_->size(),elapsedTimeCudaSeconds_inclTransfer) << "\n";
  

  delete kernel2_;
  delete weights_;
  delete [] output_;
}

BOOST_AUTO_TEST_CASE( emulate_iteration_plain_pinned  )
{ 
  int input_dims[3];
  int kernel_dims[3];
  size_t inputInByte = image_->size()*sizeof(float);
  size_t kernelInByte = kernel_->size()*sizeof(float);

  for(short i = 0;i<3;i++){
    
      input_dims[i] = image_dims_[i];
      kernel_dims[i] = kernel_dims_[i];
    
      
  }

  std::cout << std::setw(35) << boost::unit_test::framework::current_test_case().p_name << ": ";print();

  float* input = 0;
  float* kernel1 = 0;
  float* weights = 0;
  float* kernel2 = 0;
  checkCudaErrors( cudaMallocHost( (void**)&(kernel2), kernel_->size()*sizeof(float), cudaHostAllocDefault ) );
  checkCudaErrors( cudaMallocHost( (void**)&(weights), image_->size()*sizeof(float), cudaHostAllocDefault ) );
  checkCudaErrors( cudaMallocHost( (void**)&(kernel1), kernel_->size()*sizeof(float), cudaHostAllocDefault ) );
  checkCudaErrors( cudaMallocHost( (void**)&(input), image_->size()*sizeof(float), cudaHostAllocDefault ) );
  // std::vector<float>* kernel2_ = new std::vector<float>(kernel_->size());
  // std::vector<float>* weights_ = new std::vector<float>(image_->size() );
  std::fill(&kernel2[0],&kernel2[0]+kernel_->size() ,.1f);
  std::fill(&weights[0],&weights[0]+image_->size()  ,1.f);
  std::copy(kernel_->begin(), kernel_->end(),&kernel1[0]);
  std::copy(image_->begin(), image_->end(),&input[0]);

  cudaEvent_t start1_, stop1_, start2_, stop2_ ;
  cudaEventCreate(&start1_);
  cudaEventCreate(&stop1_);
  cudaEventCreate(&start2_);
  cudaEventCreate(&stop2_); 
  //-> memory consumption to here: 2*image + 2*kernel

  //////////////////////////////////////////////////////////////////////////////////////////
  //
  // Entering Loop here
  //
  float* d_image_ = 0;
  float* d_initial_ = 0;
  float* d_kernel_ = 0;
  float* d_weights_ = 0;

  size_t imSizeFFT = image_->size() + 2*input_dims[0]*input_dims[1]; //size of the R2C transform in cuFFTComplex
  size_t imSizeFFTInByte = imSizeFFT*sizeof(float);
  int gpu_device = selectDeviceWithHighestComputeCapability();

  checkCudaErrors( cudaMalloc( (void**)&(d_image_), imSizeFFTInByte ) );//a little bit larger to allow in-place FFT
  checkCudaErrors( cudaMalloc( (void**)&(d_initial_), inputInByte ) );
  checkCudaErrors( cudaMalloc( (void**)&(d_weights_), inputInByte ) );
  checkCudaErrors( cudaMalloc( (void**)&(d_kernel_), kernelInByte ) );
  cudaStream_t initial_stream, weights_stream;
  checkCudaErrors( cudaStreamCreate(&initial_stream));
  checkCudaErrors( cudaStreamCreate(&weights_stream));
  checkCudaErrors( cudaMemcpyAsync( d_weights_, weights, inputInByte , cudaMemcpyHostToDevice, weights_stream ) );
  checkCudaErrors( cudaMemcpyAsync( d_initial_, input, inputInByte , cudaMemcpyHostToDevice ,initial_stream) );

  cudaEventRecord(start1_, 0);

  checkCudaErrors( cudaMemcpy( d_kernel_, kernel1, kernelInByte , cudaMemcpyHostToDevice ) );
  checkCudaErrors( cudaMemcpy( d_image_, input, inputInByte , cudaMemcpyHostToDevice ) );
  
  
  

  cudaEventRecord(start2_, 0);
  //convolve(input) with kernel1 -> psiBlurred
  my_convolution3DfftCUDAInPlace_core(d_image_, input_dims,
				      d_kernel_,kernel_dims,
				      gpu_device);

  //computeQuotient(psiBlurred,input)
  size_t items_per_block = 128;
  size_t items_per_grid = image_->size()/items_per_block;
     
  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid); 
  fit_2Dblocks_to_threads_for_device(threads,blocks,gpu_device);

  device_divide<<<blocks,threads,0,initial_stream>>>(d_initial_,d_image_,image_->size());
  //convolve(psiBlurred) with kernel2 -> integral
  checkCudaErrors( cudaMemcpy( d_kernel_, kernel2, kernelInByte , cudaMemcpyHostToDevice ) );
  my_convolution3DfftCUDAInPlace_core(d_image_, input_dims,
				      d_kernel_,kernel_dims,
				      gpu_device);
  //computeFinalValues(input,integral,weights)
  device_finalValues_plain<<<blocks,threads,0,weights_stream>>>(d_initial_,d_image_,d_weights_,.0001f,image_->size());
  cudaEventRecord(stop2_, 0);
  cudaEventSynchronize(stop2_);
  
  checkCudaErrors(cudaMemcpyAsync(input , d_initial_ , inputInByte , cudaMemcpyDeviceToHost, weights_stream));
  checkCudaErrors(cudaStreamSynchronize(weights_stream));
  cudaEventRecord(stop1_, 0); 
  cudaEventSynchronize(stop1_);
  checkCudaErrors( cudaFree( d_image_));
  checkCudaErrors( cudaFree( d_initial_));
  checkCudaErrors( cudaFree( d_kernel_));
  checkCudaErrors( cudaFree( d_weights_));

  float elapsedTimeCudaSeconds_noTransfer =0.f;
  float elapsedTimeCudaSeconds_inclTransfer =0.f;
  cudaEventElapsedTime(&elapsedTimeCudaSeconds_noTransfer, start2_, stop2_);
  cudaEventElapsedTime(&elapsedTimeCudaSeconds_inclTransfer, start1_, stop1_);
  elapsedTimeCudaSeconds_noTransfer /=1e3;  
  elapsedTimeCudaSeconds_inclTransfer /=1e3;

  std::cout << "\n\t\t\t\t  w/o memory transfer time/s: " << std::setw(8) << elapsedTimeCudaSeconds_noTransfer
	    << ",\t throughput/Mpixel/s: " << produceMPixelThroughput(image_->size(),elapsedTimeCudaSeconds_noTransfer) << "\n"
	    << "\t\t\t\t with memory transfer time/s: " << std::setw(8) << elapsedTimeCudaSeconds_inclTransfer
	    << ",\t\t throughput/Mpixel/s: " << produceMPixelThroughput(image_->size(),elapsedTimeCudaSeconds_inclTransfer) << "\n";
  
  checkCudaErrors( cudaFreeHost( kernel2));
  checkCudaErrors( cudaFreeHost( weights));
  checkCudaErrors( cudaFreeHost( kernel1));
  checkCudaErrors( cudaFreeHost( input  ));
  checkCudaErrors( cudaStreamDestroy(initial_stream));
  checkCudaErrors( cudaStreamDestroy(weights_stream));
  
}

BOOST_AUTO_TEST_CASE( emulate_iteration_tikhonov  )
{ 
  int input_dims[3];
  int kernel_dims[3];
  size_t inputInByte = image_->size()*sizeof(float);
  size_t kernelInByte = kernel_->size()*sizeof(float);

  for(short i = 0;i<3;i++){
    
      input_dims[i] = image_dims_[i];
      kernel_dims[i] = kernel_dims_[i];
    
      
  }

  std::cout << std::setw(35) << boost::unit_test::framework::current_test_case().p_name << ": ";print();

  float* input = createArrayPtr(*image_);
  float* kernel1 = createArrayPtr(*kernel_);
  std::vector<float>* kernel2_ = new std::vector<float>(kernel_->size());
  std::vector<float>* weights_ = new std::vector<float>(image_->size());
  std::fill(kernel2_->begin(),kernel2_->end(),.1f);
  std::fill(weights_->begin(),weights_->end(),1.f);
  float* weights = createArrayPtr(*weights_);
  float* kernel2 = createArrayPtr(*kernel2_);
  cudaEvent_t start1_, stop1_, start2_, stop2_ ;
  cudaEventCreate(&start1_);
  cudaEventCreate(&stop1_);
  cudaEventCreate(&start2_);
  cudaEventCreate(&stop2_); 
  //-> memory consumption to here: 2*image + 2*kernel

  //////////////////////////////////////////////////////////////////////////////////////////
  //
  // Entering Loop here
  //
  float* d_image_ = 0;
  float* d_initial_ = 0;
  float* d_kernel_ = 0;
  float* d_weights_ = 0;

  size_t imSizeFFT = image_->size() + 2*input_dims[0]*input_dims[1]; //size of the R2C transform in cuFFTComplex
  size_t imSizeFFTInByte = imSizeFFT*sizeof(float);
  int gpu_device = selectDeviceWithHighestComputeCapability();

  checkCudaErrors( cudaMalloc( (void**)&(d_image_), imSizeFFTInByte ) );//a little bit larger to allow in-place FFT
  checkCudaErrors( cudaMalloc( (void**)&(d_initial_), inputInByte ) );
  checkCudaErrors( cudaMalloc( (void**)&(d_weights_), inputInByte ) );
  checkCudaErrors( cudaMalloc( (void**)&(d_kernel_), kernelInByte ) );
  checkCudaErrors( cudaMemcpy( d_weights_, weights, inputInByte , cudaMemcpyHostToDevice ) );

  cudaEventRecord(start1_, 0);

  checkCudaErrors( cudaMemcpy( d_kernel_, kernel1, kernelInByte , cudaMemcpyHostToDevice ) );
  checkCudaErrors( cudaMemcpy( d_image_, input, inputInByte , cudaMemcpyHostToDevice ) );
  checkCudaErrors( cudaMemcpy( d_initial_, d_image_, inputInByte , cudaMemcpyDeviceToDevice ) );

  cudaEventRecord(start2_, 0);
  //convolve(input) with kernel1 -> psiBlurred
  my_convolution3DfftCUDAInPlace_core(d_image_, input_dims,
				      d_kernel_,kernel_dims,
				      gpu_device);

  //computeQuotient(psiBlurred,input)
  size_t items_per_block = 128;
  size_t items_per_grid = image_->size()/items_per_block;
     
  dim3 threads(items_per_block);
  dim3 blocks(items_per_grid); 
  fit_2Dblocks_to_threads_for_device(threads,blocks,gpu_device);

  device_divide<<<blocks,threads>>>(d_initial_,d_image_,image_->size());
  //convolve(psiBlurred) with kernel2 -> integral
  checkCudaErrors( cudaMemcpy( d_kernel_, kernel2, kernelInByte , cudaMemcpyHostToDevice ) );
  my_convolution3DfftCUDAInPlace_core(d_image_, input_dims,
				      d_kernel_,kernel_dims,
				      gpu_device);
  //computeFinalValues(input,integral,weights)
  device_finalValues_tikhonov<<<blocks,threads>>>(d_initial_,d_image_,d_weights_,.0001f,.2f,image_->size());
  cudaEventRecord(stop2_, 0);
  cudaEventSynchronize(stop2_);
  
  checkCudaErrors(cudaMemcpy(input , d_initial_ , inputInByte , cudaMemcpyDeviceToHost));
  cudaEventRecord(stop1_, 0); 
  cudaEventSynchronize(stop1_);
  checkCudaErrors( cudaFree( d_image_));
  checkCudaErrors( cudaFree( d_initial_));
  checkCudaErrors( cudaFree( d_kernel_));
  checkCudaErrors( cudaFree( d_weights_));

  float elapsedTimeCudaSeconds_noTransfer =0.f;
  float elapsedTimeCudaSeconds_inclTransfer =0.f;
  cudaEventElapsedTime(&elapsedTimeCudaSeconds_noTransfer, start2_, stop2_);
  cudaEventElapsedTime(&elapsedTimeCudaSeconds_inclTransfer, start1_, stop1_);
  elapsedTimeCudaSeconds_noTransfer /=1e3;  
  elapsedTimeCudaSeconds_inclTransfer /=1e3;

  std::cout << "\n\t\t\t\t  w/o memory transfer time/s: " << std::setw(8) << elapsedTimeCudaSeconds_noTransfer
	    << ",\t throughput/Mpixel/s: " << produceMPixelThroughput(image_->size(),elapsedTimeCudaSeconds_noTransfer) << "\n"
	    << "\t\t\t\t with memory transfer time/s: " << std::setw(8) << elapsedTimeCudaSeconds_inclTransfer
	    << ",\t\t throughput/Mpixel/s: " << produceMPixelThroughput(image_->size(),elapsedTimeCudaSeconds_inclTransfer) << "\n";
  

  delete kernel2_;
  delete weights_;
  
}

BOOST_AUTO_TEST_SUITE_END()

//Program received signal SIGSEGV, Segmentation fault.
//0x000000000043885d in cudart::configData::addArgument(void const*, unsigned long, unsigned long) ()

// BOOST_FIXTURE_TEST_SUITE( GPU_suite, eightKpixel_noKernel_fixture )
// BOOST_AUTO_TEST_CASE( quotient_from_library  )
// { 

//   std::cout << std::setw(35) << boost::unit_test::framework::current_test_case().p_name << ": ";print();
//   const size_t size = image_dims_[0]*image_dims_[1];
//   float* output = new float[size];
//   float* output_begin = &output[0];
//   float* input = createArrayPtr(*image_);
//   float* input_begin = &input[0];
//   std::copy(input_begin,input_begin+size,output_begin);
  
//   int gpu_device = selectDeviceWithHighestComputeCapability();
//   cpu_timer timer;
//   cudaEvent_t start_, stop_;
//   cudaEventCreate(&start_);
//   cudaEventCreate(&stop_);
//   cudaEventRecord(start_, 0);

//   compute_quotient(input,output,size,0u,gpu_device);

//   cudaEventRecord(stop_, 0);
//   cudaEventSynchronize(stop_);
//   cpu_times elapsed = timer.elapsed();
//   nanosecond_type elapsedTimeNanoseconds = (elapsed.system + elapsed.user);
//   float elapsedTimeCudaMilliSeconds;
//   cudaEventElapsedTime(&elapsedTimeCudaMilliSeconds, start_, stop_);
//   std::cout << "  time/s: " << std::setw(8) << elapsedTimeNanoseconds/double(1e9) << " cuda/s: " << elapsedTimeCudaMilliSeconds/1000.
// 	    << ",\t" << produceMPixelThroughput(size,elapsedTimeNanoseconds/double(1e9)) << "\n";

//   BOOST_CHECK(true);//we know that any check of input/output is useless

//   delete [] output;

// }
// BOOST_AUTO_TEST_SUITE_END()
