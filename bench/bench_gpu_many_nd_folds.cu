#define __BENCH_GPU_DECONVOLVE_CU__
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>

#include "boost/program_options.hpp"
#include "synthetic_data.hpp"
//#include "cpu_nd_fft.hpp"
#include "multiviewnative.h"
//#include "fftw_interface.h"

#include "logging.hpp"

#include <boost/chrono.hpp>
#include <boost/thread.hpp>


#include "gpu_convolve.cuh"
#include "padd_utils.h"
#include "gpu_nd_fft.cuh"
#include "cufft_utils.cuh"

#include "cuda_profiler_api.h"


namespace mvn = multiviewnative;
typedef mvn::no_padd<mvn::image_stack> stack_padding;
typedef mvn::inplace_3d_transform_on_device<imageType>
    device_transform;
typedef mvn::gpu_convolve<stack_padding, imageType, unsigned>
    device_convolve;

template <typename Container>
void inplace_gpu_plan_many_fold(std::vector<Container>& _data, int device){

    std::vector<mvn::image_stack> forwarded_kernels(_data.size());
    std::vector<int> reshaped;
    
      for (int v = 0; v < _data.size(); ++v) {

    stack_padding local_padding(&_data[v].stack_shape_[0],
				      &_data[v].kernel_shape_[0]);

    forwarded_kernels[v].resize(local_padding.extents_);
    local_padding.wrapped_insert_at_offsets(_data[v].kernel_, forwarded_kernels[v]);

    //prepare for fft
    reshaped = multiviewnative::gpu::cufft_r2c_shape(forwarded_kernels[v].shape(),forwarded_kernels[v].shape() + 3);
    forwarded_kernels[v].resize(reshaped);
    HANDLE_ERROR(cudaHostRegister((void*)forwarded_kernels[v].data(), 
				    forwarded_kernels[v].num_elements()*sizeof(float),
				    cudaHostRegisterPortable));
  }

      //unsigned long reshaped_buffer_byte = forwarded_kernels[0].num_elements()*sizeof(float);
  
      std::vector<device_convolve*> image_folds(_data.size(),0);
      
      for (int v = 0; v < _data.size(); ++v) {
	image_folds[v] = new device_convolve(_data[v].stack_.data(),
					     &(_data[v].stack_shape_[0]),
					     &(_data[v].kernel_shape_[0])
					     );
	
      }

      std::vector<float> image_buffer(_data.size()*forwarded_kernels[0].num_elements());
      std::vector<float> kernel_buffer(image_buffer.size());

      for (int v = 0; v < _data.size(); ++v) {
	std::copy(image_folds[v]->padded_image_->data(),
		  image_folds[v]->padded_image_->data() + image_folds[v]->padded_image_->num_elements(),
		  &image_buffer[0] + (v*image_folds[v]->padded_image_->num_elements()));
	std::copy(forwarded_kernels[v].data(),
		  forwarded_kernels[v].data() + forwarded_kernels[v].num_elements(),
		  &kernel_buffer[0] + (v*forwarded_kernels[v].num_elements()));
      }

      //create plan
      std::vector<int> fftready_shape( forwarded_kernels[0].shape(), 
				       forwarded_kernels[0].shape() + 3);
      unsigned fft_size_in_byte_ = sizeof(float)*std::accumulate(fftready_shape.begin(), fftready_shape.end(),1,std::multiplies<int>());

      std::vector<int> fft_shape(_data[0].stack_shape_.begin(), 
				 _data[0].stack_shape_.end());

      
      // std::vector<int> fftready_shape_as_cufftcomplex(fftready_shape.begin(),
      // 						      fftready_shape.end());
      // for ( int & i : fftready_shape_as_cufftcomplex )
      // 	i/=2;

      
      std::vector<int> iembed(fft_shape.begin(),
			      fft_shape.end());
      std::vector<int> oembed(iembed);
      for ( int & i : oembed )
      	i/=2;


      std::copy(fft_shape.rbegin(), fft_shape.rend(),fft_shape.begin());
      
      cufftHandle* image_plan = new cufftHandle;
      cufftHandle* kernel_plan = new cufftHandle;
      
      HANDLE_CUFFT_ERROR(cufftPlanMany(image_plan,  //plan
				       3, //rank
				       (int*)&fft_shape[0], //n
				       (int*)&iembed[0],//inembed
				       1, //istride
				       fft_size_in_byte_/sizeof(cufftReal),//idist
				       (int*)&oembed[0],//onembed
				       1, //ostride
				       fft_size_in_byte_/sizeof(cufftComplex),//odist
				       CUFFT_R2C,
				       _data.size()
				       ));

      HANDLE_CUFFT_ERROR(cufftPlanMany(kernel_plan,  //plan
				       3, //rank
				       (int*)&fft_shape[0], //n
				       (int*)&iembed[0],//inembed
				       1, //istride
				       fft_size_in_byte_/sizeof(cufftReal),//idist
				       (int*)&oembed[0],//onembed
				       1, //ostride
				       fft_size_in_byte_/sizeof(cufftComplex),//odist
				       CUFFT_R2C,
				       _data.size()
				       ));

      
      //alloc on device
      float* d_images = 0;
      HANDLE_ERROR(cudaMalloc((void**)&(d_images), _data.size()*fft_size_in_byte_));
      
      float* d_kernels = 0;
      HANDLE_ERROR(cudaMalloc((void**)&(d_kernels), _data.size()*fft_size_in_byte_));
      
      //perform convolution
      std::vector<cudaStream_t*> streams(2);
      for( unsigned count = 0;count < streams.size();++count ){
	streams[count] = new cudaStream_t;
	HANDLE_ERROR(cudaStreamCreate(streams[count]));
      }
  
      //transfer to device
      HANDLE_ERROR(cudaMemcpyAsync(d_images,
				   &image_buffer[0], 
				   _data.size()*fft_size_in_byte_,
				   cudaMemcpyHostToDevice,
				   *streams[0]
				   ));
      
      HANDLE_ERROR(cudaMemcpyAsync(d_kernels,
			      &kernel_buffer[0], 
			      _data.size()*fft_size_in_byte_,
			      cudaMemcpyHostToDevice,
			      *streams[1]
			      ));

       HANDLE_CUFFT_ERROR(cufftSetStream(*image_plan,                            
					 *streams[0] )
			  );
       HANDLE_CUFFT_ERROR(cufftSetStream(*kernel_plan,                            
					 *streams[1] )
			  );
      //transform forward
      HANDLE_CUFFT_ERROR(
			 cufftExecR2C(*image_plan, d_images, (cufftComplex*)d_images));
      
      HANDLE_CUFFT_ERROR(
			 cufftExecR2C(*kernel_plan, d_kernels, (cufftComplex*)d_kernels));

      HANDLE_ERROR(cudaDeviceSynchronize());
      
      
      //multiply
      unsigned eff_fft_num_elements = fft_size_in_byte_/(2*sizeof(float));
      unsigned numThreads = 256;
      unsigned numBlocks = largestDivisor(eff_fft_num_elements, 
					  numThreads);
      
      float scale = 1.0 / float(std::accumulate(_data[0].stack_shape_.begin(),
						_data[0].stack_shape_.end(),
						1,
						std::multiplies<unsigned>()));
      
      for( unsigned v = 0;v < _data.size();++v ){
	modulateAndNormalize_kernel << <numBlocks, numThreads>>>
	  ((cufftComplex*)d_images  + (v*eff_fft_num_elements), 
	   (cufftComplex*)d_kernels + (v*eff_fft_num_elements),
	   eff_fft_num_elements, 
	   scale);
	HANDLE_ERROR(cudaPeekAtLastError());
      }

      
  //destroy old plan(s)
      HANDLE_CUFFT_ERROR(cufftDestroy(*kernel_plan));
      delete kernel_plan;

      HANDLE_CUFFT_ERROR(cufftDestroy(*image_plan));
      HANDLE_CUFFT_ERROR(cufftPlanMany(image_plan,  //plan
				       3, //n
				       (int*)&oembed[0], //
				       (int*)&oembed[0],//inembed
				       1, //istride
				       fft_size_in_byte_/sizeof(cufftComplex),//idist
				       (int*)&iembed[0],//onembed
				       1, //ostride
				       fft_size_in_byte_/sizeof(cufftReal),//odist
				       CUFFT_C2R,
				       _data.size()
				       ));

      HANDLE_CUFFT_ERROR(cufftSetStream(*image_plan,                            
					 *streams[0] )
			  );
       
  //transform back
      HANDLE_CUFFT_ERROR(
			 cufftExecC2R(*image_plan, 
				      (cufftComplex*)d_images, 
				      d_images));

  


      //transfer back
      HANDLE_ERROR(cudaMemcpyAsync(&image_buffer[0],
				   d_images, 
				   _data.size()*fft_size_in_byte_,
				   cudaMemcpyDeviceToHost,
				  *streams[0] 
			      ));

    HANDLE_ERROR(cudaDeviceSynchronize());

  
      //copy in and resize to input shape
      for ( unsigned i = 0;i<_data.size();++i){
	_data[i].stack_.resize(reshaped);
	std::copy(&image_buffer[0] + (i*_data[i].stack_.num_elements()),
		  &image_buffer[0] + ((i+1)*_data[i].stack_.num_elements()),
		  _data[i].stack_.data());
	_data[i].stack_.resize(_data[i].stack_shape_);
	
      } 
      
      //clean-up
      HANDLE_CUFFT_ERROR(cufftDestroy(*image_plan));
      delete image_plan;

      for (int v = 0; v < _data.size(); ++v) {
	HANDLE_ERROR(cudaHostUnregister((void*)forwarded_kernels[v].data()));
	delete image_folds[v];
      }
      
      for(cudaStream_t* s : streams)
	HANDLE_ERROR(cudaStreamDestroy(*s));
      
      HANDLE_ERROR(cudaFree(d_images));
      HANDLE_ERROR(cudaFree(d_kernels));
  
}

template <typename Container>
void inplace_gpu_batched_fold(std::vector<Container>& _data){
  
  std::vector<mvn::image_stack> forwarded_kernels(_data.size());

  std::vector<int> reshaped;


  for (int v = 0; v < _data.size(); ++v) {

    stack_padding local_padding(&_data[v].stack_shape_[0],
				      &_data[v].kernel_shape_[0]);

    forwarded_kernels[v].resize(local_padding.extents_);
    local_padding.wrapped_insert_at_offsets(_data[v].kernel_, forwarded_kernels[v]);

    //prepare for fft
    reshaped = multiviewnative::gpu::cufft_r2c_shape(forwarded_kernels[v].shape(),forwarded_kernels[v].shape() + 3);
    forwarded_kernels[v].resize(reshaped);
    HANDLE_ERROR(cudaHostRegister((void*)forwarded_kernels[v].data(), 
				    forwarded_kernels[v].num_elements()*sizeof(float),
				    cudaHostRegisterPortable));
  }

  unsigned long reshaped_buffer_byte = forwarded_kernels[0].num_elements()*sizeof(float);
  
  std::vector<device_convolve*> image_folds(_data.size(),0);
  for (int v = 0; v < _data.size(); ++v) {
    image_folds[v] = new device_convolve(_data[v].stack_.data(),
				      &(_data[v].stack_shape_[0]),
				      &(_data[v].kernel_shape_[0])
				      );
		     
  }

  //creating the plans
  std::vector<cufftHandle *> plans(2 //number of copy engines
				   );
  for (unsigned count = 0; count < plans.size(); ++count) {

      plans[count] = new cufftHandle;
      HANDLE_CUFFT_ERROR(cufftPlan3d(plans[count],                 //
				     (int)_data[0].stack_shape_[0], //
				     (int)_data[0].stack_shape_[1], //
				     (int)_data[0].stack_shape_[2], //
				     CUFFT_R2C)                    //
			 );
      
  }

  //requesting space on device
  std::vector<float*> src_buffers(plans.size(),0);
  for (unsigned count = 0; count < src_buffers.size(); ++count){
    HANDLE_ERROR(cudaMalloc((void**)&(src_buffers[count]), reshaped_buffer_byte));
    
  }

  
  //forward all kernels
  batched_fft_async2plans(forwarded_kernels,plans,src_buffers,false);
    
  //perform convolution
  std::vector<cudaStream_t*> streams(plans.size());
  for( unsigned count = 0;count < streams.size();++count ){
    streams[count] = new cudaStream_t;
    HANDLE_ERROR(cudaStreamCreate(streams[count]));
  }
  

  HANDLE_ERROR(cudaMemcpyAsync(src_buffers[0],
				 forwarded_kernels[0].data(), 
				 reshaped_buffer_byte,
				 cudaMemcpyHostToDevice,
				 *streams[0]
				 ));
  

  for (int v = 0; v < _data.size(); ++v) {

    image_folds[v]->half_inplace<device_transform>(src_buffers[0],src_buffers[1],
						   streams[0], streams[1],
						   v+1 < _data.size() ? forwarded_kernels[v+1].data() : 0);
  }
    
  //clean-up
  for (unsigned count = 0;count < streams.size();++count){
    HANDLE_ERROR(cudaStreamSynchronize(*streams[count]));
    HANDLE_ERROR(cudaStreamDestroy(*streams[count]));
  }

  for (unsigned count = 0;count < src_buffers.size();++count){
    HANDLE_ERROR(cudaFree(src_buffers[count]));
  }

  for (unsigned count = 0;count < plans.size();++count){

    HANDLE_CUFFT_ERROR(cufftDestroy(*plans[count]));
    delete plans[count];
    plans[count] = 0;
  }


  for (int v = 0; v < _data.size(); ++v) {
    HANDLE_ERROR(cudaHostUnregister((void*)forwarded_kernels[v].data()));
    delete image_folds[v];
  }

}

typedef boost::chrono::high_resolution_clock::time_point tp_t;
typedef boost::chrono::milliseconds ms_t;
typedef boost::chrono::nanoseconds ns_t;

namespace po = boost::program_options;

// typedef boost::multi_array<float, 3, fftw_allocator<float> > fftw_image_stack;
// typedef std::vector<float, fftw_allocator<float> > aligned_float_vector;

int main(int argc, char* argv[]) {
  unsigned num_replicas = 8;
  bool verbose = false;
  
  bool plan_many = false;
  int device_id = -1;
  


  int num_repeats = 5;
  std::string stack_dims = "";

  po::options_description desc("Allowed options");

  // clang-format off
  desc.add_options()                                                      //
      ("help,h", "produce help message")                                  //
      ("verbose,v", "print lots of information in between")               //
      ("plan_many,p", "use cufftPlanMany for transforms")               //
      ("header-only,H", "print header of stats only")                     //
                                                                          //
      ("stack_dimensions,s",                                              //
       po::value<std::string>(&stack_dims)->default_value("64x64x64"),  //
       "HxWxD of synthetic stacks to generate")                           //
                                                                          //
      ("repeats,r",                                                       //
       po::value<int>(&num_repeats)->default_value(10),                   //
       "number of repetitions per measurement")                           //
                                                                          //
      ("num_replicas,n",                                                  //
       po::value<unsigned>(&num_replicas)->default_value(8),              //
       "number of replicas to use for batched processing")                //
                                                                          //
      ("device_id,d",                                                     //
       po::value<int>(&device_id)->default_value(-1),                     //
       "cuda device to use")                                              //
      ;                                                                   //
  // clang-format on

po::variables_map vm;

  po::store(po::parse_command_line(argc, argv, desc), vm);

  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  if (vm.count("header-only")) {
    print_header();
    return 0;
  }


  verbose = vm.count("verbose");
  // out_of_place = vm.count("out-of-place");
  plan_many = vm.count("plan_many");

  std::vector<unsigned> numeric_stack_dims;
  split<'x'>(stack_dims, numeric_stack_dims);

  if (verbose) {
    std::cout << "received " << numeric_stack_dims.size() << " dimensions: ";
    for (unsigned i = 0; i < numeric_stack_dims.size(); ++i) {
      std::cout << numeric_stack_dims[i] << " ";
    }
    std::cout << "\n";
  }

  if (numeric_stack_dims.size() != 3) {
    std::cerr << ">> " << numeric_stack_dims.size()
              << "-D data, not supported yet!\n";
    return 1;
  }

  std::vector<unsigned> reshaped(numeric_stack_dims);
  reshaped.back() = (reshaped.back() / 2 + 1) * 2;

  //////////////////////////////////////////////////////////////////////////////
  // set device flags
  if(device_id<0)
    device_id = selectDeviceWithHighestComputeCapability();
  
  HANDLE_ERROR(cudaSetDevice(device_id));
  unsigned long cufft_extra_space =
      cufft_3d_estimated_memory_consumption(numeric_stack_dims);
  unsigned long cufft_data_size = multiviewnative::gpu::cufft_r2c_memory(numeric_stack_dims);
  // unsigned long data_size_byte =
  //     std::accumulate(numeric_stack_dims.begin(), numeric_stack_dims.end(), 1u,
  //                     std::multiplies<unsigned long>()) *
  //     sizeof(float);
  unsigned long memory_available_on_device = getAvailableGMemOnCurrentDevice();

  float exp_mem_mb = (cufft_extra_space + cufft_data_size) / float(1 << 20);
  float av_mem_mb = memory_available_on_device / float(1 << 20);

  if (exp_mem_mb > av_mem_mb) {
    std::cerr << "not enough memory available on device, needed " << exp_mem_mb
              << " MB (data only: " << cufft_data_size / float(1 << 20)
              << " MB), available: " << av_mem_mb << " MB\n";
    return 1;
  } else {
    if (verbose)
      std::cout << "cufft memory estimate: needed " << exp_mem_mb
                << " MB (data only: " << cufft_data_size / float(1 << 20)
                << " MB), available: " << av_mem_mb << " MB\n";
  }



  multiviewnative::image_kernel_data raw(numeric_stack_dims);
  multiviewnative::image_kernel_data reference = raw;
  
  inplace_gpu_convolution(reference.stack_.data(),
			  &reference.stack_shape_[0],
			  reference.kernel_.data(),
			  &reference.kernel_shape_[0],
			  device_id);

  std::vector<multiviewnative::image_kernel_data> stacks(num_replicas,raw);


  if (verbose) {
    std::cout << "[config]\t"
              << "\n"
              << "num_replicas\t:\t" << num_replicas << "\nnumeric size\t:\t";
    std::copy(numeric_stack_dims.begin(), numeric_stack_dims.end(),
              std::ostream_iterator<unsigned>(std::cout, " "));

    std::cout << "\nfftw size\t:\t";
    std::copy(reshaped.begin(), reshaped.end(),
              std::ostream_iterator<unsigned>(std::cout, " "));
    std::cout << "\n";
  }


    stacks[0] = raw;

  //start measurement
  std::vector<ns_t> durations(num_repeats);

  ns_t time_ns = ns_t(0);
  tp_t start, end;

  cudaProfilerStart();
    
  for (int r = 0; r < num_repeats; ++r) {

    for ( multiviewnative::image_kernel_data& s : stacks ){
      s.stack_ = raw.stack_;
      s.kernel_ = raw.kernel_;
    }
    

    start = boost::chrono::high_resolution_clock::now();

    if(!plan_many)
      inplace_gpu_batched_fold(stacks);
    else
      inplace_gpu_plan_many_fold(stacks, device_id);
    
    end = boost::chrono::high_resolution_clock::now();
    durations[r] = boost::chrono::duration_cast<ns_t>(end - start);

    time_ns += boost::chrono::duration_cast<ns_t>(end - start);
    if (verbose) {
      std::cout << r << "\t"
                << boost::chrono::duration_cast<ns_t>(durations[r]).count() /
	double(1e6) << " ms\n";
    }
  }
  cudaProfilerStop();

  bool data_valid = std::equal(reference.stack_.data(), reference.stack_.data() + reference.stack_.num_elements(),
			       stacks[0].stack_.data());
  

  std::string implementation_name = __FILE__;
  std::stringstream comments("");
  comments << "global_plan";
  if(data_valid)
    comments << ",OK";
  else
    comments << ",NA";

  if(plan_many)
    comments << ",plan_many";


  std::string device_name = get_cuda_device_name(device_id);
  std::replace(device_name.begin(), device_name.end(), ' ', '_');

  if(verbose)
    print_header();


  print_info(1,
	     implementation_name,
	     device_name,
	     num_repeats,
	     time_ns.count() / double(1e6),
	     numeric_stack_dims,
	     sizeof(float),
	     comments.str()
	     );


  return 0;
}
