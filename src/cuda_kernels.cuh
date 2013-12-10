#ifndef _COMPUTE_KERNELS_GPU_CUH_
#define _COMPUTE_KERNELS_GPU_CUH_


template <class T> 
__device__ inline T cmax(const T& _first,const T& _second){return (_first>_second)? _first : _second;};

template <class T> 
__device__ inline T cmin(const T& _first,const T& _second){return (_first<_second)? _first : _second ;};

template <typename TransferT>
__global__ void device_divide(const TransferT * _input, TransferT * _output, unsigned int _size){
  
  const size_t pixel_x = size_t(blockIdx.x)*size_t(blockDim.x) + threadIdx.x;
  const size_t pixel_y = size_t(blockIdx.y)*size_t(blockDim.y) + threadIdx.y;
  const size_t pixel_index = pixel_y*size_t(gridDim.x)*size_t(blockDim.x) + pixel_x;

  float temp_out = 0;
  float temp_in = 0;
  
  if(pixel_index<_size){
    //to exploit instruction level parallelism
    temp_out = _output[pixel_index];
    temp_in = _input[pixel_index];
    //floating point operation payload
    _output[pixel_index] = temp_in/temp_out;
  }
  
}

__global__ void device_divide_3D(cudaPitchedPtr _image , 
				 cudaPitchedPtr _output,
				 uint3 _image_dims){

  const size_t image_size = _image_dims.x*_image_dims.y*_image_dims.z;

  const unsigned int pixel_x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int pixel_y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int pixel_z = blockIdx.z*blockDim.z + threadIdx.z;
  
  const unsigned int pixel_index = pixel_z*(_image_dims.x*_image_dims.y) + pixel_y*(_image_dims.x) + pixel_x;
  
  char* input_devPtr = (char*)_image.ptr;
  char* output_devPtr = (char*)_output.ptr;

  float* input_row = (float*)(input_devPtr + pixel_z*(_image_dims.x*_image_dims.y) + pixel_y*(_image_dims.x));
  float* output_row = (float*)(output_devPtr + pixel_z*(_image_dims.x*_image_dims.y) + pixel_y*(_image_dims.x));
  
  float input = 0;
  float output = 0;

  if(pixel_index<image_size){
    input = input_row[pixel_x];
    output = output_row[pixel_x];
    output_row[pixel_x] = input/output;
  }
}


template <typename TransferT>
__global__ void device_finalValues_plain(TransferT * __restrict__ _image, 
					 const TransferT * __restrict__ _integral, 
					 const TransferT * __restrict__ _weight, 
					 TransferT _minValue,
					 size_t _size){
  
  const size_t pixel_x = size_t(blockIdx.x)*size_t(blockDim.x) + threadIdx.x;
  const size_t pixel_y = size_t(blockIdx.y)*size_t(blockDim.y) + threadIdx.y;
  const size_t pixel_index = pixel_y*size_t(gridDim.x)*size_t(blockDim.x) + pixel_x;
// const size_t pixel_index = size_t(blockIdx.x)*size_t(blockDim.x) + threadIdx.x;

  float temp_image = 0;
  
  float temp_weight = 0;;
  float new_value ;
  if(pixel_index<_size){
    //to exploit instruction level parallelism
    temp_image = _image[pixel_index];
    temp_image *= _integral[pixel_index];
    temp_weight = _weight[pixel_index];

    if(!(temp_image>0.f))
      temp_image = _minValue;

    new_value = cmax(_minValue,temp_image);
    new_value = temp_weight*(new_value - temp_image) + temp_image;

    _image[pixel_index] = new_value;
  }
  
}

template <typename TransferT>
__global__ void device_finalValues_plain_3D(cudaPitchedPtr _image , 
					    cudaPitchedPtr _integral,
					    cudaPitchedPtr _weight,
					    TransferT _minValue,
					    uint3 _image_dims){
  const size_t image_size = _image_dims.x*_image_dims.y*_image_dims.z;

  const unsigned int pixel_x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int pixel_y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int pixel_z = blockIdx.z*blockDim.z + threadIdx.z;
  
  const unsigned int pixel_index = pixel_z*(_image_dims.x*_image_dims.y) + pixel_y*(_image_dims.x) + pixel_x;

  char* image_devPtr = (char*)_image.ptr;
  char* integral_devPtr = (char*)_integral.ptr;
  char* weight_devPtr = (char*)_weight.ptr;
  
  float* image_row = (float*)(image_devPtr + pixel_z*(_image_dims.x*_image_dims.y) + pixel_y*(_image_dims.x));
  float* integral_row = (float*)(integral_devPtr + pixel_z*(_image_dims.x*_image_dims.y) + pixel_y*(_image_dims.x));
  float* weight_row = (float*)(weight_devPtr + pixel_z*(_image_dims.x*_image_dims.y) + pixel_y*(_image_dims.x));

  float temp_image = 0;
  float temp_weight = 0;
  float new_value ;
  if(pixel_index<image_size){
    //to exploit instruction level parallelism
    temp_image = image_row[pixel_x];
    temp_image *= integral_row[pixel_x];
    temp_weight = weight_row[pixel_x];

    if(!(temp_image>0.f))
      temp_image = _minValue;

    new_value = cmax(_minValue,temp_image);
    new_value = temp_weight*(new_value - temp_image) + temp_image;

    image_row[pixel_x] = new_value;
  }
  
}

template <typename TransferT>
__global__ void device_finalValues_tikhonov(TransferT * __restrict__ _image, 
					    const TransferT * __restrict__ _integral, 
					    const TransferT * __restrict__ _weight, 
					    TransferT _minValue,
					    TransferT _lambda,
					    size_t _size){
  
  const size_t pixel_x = size_t(blockIdx.x)*size_t(blockDim.x) + threadIdx.x;
  const size_t pixel_y = size_t(blockIdx.y)*size_t(blockDim.y) + threadIdx.y;
  const size_t pixel_index = pixel_y*size_t(gridDim.x)*size_t(blockDim.x) + pixel_x;
// const size_t pixel_index = size_t(blockIdx.x)*size_t(blockDim.x) + threadIdx.x;

  float temp_image = 0;
  
  float temp_weight = 0;
  float new_value ;
  if(pixel_index<_size){
    //to exploit instruction level parallelism
    temp_image = _image[pixel_index];
    temp_image *= _integral[pixel_index];
    temp_weight = _weight[pixel_index];

    if(temp_image>0.f)
      temp_image = (sqrt(1.0 + 2.0*_lambda*temp_image) - 1.)/_lambda;
    else
      temp_image = _minValue;

    new_value = cmax(_minValue,temp_image);
    new_value = temp_weight*(new_value - temp_image) + temp_image;

    _image[pixel_index] = new_value;
  }
  
}

__global__ void kernel3D_naive(cudaPitchedPtr _image , // const DimT* __restrict__ _image_dims,
			       cudaPitchedPtr _kernel, // const DimT* __restrict__ _kernel_dims,
			       cudaPitchedPtr _output,
			       uint3 _image_dims,
			       uint3 _kernel_dims){

  const unsigned int image_size = _image_dims.x*_image_dims.y*_image_dims.z;
  const unsigned int kernel_size = _kernel_dims.x*_kernel_dims.y*_kernel_dims.z;

  const unsigned int global_x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int global_y = blockIdx.y*blockDim.y + threadIdx.y;
  const unsigned int global_z = blockIdx.z*blockDim.z + threadIdx.z;

  const int pixel_x = (int)global_x + (int)_kernel_dims.x/2;
  const int pixel_y = (int)global_y + (int)_kernel_dims.y/2;
  const int pixel_z = (int)global_z + (int)_kernel_dims.z/2;
  
  const unsigned int pixel_index = pixel_z*(_image_dims.x*_image_dims.y) + pixel_y*(_image_dims.x) + pixel_x;
  unsigned int image_index;
  unsigned int kernel_index;
  float value = 0.f;
  float image_pixel = 0.f;
  float kernel_pixel = 0.f;
  
  char* input_devPtr = (char*)_image.ptr;
  char* output_devPtr = (char*)_output.ptr;
  char* kernel_devPtr = (char*)_kernel.ptr;

  char* input_slice  = 0;
  char* kernel_slice = 0;

  float* input_row  = 0;
  float* kernel_row = 0;


  for(int offset_z = -(_kernel_dims.z/2);offset_z<=(_kernel_dims.z/2);++offset_z){
    input_slice  = input_devPtr + (pixel_z+offset_z)*(_image_dims.x*_image_dims.y);    
    kernel_slice = kernel_devPtr + (offset_z + _kernel_dims.z/2)*(_kernel_dims.x*_kernel_dims.y);    

    for(int offset_y = -(_kernel_dims.y/2);offset_y<=(_kernel_dims.y/2);++offset_y){
      input_row  = (float*)(input_slice + (pixel_y + offset_y)*(_image_dims.x));
      kernel_row = (float*)(kernel_slice + (offset_y + _kernel_dims.y/2)*(_kernel_dims.x));      

      for(int offset_x = -(_kernel_dims.x/2);offset_x<=(_kernel_dims.x/2);++offset_x){
	image_index = (pixel_z+offset_z)*(_image_dims.x*_image_dims.y) + (pixel_y + offset_y)*(_image_dims.x) + pixel_x + offset_x;
	kernel_index = (offset_z + _kernel_dims.z/2)*(_kernel_dims.x*_kernel_dims.y) + (offset_y + _kernel_dims.y/2)*(_kernel_dims.x) + (offset_x + _kernel_dims.x/2);

	if(image_index < image_size)
	  image_pixel = input_row[pixel_x + offset_x];
	
	if(kernel_index < kernel_size)
	  kernel_pixel = kernel_row[offset_x + _kernel_dims.x/2];

	value += image_pixel*kernel_pixel;
      }
    }
  }

  if(pixel_index<image_size){
    char* output_slice = output_devPtr + pixel_z*(_image_dims.x*_image_dims.y);
    float* output_row = (float*)(output_slice + pixel_y*(_image_dims.x));
    output_row[0] = value;
  }
}


#endif /* _COMPUTE_KERNELS_CUH_ */
