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


__global__ void  modulateAndNormalize_kernel(cufftComplex *d_Dst, 
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


#endif /* _COMPUTE_KERNELS_CUH_ */
