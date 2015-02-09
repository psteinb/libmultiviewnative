#ifndef _CUDA_HELPERS_H_
#define _CUDA_HELPERS_H_
#include <cstring>
#include <iostream>
#include <iomanip>
#include "cuda.h"
#include "cuda_runtime.h"

template <typename ValueT>
struct multiplies : std::binary_function<ValueT, ValueT, ValueT> {

  ValueT operator()(const ValueT& _first, const ValueT& _second) const {
    return _first*_second;
  }

};

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
      std::cerr << cudaGetErrorString( err ) << " in " << file << " at line " << line << std::endl;
      cudaDeviceReset();
      exit( EXIT_FAILURE );
    }
}



#define HANDLE_ERROR_KERNEL HandleError(cudaPeekAtLastError(),__FILE__, __LINE__ )
#define HANDLE_LAST_ERROR( err ) (HandleError(cudaPeekAtLastError(),__FILE__, __LINE__ ))

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
  std::cout << "Host memory failed in "<< __FILE__ <<" at line " << __LINE__<<"\n"; \
  exit( EXIT_FAILURE );}}




template <typename T>
T largestDivisor(const T& _to_divide, const T& _divided_by){
  return (_to_divide + _divided_by -1)/(_divided_by);
};


size_t getMaxNThreadsOfDevice(short _devId){
  cudaDeviceProp prop;
  HANDLE_ERROR( cudaGetDeviceProperties(&prop, _devId));
  return (size_t)prop.maxThreadsPerBlock;
}

size_t getMaxNBlocksOfDevice(short _devId,short _dim){
  cudaDeviceProp prop;
  HANDLE_ERROR( cudaGetDeviceProperties(&prop, _devId));
  return (size_t)prop.maxGridSize[_dim];
}

void fit_2Dblocks_to_threads_for_device(dim3& _blocks, dim3& _threads, const int& _device){
  
const size_t max_blocks =  getMaxNBlocksOfDevice(_device,0);

  if(_blocks.x > max_blocks){
    _blocks.y = (_blocks.x+max_blocks-1)/max_blocks;
    _blocks.x = max_blocks;
  }
  
  
}



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
	std::copy(prop.name, prop.name + 256, name);
	//	memcpy(name,prop.name,sizeof(char)*256);
}

std::string get_cuda_device_name(int devCUDA)
{	
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties(&prop, devCUDA));
	
	return std::string(prop.name,prop.name + strlen(prop.name));
}


long long int getMemDeviceCUDA(int devCUDA)
{
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties(&prop, devCUDA));
	return ((long long int)prop.totalGlobalMem);
}

long long int getAvailableGMemOnCurrentDevice()
{
  size_t free, total;
  HANDLE_ERROR( cudaMemGetInfo(&free, &total));
  return ((long long int)free);
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

template <typename ArrT, typename DimT, typename ODimT>
void print_3d_array(ArrT* _array, DimT* _array_dims, ODimT* _order){
std::cout << "received: " << _array_dims[0] << "x" << _array_dims[1] << "x" << _array_dims[2] << " array ("
	    << _order[0] << ", "<< _order[1] << ", "<< _order[2] << ")\n";

  int precision = std::cout.precision();
  std::cout << std::setprecision(4);

  
  std::cout << std::setw(9) << "x = ";
  for(DimT x_index = 0;x_index<(_array_dims[0]);++x_index){
	std::cout << std::setw(8) << x_index << " ";
  }
  std::cout << "\n";
  std::cout << std::setfill('-') << std::setw((_array_dims[0]+1)*9) << " " << std::setfill(' ')<< std::endl ;
  unsigned flat_index = 0;
  size_t index[3];


  for(index[2] = 0;index[2]<size_t(_array_dims[2]);++index[2]){
    std::cout << "z["<< std::setw(5) << index[2] << "] \n";
    
      for(index[1] = 0;index[1]<size_t(_array_dims[1]);++index[1]){
	std::cout << "y["<< std::setw(5) << index[1] << "] ";

	for(index[0] = 0;index[0]<size_t(_array_dims[0]);++index[0]){

	  flat_index = index[_order[0]] + _array_dims[_order[0]]*(index[_order[1]] + _array_dims[_order[1]]*index[_order[2]]);

	  std::cout << std::setw(8) << _array[flat_index] << " ";
	}

	std::cout << "\n";
      }
      std::cout << "\n";
    }

  std::cout << std::setprecision(precision);
}

template <typename ArrT, typename DimT, typename ODimT>
void print_device_array(ArrT* _device_address, DimT* _array_dims, ODimT* _order){
  
  
  size_t total = std::accumulate(_array_dims,_array_dims + 3,1.,std::multiplies<int>());
  ArrT* array = new ArrT[total];
  std::cout << "transferring memory from device -> host: " << total << "items ("<< total*sizeof(ArrT)<<"B)\n";
  HANDLE_ERROR( cudaMemcpy( array, _device_address, total*sizeof(ArrT) , cudaMemcpyDeviceToHost ) );
  cudaDeviceSynchronize();

  int precision = std::cout.precision();
  std::cout << std::setprecision(4);
  std::cout << "raw data in memory:\n";
  for(unsigned idx = 0;idx < ((256 > total) ? total : 256);++idx){
    std::cout << array[idx] << (((idx+1) % 32 == 0) ? "\n" : " ");
  }

  std::cout << "\n" << std::setprecision(precision);

  print_3d_array(array, _array_dims, _order);

  delete [] array;

}

#endif /* _CUDA_HELPERS_H_ */

