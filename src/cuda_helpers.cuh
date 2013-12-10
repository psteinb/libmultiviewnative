#ifndef _CUDA_HELPERS_H_
#define _CUDA_HELPERS_H_

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

	memcpy(name,prop.name,sizeof(char)*256);
}

long long int getMemDeviceCUDA(int devCUDA)
{
	cudaDeviceProp prop;
	HANDLE_ERROR( cudaGetDeviceProperties(&prop, devCUDA));
	return ((long long int)prop.totalGlobalMem);
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
#endif /* _CUDA_HELPERS_H_ */

