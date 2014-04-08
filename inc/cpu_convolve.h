#ifndef _CPU_CONVOLVE_H_
#define _CPU_CONVOLVE_H_

#include <algorithm>
#include <numeric>
#include <iostream> 
#include <iomanip> 
#include <vector>


#include "boost/multi_array.hpp"
#include "fftw3.h"

namespace multiviewnative {

  typedef  boost::multi_array<float,              3>    image_stack;
  typedef  boost::multi_array_ref<float,          3>    image_stack_ref;
  typedef  image_stack::array_view<3>::type		image_stack_view;
  typedef  boost::multi_array_types::index_range	range;
  typedef  boost::general_storage_order<3>		storage;

  //TODO/REFACTOR: the enums could/should be replaced by objects (then use CRTP or design by policy)
  template <MemStrategy mstrategy = InPlace, CPUStrategy cstrategy = no_wisdom, typename TransferT, typename SizeT>
  struct cpu_convolve {

    typedef TransferT value_type;
    typedef SizeT size_type;

    static const MemStrategy memory_strategy = mstrategy;
    static const int num_dims = 3;
    static const CPUStrategy cpu_strategy = cstrategy;

    image_stack_ref image_;
    image_stack padded_image_;

    image_stack_ref kernel_;
    image_stack padded_kernel_;
    

    cpu_convolve(TransferT* _image_stack_arr, SizeT* _image_extents_arr,
		 TransferT* _kernel_stack_arr, SizeT* _kernel_extents_arr, 
		 SizeT* _storage_order = 0):
      image_(),
      padded_image_(),
      kernel_(),
      padded_kernel_()
    {
      std::vector<SizeT> image_shape(num_dims);
      std::copy(_image_extents_arr, _image_extents_arr+num_dims,image_shape.begin());

      std::vector<SizeT> kernel_shape(num_dims);
      std::copy(_kernel_extents_arr, _kernel_extents_arr+num_dims,kernel_shape.begin());
      
      storage local_order;
      if(!_storage_order)
	local_order = boost::multi_array::c_storage_order<num_dims>();
      else{
	bool ascending[3] = {true, true, true};
	local_order = storage(_storage_order,ascending);
      }
      
      image_ = image_stack_ref(_image_stack_arr, image_shape, local_order);
      kernel_ = kernel_stack_ref(_kernel_stack_arr, kernel_shape, local_order);
      
    };

    void perform();

    void results(TransferT* _image_stack_arr);

  private:
    
    void prepare(){
      //do padding and wrap kernel
      
      //update image depending on Inplace/OutofPlace

    };

    void fold(){
      //perform convolution
      //scale result
    };

  };

}

#endif /* _CPU_CONVOLVE_H_ */
