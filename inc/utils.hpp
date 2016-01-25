#ifndef _UTILS_H_
#define _UTILS_H_

namespace multiviewnative {

    template <typename T>
  typename std::enable_if<std::is_pointer<T>::value,
			  T>::type
    begin(T _array){

    return &_array[0];
    
  }

  template <typename container_t>
  typename std::enable_if<std::is_class<container_t>::value && sizeof(typename container_t::value_type)!=0,
			  typename container_t::value_type*>::type
  begin(container_t& _array){

    return &_array[0];
    
  }

  template <typename container_t>
  typename std::enable_if<std::is_class<container_t>::value && sizeof(typename container_t::value_type)!=0,
			  const typename container_t::value_type*>::type
  begin(const container_t& _array){

    return &_array[0];
    
  }

  template <typename container_t>
  typename std::enable_if<std::is_class<container_t>::value && sizeof(typename container_t::value_type)!=0
			  //has_member_function size,
			  ,
			  typename container_t::value_type*>::type
  end(container_t& _array){

    return &_array[_array.size()];
    
  }

  template <typename container_t>
  typename std::enable_if<std::is_class<container_t>::value && sizeof(typename container_t::value_type)!=0
			  //has_member_function size,
			  ,
			  const typename container_t::value_type*>::type
  end(const container_t& _array){

    return &_array[_array.size()];
    
  }

  
};

#endif /* _UTILS_H_ */
