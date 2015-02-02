#ifndef _PLAN_STORE_H_
#define _PLAN_STORE_H_
#include <map>
#include <sstream>
#include <iterator>
#include <stdexcept>
#include "fftw_interface.h"
#include "point.h"

namespace multiviewnative {

  namespace mvn = multiviewnative;
  
  template <typename fp_type>
  struct plan_store
  {

    typedef fftw_api_definitions<fp_type> fftw_api;
    typedef typename fftw_api::plan_type plan_t;
    typedef typename fftw_api::complex_type complex_t;

    typedef std::map<mvn::shape_t, plan_t> map_t;
    typedef typename map_t::iterator map_iter_t;
    typedef typename map_t::const_iterator map_citer_t;
    
    map_t fwd_store_;//r2c
    map_t bwd_store_;//c2r

    plan_store():
      fwd_store_(),
      bwd_store_()
    {}
    
    ~plan_store(){
      map_iter_t begin = fwd_store_.begin();
      map_iter_t end = fwd_store_.end();
      for(;begin!=end;++begin){
	fftw_api::destroy_plan(begin->second);
      }
      
      begin = bwd_store_.begin();
      end = bwd_store_.end();
      for(;begin!=end;++begin){
	fftw_api::destroy_plan(begin->second);
      }
      
    }

    bool empty() const {
      return fwd_store_.empty() && bwd_store_.empty();
    }

    void add(const mvn::shape_t& _shape,
	     fp_type* _input = 0,
	     complex_t* _output = 0
	     ) {
      
      if(fwd_store_.find(_shape)==fwd_store_.end())
	fwd_store_[_shape] = fftw_api::dft_r2c_3d(_shape[0], 
						_shape[1], 
						_shape[2],
						_input, _output,
						FFTW_MEASURE);
      
      if(bwd_store_.find(_shape)==bwd_store_.end())
	bwd_store_[_shape] = fftw_api::dft_c2r_3d(_shape[0], 
						_shape[1], 
						_shape[2],
						_output, _input,
						FFTW_MEASURE);

      
    }
    
    plan_t* const get_forward(const mvn::shape_t& _key) const {
      map_citer_t found = fwd_store_.find(_key);
      
      if(found!=fwd_store_.end())
	return &(found->second);
      else{
	std::stringstream stream;
	stream << "[multiviewnative::plan_store] key ";
	std::copy(_key.begin(), _key.end(), std::ostream_iterator<unsigned>(stream,"x"));
	stream << "not found in store\n";
	
	std::runtime_error my_x(stream.str());
	throw my_x;
      };
    }
    
    plan_t* get_forward(const mvn::shape_t& _key) {
      map_iter_t found = fwd_store_.find(_key);
      
      if(found!=fwd_store_.end())
	return &(found->second);
      else{
	std::stringstream stream;
	stream << "[multiviewnative::plan_store] key ";
	std::copy(_key.begin(), _key.end(), std::ostream_iterator<unsigned>(stream,"x"));
	stream << "not found in store\n";
	
	std::runtime_error my_x(stream.str());
	throw my_x;
      };
    }

    plan_t const * get_backward(const mvn::shape_t& _key) const {
      map_citer_t found = bwd_store_.find(_key);
      
      if(found!=bwd_store_.end())
	return &(found->second);
      else{
	std::stringstream stream;
	stream << "[multiviewnative::plan_store] key ";
	std::copy(_key.begin(), _key.end(), std::ostream_iterator<unsigned>(stream,"x"));
	stream << "not found in store\n";
	
	std::runtime_error my_x(stream.str());
	throw my_x;
      };
    }
    
    plan_t* get_backward(const mvn::shape_t& _key) {
      map_iter_t found = bwd_store_.find(_key);
      
      if(found!=bwd_store_.end())
	return &(found->second);
      else{
	std::stringstream stream;
	stream << "[multiviewnative::plan_store] key ";
	std::copy(_key.begin(), _key.end(), std::ostream_iterator<unsigned>(stream,"x"));
	stream << "not found in store\n";
	
	std::runtime_error my_x(stream.str());
	throw my_x;
      };
    }


    bool has_key(const mvn::shape_t& _key) const {
      map_citer_t fwd_found = fwd_store_.find(_key);
      map_citer_t bwd_found = bwd_store_.find(_key);
      
      return fwd_found!=fwd_store_.end() && bwd_found!=bwd_store_.end();

    }

    // map_iter_t begin() {
    //   return store_.begin();
    // }
    
    // map_iter_t end() {
    //   return store_.end();
    // }

    // map_citer_t begin() const {
    //   return store_.begin();
    // }
    
    // map_citer_t end() const {
    //   return store_.end();
    // }

  };
  
};
#endif /* _PLAN_STORE_H_ */
