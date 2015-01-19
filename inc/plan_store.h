#ifndef _PLAN_STORE_H_
#define _PLAN_STORE_H_
#include <map>
#include <sstream>
#include <iterator>
#include <stdexcept>

#include "point.h"

namespace multiviewnative {

  namespace mvn = multiviewnative;
  
  template <typename plan_type>
  struct plan_store
  {
    typedef plan_type plan_t;
    typedef std::map<mvn::shape_t, plan_type> map_t;
    typedef typename map_t::iterator map_iter_t;
    typedef typename map_t::const_iterator map_citer_t;
    
    map_t store_;
    

    bool empty() const {
      return store_.empty();
    }

    void add(const mvn::shape_t& _key,
	     plan_t& _value) {
      store_[_key] = _value;
    }
    
    plan_t const & get(const mvn::shape_t& _key) const {
      map_citer_t found = store_.find(_key);
      
      if(found!=store_.end())
	return found->second;
      else{
	std::stringstream stream;
	stream << "[multiviewnative::plan_store] key ";
	std::copy(_key.begin(), _key.end(), std::ostream_iterator<unsigned>(stream,"x"));
	stream << "not found in store\n";
	
	std::runtime_error my_x(stream.str());
	throw my_x;
	  };
    }
    
    plan_t& get(const mvn::shape_t& _key) {
      map_iter_t found = store_.find(_key);
      
      if(found!=store_.end())
	return found->second;
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
      map_citer_t found = store_.find(_key);
      
      if(found!=store_.end())
	return true;
      else
	return false;
    }
  };
  
};
#endif /* _PLAN_STORE_H_ */
