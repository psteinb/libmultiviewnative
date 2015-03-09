#ifndef _PLAN_STORE_H_
#define _PLAN_STORE_H_
#include <map>
#include <sstream>
#include <iterator>
#include <stdexcept>
#include <memory>
#include "fftw_interface.h"
#include "point.h"

namespace multiviewnative {

namespace mvn = multiviewnative;

template <typename T>
struct aligned {
  typedef fftw_allocator<T> allocator;
};

template <typename T>
struct unaligned {
  typedef std::allocator<T> allocator;
};

template <typename fp_type,
          typename MemoryAlignmentPolicy = unaligned<fp_type> >
struct plan_store {

  typedef fftw_api_definitions<fp_type> fftw_api;
  typedef typename fftw_api::plan_type plan_t;
  typedef typename fftw_api::complex_type complex_t;

  typedef std::map<mvn::shape_t, plan_t*> map_t;
  typedef typename map_t::iterator map_iter_t;
  typedef typename map_t::const_iterator map_citer_t;

  map_t fwd_store_;  // r2c
  map_t bwd_store_;  // c2r

  static plan_store* get() {
    static plan_store instance;
    return &instance;
  }

  bool empty() const { return fwd_store_.empty() && bwd_store_.empty(); }

  unsigned size() const {
    return std::max(fwd_store_.size(), bwd_store_.size());
  }

  bool has_key(const mvn::shape_t& _key) const {
    map_citer_t fwd_found = fwd_store_.find(_key);
    map_citer_t bwd_found = bwd_store_.find(_key);

    return fwd_found != fwd_store_.end() && bwd_found != bwd_store_.end();
  }

  friend std::ostream& operator<<(std::ostream& _stream,
                                  const plan_store& _self) {
    auto fwd_itr = _self.fwd_store_.cbegin();
    for (; fwd_itr != _self.fwd_store_.cend(); ++fwd_itr) {
      _stream << "fwd key shape = ";
      for (auto i : fwd_itr->first) _stream << i << " ";
      _stream << "\n";
    }

    auto bwd_itr = _self.bwd_store_.cbegin();
    for (; bwd_itr != _self.bwd_store_.cend(); ++bwd_itr) {
      _stream << "bwd key shape = ";
      for (auto i : bwd_itr->first) _stream << i << " ";
      _stream << "\n";
    }

    return _stream;
  }

  friend std::ostream& operator<<(std::ostream& _stream,
                                  const plan_store* _self) {

    _stream << *_self;

    return _stream;
  }

  /**
     \brief add a shape and prepare the corresponding backward and forward plan

     \param[in] _shape std::vector<unsigned> container typedef'ed to
     mvn::shape_t

     This function creates a temporary buffer of the size of _shape in order to
     supply something to fftw! The container itself will be unaligned and hence
     fftw will not dispatch SSE enabled functions.

     \return
     \retval

  */
  void add(const mvn::shape_t& _shape) {

    // http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags
    //"Important: the planner overwrites the input array during planning unless
    //a saved plan (see Wisdom) is available for that problem, so you should
    //initialize your input data after creating the plan."
    unsigned long total_size = std::accumulate(_shape.begin(), _shape.end() - 1,
                                               1, std::multiplies<unsigned>());
    total_size *= (_shape[_shape.size() - 1] / 2 + 1) * 2;

    // using unaligned container
    std::vector<fp_type, typename MemoryAlignmentPolicy::allocator> input(
        total_size, 0);

    complex_t* output = (complex_t*)&input[0];

    if (fwd_store_.find(_shape) == fwd_store_.end())
      fwd_store_[_shape] = new plan_t(fftw_api::dft_r2c_3d(
          _shape[0], _shape[1], _shape[2], &input[0], output, FFTW_MEASURE));

    if (bwd_store_.find(_shape) == bwd_store_.end()) {

      bwd_store_[_shape] = new plan_t(fftw_api::dft_c2r_3d(
          _shape[0], _shape[1], _shape[2], output, &input[0], FFTW_MEASURE));
    }
  }

  plan_t* const get_forward(const mvn::shape_t& _key) const {
    map_citer_t found = fwd_store_.find(_key);

    if (found != fwd_store_.end())
      return (found->second);
    else {
      std::stringstream stream;
      stream << "[multiviewnative::plan_store] key ";
      std::copy(_key.begin(), _key.end(),
                std::ostream_iterator<unsigned>(stream, "x"));
      stream << "not found in store\n";

      std::runtime_error my_x(stream.str());
      throw my_x;
    };
  }

  plan_t* get_forward(const mvn::shape_t& _key) {
    map_iter_t found = fwd_store_.find(_key);

    if (found != fwd_store_.end())
      return (found->second);
    else {
      std::stringstream stream;
      stream << "[multiviewnative::plan_store] key ";
      std::copy(_key.begin(), _key.end(),
                std::ostream_iterator<unsigned>(stream, "x"));
      stream << "not found in store\n";

      std::runtime_error my_x(stream.str());
      throw my_x;
    };
  }

  plan_t const* get_backward(const mvn::shape_t& _key) const {
    map_citer_t found = bwd_store_.find(_key);

    if (found != bwd_store_.end())
      return (found->second);
    else {
      std::stringstream stream;
      stream << "[multiviewnative::plan_store] key ";
      std::copy(_key.begin(), _key.end(),
                std::ostream_iterator<unsigned>(stream, "x"));
      stream << "not found in store\n";

      std::runtime_error my_x(stream.str());
      throw my_x;
    };
  }

  plan_t* get_backward(const mvn::shape_t& _key) {
    map_iter_t found = bwd_store_.find(_key);

    if (found != bwd_store_.end())
      return (found->second);
    else {
      std::stringstream stream;
      stream << "[multiviewnative::plan_store] key ";
      std::copy(_key.begin(), _key.end(),
                std::ostream_iterator<unsigned>(stream, "x"));
      stream << "not found in store\n";

      std::runtime_error my_x(stream.str());
      throw my_x;
    };
  }

  void clear() {
    map_iter_t begin = fwd_store_.begin();
    map_iter_t end = fwd_store_.end();
    for (; begin != end; ++begin) {
      fftw_api::destroy_plan(*begin->second);
      delete begin->second;
      begin->second = 0;
      fwd_store_.erase(begin);
    }

    begin = bwd_store_.begin();
    end = bwd_store_.end();
    for (; begin != end; ++begin) {
      fftw_api::destroy_plan(*begin->second);
      delete begin->second;
      begin->second = 0;
      bwd_store_.erase(begin);
    }
  }

 private:
  plan_store() : fwd_store_(), bwd_store_() {}

  ~plan_store() { clear(); }

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
