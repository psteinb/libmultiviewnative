#ifndef _TIFF_FIXTURES_H_
#define _TIFF_FIXTURES_H_
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <stdexcept>
#include "boost/multi_array.hpp"
#include <boost/filesystem.hpp>

#include <string>
#include <sstream>
#include "image_stack_utils.h"
#include "tiff_utils.h"
#include "tests_config.h"

////////////////////////////////////////////////////////////////////////////
// Explanation of the test images
// input :
// image_view_i.tif	.. the input frame stack from view i
// kernel1_view_i.tif	.. integrating PSF for view i
// kernel2_view_i.tif	.. conditional pdf of all views for view i
// weights_view_i.tif	.. the weights from view i
// results:
// psi_i.tif		.. the results after the i-th iteration
// psi_0		.. first guess (all pixels have the same intensity)

namespace multiviewnative {


struct tiff_stack {

  std::string stack_path_;
  image_stack stack_;
  static const int dimensionality = image_stack::dimensionality;

  tiff_stack() : stack_path_(""), stack_() {}

  tiff_stack(const std::string& _path) : stack_path_(_path), stack_() {
    if (boost::filesystem::is_regular_file(_path) || !_path.empty())
      load(_path);
    else
      std::cerr << "Unable to load file " << _path << "\n";
  }

  tiff_stack(const tiff_stack& _rhs) : stack_path_(_rhs.stack_path_), stack_() {

    std::vector<unsigned> dims(3);
    for (int i = 0; i < 3; ++i) dims[i] = _rhs.stack_.shape()[i];

    stack_.resize(dims);

    stack_ = _rhs.stack_;

    has_malformed_floats();
  }

  tiff_stack& operator=(const tiff_stack& _rhs) {

    stack_path_ = _rhs.stack_path_;

    std::vector<unsigned> dims(3);
    for (int i = 0; i < 3; ++i) dims[i] = _rhs.stack_.shape()[i];

    stack_.resize(dims);
    stack_ = _rhs.stack_;

    has_malformed_floats();

    return *this;
  }

  void load(const std::string& _path, bool _verbose = false) {
    if (!boost::filesystem::is_regular_file(_path) || _path.empty()) {
      std::stringstream msg("");
      msg << "unable to load file at path " << _path << "\n";
      throw std::runtime_error(msg.str().c_str());
    }

    stack_path_ = _path;
    TIFF* stack_tiff = TIFFOpen(_path.c_str(), "r");
    std::vector<tdir_t> stack_tdirs;
    get_tiff_dirs(stack_tiff, stack_tdirs);
    unsigned stack_size = stack_.num_elements();
    extract_tiff_to_image_stack(stack_tiff, stack_tdirs, stack_);
    TIFFClose(stack_tiff);

    if (stack_size - stack_.num_elements() != 0 && stack_.num_elements() > 0) {
      std::stringstream dim_string("");
      dim_string << "(c_storage_order: z-y-x) = [" << stack_.shape()[0] << "x"
                 << stack_.shape()[1] << "x" << stack_.shape()[2] << "]";

      if(_verbose)
	std::cout << "successfully loaded " << std::setw(20) << dim_string.str()
		  << " stack from " << _path << "\n";

      #ifdef LMVN_TRACE
      std::cout << "[trace::"<<__FILE__<<"] loaded " << std::setw(20) << dim_string.str()
		  << " stack from " << _path << "\n";
      #endif
    }
    has_malformed_floats();
  }

  bool has_malformed_floats() const {
    using namespace std;

    bool value = false;

    unsigned size = stack_.num_elements();
    bool b_is_nan = false;
    bool b_is_inf = false;
    const float* data = stack_.data();

    for (unsigned i = 0; i < size; ++i) {

      b_is_nan = std::isnan(data[i]);
      b_is_inf = std::isinf(data[i]);

      if (b_is_nan || b_is_inf) {
        std::cerr << "encountered malformed pixel in [" << stack_path_
                  << "]: index = " << i
                  << " type: " << ((b_is_nan) ? "nan" : "inf") << "\n";
        value = true;
        break;
      }
    }

    return value;
  }

  ~tiff_stack() {}

  bool empty() { return !(stack_.num_elements() > 0); }
};

struct ViewFromDisk {

  int view_number_;

  std::stringstream image_path_;
  std::stringstream kernel1_path_;
  std::stringstream kernel2_path_;
  std::stringstream weights_path_;

  tiff_stack image_;
  tiff_stack kernel1_;
  tiff_stack kernel2_;
  tiff_stack weights_;

  std::vector<int> image_dims_;
  std::vector<int> kernel1_dims_;
  std::vector<int> kernel2_dims_;
  std::vector<int> padded_dims_;
  std::vector<int> offsets_in_padded_;
  
  ViewFromDisk(const int& _view_number = -1)
      : view_number_(_view_number),
        image_path_(""),
        kernel1_path_(""),
        kernel2_path_(""),
        weights_path_(""),
        image_(),
        kernel1_(),
        kernel2_(),
        weights_(),
        image_dims_(tiff_stack::dimensionality,0),
        kernel1_dims_(tiff_stack::dimensionality,0),
        kernel2_dims_(tiff_stack::dimensionality,0),
	padded_dims_(tiff_stack::dimensionality,0),
	offsets_in_padded_(tiff_stack::dimensionality,0)
  {
    if (view_number_ > -1) this->load(view_number_);
  }

  ViewFromDisk(const ViewFromDisk& _rhs)
      : view_number_(_rhs.view_number_),
        image_path_(_rhs.image_path_.str()),
        kernel1_path_(_rhs.kernel1_path_.str()),
        kernel2_path_(_rhs.kernel2_path_.str()),
        weights_path_(_rhs.weights_path_.str()),
        image_(_rhs.image_),
        kernel1_(_rhs.kernel1_),
        kernel2_(_rhs.kernel2_),
        weights_(_rhs.weights_),
        image_dims_(_rhs.image_dims_),
        kernel1_dims_(_rhs.kernel1_dims_),
        kernel2_dims_(_rhs.kernel2_dims_),
	padded_dims_	  (_rhs.padded_dims_      ),
	offsets_in_padded_(_rhs.offsets_in_padded_) {}

  ViewFromDisk& operator=(const ViewFromDisk& _rhs) {

    view_number_ = _rhs.view_number_;
    image_path_.str(_rhs.image_path_.str());
    kernel1_path_.str(_rhs.kernel1_path_.str());
    kernel2_path_.str(_rhs.kernel2_path_.str());
    weights_path_.str(_rhs.weights_path_.str());

    image_ = _rhs.image_;
    kernel1_ = _rhs.kernel1_;
    kernel2_ = _rhs.kernel2_;
    weights_ = _rhs.weights_;

    image_dims_ = _rhs.image_dims_;
    kernel1_dims_ = _rhs.kernel1_dims_;
    kernel2_dims_ = _rhs.kernel2_dims_;

    padded_dims_      =_rhs.padded_dims_      ;
    offsets_in_padded_=_rhs.offsets_in_padded_;

    return *this;
  }

  ViewFromDisk& operator=(const int& _view_number) {

    view_number_ = _view_number;

    if (view_number_ > -1) this->load(view_number_);

    return *this;
  }

  template <typename container_type>
  void padd_with_shape(const container_type& _shape,
                       unsigned _num_kernel_widths = 1) {

    // create new image shapes
    std::copy(image_.stack_.shape(),
	      image_.stack_.shape() + tiff_stack::dimensionality,
	      padded_dims_.begin());
    // std::vector<unsigned> new_image_shape(
    //     image_.stack_.shape(),
    //     image_.stack_.shape() + tiff_stack::dimensionality);
    // std::vector<unsigned> offsets(tiff_stack::dimensionality, 0);

    // indices of interest :)
    std::vector<range> ioi(tiff_stack::dimensionality);
    unsigned count = 0;
    for (auto& d : padded_dims_) {
      offsets_in_padded_[count] = _num_kernel_widths * (_shape[count] / 2);
      d = d + 2 * offsets_in_padded_[count];
      ioi[count] = range(offsets_in_padded_[count], d - offsets_in_padded_[count]);
      count++;
    }

    image_stack padded_temp_(padded_dims_);
    padded_temp_[boost::indices[ioi[0]][ioi[1]][ioi[2]]] = image_.stack_;
    image_.stack_.resize(padded_dims_);
    image_.stack_ = padded_temp_;
    std::copy(&image_.stack_.shape()[0], &image_.stack_.shape()[0] + 3,
              image_dims_.begin());

    padded_temp_[boost::indices[ioi[0]][ioi[1]][ioi[2]]] = weights_.stack_;
    weights_.stack_.resize(padded_dims_);
    weights_.stack_ = padded_temp_;
  }

  void load(const int& _view_number) {

    // boost file system magic required here!

    view_number_ = _view_number;

    image_path_ << path_to_test_images << "input_view_" << view_number_
                << ".tif";
    kernel1_path_ << path_to_test_images << "kernel1_view_" << view_number_
                  << ".tif";
    kernel2_path_ << path_to_test_images << "kernel2_view_" << view_number_
                  << ".tif";
    weights_path_ << path_to_test_images << "weights_view_" << view_number_
                  << ".tif";

    image_.load(image_path_.str());
    kernel1_.load(kernel1_path_.str());
    kernel2_.load(kernel2_path_.str());
    weights_.load(weights_path_.str());

    std::copy(&image_.stack_.shape()[0], &image_.stack_.shape()[0] + 3,
              image_dims_.begin());
    std::copy(&kernel1_.stack_.shape()[0], &kernel1_.stack_.shape()[0] + 3,
              kernel1_dims_.begin());
    std::copy(&kernel2_.stack_.shape()[0], &kernel2_.stack_.shape()[0] + 3,
              kernel2_dims_.begin());
  }

  void load(const ViewFromDisk& _rhs) {

    view_number_ = _rhs.view_number_;
    image_path_.str(_rhs.image_path_.str());
    kernel1_path_.str(_rhs.kernel1_path_.str());
    kernel2_path_.str(_rhs.kernel2_path_.str());
    weights_path_.str(_rhs.weights_path_.str());

    image_ = _rhs.image_;
    kernel1_ = _rhs.kernel1_;
    kernel2_ = _rhs.kernel2_;
    weights_ = _rhs.weights_;

    image_dims_ = _rhs.image_dims_;
    kernel1_dims_ = _rhs.kernel1_dims_;
    kernel2_dims_ = _rhs.kernel2_dims_;
  }

  virtual ~ViewFromDisk() {};

  image_stack* image() { return &(image_.stack_); }

  const image_stack* image() const { return &(image_.stack_); }

  const image_stack* kernel1() const { return &(kernel1_.stack_); }

  image_stack* kernel1() { return &(kernel1_.stack_); }

  const image_stack* weights() const { return &(weights_.stack_); }

  image_stack* weights() { return &(weights_.stack_); }

  const image_stack* kernel2() const { return &(kernel2_.stack_); }

  image_stack* kernel2() { return &(kernel2_.stack_); }

  int* image_dims() { return &(image_dims_[0]); }

  int* kernel1_dims() { return &(kernel1_dims_[0]); }

  int* kernel2_dims() { return &(kernel2_dims_[0]); }

  const int* image_dims() const { return &(image_dims_[0]); }

  const int* kernel1_dims() const { return &(kernel1_dims_[0]); }

  const int* kernel2_dims() const { return &(kernel2_dims_[0]); }
};

template <int num_views = 6, bool padd_input_by_kernel = false>
struct ReferenceData_Impl {

  std::vector<ViewFromDisk> views_;
  static const int size = num_views;

  ReferenceData_Impl() : views_(size) {

    // check how many views are available on disk
    // boost file system

    for (unsigned i = 0; i < views_.size(); ++i) {
      views_[i].load(i);
    }

    if (padd_input_by_kernel) {
      std::vector<int> extents;
      min_kernel_shape(extents);

      for (unsigned i = 0; i < views_.size(); ++i) {
#ifdef LMVN_TRACE
	std::cout << "[trace::"<< __FILE__ <<"] padding view " << i << " ";
	for(int im = 0;im<3;++im)
	  std::cout << views_[i].image_dims()[im] << " ";
	std::cout << " with ";
	for( int ex : extents)
	  std::cout << ex << " ";
	std::cout << "\n";
	
#endif
        views_[i].padd_with_shape(extents);
	
      }
    }
  }

  ReferenceData_Impl(const ReferenceData_Impl& _rhs) : views_(_rhs.views_) {}

  void copy_in(const ReferenceData_Impl& _other) {
    for (int i = 0; i < 6; ++i) views_[i] = _other.views_[i];
  }

  template <typename container_type>
  void max_kernel_shape(container_type& _shape) {
    typedef typename container_type::value_type int_type;

    _shape.clear();
    _shape.resize(views_[0].image_dims_.size());
    std::fill(_shape.begin(), _shape.end(), 0);

    for (const ViewFromDisk& _v : views_) {
      for (int i = 0; i < _shape.size(); ++i)
        _shape[i] = std::max(_shape[i], (int_type)_v.kernel1_dims()[i]);
    }
  }

  template <typename container_type>
  void min_kernel_shape(container_type& _shape) {

    typedef typename container_type::value_type int_type;
    _shape.clear();
    _shape.resize(views_[0].image_dims_.size());
    std::fill(_shape.begin(), _shape.end(),
              std::numeric_limits<typename container_type::value_type>::max());

    for (const ViewFromDisk& _v : views_) {
      for (int i = 0; i < _shape.size(); ++i)
        _shape[i] = std::min(_shape[i], (int_type)_v.kernel1_dims()[i]);
    }
  }

  template <typename container_type>
  void min_image_shape(container_type& _shape) {
    _shape.clear();
    _shape.resize(views_[0].image_dims_.size());
    std::fill(_shape.begin(), _shape.end(),
              std::numeric_limits<typename container_type::value_type>::max());

    for (const ViewFromDisk& _v : views_) {
      for (int i = 0; i < _shape.size(); ++i)
        _shape[i] = std::min(_shape[i], _v.image_dims()[i]);
    }
  }
};

typedef ReferenceData_Impl<6, true> PaddedReferenceData;
typedef ReferenceData_Impl<6> RawReferenceData;

template <unsigned max_num_psi = 2, bool padded_images = false>
struct IterationData {

  std::vector<tiff_stack> psi_;
  std::vector<image_stack> padded_psi_;
  std::vector<std::array<range,image_stack::dimensionality> > padded_ranges_;
  std::vector<std::string> psi_paths_;

  std::string path_to_images;
  std::string iteration_basename_before_id;

  double lambda_;
  float minValue_;
  float psi0_avg_;

  IterationData(std::string _basename = "psi_")
      : psi_(),
        padded_psi_(max_num_psi),
	padded_ranges_(max_num_psi),
        psi_paths_(max_num_psi),
        path_to_images(path_to_test_images),
        iteration_basename_before_id(_basename),
        lambda_(0.0060f),  // default from plugin
        minValue_(0.0001f),
        psi0_avg_(0) {

    psi_.reserve(max_num_psi);

    for (unsigned i = 0; i < max_num_psi; ++i) {
      std::stringstream path("");
      path << path_to_images << iteration_basename_before_id << i << ".tif";
      psi_paths_[i] = path.str();

      psi_.push_back(tiff_stack(path.str()));

      // if(padded_images){

      // }
    }

    float sum = std::accumulate(
        psi_[0].stack_.data(),
        psi_[0].stack_.data() + psi_[0].stack_.num_elements(), 0.f);
    psi0_avg_ = sum / psi_[0].stack_.num_elements();
  }

  IterationData(const IterationData& _rhs)
      : psi_(_rhs.psi_.begin(), _rhs.psi_.end()),
        padded_psi_(_rhs.padded_psi_.begin(), _rhs.padded_psi_.end()),
	padded_ranges_(_rhs.padded_ranges_.begin(), _rhs.padded_ranges_.end()),
        psi_paths_(_rhs.psi_paths_.begin(), _rhs.psi_paths_.end()),
        lambda_(_rhs.lambda_),
        minValue_(_rhs.minValue_),
        psi0_avg_(_rhs.psi0_avg_) {}

  void copy_in(const IterationData& _other) {
    std::copy(_other.psi_.begin(), _other.psi_.end(), psi_.begin());
    std::copy(_other.padded_psi_.begin(), _other.padded_psi_.end(),
              padded_psi_.begin());

    std::copy(_other.padded_ranges_.begin(), _other.padded_ranges_.end(),
              padded_ranges_.begin());
    
    std::copy(_other.psi_paths_.begin(), _other.psi_paths_.end(),
              psi_paths_.begin());
    lambda_ = (_other.lambda_);
    minValue_ = (_other.minValue_);
    psi0_avg_ = (_other.psi0_avg_);
  }

  image_stack* psi(const int& _index) { return &(psi_.at(_index).stack_); }

  const image_stack* psi(const int& _index) const {
    return &(psi_.at(_index).stack_);
  }

  const std::array<range,image_stack::dimensionality>* offset(const int& _index) const {
    return &(padded_ranges_.at(_index));
  }

  template <typename container_type>
  image_stack* padded_psi(const int& _index, const container_type& _shape) {
    if (padded_psi_.at(_index).num_elements() <=
        psi_.at(_index).stack_.num_elements()) {

      container_type extended(
          psi_.at(_index).stack_.shape(),
          psi_.at(_index).stack_.shape() + image_stack::dimensionality);
      // std::vector<range> index_of_interest(image_stack::dimensionality);

      for (int d = 0; d < extended.size(); ++d) {
        extended[d] = extended[d] + 2 * (_shape[d] / 2);
        padded_ranges_[_index][d] =
            range(_shape[d] / 2, extended[d] - (_shape[d] / 2));
      }

      padded_psi_.at(_index).resize(extended);
      padded_psi_.at(
          _index)[boost::indices[padded_ranges_[_index][0]][padded_ranges_[_index][1]]
                                [padded_ranges_[_index][2]]] =
          psi_.at(_index).stack_;
    }

    return &(padded_psi_.at(_index));
  }

  
};

typedef IterationData<3> first_2_iterations;
typedef IterationData<6> first_5_iterations;
typedef IterationData<10> all_iterations;
}

#endif
