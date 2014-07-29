#ifndef _TIFF_FIXTURES_H_
#define _TIFF_FIXTURES_H_
#include <iostream> 
#include <iomanip> 
#include <vector>
#include <cmath>
#include <stdexcept>
#include "boost/multi_array.hpp"
#include <boost/filesystem.hpp>

#include <string>
#include <sstream>
#include "image_stack_utils.h"
#include "tiff_utils.h"

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

  static const std::string path_to_test_images = "/dev/shm/libmultiview_data/";

  struct tiff_stack {
    
    std::string  stack_path_  ;
    image_stack  stack_     ;

    tiff_stack():
      stack_path_(""),
      stack_(){

    }

    tiff_stack(const std::string& _path):
      stack_path_(_path),
      stack_(){
      if(boost::filesystem::is_regular_file(_path) || !_path.empty())
	load(_path);
      else
	std::cerr << "Unable to load file " << _path << "\n";
    }

    tiff_stack(const tiff_stack& _rhs):
      stack_path_(_rhs.stack_path_),
      stack_(){
      
      std::vector<unsigned> dims(3);
      for(int i = 0;i<3;++i)
	dims[i] = _rhs.stack_.shape()[i];

      stack_.resize(dims);
      
      stack_ = _rhs.stack_;

      has_malformed_floats();
    }

    tiff_stack& operator=(const tiff_stack& _rhs){
      
      stack_path_ = _rhs.stack_path_;

      std::vector<unsigned> dims(3);
      for(int i = 0;i<3;++i)
	dims[i] = _rhs.stack_.shape()[i];

      stack_.resize(dims);
      stack_ = _rhs.stack_;

      has_malformed_floats();

      return *this;
    }
    
    void load(const std::string& _path){
      if(!boost::filesystem::is_regular_file(_path) || _path.empty()){
	std::stringstream msg("");
	msg << "unable to load file at path " << _path << "\n";
	throw std::runtime_error(msg.str().c_str());
      }

      stack_path_ = _path;
      TIFF* stack_tiff   = TIFFOpen( _path.c_str() , "r" );
      std::vector<tdir_t> stack_tdirs   ;
      get_tiff_dirs(stack_tiff,   stack_tdirs  );
      unsigned stack_size = stack_.num_elements();
      extract_tiff_to_image_stack(stack_tiff,   stack_tdirs  , stack_     );
      TIFFClose(stack_tiff);
      
      if(stack_size-stack_.num_elements() != 0 && stack_.num_elements()>0){
	std::stringstream dim_string("");
	dim_string << "[" << stack_.shape()[0]  << "x" << stack_.shape()[1]  << "x" << stack_.shape()[2] << "]";
	std::cout << "successfully loaded " << std::setw(20) << dim_string.str() << " stack from "  << _path  <<  "\n";
      }
      has_malformed_floats();
    }
    
    
    bool has_malformed_floats() const {
      
      bool value = false;

      unsigned size = stack_.num_elements();
      bool is_nan = false;
      bool is_inf = false;
      const float* data = stack_.data();

      for(unsigned i = 0;i<size;++i){
	
	is_nan = std::isnan(data[i]);
	is_inf = std::isinf(data[i]);
	
	if(is_nan || is_inf){
	  std::cerr << "encountered malformed pixel in ["<< stack_path_ <<"]: index = " << i 
		    << " type: "<< ((is_nan) ? "nan" : "inf") << "\n";
	  value = true;
	  break;
	}
	  
      }

      return value;

    }

    ~tiff_stack(){

    }

    bool empty(){
      return !(stack_.num_elements() > 0);
    }
    
  };
  
  struct ViewFromDisk
  {

    int view_number_;

    std::stringstream  image_path_  ;
    std::stringstream  kernel1_path_   ;
    std::stringstream  kernel2_path_   ;
    std::stringstream  weights_path_   ;

    tiff_stack  image_     ;
    tiff_stack  kernel1_   ;
    tiff_stack  kernel2_   ;
    tiff_stack  weights_   ;

    std::vector<int>  image_dims_     ;
    std::vector<int>  kernel1_dims_   ;
    std::vector<int>  kernel2_dims_   ;
        
    ViewFromDisk(const int& _view_number = -1):
      view_number_ ( _view_number ) ,
      image_path_             (  ""  )  ,
      kernel1_path_           (  ""  )  ,
      kernel2_path_           (  ""  )  ,
      weights_path_           (  ""  )  ,
      image_                  (  )   ,
      kernel1_                (  )   ,
      kernel2_                (  )   ,
      weights_                (  ) ,  
      image_dims_                  ( 3 )   ,
      kernel1_dims_                ( 3 )   ,
      kernel2_dims_                ( 3 )  
    {
      if(view_number_>-1)
	this->load(view_number_);
    }

    ViewFromDisk(const ViewFromDisk& _rhs):
      view_number_               (              _rhs.view_number_   )                                 ,
      image_path_                (              _rhs.image_path_.str()    )                                 ,
      kernel1_path_              (              _rhs.kernel1_path_.str()  )                                 ,
      kernel2_path_              (              _rhs.kernel2_path_.str()  )                                 ,
      weights_path_              (              _rhs.weights_path_.str()  )                                 ,
      image_                     (              _rhs.image_         )                                 ,
      kernel1_                   (              _rhs.kernel1_       )                                 ,
      kernel2_                   (              _rhs.kernel2_       )                                 ,
      weights_                   (              _rhs.weights_       ),  
      image_dims_                  ( _rhs.image_dims_   )   ,
      kernel1_dims_                ( _rhs.kernel1_dims_ )   ,
      kernel2_dims_                ( _rhs.kernel2_dims_ )  
    {
    }

    ViewFromDisk& operator=(const ViewFromDisk& _rhs){
 
      view_number_   =  _rhs.view_number_   ;
      image_path_  .str(_rhs.image_path_  .str())  ;
      kernel1_path_.str(_rhs.kernel1_path_.str())  ;
      kernel2_path_.str(_rhs.kernel2_path_.str())  ;
      weights_path_.str(_rhs.weights_path_.str())  ;

      image_   = _rhs.image_  ;
      kernel1_ = _rhs.kernel1_;
      kernel2_ = _rhs.kernel2_;
      weights_ = _rhs.weights_;

      image_dims_   = _rhs.image_dims_  ;
      kernel1_dims_ = _rhs.kernel1_dims_;
      kernel2_dims_ = _rhs.kernel2_dims_;      

      return *this;
	    
    }
    
    ViewFromDisk& operator=(const int& _view_number){

      view_number_ = _view_number;
      
      if(view_number_>-1)
	this->load(view_number_);
      
      return *this;
    }

    void load(const int& _view_number){

      view_number_ = _view_number;

      image_path_    << path_to_test_images <<  "image_view_"    <<  view_number_  <<  ".tif";
      kernel1_path_  << path_to_test_images <<  "kernel1_view_"  <<  view_number_  <<  ".tif";
      kernel2_path_  << path_to_test_images <<  "kernel2_view_"  <<  view_number_  <<  ".tif";
      weights_path_  << path_to_test_images <<  "weights_view_"  <<  view_number_  <<  ".tif";
    
      image_  .load(image_path_  .str());
      kernel1_.load(kernel1_path_.str());
      kernel2_.load(kernel2_path_.str());
      weights_.load(weights_path_.str());

      std::copy(&image_.stack_.shape()[0], &image_.stack_.shape()[0] + 3, image_dims_.begin());
      std::copy(&kernel1_.stack_.shape()[0], &kernel1_.stack_.shape()[0] + 3, kernel1_dims_.begin());
      std::copy(&kernel2_.stack_.shape()[0], &kernel2_.stack_.shape()[0] + 3, kernel2_dims_.begin());

    }

    void load(const ViewFromDisk& _rhs){
      

      view_number_   =  _rhs.view_number_   ;
      image_path_  .str(_rhs.image_path_  .str())  ;
      kernel1_path_.str(_rhs.kernel1_path_.str())  ;
      kernel2_path_.str(_rhs.kernel2_path_.str())  ;
      weights_path_.str(_rhs.weights_path_.str())  ;

      image_   = _rhs.image_  ;
      kernel1_ = _rhs.kernel1_;
      kernel2_ = _rhs.kernel2_;
      weights_ = _rhs.weights_;

      image_dims_   = _rhs.image_dims_  ;
      kernel1_dims_ = _rhs.kernel1_dims_;
      kernel2_dims_ = _rhs.kernel2_dims_;     
    }



    virtual ~ViewFromDisk()  { 

    };

    image_stack* image() {
      return &(image_.stack_);
    }

    const image_stack* image() const {
      return &(image_.stack_);
    }

    const image_stack* kernel1() const {
      return &(kernel1_.stack_);
    }

    image_stack* kernel1() {
      return &(kernel1_.stack_);
    }

    const image_stack* weights() const {
      return &(weights_.stack_);
    }

    image_stack* weights() {
      return &(weights_.stack_);
    }

    const image_stack* kernel2() const {
      return &(kernel2_.stack_);
    }

    image_stack* kernel2() {
      return &(kernel2_.stack_);
    }

    int* image_dims() {
      return &(image_dims_[0]);
    }

    int* kernel1_dims() {
      return &(kernel1_dims_[0]);
    }

    int* kernel2_dims() {
      return &(kernel2_dims_[0]);
    }

    const int* image_dims() const {
      return &(image_dims_[0]);
    }

    const int* kernel1_dims() const {
      return &(kernel1_dims_[0]);
    }

    const int* kernel2_dims() const {
      return &(kernel2_dims_[0]);
    }


  };

  struct ReferenceData {
    
    std::vector<ViewFromDisk> views_;

    ReferenceData():
      views_(6){
      for(int i = 0;i<6;++i)
	views_[i].load(i);
    }

    ReferenceData(const ReferenceData& _rhs):
      views_(_rhs.views_){
    }

    void copy_in(const ReferenceData& _other){
      for(int i = 0;i<6;++i)
	views_[i] = _other.views_[i];
    }

  };

  template <unsigned max_num_psi = 2>
  struct IterationData {
    
    std::vector<tiff_stack> psi_;
    std::vector<std::string> psi_paths_;

    std::string path_to_images;
    std::string iteration_basename_before_id;

    double lambda_;
    float minValue_;
    float psi0_avg_;

    IterationData(std::string _basename="psi_"):
      psi_(),
      psi_paths_(max_num_psi),
      path_to_images(path_to_test_images),
      iteration_basename_before_id(_basename),
      lambda_(0.0060f),//default from plugin
      minValue_(0.0001f),
      psi0_avg_(0)
    {
      psi_.reserve(max_num_psi);

      for(unsigned i = 0;i<max_num_psi;++i){
	std::stringstream path("");
	path << path_to_images << iteration_basename_before_id << i << ".tif";
	psi_paths_[i] = path.str();

	psi_.push_back(tiff_stack(path.str()));

      }

      float sum = std::accumulate(psi_[0].stack_.data(), psi_[0].stack_.data() + psi_[0].stack_.num_elements(), 0.f);
      psi0_avg_ = sum/psi_[0].stack_.num_elements();
    }

    IterationData(const IterationData& _rhs):
      psi_(_rhs.psi_.begin(), _rhs.psi_.end()),
      psi_paths_(_rhs.psi_paths_.begin(), _rhs.psi_paths_.end()),
      lambda_(_rhs.lambda_),
      minValue_(_rhs.minValue_),
      psi0_avg_(_rhs.psi0_avg_)
    {
    }

    void copy_in(const IterationData& _other){
      std::copy(_other.psi_.begin(), _other.psi_.end(), psi_.begin());
      std::copy(_other.psi_paths_.begin(), _other.psi_paths_.end(), psi_paths_.begin());
      lambda_ = (_other.lambda_);
      minValue_ = (_other.minValue_);
      psi0_avg_ = (_other.psi0_avg_);
    }

    image_stack* psi(const int& _index){
      return &(psi_.at(_index).stack_);
    }

    const image_stack* psi(const int& _index) const {
      return &(psi_.at(_index).stack_);
    }
  };
  
  typedef IterationData<3> first_2_iterations;
  typedef IterationData<6> first_5_iterations;
  typedef IterationData<10> all_iterations;
  
  }

#endif














