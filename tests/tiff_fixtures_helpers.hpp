#ifndef _TIFF_FIXTURES_HELPERS_H_
#define _TIFF_FIXTURES_HELPERS_H_
#include <string>
#include <fstream>
#include <regex>
#include "boost/filesystem.hpp"
#include "boost/regex.hpp"

namespace fs = boost::filesystem;

namespace multiviewnative {

  
  struct input
  {

    static const std::string basename() { return std::string("input_view_"); }
      
  };


  struct weights{

    static const std::string basename() { return std::string("weights_view_"); }
  
  };


  struct kernel1{

    static const std::string basename() { return std::string("kernel1_view_"); }
    
  };

  
  struct kernel2{

    static const std::string basename() { return std::string("kernel2_view_"); }

  };

  template <typename tag_type>
  constexpr static fs::path path_to(int _id)  {
    fs::path value = multiviewnative::path_to_test_images;
    
    std::string fname = tag_type::basename();
    fname += std::to_string(_id);
    fname += ".tif";
    
    value /= fname;
    
    return value;
  }

  static bool contains(const std::string& _data, const std::string& _regex = "^[[:digit:]]+"){

#ifdef __GNUC__
    #if __GNUC__ < 5
    boost::regex bre(_regex,boost::regex::ECMAScript);
    return boost::regex_search(_data,bre);
    #endif

#endif
    
    std::regex re(_regex,std::regex::ECMAScript);
    return std::regex_search(_data,re);
  }
  
  template <typename tag_type>
  constexpr static std::vector<int> shape_of(int _id)  {
    
    fs::path path = multiviewnative::path_to_test_images;
    std::string fname = tag_type::basename();
    fname += std::to_string(_id);
    fname += ".shape";
    path /= fname;

    std::vector<int> value;
    if(!fs::exists(path)){
      std::cerr << path << " does not exist!\n";
      return value;
    }

    std::vector<std::string> lines;
    std::ifstream shape_file(path.string(), std::ios::in );
    

    for (std::string line; std::getline(shape_file, line); ) {
      if(contains(line))
	lines.push_back(line);
    }

    if(lines.empty()){
      std::cerr << "couldn't read any content of " << path << "\n";
      return value;
    }

    std::istringstream lines_stream(lines.back());
    int num = 0;
    for (std::string num_literal; std::getline(lines_stream, num_literal, ' '); ) {    
      try{
	num = std::stoi(num_literal);
      }
      catch(...){
	std::cerr << "unable to decipher number from " << num_literal << ", skipping it.\n";
	continue;
      }
      value.push_back(num);
    }
    
    return value;
  }
  

}



#endif /* _TIFF_FIXTURES_HELPERS_H_ */
