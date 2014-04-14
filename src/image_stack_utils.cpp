#define _IMAGE_STACK_UTILS_CPP_
#include <iomanip>
#include "image_stack_utils.h"

namespace multiviewnative {

std::ostream& operator<<(std::ostream& _cout, const image_stack& _marray){

  if(image_stack::dimensionality!=3){
    _cout << "dim!=3\n";
    return _cout;
  }


  if(_marray.empty()){
    _cout << "size=0\n";
    return _cout;
  }
  
  int precision = _cout.precision();
  _cout << std::setprecision(4);
  const size_t* shapes = _marray.shape(); 
  
  _cout << std::setw(9) << "x = ";
  for(size_t x_index = 0;x_index<(shapes[0]);++x_index){
    _cout << std::setw(8) << x_index << " ";
  }
  _cout << "\n";
  _cout << std::setfill('-') << std::setw((shapes[0]+1)*9) << " " << std::setfill(' ')<< std::endl ;

  for(size_t z_index = 0;z_index<(shapes[2]);++z_index){
    _cout << "z["<< std::setw(5) << z_index << "] \n";
    for(size_t y_index = 0;y_index<(shapes[1]);++y_index){
      _cout << "y["<< std::setw(5) << y_index << "] ";

      for(size_t x_index = 0;x_index<(shapes[0]);++x_index){
	_cout << std::setw(8) << _marray[x_index][y_index][z_index] << " ";
      }

      _cout << "\n";
    }
    _cout << "\n";
  }

  _cout << std::setprecision(precision);
  return _cout;
}

}
