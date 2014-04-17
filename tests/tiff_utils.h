#ifndef _TIFF_UTILS_H_
#define _TIFF_UTILS_H_
#include <iostream>
#include <set>
#include "tiffio.h"
#include "image_stack_utils.h"



namespace multiviewnative {

  unsigned get_num_tiff_dirs(TIFF* _tiff_handle){

    unsigned dircount = 0;
    if (_tiff_handle) {
      dircount = 1;

      while (TIFFReadDirectory(_tiff_handle)){
	dircount++;
      } 
  
    }

    return dircount;
  }

  void get_tiff_dirs(TIFF* _tiff_handle, std::vector< tdir_t >& _value){



    //rewind incase the incoming handle is not at the beginning of the file
    if(TIFFCurrentDirectory(_tiff_handle)!=0)
      TIFFSetDirectory(_tiff_handle,tdir_t(0));

    if (_tiff_handle) {
      _value.reserve(512);
      _value.push_back(TIFFCurrentDirectory(_tiff_handle));
    

      while (TIFFReadDirectory(_tiff_handle)){
	_value.push_back(TIFFCurrentDirectory(_tiff_handle));
      
      
      } 
  
    }

  }

  template <typename ExtentT>
  std::vector<ExtentT> extract_max_extents(TIFF* _tiff_handle, const std::vector< tdir_t >& _tiff_dirs ){

    std::vector<ExtentT> value(3);
    std::set<unsigned> widths;
    std::set<unsigned> heights;
    unsigned w,h;
    unsigned size_z = _tiff_dirs.size();
    for(int i = 0;i<size_z;++i)
      {
	w = h = 0;
	TIFFSetDirectory(_tiff_handle,_tiff_dirs[i]);
	TIFFGetField(_tiff_handle, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(_tiff_handle, TIFFTAG_IMAGELENGTH, &h);
	widths.insert(w);
	heights.insert(h);
      }

    value[0] = *(std::max_element(widths.begin(), widths.end()));
    value[1] = *(std::max_element(heights.begin(), heights.end()));
    value[2] = size_z;

    return value;
  }

  template <typename ValueT>
  void extract_tiff_to_vector(TIFF* _tiff_handle, const std::vector<tdir_t>& _tiff_dirs ,std::vector<ValueT>& _container){
    
    std::vector<unsigned> extents = extract_max_extents<unsigned>(_tiff_handle, _tiff_dirs );

    unsigned w,h;
    unsigned frame_offset = extents[0]*extents[1];
    unsigned total = frame_offset*extents[2];
    _container.clear();
    _container.resize(total);

    for(int frame = 0;frame<extents[2];++frame)
      {
	TIFFSetDirectory(_tiff_handle,_tiff_dirs[frame]);
	TIFFGetField(_tiff_handle, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(_tiff_handle, TIFFTAG_IMAGELENGTH, &h);
	for (unsigned y=0;y<h;++y) {
	  TIFFReadScanline(_tiff_handle,&_container[frame*frame_offset+y*w], y);
	}
      }
  }

  void extract_tiff_to_image_stack(TIFF* _tiff_handle, const std::vector<tdir_t>& _tiff_dirs ,image_stack& _container){
    
    std::vector<unsigned> extents = extract_max_extents<unsigned>(_tiff_handle, _tiff_dirs );

    
    unsigned w,h;
    unsigned frame_offset = extents[0]*extents[1];
    unsigned total = frame_offset*extents[2];
    std::vector<float> local_pixels;
    local_pixels.clear();
    local_pixels.resize(total);

    for(unsigned frame = 0;frame<extents[2];++frame)
      {
	TIFFSetDirectory(_tiff_handle,_tiff_dirs[frame]);
	TIFFGetField(_tiff_handle, TIFFTAG_IMAGEWIDTH, &w);
	TIFFGetField(_tiff_handle, TIFFTAG_IMAGELENGTH, &h);
	for (unsigned y=0;y<h;++y) {
	  TIFFReadScanline(_tiff_handle,&local_pixels[frame*frame_offset+y*w], y);
	}
      }

    _container.resize(extents);

    int tiff_order[3] = {0,1,2};
    bool ascending[3] = {true, true, true};
    storage tiff_storage(tiff_order,ascending);
    image_stack_ref  local_stack    (&local_pixels[0],    extents,    tiff_storage);
    
    _container = local_stack;
  }
  
  
}
#endif /* _TIFF_UTILS_H_ */
