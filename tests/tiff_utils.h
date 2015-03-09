#ifndef _TIFF_UTILS_H_
#define _TIFF_UTILS_H_
#include <iostream>
#include <set>
#include "tiffio.h"
#include "image_stack_utils.h"

namespace multiviewnative {

unsigned get_num_tiff_dirs(TIFF* _tiff_handle) {

  unsigned dircount = 0;
  if (_tiff_handle) {
    dircount = 1;

    while (TIFFReadDirectory(_tiff_handle)) {
      dircount++;
    }
  }

  return dircount;
}

void get_tiff_dirs(TIFF* _tiff_handle, std::vector<tdir_t>& _value) {

  // rewind incase the incoming handle is not at the beginning of the file
  if (TIFFCurrentDirectory(_tiff_handle) != 0)
    TIFFSetDirectory(_tiff_handle, tdir_t(0));

  if (_tiff_handle) {
    _value.reserve(512);
    _value.push_back(TIFFCurrentDirectory(_tiff_handle));

    while (TIFFReadDirectory(_tiff_handle)) {
      _value.push_back(TIFFCurrentDirectory(_tiff_handle));
    }
  }
}

template <typename ExtentT>
std::vector<ExtentT> extract_max_extents(
    TIFF* _tiff_handle, const std::vector<tdir_t>& _tiff_dirs) {

  std::vector<ExtentT> value(3);
  std::set<unsigned> widths;
  std::set<unsigned> heights;
  unsigned w, h;
  unsigned size_z = _tiff_dirs.size();
  for (unsigned i = 0; i < size_z; ++i) {
    w = h = 0;
    TIFFSetDirectory(_tiff_handle, _tiff_dirs[i]);
    TIFFGetField(_tiff_handle, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(_tiff_handle, TIFFTAG_IMAGELENGTH, &h);
    widths.insert(w);
    heights.insert(h);
  }

  value[2] = *(std::max_element(widths.begin(), widths.end()));
  value[1] = *(std::max_element(heights.begin(), heights.end()));
  value[0] = size_z;

  return value;
}

template <typename ValueT>
void extract_tiff_to_vector(TIFF* _tiff_handle,
                            const std::vector<tdir_t>& _tiff_dirs,
                            std::vector<ValueT>& _container) {

  std::vector<unsigned> extents =
      extract_max_extents<unsigned>(_tiff_handle, _tiff_dirs);

  unsigned w, h;
  unsigned frame_offset = extents[2] * extents[1];
  unsigned total = frame_offset * extents[0];
  _container.clear();
  _container.resize(total);

  for (int frame = 0; frame < extents[0]; ++frame) {
    TIFFSetDirectory(_tiff_handle, _tiff_dirs[frame]);
    TIFFGetField(_tiff_handle, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(_tiff_handle, TIFFTAG_IMAGELENGTH, &h);
    for (unsigned y = 0; y < h; ++y) {
      TIFFReadScanline(_tiff_handle, &_container[frame * frame_offset + y * w],
                       y);
    }
  }
}

void extract_tiff_to_image_stack(TIFF* _tiff_handle,
                                 const std::vector<tdir_t>& _tiff_dirs,
                                 image_stack& _container) {

  std::vector<unsigned> extents =
      extract_max_extents<unsigned>(_tiff_handle, _tiff_dirs);

  unsigned w, h;
  unsigned frame_offset = extents[2] * extents[1];
  unsigned total = frame_offset * extents[0];
  std::vector<float> local_pixels(total);

  for (unsigned frame = 0; frame < extents[0]; ++frame) {
    TIFFSetDirectory(_tiff_handle, _tiff_dirs[frame]);
    TIFFGetField(_tiff_handle, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(_tiff_handle, TIFFTAG_IMAGELENGTH, &h);
    for (unsigned y = 0; y < h; ++y) {
      TIFFReadScanline(_tiff_handle,
                       &local_pixels[frame * frame_offset + y * w], y);
    }
  }

  _container.resize(extents);

  image_stack_ref local_stack(&local_pixels[0], extents);

  _container = local_stack;
}

void write_image_stack(const image_stack& _stack, const std::string& _dest) {

  typedef image_stack::element value_type;
  typedef image_stack::const_array_view<1>::type stack_line;

  TIFF* output_image = TIFFOpen(_dest.c_str(), "w");
  if (!output_image) {
    std::cerr << "Unable to open " << _dest << "\n";
    return;
  } else
    std::cout << "Writing " << _dest << "\n";

  unsigned w = _stack.shape()[2];
  unsigned h = _stack.shape()[1];
  unsigned z = _stack.shape()[0];

  for (unsigned frame = 0; frame < z; ++frame) {
    TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, w);
    TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, h);
    TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(output_image, TIFFTAG_ROWSPERSTRIP,
                 TIFFDefaultStripSize(output_image, 0));
    TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(output_image, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
    TIFFSetField(output_image, TIFFTAG_PAGENUMBER, frame, z);

    std::vector<value_type> temp_row(w);

    for (unsigned y = 0; y < h; ++y) {

      stack_line temp =
          _stack[boost::indices[frame][y][multiviewnative::range(0, w)]];
      std::copy(temp.begin(), temp.end(), temp_row.begin());
      TIFFWriteScanline(output_image, &temp_row[0], y, 0);
    }

    TIFFWriteDirectory(output_image);
  }
  TIFFClose(output_image);
}
}
#endif /* _TIFF_UTILS_H_ */
