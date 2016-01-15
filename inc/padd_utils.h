#ifndef _PADD_UTILS_H_
#define _PADD_UTILS_H_
#include <vector>
#include <algorithm>
#include <type_traits>
#include <limits>
#include "boost/multi_array.hpp"

namespace multiviewnative {

template <typename ImageStackRefT, typename OtherStackT>
void wrapped_insert_at_point(
    const ImageStackRefT& _source, OtherStackT& _target,
    const std::vector<typename ImageStackRefT::size_type>& _point) {

  typedef typename std::make_signed<typename ImageStackRefT::size_type>::type
      signed_size_type;

  for (signed_size_type z = 0; z < signed_size_type(_source.shape()[0]); ++z)
    for (signed_size_type y = 0; y < signed_size_type(_source.shape()[1]); ++y)
      for (signed_size_type x = 0; x < signed_size_type(_source.shape()[2]);
           ++x) {

        // move origin to center of stack
        signed_size_type intermediate_x = x - (_source.shape()[2] / 2L);
        signed_size_type intermediate_y = y - (_source.shape()[1] / 2L);
        signed_size_type intermediate_z = z - (_source.shape()[0] / 2L);

        // if we are below 0, move by _point, otherwise keep it
        intermediate_x =
            (intermediate_x < 0) ? intermediate_x + _point[2] : intermediate_x;
        intermediate_y =
            (intermediate_y < 0) ? intermediate_y + _point[1] : intermediate_y;
        intermediate_z =
            (intermediate_z < 0) ? intermediate_z + _point[0] : intermediate_z;

        _target[intermediate_z][intermediate_y][intermediate_x] =
            _source[z][y][x];
      }
}

template <typename outT>
struct add_minus_1 {

  template <typename any_type>
  outT operator()(const any_type& _first, const any_type& _second) {
    return _first + _second - 1;
  }
};

template <typename inT, typename outT>
struct minus_1_div_2 {

  outT operator()(const inT& _first) { return (_first - 1) / 2; }
};

template <typename ImageStackT>
struct no_padd {

  typedef typename ImageStackT::value_type value_type;
  typedef typename ImageStackT::size_type size_type;
  typedef typename ImageStackT::template array_view<
      ImageStackT::dimensionality>::type image_stack_view;
  typedef boost::multi_array_types::index_range range;

  std::vector<size_type> extents_;
  std::vector<size_type> offsets_;

  no_padd()
      : extents_(ImageStackT::dimensionality, 0),
        offsets_(ImageStackT::dimensionality, 0) {}

  template <typename T, typename U>
  no_padd(T* _image_shape, U* _kernel_shape)
      : extents_(_image_shape, _image_shape + ImageStackT::dimensionality),
        offsets_(ImageStackT::dimensionality, 0) {
    static_assert(
        std::numeric_limits<T>::is_integer,
        "[no_padd] didn't receive integer type as image shape descriptor");
    static_assert(
        std::numeric_limits<U>::is_integer,
        "[no_padd] didn't receive integer type as kernel shape descriptor");
  }

  template <typename ImageStackRefT, typename OtherStackT>
  void insert_at_offsets(const ImageStackRefT& _source, OtherStackT& _target) {
    _target = _source;
  }

  template <typename ImageStackRefT, typename OtherStackT>
  void wrapped_insert_at_offsets(const ImageStackRefT& _source,
                                 OtherStackT& _target) {

    multiviewnative::wrapped_insert_at_point(_source, _target, extents_);
  }

  const size_type* offsets() const { return &offsets_[0]; }

  const size_type* extents() const { return &extents_[0]; }
};

template <typename ImageStackT>
struct zero_padd {

  typedef typename ImageStackT::value_type value_type;
  typedef typename ImageStackT::size_type size_type;
  typedef typename ImageStackT::template array_view<
      ImageStackT::dimensionality>::type image_stack_view;
  typedef boost::multi_array_types::index_range range;

  std::vector<size_type> extents_;
  std::vector<size_type> offsets_;

  zero_padd()
      : extents_(ImageStackT::dimensionality, 0),
        offsets_(ImageStackT::dimensionality, 0) {}

  zero_padd(const zero_padd& _other)
      : extents_(_other.extents_), offsets_(_other.offsets_) {}

  template <typename T, typename U>
  zero_padd(T* _image, U* _kernel)
      : extents_(ImageStackT::dimensionality, 0),
        offsets_(ImageStackT::dimensionality, 0) {
    static_assert(
        std::numeric_limits<T>::is_integer == true,
        "[zero_padd] didn't receive integer type as image shape descriptor");
    static_assert(
        std::numeric_limits<U>::is_integer,
        "[zero_padd] didn't receive integer type as kernel shape descriptor");

    //padding assumes kernel has uneven shape
    std::transform(_image, _image + ImageStackT::dimensionality, _kernel,
                   extents_.begin(), add_minus_1<T>());

    std::transform(_kernel, _kernel + ImageStackT::dimensionality,
                   offsets_.begin(), minus_1_div_2<U, size_type>());
  }

  zero_padd& operator=(const zero_padd& _other) {
    if (this != &_other) {
      extents_ = _other.extents_;
      offsets_ = _other.offsets_;
    }
    return *this;
  }

  /**
     \brief function that inserts _source in _target given limits by vector
     offsets,
     _target is expected to have dimensions
     (source.shape()[0]+2*offsets_[0])x(source.shape()[1]+2*offsets_[1])x...
     _target is hence expected to have dimensions extents_[0]xextents_[1]x...

     as an example:
     _source =
     1 1
     1 1
     given a kernel of 3x3x3
     would expecte a _target of
     0 0 0 0
     0 0 0 0
     0 0 0 0
     0 0 0 0

     and the result would look like
     0 0 0 0
     0 1 1 0
     0 1 1 0
     0 0 0 0

     \param[in] _source image stack to embed into target
     \param[in] _target image stack of size extents_[0]xextents_[1]x...

     \return
     \retval

  */
  template <typename ImageStackRefT, typename OtherStackT>
  void insert_at_offsets(const ImageStackRefT& _source, OtherStackT& _target) {

    if (std::lexicographical_compare(
            _target.shape(), _target.shape() + OtherStackT::dimensionality,
            extents_.begin(), extents_.end()))
      throw std::runtime_error(
          "multiviewnative::zero_padd::insert_at_offsets]\ttarget image stack "
          "is smaller or equal in size than source\n");

    image_stack_view subview_padded_image = _target
        [boost::indices[range(offsets_[0], offsets_[0] + _source.shape()[0])]
                       [range(offsets_[1], offsets_[1] + _source.shape()[1])]
                       [range(offsets_[2], offsets_[2] + _source.shape()[2])]];
    subview_padded_image = _source;
  }

  /**
     \brief function that inserts _source in _target given limits by vector
     offsets,
     _target is expected to have dimensions
     (source.shape()[0]+2*offsets_[0])x(source.shape()[1]+2*offsets_[1])x...
     _target is hence expected to have dimensions extents_[0]xextents_[1]x...

     as an example:
     _source =
     0 0 0
     1 2 3
     0 0 0
     given a kernel of 3x3x3
     would expecte a _target of
     0 0 0 0 0
     0 0 0 0 0
     0 0 0 0 0
     0 0 0 0 0
     0 0 0 0 0

     and the result would look like
     2 3 0 0 1
     0 0 0 0 0
     0 0 0 0 0
     0 0 0 0 0
     0 0 0 0 0

     \param[in] _source image stack to embed into target
     \param[in] _target image stack of size extents_[0]xextents_[1]x...

     \return
     \retval

  */
  template <typename ImageStackRefT, typename OtherStackT>
  void wrapped_insert_at_offsets(const ImageStackRefT& _source,
                                 OtherStackT& _target) {

    if (std::lexicographical_compare(
            _target.shape(), _target.shape() + OtherStackT::dimensionality,
            extents_.begin(), extents_.end()))
      throw std::runtime_error(
          "multiviewnative::zero_padd::insert_at_offsets]\ttarget image stack "
          "is smaller or equal in size than source\n");

    wrapped_insert_at_point(_source, _target, extents_);
  }

  const size_type* offsets() const { return &offsets_[0]; }

  const size_type* extents() const { return &extents_[0]; }

  virtual ~zero_padd() {};
};
}
#endif /* _PADD_UTILS_H_ */
