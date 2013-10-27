// ========================================================================== //
// This file is part of DO++, a basic set of libraries in C++ for computer 
// vision.
//
// Copyright (C) 2013 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public 
// License v. 2.0. If a copy of the MPL was not distributed with this file, 
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

//! @file

#ifndef DO_FEATUREDESCRIPTORS_DESCRIPTOR_HPP
#define DO_FEATUREDESCRIPTORS_DESCRIPTOR_HPP

namespace DO {

 /*!
  \ingroup Features
  @{
  */

  class DescriptorBase {
  public:
    virtual ~DescriptorBase() {}
    virtual std::ostream& print(std::ostream& os) const = 0;
    virtual std::istream& read(std::istream& in) = 0;
    
    friend
    inline std::ostream& operator<<(std::ostream& out, const DescriptorBase& d)
    { return d.print(out); }
    friend
    inline std::istream& operator>>(std::istream& in, DescriptorBase& d)
    { return d.read(in); }
  };

  //! Deprecated.
  template <typename T>
  class Descriptor : public DescriptorBase, public Map<Matrix<T, Dynamic, 1> >
  {
  public:
    typedef T bin_type;
    typedef Map<Matrix<T, Dynamic, 1> > MappedVectorType;

    inline Descriptor() : MappedVectorType(0, 0) {} 
    inline Descriptor(T *data, int sz) : MappedVectorType(data, sz) {}
    virtual inline ~Descriptor() {}

    std::ostream& print(std::ostream& os) const;
    std::istream& read(std::istream& in);
  };

  template <typename T> 
  inline std::ostream& printT(std::ostream& os, const T *array, int N)
  {
    std::copy( array, &array[N], std::ostream_iterator<T>(os," "));
    return os;
  }

  template <> 
  inline std::ostream& printT<unsigned char>(std::ostream& os, 
                                             const unsigned char *array, int N)
  {
    for(int i = 0; i < N; ++i)
      os << static_cast<int>(array[i]) << " ";
    return os;
  }

  template<typename T> 
  inline std::istream& readT(std::istream& is, T *array, int N)
  {
    for(int i = 0; i < N; ++i)
      is >> array[i];
    return is;
  }

  template <> 
  inline std::istream& readT<unsigned char>(std::istream& is,
                                            unsigned char *array, int N)
  {
    int temp = -1;
    for(int i = 0; i < N; ++i)
    {
      is >> temp;
      array[i] = static_cast<unsigned char>(temp);
    }
    return is;
  }

  template <typename T>
  std::ostream& Descriptor<T>::print(std::ostream& os) const
  { return printT<T>(os, MappedVectorType::data(),
                     static_cast<int>(MappedVectorType::size())); }

  template <typename T>
  std::istream& Descriptor<T>::read(std::istream& in)
  { return readT<T>(in, MappedVectorType::data(),
                    static_cast<int>(MappedVectorType::size())); }

  //! @}

} /* namespace DO */

#endif /* DO_FEATUREDESCRIPTORS_DESCRIPTOR_HPP */
