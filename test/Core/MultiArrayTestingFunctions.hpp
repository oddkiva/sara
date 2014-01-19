#ifndef MULTIARRAY_TESTING_FUNCTIONS_HPP
#define MULTIARRAY_TESTING_FUNCTIONS_HPP

#include <DO/Core/MultiArray.hpp>
#include <DO/Core/Color.hpp>
#include <DO/Core/DebugUtilities.hpp>

using namespace DO;

template <int StorageOrder>
void initVolume(MultiArray<Color4f, 3, StorageOrder>& volume)
{
  typedef MultiArray<Color4f, 3, StorageOrder> Volume;
  volume.resize(10, 20, 30);
  volume.check_sizes_and_strides();
  std::cout << std::endl;
  for (int i = 0; i < volume.rows(); ++i)
    for (int j = 0; j < volume.cols(); ++j)
      for (int k = 0; k < volume.depth(); ++k)
        volume(i,j,k) = Color4f(float(i),float(j),float(k),255.f);

  for (int i = 0; i < volume.rows(); ++i)
    for (int j = 0; j < volume.cols(); ++j)
      for (int k = 0; k < volume.depth(); ++k)
        ASSERT_EQ(volume(i,j,k), Color4f(float(i),float(j),float(k),255.f));
}

template <int StorageOrder>
void checkLocatorIncrement(MultiArray<Color4f, 3, StorageOrder>& volume)
{
  typedef MultiArray<Color4f, 3, StorageOrder> Volume;
  typedef typename Volume::range_iterator RangeIterator;
  typedef typename RangeIterator::vector_type Coords, Vector;

  RangeIterator it(volume.begin_range());
  if (StorageOrder == RowMajor)
  {
    for (int i = 0; i < volume.rows(); ++i)
    {
      for (int j = 0; j < volume.cols(); ++j)
      {
        for (int k = 0; k < volume.depth(); ++k, ++it)
        {
          ASSERT_EQ(*it, volume(i,j,k));
          ASSERT_EQ(it.coords(), Vector3i(i,j,k));
        }
      }
    }
  }
  else
  {
    for (int k = 0; k < volume.depth(); ++k)
    {
      for (int j = 0; j < volume.cols(); ++j)
      {
        for (int i = 0; i < volume.rows(); ++i, ++it)
        {
          ASSERT_EQ(*it, volume(i,j,k));
          ASSERT_EQ(it.coords(), Vector3i(i,j,k));
        }
      }
    }
  }
}

template <int StorageOrder>
void checkLocatorDecrement(MultiArray<Color4f, 3, StorageOrder>& volume)
{
  typedef MultiArray<Color4f, 3, StorageOrder> Volume;
  typedef typename Volume::range_iterator RangeIterator;
  typedef typename RangeIterator::vector_type Coords, Vector;

  RangeIterator it(volume.begin_range());
  it.reset_anchor( (volume.sizes().array()-1).matrix() );

  if (StorageOrder == RowMajor)
  {
    for (int i = 0; i < volume.rows(); ++i)
    {
      for (int j = 0; j < volume.cols(); ++j)
      {
        for (int k = 0; k < volume.depth(); ++k, --it)
        {
          ASSERT_EQ(*it, volume(volume.rows()-1-i,
            volume.cols()-1-j,
            volume.depth()-1-k));
          ASSERT_EQ(it.coords(), Vector3i(volume.rows()-1-i,
            volume.cols()-1-j,
            volume.depth()-1-k));
        }
      }
    }
  }
  else
  {
    for (int k = 0; k < volume.depth(); ++k)
    {
      for (int j = 0; j < volume.cols(); ++j)
      {
        for (int i = 0; i < volume.rows(); ++i, --it)
        {
          ASSERT_EQ(*it, volume(volume.rows()-1-i,
            volume.cols()-1-j,
            volume.depth()-1-k));
          ASSERT_EQ(it.coords(), Vector3i(volume.rows()-1-i,
            volume.cols()-1-j,
            volume.depth()-1-k));
        }
      }
    }
  }
}

//! TODO: make this better.
template <int StorageOrder>
void checkLocatorPotPourri(MultiArray<Color4f, 3, StorageOrder>& volume)
{
  typedef MultiArray<Color4f, 3, StorageOrder> Volume;
  typedef typename Volume::range_iterator RangeIterator;
  typedef typename RangeIterator::vector_type Coords, Vector;

  RangeIterator it(volume.begin_range());

  // Reset
  printStage("Reset anchor point of locator");
  it.reset_anchor();

  // Prefix increment.
  printStage("Incrementing locator");
  ++it;
  std::cout << "++loc = " << std::endl;
  it.check();
  std::cout << std::endl;

  // Copy constructor.
  printStage("Check copy constructor of locator");
  RangeIterator loc2(it);
  if (loc2 == it) std::cout << "Equality comparison OK!" << std::endl;

  // Postfix increment.
  RangeIterator loc3(it++);
  if (loc3 != it) std::cout << "Inequality comparison OK!" << std::endl;

  printStage("Decrementing locator");
  --it;
  it.check();
  if (loc3 == it) std::cout << "--loc OK!" << std::endl;

  printStage("Postfix increment locator");
  RangeIterator loc4(it++);
  loc4.check();

  // N-D random iterator.
  printStage("Axis iterator");
  loc4.x() += 5;
  loc4.check();

  loc4 += Vector(2, 2, 2);
  loc4.check();

  loc4 -= Vector(2, 2, 2);
  loc4.check();

  loc4.y() += 10;
  loc4.check();

  //for (int i = 0; i < 4; ++i)
  //  ++loc4.axis<i>();
  loc4.template axis<0>()[1];
  loc4.check();
  loc4.check_strides();
  std::cout << "Finished" << std::endl;

  RangeIterator& loc4bis = loc4;
  loc4bis.template axis<0>()[1];
}

#endif /* MULTIARRAY_TESTING_FUNCTIONS_HPP */