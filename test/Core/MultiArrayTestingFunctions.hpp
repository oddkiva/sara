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

  // Reset.
  it.reset_anchor();

  // Prefix increment.
  ++it;

  // Copy constructor.
  RangeIterator loc2(it);
  EXPECT_EQ(loc2, it);

  // Postfix increment.
  RangeIterator loc3(it++);
  EXPECT_NE(loc3, it);

  // Prefix decrement.
  --it;
  EXPECT_EQ(loc3, it);

  // Postfix increment.
  RangeIterator loc4(it++);

  // N-D random iterator.
  loc4.x() += 5;

  loc4 += Vector(2, 2, 2);

  loc4 -= Vector(2, 2, 2);

  loc4.y() += 10;

  loc4.template axis<0>()[1];

  RangeIterator& loc4bis = loc4;
  loc4bis.template axis<0>()[1];
}

#endif /* MULTIARRAY_TESTING_FUNCTIONS_HPP */