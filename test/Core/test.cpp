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

#include "gtest/gtest.h"
#include <DO/Core.hpp>
#include <iostream>
#include <list>
#include <utility>

using namespace DO;
using namespace std;

//#define TEST_TREE_ONLY

#ifndef TEST_TREE_ONLY
// ========================================================================== //
// scrPath test
TEST(DOCoreTest, srcPathTest)
{
  cout << "string source path: " << srcPath("") << endl;
  ASSERT_TRUE(string(srcPath("")).find("test/Core") != string::npos);
}

// ========================================================================== //
// Color test
template <class ChannelType>
class RgbTest : public testing::Test
{
protected:
  typedef testing::Test Base;
  RgbTest() : Base() {}
};

typedef testing::Types<uchar, ushort, uint, char, short, int, float, double> 
  ChannelTypes;

TYPED_TEST_CASE_P(RgbTest);

TYPED_TEST_P(RgbTest, assignmentTest)
{
  typedef TypeParam ChannelType, T;
  typedef Color<ChannelType, Rgb> Color3;

  Color3 a1(black<ChannelType>());
  ASSERT_EQ(a1, black<ChannelType>());

  Color3 a2(1, 2, 3);
  ASSERT_EQ(a2(0), 1);
  ASSERT_EQ(a2(1), 2);
  ASSERT_EQ(a2(2), 3);

  a1.template channel<R>() = 64;
  a1.template channel<G>() = 12;
  a1.template channel<B>() = 124;
  ASSERT_EQ(a1.template channel<R>(), static_cast<T>(64));
  ASSERT_EQ(a1.template channel<G>(), static_cast<T>(12));
  ASSERT_EQ(a1.template channel<B>(), static_cast<T>(124));

  const Color3& ca1 = a1;
  ASSERT_EQ(ca1.template channel<R>(), static_cast<T>(64));
  ASSERT_EQ(ca1.template channel<G>(), static_cast<T>(12));
  ASSERT_EQ(ca1.template channel<B>(), static_cast<T>(124));

  red(a1) = 89; green(a1) = 50; blue(a1) = 12;
  ASSERT_EQ(red(a1), static_cast<T>(89));
  ASSERT_EQ(green(a1), static_cast<T>(50));
  ASSERT_EQ(blue(a1), static_cast<T>(12));

  ASSERT_EQ(red(ca1), static_cast<T>(89));
  ASSERT_EQ(green(ca1), static_cast<T>(50));
  ASSERT_EQ(blue(ca1), static_cast<T>(12));
}

REGISTER_TYPED_TEST_CASE_P(RgbTest, assignmentTest);
INSTANTIATE_TYPED_TEST_CASE_P(DOCoreTest, RgbTest, ChannelTypes);

// ========================================================================== //
// Multi-array test
TEST(DOCoreTest, multiArrayTest)
{
	// Check MultiArray class.
	std::cout << "// ========================================= //" << std::endl;
	std::cout << "Check MultiArray class" << std::endl;
	typedef MultiArray<Color4f, 3, RowMajor> Volume;
	Volume volume(10, 20, 30);
	volume.check_sizes_and_strides();
	cout << endl;
	for (int i = 0; i < volume.rows(); ++i)
		for (int j = 0; j < volume.cols(); ++j)
			for (int k = 0; k < volume.depth(); ++k)
				volume(i,j,k) = Color4f(float(i),float(j),float(k),255.f);

	Volume::ArrayView array = volume.array();
	array += array;
	array = array.abs2();

	typedef MultiArray<float, 2> Mat;
	Mat M(5, 10);
	for (int i = 0; i < M.rows(); ++i)
		for (int j = 0; j < M.cols(); ++j)
			M(i,j) = float(i*M.cols()+j);

	Mat::MatrixView M2(M.matrix());
	std::cout << M2 << std::endl;

	M.sizes()*2;

  // Conversion test.
  typedef MultiArray<int, 2> Mat2i;
  typedef MultiArray<float, 2> Mat2f;
  Mat2i Ai(10, 10);
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      Ai(i,j) = i+j;

  Mat2f Af(Ai);
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      ASSERT_EQ(std::abs(Af(i,j)), float(i+j));

  Ai.array() = Ai.array() + 1;

  Af = Ai;
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      ASSERT_EQ(std::abs(Af(i,j)), float(i+j+1));
}

// ========================================================================== //
// Locator test
TEST(DOCoreTest, locatorTest)
{
	// Create coords and dims.
	int coords[] = { 2, 3, 4 };
	int dims[] = { 10, 20, 30 };

	// Check offset computations.
	std::cout << "// ========================================= //" << std::endl;
	std::cout << "Check offset computation" << std::endl;
  cout << "Column major storage" << std::endl;
  ASSERT_EQ((Offset<1, ColMajor>::eval(coords, dims)), 2);
  ASSERT_EQ((Offset<2, ColMajor>::eval(coords, dims)), 3*10+2);
  ASSERT_EQ((Offset<3, ColMajor>::eval(coords, dims)), 4*10*20+3*10+2);
	
  cout << "Row major storage" << std::endl;
  ASSERT_EQ((Offset<1, RowMajor>::eval(coords, dims)), 2);
  ASSERT_EQ((Offset<2, RowMajor>::eval(coords, dims)), 2*20+3);
  ASSERT_EQ((Offset<3, RowMajor>::eval(coords, dims)), 2*20*30+3*30+4);

	// Check stride computations.
	std::cout << "// ========================================= //" << std::endl;
	std::cout << "Check stride computation" << std::endl;
	int strides[3];
	// Column major strides
  Offset<3, ColMajor>::eval_strides(strides, dims);
  ASSERT_EQ(strides[0], 1);
  ASSERT_EQ(strides[1], 10);
  ASSERT_EQ(strides[2], 200);
	cout << "Column major strides: " << Map<Matrix<int,1,3> >(strides) << endl;
	// Row major strides
  Offset<3>::eval_strides(strides, dims);
  ASSERT_EQ(strides[0], 600);
  ASSERT_EQ(strides[1], 30);
  ASSERT_EQ(strides[2], 1);
	cout << "Row major strides: " << Map<Matrix<int,1,3> >(strides) << endl << endl;

	// Check MultiArray class.
	std::cout << "// ========================================= //" << std::endl;
	std::cout << "Check MultiArray class" << std::endl;
	typedef MultiArray<Color4f, 3, RowMajor> Volume;
	Volume volume(10, 20, 30);
	volume.check_sizes_and_strides();
	cout << endl;
	for (int i = 0; i < volume.rows(); ++i)
		for (int j = 0; j < volume.cols(); ++j)
			for (int k = 0; k < volume.depth(); ++k)
				volume(i,j,k) = Color4f(float(i),float(j),float(k),255.f);

  for (int i = 0; i < volume.rows(); ++i)
    for (int j = 0; j < volume.cols(); ++j)
      for (int k = 0; k < volume.depth(); ++k)
        ASSERT_EQ(volume(i,j,k), Color4f(float(i),float(j),float(k),255.f));

	// Check Locator class.
	std::cout << "// ========================================= //" << std::endl;
	std::cout << "Check Locator class" << std::endl;
	typedef Color4f Pixel;
	typedef Volume::Locator Loc;
	typedef Loc::coord_type Coords, Vector;
	
	Loc loc(volume.begin_locator());
	loc.check();
	cout << endl;

	// Increment
	for (int i = 0; i < volume.rows(); ++i)
		for (int j = 0; j < volume.cols(); ++j)
			for (int k = 0; k < volume.depth(); ++k, ++loc)
      {
        ASSERT_EQ(*loc, volume(i,j,k));
				ASSERT_EQ(loc.coords(), Vector3i(i,j,k));
      }
	
	// Reset
	loc.reset_anchor(loc.get_coords_of_last());
	loc.check();
	
	// Decrement.
	for (int i = 0; i < volume.rows(); ++i)
		for (int j = 0; j < volume.cols(); ++j)
			for (int k = 0; k < volume.depth(); ++k, --loc)
      {
        ASSERT_EQ(*loc, volume(volume.rows()-1-i,
                               volume.cols()-1-j,
                               volume.depth()-1-k));
        ASSERT_EQ(loc.coords(), Vector3i(volume.rows()-1-i,
                                         volume.cols()-1-j,
                                         volume.depth()-1-k));
      }

	// Reset
	loc.reset_anchor();

	++loc;
	cout << "++loc = " << endl;
	loc.check();
	cout << endl;


	Loc loc2(loc);
	if (loc2 == loc) cout << "Equality comparison OK!" << endl;

	Loc loc3(loc++);
	if (loc3 != loc) cout << "Inequality comparison OK!" << endl;

	--loc;
	loc.check();
	if (loc3 == loc) cout << "--loc OK!" << endl;

	Loc loc4(loc++);
	loc4.check();

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
  loc4.axis<0>()[1];
  loc4.check();
  loc4.check_strides();
  cout << "Finished" << endl;

  Loc& loc4bis = loc4;
  loc4bis.axis<0>()[1];
}

// ========================================================================== //
// Misc test on multi-array.
TEST(DOCoreTest, miscTest)
{
  typedef pair<Point2i, Point2i> Line;
  list<Line> lines;

  typedef Matrix2f SuperScalar;
  typedef MultiArray<SuperScalar, 2> Mat2i;
  SuperScalar a; a << 1, 2, 3, 4;
  SuperScalar b; b << 1, 1, 2, 3;

  Mat2i m(2,2);
  Mat2i n(2,2);

  // Initialize the matrices m and n.
  m.array().fill(a);
  n.array().fill(b);
  // Check m
  cout << "Check m" << endl;
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      cout << "m(" << i << "," << j << ") = " << endl << m(i,j) << endl;
  // Check n
  cout << "Check n" << endl;
  for (int i = 0; i < n.rows(); ++i)
    for (int j = 0; j < n.cols(); ++j)
      cout << "n(" << i << "," << j << ") = " << endl << n(i,j) << endl;


  // Double that matrix
  cout << "m.array() += n.array()" << endl;
  m.array() += n.array();
  // Check that matrix
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      cout << "m(" << i << "," << j << ") = " << endl << m(i,j) << endl;

  cout << "m(0,0)*n(0,0)=" << endl;
  cout << m(0,0)*n(0,0) << endl;

  // Double that matrix
  cout << "m.array() *= n.array()" << endl;
  m.array() *= n.array();
  // Check that matrix
  for (int i = 0; i < m.rows(); ++i)
    for (int j = 0; j < m.cols(); ++j)
      cout << "m(" << i << "," << j << ") = " << endl << m(i,j) << endl;

  m.matrix() += n.matrix();
  (m.array() * n.array()) + n.array() / m.array();
}

#endif TEST_TREE_ONLY

//TEST(DOCoreTest, treeTest)

template <typename T>
void checkChildrenAtTreeNode(const Tree<T>& tree, typename Tree<T>::const_node_handle v)
{
  cout << "Checking the children" << endl;
  cout << "parent = " << *v << " :" << endl;
  typename Tree<T>::const_children_iterator c = tree.children_begin(v);
  for ( ; c != tree.children_end(); ++c)
    cout << *c << " -> ";
  cout << "end" << endl << endl;
}

template <typename T>
void checkDepthFirstIterator(const Tree<T>& tree)
{
  typename Tree<T>::const_depth_first_iterator v = tree.depth_first_begin(); 
  for ( ; v != tree.depth_first_end(); ++v)
    cout << "Node " << *v << endl;
  cout << endl;
}

template <typename T>
void checkLeaves(const Tree<T>& tree)
{
  typename Tree<T>::const_leaf_iterator leaf(tree.leaf_begin());
  for ( ; leaf != tree.leaf_end(); ++leaf)
    cout << "Leaf " << *leaf << endl;
  cout << endl;
}

void testTree()
{
  typedef Tree<int>::node_handle node_handle;
  typedef Tree<int>::depth_first_iterator depth_first_iterator;
  typedef Tree<int>::breadth_first_iterator breadth_first_iterator;
  typedef Tree<int>::breadth_first_iterator breadth_first_queued_iterator;
  typedef Tree<int>::children_iterator children_iterator;
  typedef Tree<int>::leaf_iterator leaf_iterator;

  // Create the tree.
  Tree<int> tree;
  // Check it is empty.
  //ASSERT_EQ(tree.empty(), true);
  
  // Get the root vertex.
  tree.set_root(1);
  // Note that the following also works:
  //tree.insert_sibling_after(root, 1) ;
  //tree.insert_sibling_before(root, 1);
  //ASSERT_EQ(*tree.begin(), 1);
  
  // Exception is thrown for the root vertex of the tree.
  //tree.insert_sibling_after(root, 2);
  //tree.insert_sibling_before(root, 2);

  node_handle root = tree.begin();
  node_handle v2 = tree.append_child(root, 2);
  node_handle v3 = tree.append_child(root, 3);
  node_handle v4 = tree.append_child(v2, 4);
  node_handle v5 = tree.append_child(v4, 5);
  node_handle v6 = tree.insert_sibling_after(v5, 6);
  node_handle v7 = tree.insert_sibling_after(v6, 7);
  node_handle v8 = tree.append_child(v7, 8);
  node_handle v9 = tree.append_child(v3, 9);
  node_handle v10 = tree.append_child(v3, 10);
  node_handle v11 = tree.append_child(root, 11);
  node_handle v12 = tree.prepend_child(root, 12);
  node_handle v14 = tree.append_child(v10, 14);
  node_handle v13 = tree.insert_sibling_before(v14, 13);
  node_handle v15 = tree.insert_sibling_after(v14, 15);
  node_handle v16 = tree.append_child(v14, 16);
  node_handle v17 = tree.append_child(v16, 17);
  

  checkChildrenAtTreeNode<int>(tree, root);
  checkChildrenAtTreeNode<int>(tree, v2);
  checkChildrenAtTreeNode<int>(tree, v3);
  checkChildrenAtTreeNode<int>(tree, v4);
  checkChildrenAtTreeNode<int>(tree, v5);
  checkChildrenAtTreeNode<int>(tree, v6);
  checkChildrenAtTreeNode<int>(tree, v7);
  checkChildrenAtTreeNode<int>(tree, v8);
  checkChildrenAtTreeNode<int>(tree, v9);
  checkChildrenAtTreeNode<int>(tree, v10);
  checkChildrenAtTreeNode<int>(tree, v11);
  checkChildrenAtTreeNode<int>(tree, v12);
  checkChildrenAtTreeNode<int>(tree, v13);
  checkChildrenAtTreeNode<int>(tree, v14);
  checkChildrenAtTreeNode<int>(tree, v15);
  checkChildrenAtTreeNode<int>(tree, v16);
  checkChildrenAtTreeNode<int>(tree, v17);

  cout << "Checking the tree in a depth-first incremental exploration" << endl;
  for (depth_first_iterator v = tree.depth_first_begin(); 
       v != tree.depth_first_end(); ++v)
    cout << "Node " << *v << endl;
  cout << endl;

  cout << "Checking the tree in a depth-first decremental exploration" << endl;
  for (depth_first_iterator v = depth_first_iterator(v11()); 
    v != tree.depth_first_end(); --v)
    cout << "Node " << *v << endl;
  cout << endl;

  cout << "Checking the tree in a breadth-first queued exploration" << endl;
  for (breadth_first_queued_iterator v = tree.breadth_first_begin(); 
       v != tree.breadth_first_end(); ++v)
    cout << "Node " << *v << endl;
  cout << endl;


  Tree<int> clone(tree);
  cout << "Checking the cloned tree in a depth-first incremental exploration" << endl;
  for (depth_first_iterator v = clone.depth_first_begin(); 
       v != tree.depth_first_end(); ++v)
    cout << "Node " << *v << endl;
  cout << endl;

  cout << "Incrementing leaf iterator" << endl;
  leaf_iterator lastLeaf;
  for (leaf_iterator leaf(tree.leaf_begin()); leaf != tree.leaf_end(); ++leaf)
  {
    cout << "Leaf " << *leaf << endl;
    lastLeaf = leaf;
  }
  cout << endl;

  cout << "Decrementing leaf iterator" << endl;
  for ( ; lastLeaf != tree.leaf_end(); --lastLeaf)
    cout << "Leaf " << *lastLeaf << endl;
  cout << endl;



  Tree<int> tree2;
  cout << "Incrementing leaf iterator over tree2" << endl;
  checkLeaves(tree2);

  tree2.set_root(1);
  cout << "root = 1" << endl;
  checkLeaves(tree2);

  cout << "1 -> 2, 3" << endl;
  node_handle n2 = tree2.append_child(tree2.begin(), 2);
  node_handle n3 = tree2.append_child(tree2.begin(), 3);
  checkLeaves(tree2);

  cout << "2 -> 4, 5, 6" << endl;
  tree2.append_child(n2, 4);
  tree2.append_child(n2, 5);
  tree2.append_child(n2, 6);
  checkLeaves(tree2);




  saveTree(tree, srcPath("tree.gv"));
}

int main(int argc, char** argv) 
{
  //testTree();

  //using namespace std;
  //Matrix2d M;
  //M << 1, 2,
  //     3, 4;
  //cout << "M=\n" << M << endl;
  //cout << "Singular values =\n" << M.jacobiSvd().singularValues().transpose() << endl;

  //return 0;

  testing::InitGoogleTest(&argc, argv); 
	return RUN_ALL_TESTS();
}