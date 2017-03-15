// ========================================================================== //
// This file is part of Sara, a basic set of libraries in C++ for computer
// vision.
//
// Copyright (C) 2013-2016 David Ok <david.ok8@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License v. 2.0. If a copy of the MPL was not distributed with this file,
// you can obtain one at http://mozilla.org/MPL/2.0/.
// ========================================================================== //

#include <gtest/gtest.h>

#include <DO/Sara/Defines.hpp>
#include <DO/Sara/Core/Tree.hpp>


using namespace DO::Sara;
using namespace std;


using node_type = Tree<int>::node_type;
using node_handle = Tree<int>::node_handle;
using const_node_handle = Tree<int>::const_node_handle;
using sibling_iterator = Tree<int>::sibling_iterator;
using const_sibling_iterator = Tree<int>::const_sibling_iterator;
using depth_first_iterator = Tree<int>::depth_first_iterator;
using const_depth_first_iterator = Tree<int>::const_depth_first_iterator;
using breadth_first_iterator = Tree<int>::breadth_first_iterator;
using const_breadth_first_iterator = Tree<int>::const_breadth_first_iterator;
using leaf_iterator = Tree<int>::leaf_iterator;
using const_leaf_iterator = Tree<int>::const_leaf_iterator;
using node_pointer = node_handle::node_pointer;
using const_node_pointer = const_node_handle::node_pointer;


// Test Node class.
TEST(TestNode, test_constructor)
{
  node_type v = 1;
  EXPECT_EQ(v._value, 1);
  EXPECT_EQ(v._parent, node_pointer(0));
  EXPECT_EQ(v._prev_sibling, node_pointer(0));
  EXPECT_EQ(v._next_sibling, node_pointer(0));
  EXPECT_EQ(v._first_child, node_pointer(0));
  EXPECT_EQ(v._last_child, node_pointer(0));
}


// Test NodeHandle class.
TEST(TestNodeHandle, test_constructor)
{
  // Mutable.
  node_handle a;
  node_handle b(node_pointer(1));
  node_handle c(a);

  // Immutable.
  const_node_handle d;
  const_node_handle e(d);
  const_node_handle g(const_node_pointer(1));
  e = d;

  // Immutable from mutable.
  const_node_handle f(a);
  const_node_handle h(node_pointer(1));
}

TEST(TestNodeHandle, test_equality_operator)
{
  // Between mutables.
  node_handle a;
  node_handle b = node_pointer(0);
  EXPECT_EQ(a, b);

  // Between immutables.
  const_node_handle c;
  const_node_handle d;
  EXPECT_EQ(c, d);

  // Between mutables and immutables.
  EXPECT_EQ(a, c);
  EXPECT_EQ(c, a);
}

TEST(TestNodeHandle, test_inequality_operator)
{
  // Between mutables.
  node_handle a;
  node_handle b = node_pointer(1);
  EXPECT_NE(a, b);

  // Between immutables.
  const_node_handle c;
  const_node_handle d = node_pointer(1);
  EXPECT_NE(c, d);

  // Between mutables and immutables.
  EXPECT_NE(a, d);
  EXPECT_NE(d, a);
}


// Test constructors.
TEST(TestTree, test_default_constructor)
{
  Tree<int> tree;
  EXPECT_TRUE(tree.empty());
}


// Test copy.
TEST(TestTree, test_copy)
{
  /*
  We construct the following tree.
                0
               / \
              /   \
             /     \
            /       \
           /         \
          1           2
         / \         /  \
        /   \       /    \
       3     4     5      6
      / \   / \   / \    / \
     7   8 9  10 11  12 13  14
  */

  Tree<int> tree;
  tree.set_root(0);
  node_handle n0  = tree.begin();

  node_handle n1  = tree.append_child(n0, 1);
  node_handle n2  = tree.append_child(n0, 2);

  node_handle n3  = tree.append_child(n1, 3);
  node_handle n4  = tree.append_child(n1, 4);
  node_handle n5  = tree.append_child(n2, 5);
  node_handle n6  = tree.append_child(n2, 6);

  tree.append_child(n3, 7);
  tree.append_child(n3, 8);
  tree.append_child(n4, 9);
  tree.append_child(n4, 10);
  tree.append_child(n5, 11);
  tree.append_child(n5, 12);
  tree.append_child(n6, 13);
  tree.append_child(n6, 14);

  Tree<int> clone(tree);

  breadth_first_iterator bfs_tree_it = tree.breadth_first_begin();
  breadth_first_iterator bfs_clone_it = clone.breadth_first_begin();
  for (int i = 0; i <= 14; ++i, ++bfs_tree_it, ++bfs_clone_it)
  {
    EXPECT_EQ(*bfs_tree_it, i);
    EXPECT_EQ(*bfs_clone_it, i);
  }
}


TEST(TestTree, test_swap)
{
  Tree<int> tree;
  tree.set_root(0);
  node_handle n0  = tree.begin();
  node_handle n1  = tree.append_child(n0, 1);
  node_handle n2  = tree.append_child(n0, 2);
  tree.append_child(n1, 3);
  tree.append_child(n1, 4);
  tree.append_child(n2, 5);
  tree.append_child(n2, 6);

  Tree<int> another_tree;
  another_tree.set_root(0);
  n0  = another_tree.begin();
  n1  = another_tree.append_child(n0, 1);
  n2  = another_tree.append_child(n0, 2);

  tree.swap(another_tree);

  breadth_first_iterator it;
  int num_vertices;

  num_vertices = 0;
  for (it = tree.breadth_first_begin();
       it != tree.breadth_first_end(); ++it)
    ++num_vertices;
  EXPECT_EQ(num_vertices, 3);

  num_vertices = 0;
  for (it = another_tree.breadth_first_begin();
       it != another_tree.breadth_first_end(); ++it)
    ++num_vertices;
  EXPECT_EQ(num_vertices, 7);
}


// Test cut_tree and delete_subtree
// TODO.
TEST(TestTree, test_cut_tree_and_delete_subtree)
{
   /*
  We construct the following tree.
                0
               / \
              /   \
             /     \
            /       \
           /         \
          1           2
         / \         /  \
        /   \       /    \
       3     4     5      6
      / \   / \   / \    / \
     7   8 9  10 11  12 13  14
  */

  Tree<int> tree;
  tree.set_root(0);
  node_handle n0  = tree.begin();

  node_handle n1  = tree.append_child(n0, 1);
  node_handle n2  = tree.append_child(n0, 2);

  node_handle n3  = tree.append_child(n1, 3);
  node_handle n4  = tree.append_child(n1, 4);
  node_handle n5  = tree.append_child(n2, 5);
  node_handle n6  = tree.append_child(n2, 6);

  tree.append_child(n3, 7);
  tree.append_child(n3, 8);
  tree.append_child(n4, 9);
  tree.append_child(n4, 10);
  tree.append_child(n5, 11);
  tree.append_child(n5, 12);
  tree.append_child(n6, 13);
  tree.append_child(n6, 14);

  //tree.cut_tree(n4);
  //tree.delete_subtree(n2);
}


// Test basic functionalities.
TEST(TestTree, test_set_root_and_empty)
{
  Tree<int> tree;
  EXPECT_TRUE(tree.empty());
  tree.set_root(1);
  EXPECT_FALSE(tree.empty());
}

TEST(TestTree, test_begin)
{
  Tree<int> tree;
  const Tree<int> & const_tree = tree;
  EXPECT_EQ(tree.begin(), node_handle(0));
  EXPECT_EQ(const_tree.begin(), node_handle(0));

  tree.set_root(1);
  EXPECT_NE(tree.begin(), node_handle(0));
  EXPECT_NE(const_tree.begin(), node_handle(0));
  EXPECT_EQ(*tree.begin(), 1);
}

TEST(TestTree, test_end)
{
  Tree<int> tree;
  const Tree<int> & const_tree = tree;

  EXPECT_EQ(tree.end(), tree.begin());
  EXPECT_EQ(const_tree.end(), tree.begin());

  tree.set_root(1);
  EXPECT_NE(tree.end(), tree.begin());
  EXPECT_NE(const_tree.end(), tree.begin());
}

TEST(TestTree, test_append_child)
{
  Tree<int> tree;

  tree.set_root(0);
  node_handle v0 = tree.begin();

  tree.append_child(v0, 1);
  tree.append_child(v0, 2);
  tree.append_child(v0, 3);

  sibling_iterator child = tree.children_begin(v0);
  for (int i = 1; i <= 3; ++i, ++child)
    EXPECT_EQ(*child, i);
}

TEST(TestTree, test_prepend_child)
{
  Tree<int> tree;

  tree.set_root(0);
  node_handle v0 = tree.begin();

  tree.prepend_child(v0, 1);
  tree.prepend_child(v0, 2);
  tree.prepend_child(v0, 3);

  sibling_iterator child = tree.children_begin(v0);
  for (int i = 3; i > 0; --i, ++child)
    EXPECT_EQ(*child, i);
}

TEST(TestTree, test_insert_child_before)
{
  Tree<int> tree;

  tree.set_root(0);

  node_handle root = tree.begin();
  EXPECT_THROW(tree.insert_sibling_before(root, 2), std::exception);

  node_handle child1 = tree.append_child(root, 1);
  node_handle child2 = tree.append_child(root, 3);
  tree.insert_sibling_before(child1, 0);
  tree.insert_sibling_before(child2, 2);

  sibling_iterator child = tree.children_begin(root);
  for (int i = 0; i <= 3; ++i, ++child)
    EXPECT_EQ(*child, i);
}

TEST(TestTree, test_insert_child_after)
{
  Tree<int> tree;

  tree.set_root(0);

  node_handle root = tree.begin();
  EXPECT_THROW(tree.insert_sibling_after(root, 2), std::exception);

  node_handle child1 = tree.append_child(root, 1);
  node_handle child2 = tree.append_child(root, 3);
  tree.insert_sibling_after(child1, 2);
  tree.insert_sibling_after(child2, 4);

  sibling_iterator child = tree.children_begin(root);
  for (int i = 1; i <= 4; ++i, ++child)
    EXPECT_EQ(*child, i);
}


// Test sibling iterator.
TEST(TestTree, test_sibling_iterator)
{
  Tree<int> tree;

  tree.set_root(0);
  node_handle v0 = tree.begin();

  tree.append_child(v0, 1);
  tree.append_child(v0, 2);
  tree.append_child(v0, 3);

  sibling_iterator child = tree.children_begin(v0);
  for (int i = 1; i <= 3; ++i, ++child)
    EXPECT_EQ(*child, i);
  EXPECT_EQ(child, tree.children_end());

  child = tree.children_rbegin(v0);
  for (int i = 3; i >= 1; --i, --child)
    EXPECT_EQ(*child, i);
  EXPECT_EQ(child, tree.children_rend());
}

TEST(TestTree, test_const_sibling_iterator)
{
  Tree<int> tree;

  tree.set_root(0);
  node_handle v0 = tree.begin();

  tree.append_child(v0, 1);
  tree.append_child(v0, 2);
  tree.append_child(v0, 3);

  const Tree<int>& const_tree = tree;
  const_sibling_iterator const_child = const_tree.children_begin(v0);
  for (int i = 1; i <= 3; ++i, ++const_child)
    EXPECT_EQ(*const_child, i);
  EXPECT_EQ(const_child, tree.children_end());

  const_child = tree.children_rbegin(v0);
  for (int i = 3; i >= 1; --i, --const_child)
    EXPECT_EQ(*const_child, i);
  EXPECT_EQ(const_child, tree.children_rend());
}


// Test depth-first iterator.
TEST(TestDepthFirstIterator, test_depth_first_tree)
{
  /*
    We construct the following tree.

             0________
            / \       \
           1   5       12
          /   /|\       \
         2   6 7 8___    13
        /       / \  \    \
       3       9   10 11   14
      /
     4

   */

  Tree<int> tree;

  tree.set_root(0);
  node_handle n0  = tree.begin();
  node_handle n1  = tree.append_child(n0, 1);
  node_handle n2  = tree.append_child(n1, 2);
  node_handle n3  = tree.append_child(n2, 3);
  tree.append_child(n3, 4);
  node_handle n5  = tree.append_child(n1, 5);
  tree.append_child(n5, 6);
  tree.append_child(n5, 7);
  node_handle n8  = tree.append_child(n5, 8);
  tree.append_child(n8, 9);
  tree.append_child(n8, 10);
  tree.append_child(n8, 11);
  node_handle n12 = tree.append_child(n0, 12);
  node_handle n13 = tree.append_child(n12, 13);
  tree.append_child(n13, 14);

  depth_first_iterator dfs_it = tree.depth_first_begin();
  const_depth_first_iterator const_dfs_it = tree.depth_first_begin();
  for (int i = 0; i <= 14; ++i, ++dfs_it, ++const_dfs_it)
  {
    EXPECT_EQ(*dfs_it, i);
    EXPECT_EQ(*const_dfs_it, i);
  }
  EXPECT_EQ(dfs_it, tree.depth_first_end());
  EXPECT_EQ(const_dfs_it, tree.depth_first_end());

  dfs_it = tree.depth_first_rbegin();
  const_dfs_it = tree.depth_first_rbegin();
  for (int i = 14; i >= 0; --i, --dfs_it, --const_dfs_it)
  {
    EXPECT_EQ(*dfs_it, i);
    EXPECT_EQ(*const_dfs_it, i);
  }
  EXPECT_EQ(dfs_it, tree.depth_first_rend());
  EXPECT_EQ(const_dfs_it, tree.depth_first_rend());
}


// Test breadth-first iterator.
TEST(TestTree, test_breadth_first_iterator)
{
  /*
  We construct the following tree.
                0
               / \
              /   \
             /     \
            /       \
           /         \
          1           2
         / \         /  \
        /   \       /    \
       3     4     5      6
      / \   / \   / \    / \
     7   8 9  10 11  12 13  14
  */

  Tree<int> tree;
  tree.set_root(0);
  node_handle n0  = tree.begin();

  node_handle n1  = tree.append_child(n0, 1);
  node_handle n2  = tree.append_child(n0, 2);

  node_handle n3  = tree.append_child(n1, 3);
  node_handle n4  = tree.append_child(n1, 4);
  node_handle n5  = tree.append_child(n2, 5);
  node_handle n6  = tree.append_child(n2, 6);

  tree.append_child(n3, 7);
  tree.append_child(n3, 8);
  tree.append_child(n4, 9);
  tree.append_child(n4, 10);
  tree.append_child(n5, 11);
  tree.append_child(n5, 12);
  tree.append_child(n6, 13);
  tree.append_child(n6, 14);

  breadth_first_iterator bfs_it = tree.breadth_first_begin();
  const_breadth_first_iterator const_bfs_it = tree.breadth_first_begin();
  for (int i = 0; i <= 14; ++i, ++bfs_it, ++const_bfs_it)
  {
    EXPECT_EQ(*bfs_it, i);
    EXPECT_EQ(*const_bfs_it, i);
  }
  EXPECT_EQ(bfs_it, tree.breadth_first_end());
  EXPECT_EQ(const_bfs_it, tree.breadth_first_end());
}


// Test leaf iterator.
TEST(TestTree, test_leaf_iterator)
{
  /*
  We construct the following tree.
                0
               / \
              /   \
             /     \
            /       \
           /         \
          1           2
         / \         /  \
        /   \       /    \
       3     4     5      6
      / \   / \   / \    / \
     7   8 9  10 11  12 13  14
  */

  Tree<int> tree;
  tree.set_root(0);
  node_handle n0  = tree.begin();

  node_handle n1  = tree.append_child(n0, 1);
  node_handle n2  = tree.append_child(n0, 2);

  node_handle n3  = tree.append_child(n1, 3);
  node_handle n4  = tree.append_child(n1, 4);
  node_handle n5  = tree.append_child(n2, 5);
  node_handle n6  = tree.append_child(n2, 6);

  tree.append_child(n3, 7);
  tree.append_child(n3, 8);
  tree.append_child(n4, 9);
  tree.append_child(n4, 10);
  tree.append_child(n5, 11);
  tree.append_child(n5, 12);
  tree.append_child(n6, 13);
  tree.append_child(n6, 14);

  breadth_first_iterator bfs_it = tree.breadth_first_begin();
  const_breadth_first_iterator const_bfs_it = tree.breadth_first_begin();
  for (int i = 0; i <= 14; ++i, ++bfs_it, ++const_bfs_it)
  {
    EXPECT_EQ(*bfs_it, i);
    EXPECT_EQ(*const_bfs_it, i);
  }
  EXPECT_EQ(bfs_it, tree.breadth_first_end());
  EXPECT_EQ(const_bfs_it, tree.breadth_first_end());

  leaf_iterator leaf_it = tree.leaf_begin();
  const_leaf_iterator const_leaf_it = tree.leaf_begin();
  for (int i = 7; i <= 14; ++i, ++leaf_it, ++const_leaf_it)
  {
    EXPECT_EQ(*leaf_it, i);
    EXPECT_EQ(*const_leaf_it, i);
  }
  EXPECT_EQ(leaf_it, tree.leaf_end());
  EXPECT_EQ(const_leaf_it, tree.leaf_end());
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
