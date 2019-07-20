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

#define BOOST_TEST_MODULE "Core/Tree Class"

#include <boost/test/unit_test.hpp>

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
BOOST_AUTO_TEST_SUITE(TestTreeNode)

BOOST_AUTO_TEST_CASE(test_constructor)
{
  node_type v = 1;
  BOOST_CHECK_EQUAL(v._value, 1);
  BOOST_CHECK_EQUAL(v._parent, node_pointer(0));
  BOOST_CHECK_EQUAL(v._prev_sibling, node_pointer(0));
  BOOST_CHECK_EQUAL(v._next_sibling, node_pointer(0));
  BOOST_CHECK_EQUAL(v._first_child, node_pointer(0));
  BOOST_CHECK_EQUAL(v._last_child, node_pointer(0));
}

BOOST_AUTO_TEST_SUITE_END()


// Test NodeHandle class.
BOOST_AUTO_TEST_SUITE(TestNodeHandle)

BOOST_AUTO_TEST_CASE(test_constructor)
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

BOOST_AUTO_TEST_CASE(test_equality_operator)
{
  // Between mutables.
  node_handle a;
  node_handle b = node_pointer(0);
  BOOST_CHECK(a == b);

  // Between immutables.
  const_node_handle c;
  const_node_handle d;
  BOOST_CHECK(c == d);

  // Between mutables and immutables.
  BOOST_CHECK(a == c);
  BOOST_CHECK(c == a);
}

BOOST_AUTO_TEST_CASE(test_inequality_operator)
{
  // Between mutables.
  node_handle a;
  node_handle b = node_pointer(1);
  BOOST_CHECK(a != b);

  // Between immutables.
  const_node_handle c;
  const_node_handle d = node_pointer(1);
  BOOST_CHECK(c != d);

  // Between mutables and immutables.
  BOOST_CHECK(a != d);
  BOOST_CHECK(d != a);
}

BOOST_AUTO_TEST_SUITE_END()


// Test constructors.
BOOST_AUTO_TEST_SUITE(TestTree)

BOOST_AUTO_TEST_CASE(test_default_constructor)
{
  Tree<int> tree;
  BOOST_CHECK(tree.empty());
}

// Test copy.
BOOST_AUTO_TEST_CASE(test_copy)
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
    BOOST_CHECK_EQUAL(*bfs_tree_it, i);
    BOOST_CHECK_EQUAL(*bfs_clone_it, i);
  }
}


BOOST_AUTO_TEST_CASE(test_swap)
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
  BOOST_CHECK_EQUAL(num_vertices, 3);

  num_vertices = 0;
  for (it = another_tree.breadth_first_begin();
       it != another_tree.breadth_first_end(); ++it)
    ++num_vertices;
  BOOST_CHECK_EQUAL(num_vertices, 7);
}


// Test cut_tree and delete_subtree
// TODO.
BOOST_AUTO_TEST_CASE(test_cut_tree_and_delete_subtree)
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
BOOST_AUTO_TEST_CASE(test_set_root_and_empty)
{
  Tree<int> tree;
  BOOST_CHECK(tree.empty());
  tree.set_root(1);
  BOOST_CHECK(!tree.empty());
}

BOOST_AUTO_TEST_CASE(test_begin)
{
  Tree<int> tree;
  const Tree<int> & const_tree = tree;
  BOOST_CHECK(tree.begin() == node_handle(0));
  BOOST_CHECK(const_tree.begin() == node_handle(0));

  tree.set_root(1);
  BOOST_CHECK(tree.begin() != node_handle(0));
  BOOST_CHECK(const_tree.begin() != node_handle(0));
  BOOST_CHECK_EQUAL(*tree.begin(), 1);
}

BOOST_AUTO_TEST_CASE(test_end)
{
  Tree<int> tree;
  const Tree<int> & const_tree = tree;

  BOOST_CHECK(tree.end() == tree.begin());
  BOOST_CHECK(const_tree.end() == tree.begin());

  tree.set_root(1);
  BOOST_CHECK(tree.end() != tree.begin());
  BOOST_CHECK(const_tree.end() != tree.begin());
}

BOOST_AUTO_TEST_CASE(test_append_child)
{
  Tree<int> tree;

  tree.set_root(0);
  node_handle v0 = tree.begin();

  tree.append_child(v0, 1);
  tree.append_child(v0, 2);
  tree.append_child(v0, 3);

  sibling_iterator child = tree.children_begin(v0);
  for (int i = 1; i <= 3; ++i, ++child)
    BOOST_CHECK_EQUAL(*child, i);
}

BOOST_AUTO_TEST_CASE(test_prepend_child)
{
  Tree<int> tree;

  tree.set_root(0);
  node_handle v0 = tree.begin();

  tree.prepend_child(v0, 1);
  tree.prepend_child(v0, 2);
  tree.prepend_child(v0, 3);

  sibling_iterator child = tree.children_begin(v0);
  for (int i = 3; i > 0; --i, ++child)
    BOOST_CHECK_EQUAL(*child, i);
}

BOOST_AUTO_TEST_CASE(test_insert_child_before)
{
  Tree<int> tree;

  tree.set_root(0);

  node_handle root = tree.begin();
  BOOST_CHECK_THROW(tree.insert_sibling_before(root, 2), std::exception);

  node_handle child1 = tree.append_child(root, 1);
  node_handle child2 = tree.append_child(root, 3);
  tree.insert_sibling_before(child1, 0);
  tree.insert_sibling_before(child2, 2);

  sibling_iterator child = tree.children_begin(root);
  for (int i = 0; i <= 3; ++i, ++child)
    BOOST_CHECK_EQUAL(*child, i);
}

BOOST_AUTO_TEST_CASE(test_insert_child_after)
{
  Tree<int> tree;

  tree.set_root(0);

  node_handle root = tree.begin();
  BOOST_CHECK_THROW(tree.insert_sibling_after(root, 2), std::exception);

  node_handle child1 = tree.append_child(root, 1);
  node_handle child2 = tree.append_child(root, 3);
  tree.insert_sibling_after(child1, 2);
  tree.insert_sibling_after(child2, 4);

  sibling_iterator child = tree.children_begin(root);
  for (int i = 1; i <= 4; ++i, ++child)
    BOOST_CHECK_EQUAL(*child, i);
}


// Test sibling iterator.
BOOST_AUTO_TEST_CASE(test_sibling_iterator)
{
  Tree<int> tree;

  tree.set_root(0);
  node_handle v0 = tree.begin();

  tree.append_child(v0, 1);
  tree.append_child(v0, 2);
  tree.append_child(v0, 3);

  sibling_iterator child = tree.children_begin(v0);
  for (int i = 1; i <= 3; ++i, ++child)
    BOOST_CHECK_EQUAL(*child, i);
  BOOST_CHECK(child == tree.children_end());

  child = tree.children_rbegin(v0);
  for (int i = 3; i >= 1; --i, --child)
    BOOST_CHECK_EQUAL(*child, i);
  BOOST_CHECK(child == tree.children_rend());
}

BOOST_AUTO_TEST_CASE(test_const_sibling_iterator)
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
    BOOST_CHECK_EQUAL(*const_child, i);
  BOOST_CHECK(const_child == tree.children_end());

  const_child = tree.children_rbegin(v0);
  for (int i = 3; i >= 1; --i, --const_child)
    BOOST_CHECK_EQUAL(*const_child, i);
  BOOST_CHECK(const_child == tree.children_rend());
}


// Test depth-first iterator.
BOOST_AUTO_TEST_CASE(test_depth_first_tree)
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
    BOOST_CHECK_EQUAL(*dfs_it, i);
    BOOST_CHECK_EQUAL(*const_dfs_it, i);
  }
  BOOST_CHECK(dfs_it == tree.depth_first_end());
  BOOST_CHECK(const_dfs_it == tree.depth_first_end());

  dfs_it = tree.depth_first_rbegin();
  const_dfs_it = tree.depth_first_rbegin();
  for (int i = 14; i >= 0; --i, --dfs_it, --const_dfs_it)
  {
    BOOST_CHECK_EQUAL(*dfs_it, i);
    BOOST_CHECK_EQUAL(*const_dfs_it, i);
  }
  BOOST_CHECK(dfs_it == tree.depth_first_rend());
  BOOST_CHECK(const_dfs_it == tree.depth_first_rend());
}


// Test breadth-first iterator.
BOOST_AUTO_TEST_CASE(test_breadth_first_iterator)
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
    BOOST_CHECK_EQUAL(*bfs_it, i);
    BOOST_CHECK_EQUAL(*const_bfs_it, i);
  }
  BOOST_CHECK(bfs_it == tree.breadth_first_end());
  BOOST_CHECK(const_bfs_it == tree.breadth_first_end());
}


// Test leaf iterator.
BOOST_AUTO_TEST_CASE(test_leaf_iterator)
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
    BOOST_CHECK_EQUAL(*bfs_it, i);
    BOOST_CHECK_EQUAL(*const_bfs_it, i);
  }
  BOOST_CHECK(bfs_it == tree.breadth_first_end());
  BOOST_CHECK(const_bfs_it == tree.breadth_first_end());

  leaf_iterator leaf_it = tree.leaf_begin();
  const_leaf_iterator const_leaf_it = tree.leaf_begin();
  for (int i = 7; i <= 14; ++i, ++leaf_it, ++const_leaf_it)
  {
    BOOST_CHECK_EQUAL(*leaf_it, i);
    BOOST_CHECK_EQUAL(*const_leaf_it, i);
  }
  BOOST_CHECK(leaf_it == tree.leaf_end());
  BOOST_CHECK(const_leaf_it == tree.leaf_end());
}

BOOST_AUTO_TEST_SUITE_END()
