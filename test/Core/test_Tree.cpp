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

#include <gtest/gtest.h>
#include <DO/Defines.hpp>
#include <DO/Core/Tree.hpp>

using namespace DO;
using namespace std;

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

TEST(DO_Core_Test, treeTest)
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
  ASSERT_EQ(tree.empty(), true);
  
  // Get the root vertex.
  tree.set_root(1);
  // Note that the following also works:
  //tree.insert_sibling_after(root, 1) ;
  //tree.insert_sibling_before(root, 1);
  ASSERT_EQ(*tree.begin(), 1);
  
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
  
  cout << "3 -> 7" << endl;
  tree2.append_child(n3, 7);
  checkLeaves(tree2);

  saveTree(tree, srcPath("tree.gv"));
}

int main(int argc, char** argv) 
{
  testing::InitGoogleTest(&argc, argv); 
  return RUN_ALL_TESTS();
}