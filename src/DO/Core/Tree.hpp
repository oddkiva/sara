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
//! \brief This contains the implementation of the tree data structure.

#ifndef DO_CORE_TREE_HPP
#define DO_CORE_TREE_HPP

#include "Meta.hpp"
#include <queue>
#include <stack>
#include <fstream>

namespace DO {

  //! \ingroup Core
  //! \defgroup Tree Tree
  //! @{

  //! \brief The tree data structure is by definition an arborescence, in graph 
  //! theory, i.e., an directed graph with a root vertex 'u' such that there is 
  //! a unique path from 'u' to any vertex 'v' in the tree.
  //!
  //! \todo: finish testing (cf. methods' description where TODO is present.).
  template <typename T>
  class Tree
  {
  private: /* internal data structures */
    class Node;
    class NullNodeHandleException : public std::exception {};
    template <bool IsConst> class NodeHandle;
    template <bool IsConst> class ChildrenIterator;
    template <bool IsConst> class DepthFirstIterator;
    template <bool IsConst> class BreadthFirstIterator;
    template <bool IsConst> class BreadthFirstIterator;
    template <bool IsConst> class LeafIterator;

  public: /* STL-style typedefs */
    typedef T value_type;
    typedef T * pointer;
    typedef const T * const_pointer;
    typedef T& reference;
    typedef const T& const_reference;

    typedef Node node_type;
    typedef NodeHandle<false> node_handle;
    typedef ChildrenIterator<false> children_iterator;
    typedef DepthFirstIterator<false> depth_first_iterator;
    typedef BreadthFirstIterator<false> breadth_first_iterator;
    typedef LeafIterator<false> leaf_iterator;

    typedef NodeHandle<true> const_node_handle;
    typedef ChildrenIterator<true> const_children_iterator;
    typedef DepthFirstIterator<true> const_depth_first_iterator;
    typedef BreadthFirstIterator<true> const_breadth_first_iterator;
    typedef LeafIterator<true> const_leaf_iterator;

  public: /* interface */
    //! Default constructor
    inline Tree()
      : _root_node_ptr(0) {}

    //! Constructor with root vertex.
    inline Tree(const T& v)
      : _root_node_ptr(new Node(v)) {}

    //! Copy constructor.
    inline Tree(const Tree& t)
      : _root_node_ptr(0)
    { *this = t; }

    //! Destructor.
    inline ~Tree()
    { clear(); }

    //! Assignment operator.
    inline Tree& operator=(const Tree& t)
    {
      clear();
      
      const_node_handle src_node = t.begin();
      set_root(*src_node);
      node_handle dst_node = begin();

      while (src_node != 0)
      {
        if (src_node.first_child() != 0)
        {
          const_node_handle child = src_node.first_child();
          dst_node = append_child(dst_node, *child);
          src_node = child;
        }
        else
        {
          while (src_node.next_sibling() == 0 && src_node != t.begin())
          {
            src_node = src_node.parent();
            dst_node = dst_node.parent();
          }
          if (src_node == t.begin())
            src_node = 0;
          else if (src_node.next_sibling() != 0)
          {
            const_node_handle sibling = src_node.next_sibling();
            dst_node = insert_sibling_after(dst_node, *sibling);
            src_node = sibling;
          }
        }
      }

      return *this;      
    }

    //! Equality operator.
    // \todo: you, dummy! That's false. Because equality can happen even if 
    // the tree structures differs. 
    // Check that each node also has the same number of children. Proof?
    /*bool operator==(const Tree& t) const
    {
      const_depth_first_iterator
        v1 = depth_first_begin(),
        v2 = t.depth_first_begin();

      while ( v1 != depth_first_end() || v2 != t.depth_first_end() )
      {
        if (*v1 != *v2)
          return false;
        ++v1;
        ++v2;
      }

      return true;
    }*/

    //! Inequality operator.
    inline bool operator!=(const Tree& t) const
    { return !(*this == t); }

    //! Swap function.
    inline void swap(const Tree& t)
    { swap(_root_node_ptr, t._root_node_ptr); }

    //! Clear function.
    inline void clear()
    {
      if (empty())
        return;

      //std::cout << "Clearing" << std::endl;
      std::stack<Node *> nodes;
      for (depth_first_iterator n = depth_first_begin(); n != depth_first_end(); ++n)
      {
        //std::cout << "Pushing " << *n << std::endl;
        nodes.push(n());
      }

      while (!nodes.empty())
      {
        Node *n = nodes.top();
        nodes.pop();
        delete n;
      }

      _root_node_ptr = 0;
    }

    //! Returns if the tree is empty.
    inline bool empty() const
    { return begin() == end(); }

    //! Set the root of the tree with value 'v'.
    inline void set_root(const T& v)
    {
      if (empty())
        _root_node_ptr = new Node;
      _root_node_ptr->set_value(v);
    }

    //! Insert a sibling with value 'v' before the specified node and returns
    //! the child node handle.
    inline node_handle insert_sibling_before(node_handle n, const T& v)
    {
      // If the tree is empty
      if (n == begin() && empty())
      {
        set_root(v);
        return begin();
      }
      
      if (n == node_handle())
        throw NullNodeHandleException();

      node_handle sibling(new Node(v));
      n()->insert_sibling_before(sibling());
      return sibling;
    }

    //! Insert a sibling with value 'v' after the specified node and returns
    //! the child node handle.
    inline node_handle insert_sibling_after(node_handle n, const T& v)
    {
      // If the tree is empty
      if (n == begin() && empty())
      {
        set_root(v);
        return begin();
      }

      if (n == node_handle())
        throw NullNodeHandleException();

      node_handle sibling(new Node(v));
      n()->insert_sibling_after(sibling());
      return sibling;
    }

    //! Append child to specified node and returns the child node handle.
    inline node_handle append_child(node_handle n, const T& v)
    { 
      if (n == node_handle())
        throw NullNodeHandleException();
      node_handle child(new Node(v));
      n()->append_child(child());
      return child;
    }

    //! Prepend child to specified node.
    inline node_handle prepend_child(node_handle n, const T& v)
    {
      if (n == node_handle())
        throw NullNodeHandleException();
      node_handle child(new Node(v));
      n()->prepend_child(child());
      return child;
    }

    //! Append child tree to specified node.
    inline void append_child_tree(node_handle node, Tree& tree)
    { node()->append_child(tree.begin()()); }

    //! Prepend child tree to specified node.
    inline void prepend_child_tree(node_handle node, Tree& tree)
    { node()->prepend_child(tree.begin()()); }

    //! Cut the tree at the specified node which becomes the root of the subtree.
    //! \todo: check if the implementation is correct.
    inline Tree cut_tree(node_handle node)
    {
      node.parent().remove_child(node);
      Tree t;
      t._root_node_ptr = node();
      return t;
    }

    //! Delete the subtree at the specified node being the root of the subtree.
    //! \todo: check if the implementation is correct.
    inline void delete_subtree(node_handle node)
    {
      node.parent().remove_child(node);
      Tree t;
      t._root_node_ptr = node();
      t.clear();
    }

    //! Returns the root of the tree.
    inline node_handle begin()
    { return _root_node_ptr; }
    //! Returns the last node of the tree.
    inline node_handle end()
    { return 0; };

    //! Returns the parent of the input node.
    inline node_handle parent_of(node_handle v)
    {
      if (v == node_handle())
        throw NullNodeHandleException();
      return v()->_parent;
    }
    //! Returns the first child iterator.
    inline children_iterator children_begin(node_handle v)
    { return children_iterator(v()); }
    //! Returns the last child iterator.
    inline children_iterator children_end()
    { return children_iterator(); }

    //! Returns the first depth-first iterator.
    inline depth_first_iterator depth_first_begin()
    { return depth_first_iterator(_root_node_ptr); }
    //! Returns the last depth-first iterator.
    inline depth_first_iterator depth_first_end()
    { return depth_first_iterator(); }

    //! Returns the first breadth-first iterator.
    inline breadth_first_iterator breadth_first_begin()
    { return breadth_first_iterator(_root_node_ptr); }
    //! Returns the last breadth-first iterator.
    inline breadth_first_iterator breadth_first_end()
    { return breadth_first_iterator(); }

    //! Returns the first leaf iterator
    inline leaf_iterator leaf_begin()
    { return leaf_iterator(_root_node_ptr); }
    //! Returns the last leaf iterator.
    inline leaf_iterator leaf_end()
    { return leaf_iterator(); }

    //! Returns the root of the tree (constant accessor).
    inline const_node_handle begin() const
    { return _root_node_ptr; }
    //! Returns the last node of the tree (constant access).
    inline const_node_handle end() const
    { return 0; }

    //! Returns the parent of the input node.
    inline const_node_handle parent_of(node_handle v) const
    {
      if (v == const_node_handle())
        throw NullNodeHandleException();
      return v()->_parent;
    }
    //! Returns the first constant child iterator.
    inline const_children_iterator children_begin(const_node_handle v) const
    { return const_children_iterator(v()); }
    //! Returns the last constant child iterator.
    inline const_children_iterator children_end() const
    { return const_children_iterator(); }

    //! Returns the first constant depth-first iterator.
    inline const_depth_first_iterator depth_first_begin() const
    { return const_depth_first_iterator(_root_node_ptr); }
    //! Returns the last constant depth-first iterator.
    inline const_depth_first_iterator depth_first_end() const
    { return const_depth_first_iterator(); }

    //! Returns the first constant breadth-first iterator.
    inline const_breadth_first_iterator breadth_first_begin() const
    { return const_breadth_first_iterator(_root_node_ptr); }
    //! Returns the last constant breadth-first iterator.
    inline const_breadth_first_iterator breadth_first_end() const
    { return const_breadth_first_iterator(); }

    //! Returns the first leaf iterator
    inline const_leaf_iterator leaf_begin() const
    { return const_leaf_iterator(_root_node_ptr); }
    //! Returns the last leaf iterator.
    inline const_leaf_iterator leaf_end() const
    { return const_leaf_iterator(); }

  private: /* classes */
    // Node implementation.
    class Node
    {
    public:
      inline Node()
        : _value()
        , _parent(0)
        , _prev_sibling(0), _next_sibling(0)
        , _first_child(0), _last_child(0)
      {
      }

      inline Node(const T& v)
        : _value(v)
        , _parent(0)
        , _prev_sibling(0), _next_sibling(0)
        , _first_child(0), _last_child(0)
      {
      }

      void set_value(const T& v)
      { _value = v; }

      void insert_sibling_before(Node *sibling)
      {
        if (_prev_sibling)
          _prev_sibling->_next_sibling = sibling;
        else
          _parent->_first_child = sibling;
        
        _prev_sibling = sibling;
        sibling->_next_sibling = this;
        sibling->_parent = this->_parent;
      }

      void insert_sibling_after(Node *sibling)
      {
        if (_next_sibling)
          _next_sibling->_prev_sibling = sibling;
        else
          _parent->_last_child = sibling;
        _next_sibling = sibling;
        sibling->_prev_sibling = this;
        sibling->_parent = this->_parent;
      }

      void append_child(Node *child)
      {
        child->_parent = this;
        child->_prev_sibling = _last_child;
        child->_next_sibling = 0;

        if (_first_child == 0)
          _first_child = child;
        if (_last_child)
          _last_child->_next_sibling = child;
        _last_child = child;
      }

      void prepend_child(Node *child)
      {
        child->_parent = this;
        child->_prev_sibling = 0;
        child->_next_sibling = _first_child;

        if (_last_child == 0)
          _last_child = child;
        if (_first_child)
          _first_child->_prev_sibling = child;
        _first_child = child;
      }

      //! Caution: this does not destroy the child node, which can actually be
      //! a subtree, since it may have descendants.
      void remove_child(Node *child)
      {
        Node *prev_sibling = child->_prev_sibling;
        Node *next_sibling = child->_next_sibling;

        // Modify myself, the parent of this child.
        if (_first_child == child)
          _first_child = next_sibling;
        if (_last_child == child)
          _last_child = prev_sibling;

        // Modify the siblings.
        prev_sibling->_next_sibling = next_sibling;
        next_sibling->_prev_sibling = prev_sibling;

        // Make this child an orphan.
        child->_parent = 0;
        // ...without siblings
        child->_prev_sibling = 0;
        child->_next_sibling = 0;
      }

      T _value;
      Node *_parent;
      Node *_prev_sibling, *_next_sibling;
      Node *_first_child, *_last_child;
    };

    // Node handle implementation.
    template <bool IsConst>
    class NodeHandle
    {
    public:
      // STL style typedefs.
      typedef NodeHandle self_type;
      typedef typename Meta::Choose<IsConst, const Node *, Node *>::Type 
        node_pointer;
      typedef typename Meta::Choose<IsConst, const T&, T&>::Type reference;
      typedef typename Meta::Choose<IsConst, const T *, T *>::Type pointer;

      inline NodeHandle(node_pointer node_ptr = 0)
        : _node_ptr(node_ptr)
      {}

      inline NodeHandle(const NodeHandle<false>& node)
        : _node_ptr(node())
      {
      }

      inline reference operator*() const
      {
        if (!_node_ptr)
          throw NullNodeHandleException();
        return _node_ptr->_value;
      }

      inline pointer operator->() const
      {
        if (!_node_ptr)
          throw NullNodeHandleException();
        return &(_node_ptr->_value);
      }

      // Equality and difference comparisons
      template <bool is_const_2>
      inline bool operator==(const NodeHandle<is_const_2>& iter2) const
      { return _node_ptr == iter2(); }

      template <bool is_const_2>
      inline bool operator!=(const NodeHandle<is_const_2>& iter2) const
      { return _node_ptr != iter2(); }

      inline bool operator==(node_pointer node) const
      { return _node_ptr == node; }

      inline bool operator!=(node_pointer node) const
      { return _node_ptr != node; }

      inline node_pointer operator()() const
      { return _node_ptr; }

      inline self_type parent() const
      { return _node_ptr->_parent; }

      inline self_type prev_sibling() const
      { return _node_ptr->_prev_sibling; }

      inline self_type next_sibling() const
      { return _node_ptr->_next_sibling; }

      inline self_type first_child() const
      { return _node_ptr->_first_child; }

      inline self_type last_child() const
      { return _node_ptr->_last_child; }

      inline bool is_first_child() const
      {
        if (!_node_ptr)
          throw NullNodeHandleException();
        if (!_node_ptr->_parent)
          return true;
        return _node_ptr == _node_ptr->_parent->_first_child;
      }

      inline bool is_last_child() const
      {
        if (!_node_ptr)
          throw NullNodeHandleException();
        if (!_node_ptr->_parent)
          return true;
        return _node_ptr == _node_ptr->_parent->_last_child;
      }

    protected:
      node_pointer _node_ptr;
    };

    // Children iterator implementation.
    template <bool IsConst>
    class ChildrenIterator : public NodeHandle<IsConst>
    {
      typedef ChildrenIterator self_type;
      typedef NodeHandle<IsConst> base_type;
      typedef typename base_type::node_pointer node_pointer;
      using base_type::_node_ptr;

    public:
      inline ChildrenIterator()
        : base_type(0) {}

      inline ChildrenIterator(node_pointer parent)
        : base_type(parent->_first_child) {}
      
      inline ChildrenIterator(const NodeHandle<false>& node)
        : base_type(node()->_first_child) {}

      inline self_type& operator++()
      {
        if(_node_ptr)
          _node_ptr = _node_ptr->_next_sibling;
        return *this;
      }

      inline self_type& operator--()
      {
        if(_node_ptr)
          _node_ptr = _node_ptr->_prev_sibling;
        return *this;
      }

      inline self_type operator++(int) const
      {
        ChildrenIterator prev(*this);
        operator++();
        return prev;
      }

      inline self_type operator--(int) const
      {
        ChildrenIterator prev(*this);
        operator--();
        return prev;
      }
    };

    // Depth-first iterator implementation.
    template <bool IsConst>
    class DepthFirstIterator : public NodeHandle<IsConst>
    {
      typedef DepthFirstIterator self_type;
      typedef NodeHandle<IsConst> base_type;

    protected:
      typedef typename base_type::node_pointer node_pointer;
      using base_type::_node_ptr;

    public:
      inline DepthFirstIterator()
        : base_type(0) {}

      inline DepthFirstIterator(node_pointer node_ptr)
        : base_type(node_ptr), _root_node_ptr(node_ptr) {}

      inline DepthFirstIterator(const NodeHandle<false>& node)
        : base_type(node), _root_node_ptr(node()) {}

      self_type& operator++()
      {
        if (_node_ptr == 0)
          return *this;

        if (_node_ptr->_first_child)
          _node_ptr = _node_ptr->_first_child;
        else
        {
          while (_node_ptr->_next_sibling == 0 && _node_ptr != _root_node_ptr)
            _node_ptr = _node_ptr->_parent;
          if (_node_ptr == _root_node_ptr)
            _node_ptr = 0;
          else if (_node_ptr->_next_sibling)
            _node_ptr = _node_ptr->_next_sibling;
        }

        return *this;
      }

      self_type& operator--()
      {
        if (_node_ptr == 0)
          return *this;

        if (_node_ptr->_prev_sibling)
        {
          _node_ptr = _node_ptr->_prev_sibling;
          while (_node_ptr->_last_child)
            _node_ptr = _node_ptr->_last_child;
        }
        else
          _node_ptr = _node_ptr->_parent;

        return *this;
      }

      self_type operator++(int)
      {
        DepthFirstIterator prev(*this);
        operator++();
        return prev;
      }

      self_type operator--(int)
      {
        DepthFirstIterator prev(*this);
        operator--();
        return prev;
      }

    protected:
      node_pointer _root_node_ptr;
    };

    // Breadth-first queued iterator implementation.
    template <bool IsConst>
    class BreadthFirstIterator : public NodeHandle<IsConst>
    {
      typedef BreadthFirstIterator self_type;
      typedef NodeHandle<IsConst> base_type;
      typedef typename base_type::node_pointer node_pointer;
      using base_type::_node_ptr;

    public:
      inline BreadthFirstIterator()
        : base_type(0) {}

      inline BreadthFirstIterator(node_pointer node_ptr)
        : base_type(node_ptr)
      { _queue.push(node_ptr); }

      inline BreadthFirstIterator(const NodeHandle<false>& node)
        : base_type(node())
      { _queue.push(node()); }

      self_type& operator++()
      {
        for (node_pointer n = _queue.front()->_first_child;
             n != 0; n = n->_next_sibling)
          _queue.push(n);
        _queue.pop();
        _node_ptr = _queue.empty() ? 0 : _queue.front();
        return *this;
      }

      self_type operator++(int)
      {
        BreadthFirstIterator prev(*this);
        operator++();
        return prev;
      }

    private:
      std::queue<node_pointer> _queue;
    };

    // Leaf iterator implementation
    template <bool IsConst>
    class LeafIterator : public DepthFirstIterator<IsConst>
    {
      typedef LeafIterator self_type;
      typedef DepthFirstIterator<IsConst> base_type;
      typedef typename base_type::node_pointer node_pointer;
      using base_type::_node_ptr;
      using base_type::_root_node_ptr;

    public:
      inline LeafIterator()
        : base_type() {}

      inline LeafIterator(node_pointer node_ptr)
        : base_type(node_ptr)
      {
        init();
      }

      inline LeafIterator(const NodeHandle<IsConst>& node)
        : base_type(node())
      {
        init();
      }

      self_type& operator++()
      {
        base_type::operator++();
        if (_node_ptr == 0)
          return *this;
        while (_node_ptr->_first_child)
          base_type::operator++();
        return *this;
      }

      self_type& operator--()
      {
        if (!_node_ptr)
          return *this;

        base_type::operator--();
        while (_node_ptr->_first_child)
        {
          base_type::operator--();
          if (!_node_ptr)
            return *this;
        }
        return *this;
      }

      self_type operator++(int)
      {
        LeafIterator prev(*this);
        operator++();
        return prev;
      }

      self_type operator--(int)
      {
        LeafIterator prev(*this);
        operator--();
        return prev;
      }

    private:
      void init()
      {
        if (!_node_ptr)
          return;
        while (_node_ptr->_first_child)
          base_type::operator++();
      }
    };

  private: /* data members */
    Node *_root_node_ptr; //!< Root node of the tree.
  };

  //! Save the tree content in GraphViz format
  template <typename T>
  bool saveTree(const Tree<T>& tree, const std::string& name)
  {
    using namespace std;

    ofstream f(name.c_str());
    if (!f.is_open())
    {
      cerr << "Error: cannot create file:\n" << name << endl;
      return false;
    }

    f << "digraph G {" << endl;

    typedef typename Tree<T>::const_breadth_first_iterator
      const_breadth_first_iterator;
    typedef typename Tree<T>::const_children_iterator const_children_iterator;
    for (const_breadth_first_iterator n = tree.breadth_first_begin();
         n != tree.breadth_first_end(); ++n)
    {
      const_children_iterator child = tree.children_begin(n);
      const_children_iterator end = tree.children_end();
      for ( ; child != end; ++child)
        f << "\t" << *n << " -> " << *child << ";" << endl;
    }

    f << "}" << endl;

    f.close();
    return true;
  }

  //! @}
 
}

#endif /* DO_CORE_TREE_HPP */