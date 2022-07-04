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

//! @file
//! @brief This contains the implementation of the tree data structure.

#pragma once

#include <exception>
#include <fstream>
#include <iostream>
#include <queue>
#include <map>
#include <stack>

#include <DO/Sara/Core/Meta.hpp>


namespace DO { namespace Sara {

  //! @ingroup Core
  //! @defgroup Tree Tree
  //! @{

  //! @brief Tree data structure.
  //!
  //! The tree data structure is by definition an arborescence, in graph
  //! theory, i.e., an directed graph with a root vertex 'u' such that there is
  //! a unique path from 'u' to any vertex 'v' in the tree.  !
  //!
  //! @todo: finish testing (cf. methods' description where TODO is present.).
  template <typename T>
  class Tree
  {
  private: /* internal data structures */
    class Node;
    class NullNodeHandleException : public std::exception {};
    template <bool IsConst> class NodeHandle;
    template <bool IsConst> class SiblingIterator;
    template <bool IsConst> class DepthFirstIterator;
    template <bool IsConst> class BreadthFirstIterator;
    template <bool IsConst> class BreadthFirstIterator;
    template <bool IsConst> class LeafIterator;

  public: /* STL-style typedefs */
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using reference = T&;
    using const_reference = const T&;

    using node_type = Node;
    using node_handle = NodeHandle<false>;
    using sibling_iterator = SiblingIterator<false>;
    using depth_first_iterator = DepthFirstIterator<false>;
    using breadth_first_iterator = BreadthFirstIterator<false>;
    using leaf_iterator = LeafIterator<false>;

    using const_node_handle = NodeHandle<true>;
    using const_sibling_iterator = SiblingIterator<true>;
    using const_depth_first_iterator = DepthFirstIterator<true>;
    using const_breadth_first_iterator = BreadthFirstIterator<true>;
    using const_leaf_iterator= LeafIterator<true>;

  public: /* interface */
    //! @brief Default constructor
    Tree() = default;

    //! @brief Copy constructor.
    inline Tree(const Tree& other)
    {
      copy(other);
    }

    //! @brief Move constructor.
    inline Tree(Tree&& other)
    {
      swap(other);
    }

    //! @brief Destructor.
    inline ~Tree()
    {
      clear();
    }

    //! @brief Assignment operator.
    inline Tree& operator=(Tree other)
    {
      swap(other);
      return *this;
    }

#ifndef FIXME
    //! @brief Equality operator.
    //! @todo: you, dummy! That's false. Because equality can happen even if
    //! the tree structures differs.
    //! Check that each node also has the same number of children. Proof?
    bool operator==(const Tree& t) const
    {
      auto v1 = depth_first_begin();
      auto v2 = t.depth_first_begin();

      while (v1 != depth_first_end() || v2 != t.depth_first_end())
      {
        if (*v1 != *v2)
          return false;
        ++v1;
        ++v2;
      }

      return true;
    }
#endif

    //! @brief Inequality operator.
    inline bool operator!=(const Tree& t) const
    {
      return !(*this == t);
    }

    //! @brief Swaps the contents with an other tree.
    inline void swap(Tree& other)
    {
      std::swap(_root_node_ptr, other._root_node_ptr);
    }

    //! @brief Destroys the content of the tree.
    inline void clear()
    {
      if (empty())
        return;

      auto nodes = std::stack<Node *>{};
      for (auto n = depth_first_begin(); n != depth_first_end(); ++n)
        nodes.push(n);

      while (!nodes.empty())
      {
        auto n = nodes.top();
        nodes.pop();
        delete n;
      }

      _root_node_ptr = nullptr;
    }

    //! @brief Returns if the tree is empty.
    inline bool empty() const
    {
      return begin() == end();
    }

    //! @brief Sets the root of the tree with value 'v'.
    inline void set_root(const T& v)
    {
      if (empty())
        _root_node_ptr = new Node{};
      _root_node_ptr->_value = v;
    }

    //! @brief Inserts a sibling with value 'v' before the specified node and
    //! returns the child node handle.
    inline node_handle insert_sibling_before(node_handle n, const T& v)
    {
      // If I am the root, not allowed.
      if (!n || n.parent() == node_handle{})
        throw NullNodeHandleException{};

      node_handle sibling{ new Node(v) };
      n.self()->insert_sibling_before(sibling);
      return sibling;
    }

    //! @brief Inserts a sibling with value 'v' after the specified node and
    //! returns the child node handle.
    inline node_handle insert_sibling_after(node_handle n, const T& v)
    {
      // If I am the root, not allowed.
      if (!n || n.parent() == node_handle{})
        throw NullNodeHandleException{};

      node_handle sibling(new Node{ v });
      n.self()->insert_sibling_after(sibling);
      return sibling;
    }

    //! @brief Append child to specified node and returns the child node handle.
    inline node_handle append_child(node_handle n, const T& v)
    {
      if (n == node_handle{})
        throw NullNodeHandleException{};

      node_handle child{ new Node{ v } };
      n.self()->append_child(child);
      return child;
    }

    //! @brief Prepend child to specified node.
    inline node_handle prepend_child(node_handle n, const T& v)
    {
      if (n == node_handle{})
        throw NullNodeHandleException{};

      node_handle child{ new Node(v) };
      n.self()->prepend_child(child);
      return child;
    }

    //! @brief Append child tree to specified node.
    inline void append_child_tree(node_handle node, Tree& tree)
    {
      node.self()->append_child(tree.begin());
    }

    //! @brief Prepend child tree to specified node.
    inline void prepend_child_tree(node_handle node, Tree& tree)
    {
      node.self()->prepend_child(tree.begin());
    }

#ifdef FIXME
    //! @brief Cut the tree at the specified node which becomes the root of the subtree.
    //! @todo: check if the implementation is correct.
    inline Tree cut_tree(node_handle node)
    {
      node.parent().self()->remove_child(node);
      Tree t{};
      t._root_node_ptr = node;
      return t;
    }

    //! @brief Delete the subtree at the specified node being the root of the subtree.
    //! @todo: check if the implementation is correct.
    inline void delete_subtree(node_handle node)
    {
      node.parent().self()->remove_child(node);
      Tree t{};
      t._root_node_ptr = node.self();
      t.clear();
    }
#endif

    //! @brief Returns the root of the tree.
    inline node_handle begin()
    {
      return _root_node_ptr;
    }

    //! @brief Returns the last node of the tree.
    inline node_handle end()
    {
      return nullptr;
    };

    //! @brief Returns the first child iterator.
    inline sibling_iterator children_begin(node_handle v)
    {
      return sibling_iterator{ v.first_child() };
    }

    //! @brief @brief Returns the last child iterator.
    inline sibling_iterator children_rbegin(node_handle v)
    {
      return sibling_iterator{ v.last_child() };
    }

    //! @brief @brief Returns the last child iterator.
    inline sibling_iterator children_end()
    {
      return sibling_iterator{};
    }

    //! @brief @brief Returns the last child iterator.
    inline sibling_iterator children_rend()
    {
      return sibling_iterator{};
    }

    //! @brief @brief Returns the first depth-first iterator.
    inline depth_first_iterator depth_first_begin()
    {
      return depth_first_iterator{ _root_node_ptr };
    }

    //! @brief @brief Returns the first depth-first reverse iterator.
    inline depth_first_iterator depth_first_rbegin()
    {
      auto last = _root_node_ptr;
      while (last->_last_child)
        last = last->_last_child;
      return depth_first_iterator{ last };
    }

    //! @brief @brief Returns the last depth-first iterator.
    inline depth_first_iterator depth_first_end()
    {
      return depth_first_iterator{};
    }

    //! @brief Returns the last depth-first reverse iterator.
    inline depth_first_iterator depth_first_rend()
    {
      return depth_first_iterator{};
    }

    //! @brief Returns the first breadth-first iterator.
    inline breadth_first_iterator breadth_first_begin()
    {
      return breadth_first_iterator{ _root_node_ptr };
    }

    //! @brief Returns the last breadth-first iterator.
    inline breadth_first_iterator breadth_first_end()
    {
      return breadth_first_iterator{};
    }

    //! @brief Returns the first leaf iterator
    inline leaf_iterator leaf_begin()
    {
      return leaf_iterator{ _root_node_ptr };
    }

    //! @brief Returns the last leaf iterator.
    inline leaf_iterator leaf_end()
    {
      return leaf_iterator{};
    }

    //! @brief Returns the root of the tree (constant accessor).
    inline const_node_handle begin() const
    {
      return _root_node_ptr;
    }

    //! @brief Returns the last node of the tree (constant access).
    inline const_node_handle end() const
    {
      return nullptr;
    }

    //! @brief Returns the first constant child iterator.
    inline const_sibling_iterator children_begin(const_node_handle v) const
    {
      return const_sibling_iterator{ v.first_child() };
    }

    //! @brief Returns the last child iterator.
    inline sibling_iterator children_rbegin(node_handle v) const
    {
      return sibling_iterator{ v.last_child() };
    }

    //! @brief Returns the last constant child iterator.
    inline const_sibling_iterator children_end() const
    {
      return const_sibling_iterator{};
    }

    //! @brief Returns the last constant child iterator.
    inline const_sibling_iterator children_rend() const
    {
      return const_sibling_iterator{};
    }

    //! @brief Returns the first constant depth-first iterator.
    inline const_depth_first_iterator depth_first_begin() const
    {
      return const_depth_first_iterator{ _root_node_ptr };
    }

    //! @brief Returns the last const depth-first iterator.
    inline depth_first_iterator depth_first_rbegin() const
    {
      auto last = _root_node_ptr;
      while (last->_last_child)
        last = last->_last_child;
      return depth_first_iterator{ last };
    }

    //! @brief Returns the last constant depth-first iterator.
    inline const_depth_first_iterator depth_first_end() const
    {
      return const_depth_first_iterator{};
    }

    //! @brief Returns the last const depth-first iterator.
    inline depth_first_iterator depth_first_rend() const
    {
      return const_depth_first_iterator{};
    }

    //! @brief Returns the first constant breadth-first iterator.
    inline const_breadth_first_iterator breadth_first_begin() const
    {
      return const_breadth_first_iterator{ _root_node_ptr };
    }

    //! @brief Returns the last constant breadth-first iterator.
    inline const_breadth_first_iterator breadth_first_end() const
    {
      return const_breadth_first_iterator{};
    }

    //! @brief Returns the first leaf iterator
    inline const_leaf_iterator leaf_begin() const
    {
      return const_leaf_iterator{ _root_node_ptr };
    }

    //! @brief Returns the last leaf iterator.
    inline const_leaf_iterator leaf_end() const
    {
      return const_leaf_iterator{};
    }

  private:
    //! @brief Copies the contents of another tree.
    inline void copy(const Tree& other)
    {
      clear();

      if (other.empty())
        return;

      // Mapping between source node and destination nodes.
      auto src_to_dst = std::map<const Node *, Node *>{};

      // Initialize the copy.
      set_root(*other.begin());
      src_to_dst[other.begin()] = begin();

      // Loop.
      auto src_node = ++other.depth_first_begin();
      auto src_node_end = other.depth_first_end();
      for ( ; src_node != src_node_end; ++src_node)
      {
        // Parent of the source node.
        auto src_parent_node = src_node.parent();
        // Parent of the destination node.
        auto dst_parent_node = src_to_dst[src_parent_node];

        // Create the new node.
        auto dst_node = new Node(*src_node);

        // Connect the parent and the new child.
        dst_parent_node->append_child(dst_node);

        // Update the mapping.
        src_to_dst[src_node] = dst_node;
      }
    }

  private: /* classes */
    //! @brief Tree node class.
    class Node
    {
    public:
      inline Node() = default;

      inline Node(const T& v)
        : _value{ v }
      {
      }

      inline Node(T&& v)
        : _value{ v }
      {
      }

      void insert_sibling_before(Node *sibling)
      {
        sibling->_parent = this->_parent;
        sibling->_prev_sibling = _prev_sibling;
        sibling->_next_sibling = this;

        if (_prev_sibling)
          _prev_sibling->_next_sibling = sibling;
        else
          _parent->_first_child = sibling;
        _prev_sibling = sibling;
      }

      void insert_sibling_after(Node *sibling)
      {
        sibling->_parent = this->_parent;
        sibling->_prev_sibling = this;
        sibling->_next_sibling = _next_sibling;

        if (_next_sibling)
          _next_sibling->_prev_sibling = sibling;
        else
          _parent->_last_child = sibling;
        _next_sibling = sibling;
      }

      void append_child(Node *child)
      {
        child->_parent = this;
        child->_prev_sibling = _last_child;
        child->_next_sibling = nullptr;

        if (!_first_child)
          _first_child = child;
        if (_last_child)
          _last_child->_next_sibling = child;
        _last_child = child;
      }

      void prepend_child(Node *child)
      {
        child->_parent = this;
        child->_prev_sibling = nullptr;
        child->_next_sibling = _first_child;

        if (!_last_child)
          _last_child = child;
        if (_first_child)
          _first_child->_prev_sibling = child;
        _first_child = child;
      }

      //! @brief Caution: this does not destroy the child node, which can actually be
      //! a subtree, since it may have descendants.
      void remove_child(Node *child)
      {
        auto prev_sibling = child->_prev_sibling;
        auto next_sibling = child->_next_sibling;

        // Modify myself, the parent of this child.
        if (_first_child == child)
          _first_child = next_sibling;
        if (_last_child == child)
          _last_child = prev_sibling;

        // Modify the siblings.
        prev_sibling->_next_sibling = next_sibling;
        next_sibling->_prev_sibling = prev_sibling;

        // Make this child an orphan.
        child->_parent = nullptr;
        // ...without siblings
        child->_prev_sibling = nullptr;
        child->_next_sibling = nullptr;
      }

      T _value;
      Node *_parent{ nullptr };
      Node *_prev_sibling{ nullptr };
      Node *_next_sibling{ nullptr };
      Node *_first_child{ nullptr };
      Node *_last_child{ nullptr };
    };

    //! @brief Node handle.
    template <bool IsConst>
    class NodeHandle
    {
    public:
      //! @{
      //! @brief STL style typedefs.
      using self_type = NodeHandle;
      using node_pointer = std::conditional_t<IsConst, const Node*, Node*>;
      using reference = std::conditional_t<IsConst, const T&, T&>;
      using pointer = std::conditional_t<IsConst, const T *, T *>;
      //! @}

      inline NodeHandle(node_pointer node_ptr = nullptr)
        : _node_ptr{ node_ptr }
      {
      }

      inline NodeHandle(const NodeHandle<false>& other)
        : _node_ptr{ other }
      {
      }

      inline operator node_pointer() const
      {
        return _node_ptr;
      }

      inline reference operator*() const
      {
        if (!_node_ptr)
          throw NullNodeHandleException{};
        return _node_ptr->_value;
      }

      inline pointer operator->() const
      {
        if (!_node_ptr)
          throw NullNodeHandleException{};
        return &(_node_ptr->_value);
      }

      // Comparison operators.
      inline bool operator==(const self_type& other) const
      {
        return _node_ptr == other._node_ptr;
      }

      inline bool operator!=(const self_type& other) const
      {
        return !this->operator==(other);
      }

      template <bool IsConst2>
      inline bool operator==(const NodeHandle<IsConst2>& other) const
      {
        return _node_ptr == other.self();
      }

      template <bool IsConst2>
      inline bool operator!=(const NodeHandle<IsConst2>& other) const
      {
        return !this->operator==<IsConst2>(other);
      }

      inline node_pointer self() const
      {
        return _node_ptr;
      }

      inline self_type parent() const
      {
        return _node_ptr->_parent;
      }

      inline self_type prev_sibling() const
      {
        return _node_ptr->_prev_sibling;
      }

      inline self_type next_sibling() const
      {
        return _node_ptr->_next_sibling;
      }

      inline self_type first_child() const
      {
        return _node_ptr->_first_child;
      }

      inline self_type last_child() const
      {
        return _node_ptr->_last_child;
      }

    protected:
      node_pointer _node_ptr;
    };

    //! @brief Children iterator implementation.
    template <bool IsConst>
    class SiblingIterator : public NodeHandle<IsConst>
    {
      using self_type = SiblingIterator;
      using base_type = NodeHandle<IsConst>;
      using node_pointer = typename base_type::node_pointer;

      using base_type::_node_ptr;

    public:
      inline SiblingIterator() = default;

      inline SiblingIterator(const Node *node)
        : base_type{ node }
      {
      }

      inline SiblingIterator(const NodeHandle<false>& node)
        : base_type{ node.self() }
      {
      }

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

      inline self_type operator++(int)
      {
        SiblingIterator prev(*this);
        operator++();
        return prev;
      }

      inline self_type operator--(int)
      {
        SiblingIterator prev(*this);
        operator--();
        return prev;
      }
    };

    //! @brief Depth-first iterator.
    template <bool IsConst>
    class DepthFirstIterator : public NodeHandle<IsConst>
    {
      using self_type = DepthFirstIterator;
      using base_type = NodeHandle<IsConst>;

    protected:
      using node_pointer = typename base_type::node_pointer;
      using base_type::_node_ptr;

    public:
      inline DepthFirstIterator() = default;

      inline DepthFirstIterator(node_pointer node_ptr)
        : base_type{ node_ptr }
        , _root_node_ptr{ node_ptr }
      {
      }

      inline DepthFirstIterator(const NodeHandle<false>& node)
        : base_type{ node }
        , _root_node_ptr{ node.self() }
      {
      }

      self_type& operator++()
      {
        // End of depth-first search?
        if (_node_ptr == nullptr)
          return *this;

        // Go to the first child.
        if (_node_ptr->_first_child)
          _node_ptr = _node_ptr->_first_child;
        else
        {
          while (_node_ptr->_next_sibling == nullptr &&
                 _node_ptr != _root_node_ptr)
            _node_ptr = _node_ptr->_parent;
          if (_node_ptr == _root_node_ptr)
            _node_ptr = nullptr;
          else if (_node_ptr->_next_sibling)
            _node_ptr = _node_ptr->_next_sibling;
        }

        return *this;
      }

      self_type& operator--()
      {
        if (_node_ptr == nullptr)
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
        DepthFirstIterator prev{ *this };
        operator++();
        return prev;
      }

      self_type operator--(int)
      {
        DepthFirstIterator prev{ *this };
        operator--();
        return prev;
      }

    protected:
      node_pointer _root_node_ptr;
    };

    //! @brief Breadth-first queued iterator.
    template <bool IsConst>
    class BreadthFirstIterator : public NodeHandle<IsConst>
    {
      using self_type = BreadthFirstIterator;
      using base_type = NodeHandle<IsConst>;
      using node_pointer = typename base_type::node_pointer;
      using base_type::_node_ptr;

    public:
      inline BreadthFirstIterator() = default;

      inline BreadthFirstIterator(node_pointer node_ptr)
        : base_type{ node_ptr }
      {
        _queue.push(node_ptr);
      }

      inline BreadthFirstIterator(const NodeHandle<false>& node)
        : base_type{ node }
      {
        _queue.push(node.self());
      }

      self_type& operator++()
      {
        auto n = _queue.front()->_first_child;
        for ( ; n != 0; n = n->_next_sibling)
          _queue.push(n);
        _queue.pop();
        _node_ptr = _queue.empty() ? nullptr : _queue.front();
        return *this;
      }

      self_type operator++(int)
      {
        BreadthFirstIterator prev{ *this };
        operator++();
        return prev;
      }

    private:
      std::queue<node_pointer> _queue;
    };

    //! @brief Leaf iterator.
    template <bool IsConst>
    class LeafIterator : public DepthFirstIterator<IsConst>
    {
      template <bool IsConst2> friend class LeafIterator;

      using self_type = LeafIterator;
      using base_type = DepthFirstIterator<IsConst>;
      using node_pointer = typename base_type::node_pointer;
      using base_type::_node_ptr;
      using base_type::_root_node_ptr;

    public:
      inline LeafIterator() = default;

      inline LeafIterator(node_pointer node_ptr)
        : base_type{ node_ptr }
      {
        if (_node_ptr == nullptr)
          return;
        while (_node_ptr->_first_child)
          base_type::operator++();
      }

      inline LeafIterator(const LeafIterator<false>& node)
        : base_type{}
      {
        _node_ptr = node._node_ptr;
        _root_node_ptr = node._root_node_ptr;
      }

      self_type& operator++()
      {
        base_type::operator++();
        if (_node_ptr == nullptr)
          return *this;
        while (_node_ptr->_first_child)
          base_type::operator++();
        return *this;
      }

      self_type& operator--() const
      {
        if (_node_ptr == nullptr)
          return *this;

        base_type::operator--();
        while (_node_ptr->_first_child)
        {
          base_type::operator--();
          if (_node_ptr == nullptr)
            return *this;
        }
        return *this;
      }

      self_type operator++(int)
      {
        LeafIterator prev{ *this };
        operator++();
        return prev;
      }

      self_type operator--(int)
      {
        LeafIterator prev{ *this };
        operator--();
        return prev;
      }
    };

  private: /* data members */
    //! @brief Root node of the tree.
    Node *_root_node_ptr{ nullptr };
  };

  //! @brief Save the tree content in GraphViz format
  template <typename T>
  bool save_tree(const Tree<T>& tree, const std::string& name)
  {
    using namespace std;

    ofstream f{ name };
    if (!f)
    {
      cerr << "Error: cannot create file:\n" << name << endl;
      return false;
    }

    f << "digraph G {" << endl;

    for (auto n = tree.breadth_first_begin();
         n != tree.breadth_first_end(); ++n)
    {
      auto child = tree.children_begin(n);
      auto end = tree.children_end();
      for ( ; child != end; ++child)
        f << "\t" << *n << " -> " << *child << ";" << endl;
    }

    f << "}" << endl;

    f.close();
    return true;
  }

  //! @}


} /* namespace Sara */
} /* namespace DO */
