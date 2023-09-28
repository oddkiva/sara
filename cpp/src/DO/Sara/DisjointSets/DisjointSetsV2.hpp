/*
 *	 Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>
 *
 *	 This software is provided 'as-is', without any express or implied
 *	 warranty.  In no event will the authors be held liable for any damages
 *	 arising from the use of this software.
 *
 *	 Permission is granted to anyone to use this software for any purpose,
 *	 including commercial applications, and to alter it and redistribute it
 *	 freely, subject to the following restrictions:
 *
 *	 1. The origin of this software must not be misrepresented; you must not
 *	 claim that you wrote the original software. If you use this software
 *	 in a product, an acknowledgment in the product documentation would be
 *	 appreciated but is not required.
 *	 2. Altered source versions must be plainly marked as such, and must not be
 *	 misrepresented as being the original software.
 *	 3. This notice may not be removed or altered from any source distribution.
 *
 */

#pragma once

#include <atomic>
#include <iostream>
#include <numeric>
#include <vector>


namespace DO::Sara::v2 {

  //! @brief Concurrent implementation of the disjoint set.
  /*!
   *  Note below from the original author:
   *
   *  Lock-free parallel disjoint set data structure (aka UNION-FIND)
   *  with path compression and union by rank
   *
   *  Supports concurrent find_set(), same_set() and join() calls as described
   *  in the paper
   *
   *  "Wait-free Parallel Algorithms for the Union-Find Problem"
   *  by Richard J. Anderson and Heather Woll
   *
   *  In addition, this class supports optimistic locking (try_lock/unlock)
   *  of disjoint sets and a combined join+unlock operation.
   *
   *  @author Wenzel Jakob
   *
   *
   *  My notes:
   *  I have re-read the code to study the implementation details and compare
   *  the similarities of the synchronous version which I have implemented.
   */
  class DisjointSets
  {
  public:
    DisjointSets(std::uint32_t size)
      : _data(size)
    {
      std::iota(_data.begin(), _data.end(), std::uint32_t{});
    }

    // Find the component ID of the vertex.
    std::uint32_t find_set(std::uint32_t id) const
    {
      // 8-byte long mask:                     7 6 5 4 3 2 1 0
      static constexpr std::uint64_t mask = 0xFFFFFFFF00000000ULL;
      // That is:
      // - bits are 0 on the first four bytes.
      // - bits are 1 on the last four bytes.

      // The component id is the vertex that originates the component and it
      // means that the parent of this vertex is the vertex itself.
      while (id != parent(id))
      {
        std::uint64_t value = _data[id];
        std::uint32_t new_parent = parent(std::uint32_t(value));

        const auto rank_shifted = value & mask;
        std::uint64_t new_value = rank_shifted | new_parent;

        /* Try to update parent (may fail, that's ok) */
        if (value != new_value)
          _data[id].compare_exchange_weak(value, new_value);

        id = new_parent;
      }
      return id;
    }

    bool same_set(std::uint32_t id1, std::uint32_t id2) const
    {
      for (;;)
      {
        id1 = find_set(id1);
        id2 = find_set(id2);
        if (id1 == id2)
          return true;
        if (parent(id1) == id1)
          return false;
      }
    }

    std::uint32_t join(std::uint32_t id1, std::uint32_t id2)
    {
      for (;;)
      {
        id1 = find_set(id1);
        id2 = find_set(id2);
        static_assert(std::is_same_v<decltype(id1), std::uint32_t>);
        static_assert(std::is_same_v<decltype(id2), std::uint32_t>);

        if (id1 == id2)
          return id1;

        // Link phase.
        auto r1 = rank(id1);
        auto r2 = rank(id2);
        static_assert(std::is_same_v<decltype(r1), std::uint32_t>);
        static_assert(std::is_same_v<decltype(r2), std::uint32_t>);

        if (r1 > r2 || (r1 == r2 && id1 < id2))
        {
          std::swap(r1, r2);
          std::swap(id1, id2);
        }

        auto old_entry = (std::uint64_t(r1) << 32) | id1;
        auto new_entry = (std::uint64_t(r1) << 32) | id2;

        if (!_data[id1].compare_exchange_strong(old_entry, new_entry))
          continue;

        if (r1 == r2)
        {
          old_entry = (std::uint64_t(r2) << 32) | id2;
          new_entry = (std::uint64_t(r2 + 1) << 32) | id2;

          /* Try to update the rank (may fail, that's ok) */
          _data[id2].compare_exchange_weak(old_entry, new_entry);
        }

        break;
      }
      return id2;
    }

    /*!
     *  Try to lock the a disjoint union identified by one of its elements (this
     *  can occasionally fail when there are concurrent operations). The
     *  parameter 'id' will be updated to store the current representative ID of
     *  the union.
     */
    bool try_lock(std::uint32_t& id)
    {
      constexpr std::uint64_t lock_flag = 1ULL << 63;

      id = find_set(id);
      std::uint64_t value = _data[id];
      if ((value & lock_flag) || (std::uint32_t) value != id)
        return false;
// On IA32/x64, a PAUSE instruction is recommended for CAS busy loops
#if defined(__i386__) || defined(__amd64__)
      __asm__ __volatile__("pause\n");
#endif
      return _data[id].compare_exchange_strong(value, value | lock_flag);
    }

    void unlock(std::uint32_t id)
    {
      const std::uint64_t lock_flag = 1ULL << 63;
      _data[id] &= ~lock_flag;
    }

    /*!
     *  Return the representative index of the set that results from merging
     *  locked disjoint sets 'id1' and 'id2'
     */
    std::uint32_t join_index_locked(std::uint32_t id1, std::uint32_t id2) const
    {
      std::uint32_t r1 = rank(id1), r2 = rank(id2);
      return (r1 > r2 || (r1 == r2 && id1 < id2)) ? id1 : id2;
    }

    /*!
     *  Atomically join two locked disjoint sets and unlock them. Assumes that
     *  here are no other concurrent join() involving the same_set sets
     */
    std::uint32_t join_unlock(std::uint32_t id1, std::uint32_t id2)
    {
      std::uint32_t r1 = rank(id1), r2 = rank(id2);

      if (r1 > r2 || (r1 == r2 && id1 < id2))
      {
        std::swap(r1, r2);
        std::swap(id1, id2);
      }

      _data[id1] = ((std::uint64_t) r1 << 32) | id2;
      _data[id2] = ((std::uint64_t)(r2 + ((r1 == r2) ? 1 : 0)) << 32) | id2;

      return id2;
    }

    std::uint32_t size() const
    {
      return std::uint32_t(_data.size());
    }

    std::uint32_t rank(std::uint32_t id) const
    {
      // Shift to the right
      //                                          7   here probably because of
      //                                          |   the lockflag.
      //                                          |
      return (std::uint32_t(_data[id] >> 32)) & 0x7FFFFFFFu;

      // That means:
      // - there must be a (relatively large) maximum number on the number of
      // vertices after which the implementation can fail, and
      // - there must be preferably at least no singleton component.
    }

    std::uint32_t parent(std::uint32_t id) const
    {
      // The last 4 bytes of the datum is the rank value and is discarded by
      // type casting.
      return std::uint32_t(_data[id]);
    }

    friend std::ostream& operator<<(std::ostream& os, const DisjointSets& f)
    {
      for (auto i = 0u; i < f._data.size(); ++i)
        os << i << ": parent=" << f.parent(i) << ", rank=" << f.rank(i)
           << std::endl;
      return os;
    }

    //! @brief List of rank-parent pairs.
    /*!
     *  The datum associated to vertex `i` contains two informations:
     *  - the parent on the first 4 bytes.
     *  - the rank on the last 4 bytes.
     */
    mutable std::vector<std::atomic<std::uint64_t>> _data;
  };

}  // namespace DO::Sara::v2
