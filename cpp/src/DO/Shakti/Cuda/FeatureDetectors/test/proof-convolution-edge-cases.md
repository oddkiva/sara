# Column-based convolution

## Edge-case where: w < r

Example:
- r = 5
- w = 3

Intermediate calculus:
-  left border : [0  , r      [ =                 [0,  5[
-    main data : [r+0, r+w    [ =                 [5,  8[
- right border : [w+r, r+w+r+1[ = [w+r, w+2r+1[ = [8, 14[

How many necessary reads.

Indexing |  Left Border       |  Main Data      |  Right border     | Stop
---------|--------------------|-----------------|-------------------|----------
Absolute |  0             r-1 |  0+r ...  w-1+r |  w           w+2r | w+2r+1
         |  0               4 |  5   ...      8 |                   |     14
---------|--------------------------------------|-------------------|----------
Relative | -r              -1 |  0   ...    w-1 |  w  ...      w+r  |      9
         | -5  -4  -3  -2  -1 |  0            3 |  4  5  6  7    8  |      9
------------------------------------------------|-------------------|----------

## Load-Store Operation into the Shared Memory (SMem)

- tx = thread index
-  x = global index

### Not caring about the edge case: w < r

Load-store operation happens if: x < w

   Active Thread | Active   |  Ignored
-----------------|----------|-----------------------
               x |  0  1  2 |  3  4  5  6  7  8  9
-----------------|----------|-----------------------
      Load-Store |          | Missing indices in shared memory.
-----------------|----------|-----------------------
   Left border l | -5 -4 -3 | -2 -1  0  1  2  3  4(AIE)            
-----------------|----------|-----------------------
  Right border r |  5  6  7 |  8  9 11


### Caring about the edge case: w < r

Load-store operation happens if: x < max(w, r) + r
                                 x < max(3, 5) + 5
                                 x < 10

   Active Thread | Active    
-----------------|--------------------------------
               x |  0  1  2  3  4  5  6  7  8  9
-----------------|--------------------------------
      Load-Store |          
-----------------|--------------------------------
   Left border l | -5 -4 -3 -2 -1  0  1  2  3  4  
-----------------|--------------------------------
  Right border r |  5  6  7  8  9 11 12 13 14 15


### w == r

Load-store operation happens if: x < max(w, r) + r
                                 x < 10

   Active Thread | Active    
-----------------|--------------------------------
               x |  0  1  2  3  4  5  6  7  8  9
-----------------|--------------------------------
      Load-Store |          
-----------------|--------------------------------
   Left border l | -5 -4 -3 -2 -1  0  1  2  3  4  
-----------------|--------------------------------
  Right border r |  5  6  7  8  9 11 12 13 14 15
