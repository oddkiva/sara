# Chessboard Detection

- The vanilla junction detector works very well in practice.

- Harris' corner detection works very well.

- Hessian Detection: it's a good idea, but it turns out to be quite tricky to
  filter and set the appropriate threshold for every case.

- Color-based grouping via watershed works, but unstable w.r.t. illumination
  changes: possible but not worth exploring

- The approach uses an edge-linking approach to try to deal with 
  illumination changes more robustly.

- We can reconstruct squares, but we still need the network of corners in the
  form: (i, j) -> (x_ij, y_ij)

- TODO: robustify the line segment detection...


- Connecting end points of edges: use Harris's corner detector to join edges,
  which is better justified.
