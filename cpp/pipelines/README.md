SfM pipeline usage
==================

1. Detect SIFT keypoints.
```
${SARA_BUILD_DIR}/bin/detect_sift
  --dirpath ${HOME}/Desktop/Datasets/sfm/castle_int
  --out_h5_file ${HOME}/Desktop/Datasets/sfm/castle_int.h5
  --overwrite  # explicitly overwrite if you need to.
```

2. View detected SIFT keypoints.
```
${SARA_BUILD_DIR}/bin/detect_sift
  --dirpath ${HOME}/Desktop/Datasets/sfm/castle_int
  --out_h5_file ${HOME}/Desktop/Datasets/sfm/castle_int.h5
  --read
```

3. Match keypoints.
```
${SARA_BUILD_DIR}/bin/detect_sift
  --dirpath ${HOME}/Desktop/Datasets/sfm/castle_int
  --out_h5_file ${HOME}/Desktop/Datasets/sfm/castle_int.h5
  --overwrite  # if needed.
```

4. Estimate fundamental matrices.
```
${SARA_BUILD_DIR}/bin/estimate_essential_matrices \
  --dirpath ${HOME}/Desktop/Datasets/sfm/castle_int \
  --out_h5_file ${HOME}/Desktop/Datasets/sfm/castle_int.h5 \
  --overwrite  # if needed.
```

5. Check fundamental matrices.
```
${SARA_BUILD_DIR}/bin/estimate_essential_matrices \
  --dirpath ${HOME}/Desktop/Datasets/sfm/castle_int \
  --out_h5_file ${HOME}/Desktop/Datasets/sfm/castle_int.h5 \
  --read --wait_key
```

6. Estimate essential matrices.
```
${SARA_BUILD_DIR}/bin/estimate_essential_matrices \
  --dirpath ${HOME}/Desktop/Datasets/sfm/castle_int \
  --out_h5_file ${HOME}/Desktop/Datasets/sfm/castle_int.h5 \
  --overwrite  # if needed.
```

7. Check essential matrices.
```
${SARA_BUILD_DIR}/bin/estimate_essential_matrices \
  --dirpath ${HOME}/Desktop/Datasets/sfm/castle_int \
  --out_h5_file ${HOME}/Desktop/Datasets/sfm/castle_int.h5 \
  --read --wait_key
```

8. Triangulate 3D points based on the essential matrices.
```
${SARA_BUILD_DIR}/bin/triangulate \
  --dirpath ${HOME}/Desktop/Datasets/sfm/castle_int \
  --out_h5_file ${HOME}/Desktop/Datasets/sfm/castle_int.h5 \
  --debug  # optional
```
