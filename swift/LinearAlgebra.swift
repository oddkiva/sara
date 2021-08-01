import Accelerate


func gemm<T>(A: [T], B: [T], AB: inout [T], m: Int, n: Int, k: Int) {
    let order = CBLAS_ORDER::CblasColMajor;
    let transa = CBLAS_TRANSPOSE::CblasNoTrans;
    let transb = CBLAS_TRANSPOSE::CblasNoTrans;
    let lda = m;
    let ldb = k;
    let ldc = m;

    switch T.type {
    case is Double:
        cblas_dgemm(order, transa, transb, m, n, k, alpha, A.data(), lda,
                    B.data(), ldb, beta, C.data(), ldc);
    case is Float:
        cblas_sgemm(order, transa, transb, m, n, k, alpha, A.data(), lda,
                    B.data(), ldb, beta, C.data(), ldc);
    }
}
