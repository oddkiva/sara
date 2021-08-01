protocol TensorLike<T> {
    var dataPtr: UnsafeMutableBufferPointer<T>? = nil
    var sizes = [Int]()
    var strides = [Int]()
}

struct Tensor<T> : TensorView<T> {
    var data = [T]()
}
