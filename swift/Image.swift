import SaraGraphics


struct ImageView<T> {
    var dataPtr: UnsafeMutableBufferPointer<T>? = nil
    var width: Int = 0
    var height: Int = 0
    var numChannels: Int = 0

    func pixel(_ x: Int, _ y: Int) -> (T, T, T) {
        let index = (y * width + x) * numChannels
        let r = self.dataPtr![index + 0]
        let g = self.dataPtr![index + 1]
        let b = self.dataPtr![index + 2]
        return (r, g, b)
    }
}


struct Image<T> {
    var data: [T] = []
    var width: Int = 0
    var height: Int = 0
    var numChannels: Int = 0

    func rgb(_ x: Int, _ y: Int) -> (T, T, T) {
        let index = (y * width + x) * numChannels
        let r = self.data[index + 0]
        let g = self.data[index + 1]
        let b = self.data[index + 2]
        return (r, g, b)
    }

    func gray(_ x: Int, _ y: Int) -> T {
        let index = y * width + x
        let gray = self.data[index]
        return gray
    }

    func numElements() -> Int {
        return self.width * self.height * self.numChannels
    }

    mutating func view() -> ImageView<T> {
        var imageView = ImageView<T>()

        self.data.withUnsafeMutableBufferPointer {
            (data) in
            imageView.dataPtr = data
        }
        imageView.width = self.width
        imageView.height = self.height
        imageView.numChannels = self.numChannels

        return imageView
    }
}
