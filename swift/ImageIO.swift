import SaraGraphics


public class JpegImageReader {
    private var _reader: UnsafeMutableRawPointer

    init(filePath: String) {
        self._reader = JpegImageReader_init(filePath.cString(using: .utf8))
    }

    func read(image: inout Image<UInt8>) {
        var w: Int32 = 0
        var h: Int32 = 0
        var c: Int32 = 0
        JpegImageReader_imageSizes(_reader, &w, &h, &c);
        image.width = Int(w)
        image.height = Int(h)
        image.numChannels = Int(c)

        let imageSize = image.numElements()
        image.data = [UInt8](repeating: 0, count: imageSize)
        image.data.withUnsafeMutableBufferPointer {
            (data) in
            let ptr = UnsafeMutableRawPointer(data.baseAddress!).bindMemory(
                to: UInt8.self,
                capacity: imageSize)
            JpegImageReader_readImageData(self._reader, ptr)
        }
    }

    deinit {
        JpegImageReader_deinit(self._reader)
    }
}


func imread(filePath: String) -> Image<UInt8> {
    let imreader = JpegImageReader(filePath: filePath)
    var image = Image<UInt8>()
    imreader.read(image: &image)
    return image
}
