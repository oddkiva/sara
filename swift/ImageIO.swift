import SaraGraphics


public class ImageReader {
  private var _reader: UnsafeMutableRawPointer

  init(filepath: String) {
    self._reader = ImageReader_init(filepath.cString(using: .utf8))
  }

  func read(image: inout Image<UInt8>) {
    var w: Int32 = 0
    var h: Int32 = 0
    var c: Int32 = 0
    ImageReader_imageSizes(_reader, &w, &h, &c);
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
      ImageReader_readImageData(self._reader, ptr)
    }
  }

  deinit {
    ImageReader_deinit(self._reader)
  }
}


func imread(filepath: String) -> Image<UInt8> {
  let imreader = ImageReader(filepath: filepath)
  var image = Image<UInt8>()
  imreader.read(image: &image)
  return image
}
