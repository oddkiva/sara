import CxxStdlib
import SaraGraphics

func imread(filePath: String) -> Image<UInt8> {
  var imageReader = JpegImageReader(std.string(filePath))

  var image = Image<UInt8>()
  let imageSizes = imageReader.imageSizes()

  image.width = Int(imageSizes[0])
  image.height = Int(imageSizes[1])
  image.numChannels = Int(imageSizes[2])

  let imageSize = image.numElements()
  image.data = [UInt8](repeating: 0, count: imageSize)
  image.data.withUnsafeMutableBufferPointer { (data) in
    let ptr = UnsafeMutableRawPointer(data.baseAddress!).bindMemory(
      to: UInt8.self,
      capacity: imageSize)
    imageReader.read(ptr)
  }
  return image
}
