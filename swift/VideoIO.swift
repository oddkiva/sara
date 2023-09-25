import CxxStdlib
import SaraGraphics

public class VideoStream {
  private var impl: VideoStreamImpl

  var frame: ImageView<UInt8>

  init(filePath: String) {
    print("Initing...")
    self.impl = VideoStreamImpl(std.string(filePath))
    print("Initing done...")
    self.frame = ImageView<UInt8>()

    let w = Int(impl.frameWidth())
    let h = Int(impl.frameHeight())
    let framePtr = impl.framePointer()
    let frameByteSize = w * h * 3 /* RGB */

    self.frame.dataPtr = UnsafeMutableBufferPointer<UInt8>(
      start: framePtr,
      count: frameByteSize
    )
    self.frame.width = w
    self.frame.height = h
    self.frame.numChannels = 3
  }

  func read() -> Bool {
    return impl.read()
  }
}
