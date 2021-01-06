import SaraGraphics


public class VideoStream {
    private var _stream: UnsafeMutableRawPointer

    var frame: ImageView<UInt8>

    init(filePath: String) {
        self._stream = VideoStream_init(filePath.cString(using: .utf8))
        self.frame = ImageView<UInt8>()

        let framePtr = VideoStream_getFramePtr(self._stream)
        let frameWidth = VideoStream_getFrameWidth(self._stream)
        let frameHeight = VideoStream_getFrameHeight(self._stream)
        let frameByteSize = Int(frameWidth) * Int(frameHeight) * 3

        self.frame.dataPtr = UnsafeMutableBufferPointer<UInt8>(
            start: framePtr,
            count: frameByteSize
        )
        self.frame.width = Int(frameWidth)
        self.frame.height = Int(frameHeight)
        self.frame.numChannels = 3
    }

    func read() -> Bool {
        return Bool(VideoStream_readFrame(self._stream) == 1)
    }

    deinit {
        VideoStream_deinit(self._stream)
    }
}
