import SaraGraphics


func runGraphics(userMainFn: @escaping (@convention(c) () -> Void)) {
    var ctx = GraphicsContext()
    ctx.registerUserMainFunc(userMainFn)
    ctx.exec()
}


func rgb(_ r: UInt8, _ g: UInt8, _ b: UInt8) -> Color {
    return Color(r: r, g: g, b: b, a: UInt8.max)
}


func drawImage(image: Image<UInt8>) {
    image.data.withUnsafeBufferPointer { (data) in
        let ptr = UnsafeRawPointer(data.baseAddress!).bindMemory(
            to: UInt8.self,
            capacity: image.numElements())
        drawImage(ptr, Int32(image.width), Int32(image.height), 0, 0, 1)
    }
}

func drawImage(image: ImageView<UInt8>) {
    drawImage(image.dataPtr?.baseAddress,
              Int32(image.width), Int32(image.height),
              0, 0, 1)
}
