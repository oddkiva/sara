import SaraGraphics


public class GraphicsContext {
    private var _qApp: UnsafeMutableRawPointer
    private var _widgetList: UnsafeMutableRawPointer

    init() {
        print("Init Graphics Context...")
        self._qApp = GraphicsContext_initQApp()
        self._widgetList = GraphicsContext_initWidgetList()
    }

    func registerUserMainFunc(userMainFn: @escaping (@convention(c) () -> Void)) {
        GraphicsContext_registerUserMainFunc(userMainFn)
    }

    func exec() {
        GraphicsContext_exec(self._qApp)
    }

    deinit {
        print("Deinit Graphics Context...")
        GraphicsContext_deinitWidgetList(self._widgetList)
    }
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
