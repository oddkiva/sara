import Foundation

import SaraCore
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


func main() {
  var numbers: [Int32] = [1, 2, 3, 5]
  print("Before squaring: \(numbers)")

  tic()
  // Call a C function from Swift.
  numbers.withUnsafeMutableBufferPointer {
    (numbers) in
    let ptr = UnsafeMutableRawPointer(numbers.baseAddress!).bindMemory(
      to: CInt.self,
      capacity: numbers.count)
    square(ptr, CInt(numbers.count))
  }
  toc("Squaring Operation")

  print("After squaring: \(numbers)")

  usleep(1000*1000)

  let w: Int32 = 320
  let h: Int32 = 240
  createWindow(w, h)
  for y in 0..<h {
    for x in 0..<w {
      drawPoint(x, y,
                Int32.random(in: 0...255),
                Int32.random(in: 0...255),
                Int32.random(in: 0...255))
    }
  }

  clearWindow()

  let p1: [Int32] = [10, 10]
  let p2: [Int32] = [200, 200]
  drawLine(p1[0], p1[1], p2[0], p2[1],
           Int32.random(in: 0...255),
           Int32.random(in: 0...255),
           Int32.random(in: 0...255), Int32(10))

  getKey()
}


let ctx = GraphicsContext()
ctx.registerUserMainFunc(userMainFn: main)
ctx.exec()
