import Foundation

import SaraCore
import SaraGraphics


// Swift wrapper.
public class GraphicsApplication {
  fileprivate var _cppObject: UnsafeMutableRawPointer
  fileprivate var _argc: CInt = 0
  fileprivate var _argv: [CChar] = []

  init() {
    self._cppObject = GraphicsApplication_initialize()
  }

  func registerUserMainFunc(userMainFn: @escaping (@convention(c) () -> Void)) {
    // The swift function will be called from C++.
    GraphicsApplication_registerUserMainFunc(app._cppObject, userMainFn)
  }

  func exec() {
    GraphicsApplication_exec(self._cppObject)
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

  var i = 0
  while i < 20 {
    print("\(i)")
    usleep(10*1000)
    i += 1
  }

  let w = createWindow(300, 300)
  for y in 0..<300 {
    for x in 0..<300 {
      drawPoint(Int32(x), Int32(y),
                Int32.random(in: 0...255),
                Int32.random(in: 0...255),
                Int32.random(in: 0...255))
    }
  }

  getKey();
  closeWindow(w);
}


let app = GraphicsApplication()
app.registerUserMainFunc(userMainFn: main)
app.exec()
