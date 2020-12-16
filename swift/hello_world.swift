import Foundation

import SaraCore
import SaraGraphics


typealias UserMainFunc = () -> Void


public class GraphicsApplication {
  fileprivate var _cppObject: UnsafeMutableRawPointer
  fileprivate var _argc: CInt = 0
  fileprivate var _argv: [CChar] = []

  init() {
    print("[GraphicsApplication] Init...")
    self._cppObject = initialize_graphics_application()
  }

  func registerUserMainFunc(userMainFn: @escaping UserMainFunc) {
    // let mainp: @convention(c) () -> Void = userMainFn
    // register_user_main(app._cppObject, mainp)
  }

  func exec() {
    exec_graphics_application(self._cppObject)
  }

  deinit {
    print("[GraphicsApplication] Deinit...")
    deinitialize_graphics_application(self._cppObject)
  }

}

func main() {
  let message = "Hello World!"
  tic()
  print("\(message)")
  toc("\(message)")

  var numbers: [Int32] = [1, 2, 3, 5]
  numbers.withUnsafeMutableBufferPointer {
    (numbers) in
    let ptr = UnsafeMutableRawPointer(numbers.baseAddress!).bindMemory(
      to: CInt.self,
      capacity: numbers.count)
    square(ptr, CInt(numbers.count))
  }

  print("\(numbers)")

  var i = 0
  while i < 20 {
    print("\(i)")
    usleep(10*1000)
    i += 1
  }
}


let app = GraphicsApplication()
let mainp: @convention(c) () -> Void = main
register_user_main(app._cppObject, mainp)
app.exec()
