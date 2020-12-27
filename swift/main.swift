import Foundation

import SaraCore
import SaraGraphics


func main() {
  var numbers: [Int32] = [1, 2, 3, 5]
  print("Before squaring: \(numbers)")

  tic()
  numbers.withUnsafeMutableBufferPointer {
    (numbers) in
    let ptr = UnsafeMutableRawPointer(numbers.baseAddress!).bindMemory(
      to: CInt.self,
      capacity: numbers.count)
    square(ptr, CInt(numbers.count))
  }
  toc("Squaring Operation")

  print("After squaring: \(numbers)")

  usleep(1000)

  let image_filepath = "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg"
  let image = imread(filepath: image_filepath)

  let w: Int32 = Int32(image.width)
  let h: Int32 = Int32(image.height)
  createWindow(w, h)

  for y in 0..<Int32(10) {
    for x in 0..<Int32(10) {
      var color = rgb(UInt8.random(in: 0...UInt8.max),
                      UInt8.random(in: 0...UInt8.max),
                      UInt8.random(in: 0...UInt8.max))
      drawPoint(x, y, &color)
    }
  }
  getKey()

  drawImage(image: image)
  getKey()

  clearWindow()
  typealias Point = (x: Int32, y: Int32)
  let p1: Point = (10, 10)
  let p2: Point = (200, 200)
  var color = rgb(UInt8.random(in: 0...UInt8.max),
                  UInt8.random(in: 0...UInt8.max),
                  UInt8.random(in: 0...UInt8.max))
  let penWidth: Int32 = 10
  drawLine(p1.x, p1.y, p2.x, p2.y, &color, penWidth)

  getKey()
}


let ctx = GraphicsContext()
ctx.registerUserMainFunc(userMainFn: main)
ctx.exec()
