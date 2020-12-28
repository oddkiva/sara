import Foundation

import SaraCore
import SaraGraphics


func testCBasicFunction() {
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
}

func testDrawFunctions() {
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

func testImageRead() {
    let imageFilepath = "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg"
    let image = imread(filepath: imageFilepath)

    let w = Int32(image.width)
    let h = Int32(image.height)
    resizeWindow(w, h)

    // Draw a bit of the image pixel by pixel.
    for y in 0..<h / 8 {
        for x in 0..<w / 8 {
            let (r, g, b) = image.pixel(Int(x), Int(y))
            var color = rgb(UInt8(r), UInt8(g), UInt8(b))
            drawPoint(x, y, &color)
        }
    }
    getKey()

    // Draw the image.
    drawImage(image: image)
    getKey()
}

func testVideoRead() {
    let videoFilePath = "/Users/david/Desktop/Datasets/videos/sample10.mp4"
    let videoStream = VideoStream(filepath: videoFilePath)
    resizeWindow(Int32(videoStream.frame.width), Int32(videoStream.frame.height))
    while videoStream.read() {
        drawImage(image: videoStream.frame)
    }
}

func main() {
    createWindow(300, 300)
    testCBasicFunction()
    testDrawFunctions()
    testImageRead()
    testVideoRead()
}


let ctx = GraphicsContext()
ctx.registerUserMainFunc(userMainFn: main)
ctx.exec()
