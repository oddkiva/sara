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
#if os(macOS)
    let imageFilepath = "/Users/david/GitLab/DO-CV/sara/data/sunflowerField.jpg"
#else
    let imageFilepath = "/home/david/GitLab/DO-CV/sara/data/sunflowerField.jpg"
#endif
    let image = imread(filePath: imageFilepath)

    let w = Int32(image.width)
    let h = Int32(image.height)
    resizeWindow(w, h)

    // Draw a bit of the image pixel by pixel.
    for y in 0..<h / 8 {
        for x in 0..<w / 8 {
            let (r, g, b) = image.pixel(Int(x), Int(y))
            var color = rgb(r, g, b)
            drawPoint(x, y, &color)
        }
    }
    getKey()

    // Draw the image.
    drawImage(image: image)
    getKey()
}

func testVideoRead() {
#if os(macOS)
    let videoFilePath = "/Users/david/Desktop/Datasets/videos/sample10.mp4"
#else
    let videoFilePath = "/home/david/Desktop/Datasets/sfm/oddkiva/bali-excursion.MP4"
#endif
    let videoStream = VideoStream(filePath: videoFilePath)
    resizeWindow(Int32(videoStream.frame.width),
                 Int32(videoStream.frame.height))
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


runGraphics {
    if (CommandLine.arguments.count > 1) {
        print("Command line arguments for '\(CommandLine.arguments[0])'")
        for arg in CommandLine.arguments[1...] {
            print("* \(arg)")
        }
    }
    main()
}
