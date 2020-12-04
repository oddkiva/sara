import SaraSwiftCore

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
}


main()
