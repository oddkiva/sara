import SaraGraphics

struct Image<T> {
  var data: [T] = []
  var width: Int = 0
  var height: Int = 0
  var numChannels: Int = 0

  func pixel(_ x: Int, _ y: Int) -> (T, T, T) {
    let index = (y * width + x) * numChannels
    let r = self.data[index + 0]
    let g = self.data[index + 1]
    let b = self.data[index + 2]
    return (r, g, b)
  }

  func numElements() -> Int {
    return self.width * self.height * self.numChannels
  }
}
