struct Image<T> {
  var data: [T] = []
  var width: Int = 0
  var height: Int = 0
  var numChannels: Int = 0
  func numElements() -> Int {
    return self.width * self.height * self.numChannels
  }
}
