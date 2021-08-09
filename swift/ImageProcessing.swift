import SaraImageProcessing


func rgb8ToGray32f(src: ImageView<UInt8>, dst: ImageView<Float32>) {
  shakti_rgb8_to_gray32f_cpu(src.dataPtr?.baseAddress, dst.dataPtr?.baseAddress,
                             Int32(src.width), Int32(src.height))
}

func gray32fToRgb8(src: ImageView<Float32>, dst: ImageView<UInt8>) {
  shakti_gray32f_to_rgb8_cpu(src.dataPtr?.baseAddress, dst.dataPtr?.baseAddress,
                             Int32(src.width), Int32(src.height))
}
