__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE |
                               CLK_FILTER_NEAREST;

__kernel void gradient(read_only image2d_t f,
                       write_only image2d_t abs_nabla_f)
{
  int2 coord = (int2) (get_global_id(0), get_global_id(1));

  // x-gradient.
  float4 f_x = read_imagef(f, sampler, coord + (int2)(1, 0))
             - read_imagef(f, sampler, coord - (int2)(1, 0));
  // y-gradient.
  float4 f_y = read_imagef(f, sampler, coord + (int2)(0, 1))
             - read_imagef(f, sampler, coord - (int2)(0, 1));

  // Result.
  float res = f_x.x;
  float4 color = (float4) (res, res, res, res);
  write_imagef(abs_nabla_f, coord, color);
}