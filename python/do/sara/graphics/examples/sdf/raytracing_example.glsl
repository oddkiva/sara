struct Sphere
{
  vec3 center;
  float radius;
};

vec4 hit(vec2 p, float focal, Sphere s)
{
  vec3 camera_pos = vec3(0, 0, 0);
  vec3 dist = camera_pos - s.center;

  // Calculate the ray.
  vec3 ray;
  ray.xy = p;
  ray.z = -focal;

  // The ray hits the sphere if:
  float a = dot(ray, ray);
  float b = 2 * dot(ray, dist);
  float c = dot(dist, dist) - s.radius * s.radius;
  float delta = b * b - 4 * a*c;
  return delta < 1e-6f ? vec4(0, 0, 0, 1) : vec4(1, 0, 0, 1);
}

void main() {
  // Create the scene: a single sphere.
  Sphere s;
  s.center = vec3(0, 0, -10);
  s.radius = 1;

  float focal = 2;

  // Raytracing.
  gl_FragColor = hit(v_position, focal, s);
}
