import numpy as np

from vispy import app, gloo
from vispy.gloo import Program


# GOAL: render a 3D sphere rendering.

vertex = """
    attribute vec2 position;
    varying vec2 v_position;
    void main(){
        v_position = position;
        gl_Position = vec4(position, 0.0, 1.0);
    }
    """

fragment = """
    // Math calculations.
    const float focal = 1e-1;
    const float eps = 1e-4;
    const float max_steps = 500;

    // First I need a sphere.
    float sdf_sphere(vec3 p, vec3 center, float radius) {
        return length(p - center) - radius;
    }

    float sphere1(vec3 p)
    {
        vec3 c = vec3(0, 0, -3.3);
        float r = 2.9;
        return sdf_sphere(p, c, r);
    }

    float sphere2(vec3 p)
    {
        vec3 c = vec3(3, 0, -2.9);
        float r = 2;
        return sdf_sphere(p, c, r);
    }

    float smooth_transition(float k, float x1, float x2)
    {
        float c = 1 / (6 * k * k);
        return pow(max(k - abs(x1 - x2), 0), 3) * c;
    }

    float sdf_scene(vec3 p)
    {
        float s1 = sphere1(p);
        float s2 = sphere2(p);

        float s3 = sdf_sphere(p, vec3(2.2, 2.2, -2), 1);

        // Smooth transition between the two spheres.
        float k = 2;

        //return min(s3, min(s1, s2) - smooth_transition(k, s1, s2));
        return s3;
    }

    vec3 sdf_normal(vec3 p)
    {
        return normalize(
            vec3(
                sdf_scene(p + vec3(eps, 0, 0)) - sdf_scene(p - vec3(eps, 0, 0)),
                sdf_scene(p + vec3(0, eps, 0)) - sdf_scene(p - vec3(0, eps, 0)),
                sdf_scene(p + vec3(0, 0, eps)) - sdf_scene(p - vec3(0, 0, eps))
            )
        );
    }

    vec4 raymarch(vec2 uv, float focal)
    {
        vec3 p;
        p.xy = uv;
        p.z = -focal;

        // Calculate the inverse perspective matrix.

        vec3 ray = normalize(p);

        int iter;
        for (iter = 0; iter < max_steps; ++iter)
        {
            float distance = sdf_scene(p);
            if (distance < eps)
            {
                // Quick-and-dirty Phong shading: eye = light source as well.
                vec3 n = sdf_normal(p);
                float i = abs(dot(n, ray)) / (length(ray));

                return vec4(i, i, i, 1);
            }

            if (distance > 1000)
                break;

            // The SDF returns the distance to the closest point of the 0
            // level-set.
            p += distance * ray;
        }

        return vec4(0, 0, 0, 1);
    }

    varying vec2 v_position;
    uniform float radius;

    void main() {
        // Create the scene: a single sphere.

        // TODO: Projection matrix
        // Just set the focal length.
        gl_FragColor = raymarch(v_position, focal);
    }
    """


class Canvas(app.Canvas):
    def __init__(self):
        self.phi = 0

        super().__init__(size=(512, 512), title='Sphere SDF',
                         keys='interactive')

        # Build program
        self.program = Program(vertex, fragment, count=4)

        # Set uniforms and attributes
        self.program['color'] = [(1, 0, 0, 1), (0, 1, 0, 1),
                                 (0, 0, 1, 1), (1, 1, 0, 1)]
        self.program['position'] = [(-1, -1), (-1, +1),
                                    (+1, -1), (+1, +1)]
        self.program['radius'] = 0.5 + 0.25 * np.cos(np.pi * self.phi / 180.)

        gloo.set_viewport(0, 0, *self.physical_size)

        self.timer = app.Timer('auto', self.on_timer)
        self.timer.start()

        self.show()

    def on_timer(self, event):
        self.program['radius'] = 0.5 + 0.25 * np.cos(np.pi * self.phi / 180.)
        self.update()
        self.phi += 1.0

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('triangle_strip')

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

if __name__ == '__main__':
    c = Canvas()
    app.run()
