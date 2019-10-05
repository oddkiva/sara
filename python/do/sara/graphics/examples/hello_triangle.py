import sys

from PySide2 import QtGui


class TriangleWindow(QtGui.QOpenGLWindow):

    def __init__(self):
        super(TriangleWindow, self).__init__()

    def setup_shader_program(self):
        self._program = QtGui.QOpenGLShaderProgram(self.context())
        self._vertex_shader_source = """
#version 330 core
  layout (location = 0) in vec3 in_coords;
  layout (location = 1) in vec3 in_color;

  out vec3 out_color;

  void main()
  {
    gl_Position = vec4(in_coords, 1.0);
    gl_PointSize = 200.0;
    out_color = in_color;
  }
        """

        self._fragment_shader_source = """
#version 330 core
  in vec3 out_color;
  out vec4 frag_color;

  void main()
  {
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;

    float dist = length(gl_PointCoord - vec2(0.5));

    if (dot(circCoord, circCoord) > 1.0)
        discard;
    float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
    frag_color = vec4(out_color, alpha);
  }
        """

        self._arg_pos = {'in_coords': 0, 'in_color': 1, 'out_color': 0}

        self._program.addCacheableShaderFromSourceCode(
            QtGui.QOpenGLShader.Vertex, self._vertex_shader_source)
        self._program.addCacheableShaderFromSourceCode(
            QtGui.QOpenGLShader.Fragment, self._vertex_shader_source)


    def initializeGL(self):
        super(TriangleWindow, self).initializeGL()
        self.setup_shader_program()



if __name__ == '__main__':
    app = QtGui.QGuiApplication(sys.argv)

    win = TriangleWindow()
    win.show()

    sys.exit(app.exec_())
