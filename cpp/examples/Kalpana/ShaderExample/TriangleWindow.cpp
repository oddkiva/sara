#include "TriangleWindow.hpp"

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QScreen>

#include <string>


static const std::string vertexShaderSource = R"""(
attribute highp vec4 posAttr;
attribute lowp vec4 colAttr;
varying lowp vec4 col;
uniform highp mat4 matrix;

void main()
{
  col = colAttr;
  gl_Position = matrix * posAttr;
}
)""";

static const std::string fragmentShaderSource = R"""(
varying lowp vec4 col;

void main()
{
  gl_FragColor = col;
}
)""";


TriangleWindow::TriangleWindow()
  : m_program(nullptr)
  , m_frame(0)
{
}

void TriangleWindow::initialize()
{
  m_program = new QOpenGLShaderProgram{this};
  m_program->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                     vertexShaderSource.c_str());
  m_program->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                     fragmentShaderSource.c_str());
  m_program->link();

  m_posAttr = m_program->attributeLocation("posAttr");
  m_colAttr = m_program->attributeLocation("colAttr");
  m_matrixUniform = m_program->uniformLocation("matrix");
}

void TriangleWindow::render()
{
  const qreal retinaScale = devicePixelRatio();
  glViewport(0, 0, width() * retinaScale, height() * retinaScale);

  glClear(GL_COLOR_BUFFER_BIT);

  m_program->bind();

  QMatrix4x4 matrix;
  matrix.perspective(60.0f, 4.0f / 3.0f, 0.1f, 100.0f);
  matrix.translate(0, 0, -2);
  matrix.rotate(100.0f * m_frame / screen()->refreshRate(), 0, 1, 0);

  m_program->setUniformValue(m_matrixUniform, matrix);

  GLfloat vertices[] = {
    0.0f, 0.707f,
    -0.5f, -0.5f,
    0.5f, -0.5f
  };

  GLfloat colors[] = {
    1.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f
  };

  glVertexAttribPointer(m_posAttr, 2, GL_FLOAT, GL_FALSE, 0, vertices);
  glVertexAttribPointer(m_colAttr, 3, GL_FLOAT, GL_FALSE, 0, colors);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  glDrawArrays(GL_TRIANGLES, 0, 3);

  glDisableVertexAttribArray(0);
  glDisableVertexAttribArray(1);

  m_program->release();

  ++m_frame;
}
