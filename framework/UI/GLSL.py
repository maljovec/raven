# PyOpenGL imports
from OpenGL.GL import *
import OpenGL.arrays.vbo as glvbo
import OpenGL.GLUT as glut
import OpenGL.GLU as glu

def compile_shader(source,shader_type):
    """Compile a shader from source."""
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if not(result):
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

def create_shader_program(shaderList):
    """Create a shader program with from compiled shaders."""
    program = glCreateProgram()
    for shader in shaderList:
      glAttachShader(program, shader)
    glLinkProgram(program)
    result = glGetProgramiv(program, GL_LINK_STATUS)
    if not(result):
        raise RuntimeError(glGetProgramInfoLog(program))
    return program