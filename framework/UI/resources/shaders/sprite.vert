attribute vec2 coord2d;
varying vec4 f_color;
uniform float offset_x;
uniform float scale_x;
uniform float point_size;

void main(void) {
  //gl_Position = vec4((coord2d.x + offset_x) * scale_x, coord2d.y, 0, 1);
  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
  f_color = gl_Color;
  gl_PointSize = point_size;
}