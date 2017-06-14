#version 120

varying vec4 f_color;

void main(void) {
  //if ((gl_PointCoord.s-.5)*(gl_PointCoord.s-.5)+(gl_PointCoord.t-.5)*(gl_PointCoord.t-.5) <= 1.)
  float alpha = clamp(1.-2.*distance(gl_PointCoord,vec2(0.5,0.5)),0.,1);
  //Soft cutoff and blend with whatever was already there
  //gl_FragColor = vec4(vec3(f_color),f_color.a*alpha);
  //gl_FragColor = f_color;
  //Hard cutoff
  gl_FragColor = vec4(vec3(f_color),f_color.a*ceil(alpha));
  if (ceil(alpha) == 0) { discard; }
}