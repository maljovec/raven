def run(self, Input):
  x = 0
  for i in range(1,100000):
    for j in range(424):
      sgn = 1.
      if j % 2:
        sgn = -1.

      x += sgn/(float(i+j))

  self.Bout = self.Bin + x
