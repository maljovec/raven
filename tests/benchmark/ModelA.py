def run(self, Input):
  x = 0
  for i in range(1,100000):
    for j in range(424):
      x += 1./(float(i+j))

  self.Bin = x
