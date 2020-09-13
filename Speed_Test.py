import numpy as np
import time

times = []
for n in range(10):
    x = np.random.randn(1000, 1000)
    y = np.random.randn(1000, 1000)

    t0 = time.time()
    for i in range(10):
        z = x @ y
    t1 = time.time()

    print(t1-t0)
    times.append(t1-t0)

print("")
print(np.average(times))
