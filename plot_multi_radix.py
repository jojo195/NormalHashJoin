import os
import re
import numpy as np
import matplotlib.pyplot as plt

tuple_num = 12800000
cpu_time = 20
cpu_speed = tuple_num/cpu_time
dir = "./profile"
p = re.compile('PRH_nradix([0-9]+)')
part_list = list(range(19))
probe_list = list(range(19))
speed_list = list(range(19))
plt.rcParams['figure.constrained_layout.use'] = True
for root, dirs, files in os.walk(dir):
    for filename in files:
        m = p.match(filename)
        if not m:
            continue
        nr_tasklet = int(m.groups()[0])
        f = open(os.path.join(dir, filename))
        raw_data = f.read()
        # cycles = float(re.search(r'DPU cycles\s*=\s*([0-9\.e\+]*)', raw_data).groups()[0])
        total = int(re.search(r'TOTAL: ([0-9\.]*)', raw_data).groups()[0])
        part = int(re.search(r'PART \(cycles\): ([0-9\.]*)', raw_data).groups()[0])
        speed = float(re.search(r'CYCLES-PER-TUPLE: ([0-9\.]*)', raw_data).groups()[0])
        part_list[nr_tasklet] = part
        probe_list[nr_tasklet] = total - part
        speed_list[nr_tasklet] = speed

x_data = np.array(range(19))
x_ticks = x_data

width = 0.1

plt.title('Speed go with number of number of radix bit')
plt.ylabel('Speed(Cycles per tuple)')
plt.xlabel('#radix bit (log2 of fanout)')
plt.grid(linestyle='--', alpha=0.5)
plt.xticks(x_ticks, x_data)
plt.bar(x_data, part_list)
plt.bar(x_data, probe_list, bottom=part_list)
# plt.bar(0, cpu_speed, width=0.5)
            
plt.savefig("radix bit inference.jpg")
