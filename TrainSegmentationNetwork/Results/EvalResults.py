import numpy as np



with open("TotalResult", "rt") as openfile:

	lines = openfile.readlines()

	data = []

	for idx in range(10):
		numbers = lines[idx*5 + 1 : idx*5 + 4]

		sample = []

		for line in numbers:
			sample.append(list(map(float, line.split())))

		mean = np.mean(sample, 0)
		data.append(mean)

		print("%f & %f & %f & %f \\\\" % (mean[0], mean[1], mean[2], mean[3]))
		pass
