import numpy
import re

FILEPATH = "Summary_Text_DenseNet"

EXTENSION = "_Auswertung"


new_file = FILEPATH + EXTENSION

write_file = open(new_file, "w+")

with open(FILEPATH) as openfile:

	all_lines = openfile.read()
	all_lines = all_lines.split("\n")

	for i in range(0, len(all_lines), 5):
		lines = all_lines[i+1:i+4]

		new_arr = []

		for j in range(3):
			strings = re.findall("(?:\d\.)?\d+", lines[j])
			new_arr.append(list(map(float, strings)))
			pass

		new_arr = numpy.array(new_arr)

		sum_arr = list(numpy.sum(new_arr, 0))

		sensitivity = sum_arr[1] / (sum_arr[1] + sum_arr[2])
		specifity = sum_arr[3] / (sum_arr[3] + sum_arr[4])

		f_val = 2 * sensitivity * specifity / (sensitivity + specifity)

		line = f"{i / 5 + 1} & {sum_arr[0] / 3} & {sum_arr[3]} & {sum_arr[4]} & {sum_arr[1]} & {sum_arr[2]} & {sensitivity} & {specifity} & {f_val} \\\\"

		write_file.write(line + "\n")
		print(line)

		pass

