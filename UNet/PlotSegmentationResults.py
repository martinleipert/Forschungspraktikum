# And also do it as overlays!

import numpy as np
from matplotlib import pyplot as plt
import torch
import os

CLASS_LABELS = {
	0: "Background",
	1: "Text",
	2: "Image Region",
	3: "Graphic Region"
}


def plot_result(segmented_data_set, model_name, file_names, class_labels=CLASS_LABELS):
	result_dir = f"{model_name}_results"
	if not os.path.exists(result_dir):
		os.mkdir(result_dir)

	sum_set = torch.sum(torch.sigmoid(segmented_dataset), 1)

	numpy_set = torch.sigmoid(segmented_data_set)

	for i in range(sum_set.size(0)):
		numpy_set[i, :, :, :] = np.divide(numpy_set[i, :, :, :] / sum_set[i, :, :])

	segmented_data_set = segmented_data_set.cpu().detach().numpy()

	dimensions = np.shape(segmented_data_set)

	# Get the images from the batch
	for im_idx in range(dimensions[0]):
		file_name = file_names[im_idx]

		base_name = os.path.basename(file_name)

		segmented_image = segmented_data_set[im_idx, :, :, :]

		# Create a figure
		fig = plt.figure(figsize=(15, 7), dpi=200)
		fig.set_title(f"{model_name} Segmentation")

		# Plot the segmented classes
		for cls_idx, cls_label in class_labels:

			cls_data = segmented_image[cls_idx, :, :]

			# Add at subsequent positions
			ax = fig.add_sublot(220 + cls_idx)
			ax.imshow(cls_data, vmin=0, vmax=1)
			ax.set_title(cls_label)

		fig.savefig(f"Results/{result_dir}/{base_name}_segmentation.png")
