# Plot the results of the segmentations
# And also do it as overlays!

import numpy as np
from matplotlib import pyplot as plt
import os
import torch

CLASS_LABELS = {
	0: "Background",
	1: "Text",
	2: "Image Region",
	3: "Graphic Region"
}


def plot_result(segmented_data_set, images, model_name, file_names, class_labels=CLASS_LABELS, store=True, show=False):

	if store:
		result_dir = f"Results/{model_name}_result"
		if not os.path.exists(result_dir):
			os.mkdir(result_dir)

	numpy_set = torch.sigmoid(segmented_data_set).cpu().detach().numpy()
	sum_set = np.sum(numpy_set, axis=1)

	for i in range(sum_set.shape[0]):
		numpy_set[i, :, :, :] = np.divide(numpy_set[i, :, :, :], sum_set[i, :, :])
		pass

	dimensions = np.shape(numpy_set)

	# Get the images from the batch
	for im_idx in range(dimensions[0]):
		file_name = file_names[im_idx]
		base_name = os.path.basename(file_name)

		segmented_image = numpy_set[im_idx, :, :, :]
		combination_image = np.zeros(list(np.shape(segmented_image)[1:3]) + [3])

		org_image = images[im_idx, :, :, :].cpu().detach().numpy()
		org_image = to_gray_scale(org_image)

		# Create a figure
		fig = plt.figure(figsize=(8, 9), dpi=200)
		fig.suptitle(f"{model_name} Segmentation")

		# Plot the segmented classes
		for cls_idx, cls_label in class_labels.items():

			cls_data = segmented_image[cls_idx, :, :]

			if cls_idx < 2:
				combination_image[:, :, cls_idx] = cls_data
			elif cls_idx == 3:
				combination_image[:, :, cls_idx-1] = cls_data

			# Add at subsequent positions
			ax = fig.add_subplot(221 + cls_idx)
			ax.imshow(cls_data, vmin=0, vmax=1)
			ax.set_title(cls_label)

		if show:
			fig.show()
		if store:
			fig.tight_layout()
			fig.savefig(f"{result_dir}/{base_name}_segmentation.png")
		fig.clf()
		plt.close(fig)

		# Create another figure
		fig = plt.figure(figsize=(7, 8), dpi=200)
		ax = fig.add_subplot(111)
		ax.set_title(f"{model_name} Segmentation")

		ax.imshow(org_image, 'gray', interpolation='bicubic')
		ax.imshow(combination_image, 'jet', interpolation='bicubic', vmin=0, vmax=3, alpha=0.5)

		if show:
			fig.show()
		if store:
			fig.tight_layout()
			fig.savefig(f"{result_dir}/{base_name}_segmentation_overlay.png")
		fig.clf()
		plt.close(fig)


def to_gray_scale(im_array):
	r = 0.2125 * im_array[0, :, :]
	g = 0.7154 * im_array[1, :, :]
	b = 0.0721 * im_array[2, :, :]

	return r + g + b
