import numpy


def calculateIoU(prediction, target):

	result = numpy.zeros((4, 1))


	for j in range(prediction.shape[0]):
		this_prediction = numpy.moveaxis(prediction[0, :, :, :], 0, 2)
		labels = numpy.argmax(this_prediction, axis=2)
		for i in range(4):
			this_prediction = numpy.where(labels == i, 1, 0)

			intersection = numpy.logical_and(this_prediction, target[j, i, :, :])
			union = numpy.logical_or(this_prediction, target[j, i, :, :])

			intersection = intersection.sum()
			union = union.sum()
			iou = intersection / union

			iou = numpy.nan_to_num(iou, 0)

			result[i] += iou

	return result


if __name__ == "__main__":
	prediction = numpy.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
	target = numpy.array([[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]])

	target = numpy.array([[target, target, target, target]])
	prediction = numpy.array([[prediction, prediction, prediction, prediction]])

	res = calculateIoU(prediction, target)

	print(res)
