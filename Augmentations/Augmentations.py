"""
By Martin Leipert
martin.leipert@fau.de
15.06.2019

Augmentation sets for the calculations

-> As defined for the evaluation
"""

from albumentations import (
	HorizontalFlip, IAAPerspective, CLAHE, RandomRotate90,
	Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
	IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
	IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, ElasticTransform, RGBShift, ChannelShuffle,
	ToGray, RandomShadow, GaussianBlur, RandomSnow, JpegCompression
)


def weak_augmentation():
	# Few documents are flipped or rotated
	flip = OneOf([
		Flip(p=0.1),
		HorizontalFlip(p=0.1),
		# Rotated around 90 deg sometimes happens
		RandomRotate90(p=0.8)],
		p=0.1)

	# Light rotation and shifts on the scanner occur,
	# As the documents are of different size, scale is sth. which frequently occurs in the set
	affine = ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=5, p=0.7)

	# Few distorsion occurs due to not perfectly planar documents
	distortion = OneOf([
		OpticalDistortion(distort_limit=0.025, shift_limit=0.025, p=0.5),
		GridDistortion(p=0.5)
		], p=0.15)

	# Sometimes shadows occur in the image due to not perfect plainness
	effects = RandomShadow(p=0.25)

	# Contrast and lighting changes are frequent
	brightness_contrast = RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.2, p=0.3)

	# HSV varies a little over the documents
	color = OneOf([
		HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5),
		RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5)],
		p=0.4)

	augmentation = Compose([flip, affine, distortion, brightness_contrast, distortion, color, effects], p=1)

	return augmentation


# Moderate
# augmentation -> not only dataset variability but also additional
def moderate_augmentation():

	# Flipping -> To prevent searching always in the same image region for a seal
	# -> Improves generalization
	# -> However not that relevant augmentation
	flip = OneOf([
		Flip(p=0.3),
		HorizontalFlip(p=0.3),
		Transpose(p=0.4)
	],
	p=0.2)

	# Generalization -> But large rotations or shiftt's dont make sense as documents are scanned
	affine = ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=7, p=0.7)

	noise_blur = OneOf([
			# Gaussian Noise -> Robustness
			GaussNoise(var_limit=(10., 50.), p=0.2),
			# Blur -> Robustness
			MotionBlur(blur_limit=5, p=0.2),
			MedianBlur(blur_limit=5, p=0.1),
			Blur(blur_limit=7, p=0.1),
			GaussianBlur(blur_limit=7, p=0.4)
		], p=0.5)

	distortion = OneOf([
		OpticalDistortion(distort_limit=0.04, shift_limit=0.04, p=0.2),
		GridDistortion(p=0.2),
		ElasticTransform(p=0.6)
	], p=0.3)

	# Sometimes shadows occur in the image due to not perfect plainness
	effects = RandomShadow(p=0.25)

	brightness_contrast = RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3)

	color = OneOf([
		# Histogram equalization
		CLAHE(clip_limit=3, p=0.1),
		HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.4),
		RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.4),
		ToGray(p=0.1)
	])

	augmentation = Compose([flip, affine, noise_blur, distortion, effects, brightness_contrast, color], p=1)

	return augmentation


# Moderate
# augmentation -> not only dataset variability but also additional
def heavy_augmentation():

	# Flipping -> To prevent searching always in the same image region for a seal
	# -> Improves generalization
	# -> However not that relevant augmentation
	flip = OneOf([
		Flip(p=0.3),
		HorizontalFlip(p=0.3),
		Transpose(p=0.4)
	],
	p=0.2)

	# Generalization -> But large rotations or shiftt's dont make sense as documents are scanned
	affine = ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=7, p=0.7)

	noise_blur = OneOf([
			# Gaussian Noise -> Robustness
			GaussNoise(var_limit=(10., 70.), p=0.2),
			# Blur -> Robustness
			MotionBlur(blur_limit=10, p=0.2),
			MedianBlur(blur_limit=10, p=0.1),
			Blur(blur_limit=20, p=0.1),
			GaussianBlur(blur_limit=20, p=0.4)
		], p=0.5)

	distortion = OneOf([
		OpticalDistortion(distort_limit=0.06, shift_limit=0.06, p=0.2),
		GridDistortion(p=0.2),
		ElasticTransform(p=0.6)
	], p=0.6)

	# Sometimes shadows occur in the image due to not perfect plainness
	effects = OneOf([
		RandomShadow(p=0.3),
		JpegCompression(p=0.3),
		# CoarseDropout(p=0.3),
		RandomSnow(p=0.4)
	], p=0.6)

	brightness_contrast = RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4)

	color = OneOf([
		# Histogram equalization
		CLAHE(clip_limit=5, p=0.1),
		HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
		RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
		ToGray(p=0.1)
	])

	augmentation = Compose([flip, affine, noise_blur, distortion, effects, brightness_contrast, color], p=1)

	return augmentation
