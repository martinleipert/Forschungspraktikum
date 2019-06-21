"""
By Martin Leipert
martin.leipert@fau.de
15.06.2019

Augmentation sets for the calculations
"""

from albumentations import (
	HorizontalFlip, IAAPerspective, CLAHE, RandomRotate90,
	Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
	IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
	IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, ElasticTransform, RGBShift, ChannelShuffle,
	ToGray, RandomShadow
)


def weak_augmentation():

	affine = OneOf([
		# Flipping -> To prevent searching always in the same image region for a seal
		# -> Improves generalization
		Compose([
			Flip(p=0.7),
			HorizontalFlip(p=0.7)],
		p=0.3),
		# Generalization -> But large rotations or shiftt's dont make sense as documents are scanned
		ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1, rotate_limit=5, p=0.7),
			], p=0.5)

	noise_blur = OneOf([
			# Gaussian Noise -> Robustness
			GaussNoise(p=0.2),
			# Blur -> Robustness
			MotionBlur(p=0.2),
			MedianBlur(blur_limit=3, p=0.1),
			Blur(blur_limit=3, p=0.1),
		], p=0.5)

	distortion = OneOf([
		OpticalDistortion(p=0.2),
		GridDistortion(p=0.2),
		ElasticTransform(p=0.6)
	], p=0.2)

	color = OneOf([
		# Histogram equalization
		CLAHE(clip_limit=2),
		RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.),
		HueSaturationValue(p=0.3),
		RGBShift,
		ToGray(p=0.5)
	], p=0.7),

	augmentation = Compose([affine, noise_blur, distortion, color],p=0.9)

	return augmentation
