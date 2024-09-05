"""
Definitions of data preprocessing pipelines.
"""
import torch
from monai.transforms import (
	Compose,
	CropForegroundd,
	EnsureTyped,
	EnsureChannelFirstd,
	LoadImaged,
	NormalizeIntensityd,
	Orientationd,
	RandAxisFlipd,
	RandRotated,
	RandScaleIntensityd,
	RandShiftIntensityd,
	Resized,
	ScaleIntensityd,
	Spacingd,
	SpatialPadd
)


__all__ = ['get_transformations']


def get_transformations(size):
	"""
	Get data transformation pipelines.
	Args:
		size (int): size for the input image. Final input shape will be (`size`, `size`, `size`).
	Returns:
		train_transform (monai.transforms.Compose): pipeline for the training input data.
		eval_transform (monai.transforms.Compose): pipeline for the evaluation/testing input data.
	"""
	train_transform = Compose([
		LoadImaged(keys='image'),
		EnsureChannelFirstd(keys='image'),
		EnsureChannelFirstd(keys='data', channel_dim='no_channel'),
		EnsureTyped(keys=['image', 'data'], dtype=torch.float32),
		EnsureTyped(keys=['label'], dtype=torch.long),
		Orientationd(keys='image', axcodes='RAS'),
		Spacingd(
			keys='image',
			pixdim=(1.0, 1.0, 1.0),
			mode='bilinear',
			align_corners=True,
			scale_extent=True
		),
		ScaleIntensityd(keys='image', channel_wise=True),
		CropForegroundd(
			keys='image',
			source_key='image',
			select_fn=(lambda x: x > .3),
			allow_smaller=True
		),
		Resized(
			keys='image',
			spatial_size=size,
			size_mode='longest',
			mode='bilinear',
			align_corners=True
		),
		SpatialPadd(keys='image', spatial_size=(size, size, size), mode='minimum'),
		RandAxisFlipd(keys='image', prob=0.5),
		RandRotated(
			keys='image',
			prob=0.5,
			range_x=[.4, .4],
			range_y=[.4, .4],
			range_z=[.4, .4],
			padding_mode='zeros',
			align_corners=True
		),
		NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
		RandScaleIntensityd(keys='image', factors=0.1, prob=1.0),
		RandShiftIntensityd(keys='image', offsets=0.1, prob=1.0)
	])
	eval_transform = Compose([
		LoadImaged(keys='image'),
		EnsureChannelFirstd(keys='image'),
		EnsureChannelFirstd(keys='data', channel_dim='no_channel'),
		EnsureTyped(keys=['image', 'data'], dtype=torch.float32),
		EnsureTyped(keys=['label'], dtype=torch.long),
		Orientationd(keys='image', axcodes='RAS'),
		Spacingd(
			keys='image',
			pixdim=(1.0, 1.0, 1.0),
			mode='bilinear',
			align_corners=True,
			scale_extent=True
		),
		ScaleIntensityd(keys='image', channel_wise=True),
		CropForegroundd(
			keys='image',
			source_key='image',
			select_fn=(lambda x: x > .3),
			allow_smaller=True
		),
		Resized(
			keys='image',
			spatial_size=size,
			size_mode='longest',
			mode='bilinear',
			align_corners=True
		),
		SpatialPadd(keys='image', spatial_size=(size, size, size), mode='minimum'),
		NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True)
	])
	return train_transform, eval_transform