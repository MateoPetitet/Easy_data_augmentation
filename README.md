# Easy Data Augmentation for fish pictures

This script aims to automaticaly do data augmentation on fish pictures using the [albumentations python library](https://albumentations.ai/). This is useful to train AI on fish recognition when you don't have a lot of data. The transformations and their ranges have been choosen to best reflect underwater conditions.

## Data structure
Data is expected to follow this structure : 
```bash
├─── data_folder
│   ├─── first_folder
│   │   ├─── file_1_1.png
│   │   ├─── file_1_2.png
│   │   ├─── file_1_3.png
│   ├─── second_folder
│   │   ├─── file_2_1.png
│   │   ├─── file_2_2.png
│   │   ├─── file_2_3.png
│   ├─── third_folder
│   │   ├─── file_3_1.png
│   │   ├─── file_3_2.png
│   │   ├─── file_3_3.png
```
The script will create one folder per augmented picture.

## Parameters Explained
All optional parameters are controlled by 0 or 1 except for **- -nb**. They all default to 1, except for **- -test**.
 All the transformations happens in the same order. It is the most logical for our use case in marine biology.

**- -path** Path to the dataset directory
**- -nb (optional)** The number of new pictures to generate per picture. If not precised, all possible pictures will be generated.
**- -flips (optional)** Add 90° flips rotations and mirrors to the transformations, except the one that would put the fish on its back (assuming it is in the correct orientation in the picture). Uses : 
> - Transpose(p=1.0), A.HorizontalFlip(p=1.0)
> - Transpose(p=1.0), VerticalFlip(p=1.0)
> - HorizontalFlip(p=1.0)
> - Transpose(p=1.0), Rotate(limit=[180,180], p=1.0)
> - Transpose(p=1.0)

 **- -crops_move (optional)** Add center and random crop to the transformations, as well as a translation. Uses : 
> - CenterCrop(height=int(height\*0.75), width=int(width\*0.75), p=1.0)
> - RandomCrop(height=int(height\*0.75), width=int(width\*0.75), p=1.0)
> - Affine(translate_percent=(-0.25, 0.25), p=1.0)

 **- -rotate (optional)** Add random rotation between -45° and 45° to the transformations. Uses : 
> - Rotate(limit=45, p=1.0)

 **- -lights_colors (optional)** Add random color and light changes to the transformations. Uses : 
> - HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1.0)
>  - Illumination(mode='linear', intensity_range=(0.1, 0.2), effect_type='darken', angle_range=(0, 180), p=1.0) PlanckianJitter(mode="blackbody", temperature_limit=(3000,6000), sampling_method="uniform", p=1.0)
> - RandomBrightnessContrast(p=1.0) ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.05, 0.05), p=1.0)
> - RGBShift(r_shift_limit=30, g_shift_limit=30,b_shift_limit=30, p=1.0)
> - Posterize(num_bits=(3,5), p=1.0)

 **- -blurs (optional)** Add random blurs to the transformations. Uses : 
> - Defocus(radius=(4, 6), alias_blur=(0.2, 0.4), p=1.0)
> - ZoomBlur(max_factor=1.1, p=1) MedianBlur(blur_limit=(3, 7), p=1.0)
> - MotionBlur(blur_limit=(5,21), p=1.0) GlassBlur(sigma=0.7, max_delta=1, iterations=2, mode="fast", p=1)

 **- -noises_objects (optional)** Add random noises, particles or objects to the transformation. Uses : 
> - GaussNoise(std_range=(0.05, 0.1), p=1.0) SaltAndPepper(amount=(0.05, 0.1), p=1.0) ShotNoise(scale_range=(0.05, 0.1), p=1.0) PixelDropout(dropout_prob=0.075, per_channel=True, p=1.0)
> - RandomFog(fog_coef_range=(0.2, 0.5), alpha_coef=0.1, p=1.0)
> - RandomGravel(gravel_roi=(0.2, 0.2, 0.8, 0.8), number_of_patches=5, p=1.0) 
> - RandomRain(slant_range=(-15, 15), drop_length=30, drop_width=2, drop_color=(180, 180, 180), blur_value=5, brightness_coefficient=0.8, p=1.0) 
> - RandomShadow(shadow_roi=(0.01, 0.01, 0.99, 0.99), num_shadows_limit=(1, 4), shadow_dimension=5, shadow_intensity_range=(0.1, 0.6), p=1.0)
> - RandomSnow(snow_point_range=(0.01, 0.4), brightness_coeff=2.0, method="texture", p=1.0) 
> - RandomSunFlare(flare_roi=(0.1, 0, 0.9, 0.3), angle_range=(0.25, 0.75), num_flare_circles_range=(5, 15), src_radius=200, src_color=(255, 200, 100), method="physics_based", p=1.0) 
> - PlasmaShadow(shadow_intensity_range=(0.5, 0.9), roughness=0.3, p=1.0) 
> - CoarseDropout(num_holes_range=(3, 6), hole_height_range=(0.05, 0.1), hole_width_range=(0.05, 0.1), p=1.0) 
> - GridDropout(ratio=0.2, p=1.0) Erasing(scale=(0.2, 0.5), ratio=(0.5, 2.0), p=1.0)

 **- -transforms (optional)** Add miscellaneous transformations to the transformations. Uses : 
> - ChromaticAberration(primary_distortion_limit=0.9, secondary_distortion_limit=0.9, mode='random', interpolation=cv2.INTER_LINEAR, p=1.0) 
> - Downscale(scale_range=(0.1, 0.9), interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR}, p=1.0) 
> - Emboss(alpha=(0.8, 0.9), strength=(0.8, 1.1), p=1.0) 
> - ImageCompression(quality_range=(5, 30), compression_type='jpeg', p=1.0) 
> - ImageCompression(quality_range=(1, 20), compression_type='webp', p=1.0) 
> - Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), method='kernel', p=1.0)
> - ElasticTransform(alpha=40, sigma=40, p=1.0)
> - GridDistortion(num_steps=5, distort_limit=0.1, p=1.0)
> - GridElasticDeform(num_grid_xy=(4, 4), magnitude=2, p=1.0)
> - LongestMaxSize(max_size=int(max(height,width)\*0.5))
> - LongestMaxSize(max_size=int(max(height,width)\*2))
> - Morphological(scale=(2, 3), operation='erosion', p=1.0)
> - Morphological(scale=(2, 3), operation='dilation', p=1.0)
> - OpticalDistortion(distort_limit=0.3, mode='fisheye', p=1.0)
> - OpticalDistortion(distort_limit=0.3, mode='camera', p=1.0)
> - Perspective(scale=(0.05, 0.1), keep_size=True, p=1.0)
> - ThinPlateSpline(scale_range=(0.1, 0.2), num_control_points=3, p=1.0)

 **- -no_transfo (optional)** Add, for each category of transformation, the possibility of skipping it entirely. Uses :
> - NoOp(p=1.0)

 **- -test_mode (optional)** Computes the total number of new images that can be created with current settings, without actually creating any.
 
 
## How to Use
Just run :
```bash
python easy_data_augmentation.py --path "C:/Data_folder" --nb 1000 --noises_objects 0 --no_transfo 0
```
And change the parameters you want.
