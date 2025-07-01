"""
Created on Tue Jul  1 10:55:36 2025

@author: Matéo Petitet for OFB/Parc Naturel Marin de Martinique
"""
# -*- coding: utf-8 -*-
import cv2
import albumentations as A
import matplotlib.pyplot as plt


def apply_transformations(combinaison, picture):
    for i, list_transform in enumerate(combinaison):
        augmented_image = picture.copy()  #copie pour conserver l'originale
        for transform in list_transform:
            augmented_image = transform(image=augmented_image)["image"]
        
        #une nouvelle figure pour chaque résultat
        plt.figure(i, frameon=False)  # identifiant unique par index + pas de bordure
        plt.imshow(augmented_image)
        plt.axis('off')

    
    plt.show()  # Afficher toutes les figures ensemble


img = cv2.imread("/home/mateo/Travail/Data_Augmentation/Easy_data_augmentation/atlas_transformations/images/OG.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hauteur, largeur = img.shape[:2]

Rotations_90_flip = [[A.Transpose(p=1.0), A.HorizontalFlip(p=1.0)],
                     [A.Transpose(p=1.0), A.VerticalFlip(p=1.0)],
                     [A.HorizontalFlip(p=1.0)],
                     [A.Transpose(p=1.0), A.Rotate(limit=[180,180], p=1.0)],
                     [A.Transpose(p=1.0)]]

#apply_transformations(Rotations_90_flip, img)

Crop_Move = [[A.CenterCrop(height=int(hauteur*0.75), width=int(largeur*0.75), p=1.0)],
        [A.RandomCrop(height=int(hauteur*0.75), width=int(largeur*0.75), p=1.0)],
        [A.Affine(translate_percent=(-0.25, -0.25), p=1.0)],
        [A.Affine(translate_percent=(0.25, 0.25), p=1.0)]]

#apply_transformations(Crop_Move, img)

Rotation=[[A.Rotate(limit=45, p=1.0)]]

#apply_transformations(Rotation, img)

Transformations=[[A.ChromaticAberration(primary_distortion_limit=0.9, secondary_distortion_limit=0.9, mode='random', interpolation=cv2.INTER_LINEAR, p=1.0)],
                 [A.Downscale(scale_range=(0.1, 0.1), interpolation_pair={'downscale': cv2.INTER_NEAREST, 'upscale': cv2.INTER_LINEAR}, p=1.0)],
                 [A.Emboss(alpha=(0.8, 0.9), strength=(1.1, 1.1), p=1.0)],
                 [A.ImageCompression(quality_range=(5, 5), compression_type='jpeg', p=1.0)],
                 [A.ImageCompression(quality_range=(1, 1), compression_type='webp', p=1.0)],
                 [A.Sharpen(alpha=(0.4, 0.5), lightness=(1.0, 1.0), method='kernel', p=1.0)],
                 [A.ElasticTransform(alpha=40, sigma=40, p=1.0)],
                 [A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0)],
                 [A.GridElasticDeform(num_grid_xy=(4, 4), magnitude=2, p=1.0)],
                 [A.LongestMaxSize(max_size=int(max(hauteur,largeur)*0.5))],
                 [A.LongestMaxSize(max_size=int(max(hauteur,largeur)*2))],
                 [A.Morphological(scale=(3, 3), operation='erosion', p=1.0)],
                 [A.Morphological(scale=(3, 3), operation='dilation', p=1.0)],
                 [A.OpticalDistortion(distort_limit=0.3, mode='fisheye', p=1.0)],
                 [A.OpticalDistortion(distort_limit=0.3, mode='camera', p=1.0)],
                 [A.Perspective(scale=(0.2, 0.2), keep_size=True, p=1.0)],
                 [A.ThinPlateSpline(scale_range=(0.2, 0.2), num_control_points=3, p=1.0)]]

#apply_transformations(Transformations, img)

Freres_couleur_Couleur_Lumiere = [[A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=1.0)],
                   [A.Illumination(mode='linear', intensity_range=(0.2, 0.2), effect_type='darken', angle_range=(0, 180), p=1.0)],
                   [A.PlanckianJitter(mode="blackbody", temperature_limit=(6000,6000), sampling_method="uniform", p=1.0)],
                   [A.RandomBrightnessContrast(p=1.0)],
                   [A.ColorJitter(brightness=(0.9, 1.1), contrast=(0.9, 1.1), saturation=(0.9, 1.1), hue=(-0.05, 0.05), p=1.0)],
                   [A.RGBShift(r_shift_limit=30, g_shift_limit=30,b_shift_limit=30, p=1.0)],
                   [A.Posterize(num_bits=(3,3), p=1.0)]]

#apply_transformations(Freres_couleur_Couleur_Lumiere, img)

Flou = [[A.Defocus(radius=(4, 6), alias_blur=(0.4, 0.4), p=1.0)],
         [A.ZoomBlur(max_factor=1.1, p=1)],
         [A.MedianBlur(blur_limit=(6, 6), p=1.0)],
         [A.MotionBlur(blur_limit=(5,21), p=1.0)],
         [A.GlassBlur(sigma=0.7, max_delta=1, iterations=2, mode="fast", p=1)]]

#apply_transformations(Flou, img)

Bruits_Particules_Objets = [[A.GaussNoise(std_range=(0.1, 0.1), p=1.0)],
                            [A.SaltAndPepper(amount=(0.05, 0.1), p=1.0)],
                            [A.ShotNoise(scale_range=(0.05, 0.1), p=1.0)],
                            [A.PixelDropout(dropout_prob=0.075, per_channel=True, p=1.0)],
                            [A.RandomFog(fog_coef_range=(0.2, 0.5), alpha_coef=0.1, p=1.0)],
                            [A.RandomGravel(gravel_roi=(0.2, 0.2, 0.8, 0.8), number_of_patches=5, p=1.0)],
                            [A.RandomRain(slant_range=(-15, 15), drop_length=30, drop_width=2, drop_color=(180, 180, 180), blur_value=5, brightness_coefficient=0.8, p=1.0)],
                            [A.RandomShadow(shadow_roi=(0.01, 0.01, 0.99, 0.99), num_shadows_limit=(1, 4), shadow_dimension=5, shadow_intensity_range=(0.1, 0.6), p=1.0)],
                            [A.RandomSnow(snow_point_range=(0.01, 0.4), brightness_coeff=2.0, method="texture", p=1.0)],
                            [A.RandomSunFlare(flare_roi=(0.1, 0, 0.9, 0.3), angle_range=(0.25, 0.75), num_flare_circles_range=(5, 15), src_radius=200, src_color=(255, 200, 100), method="physics_based", p=1.0)],
                            [A.PlasmaShadow(shadow_intensity_range=(0.5, 0.9), roughness=0.3, p=1.0)],
                            [A.CoarseDropout(num_holes_range=(3, 6), hole_height_range=(0.05, 0.1), hole_width_range=(0.05, 0.1), p=1.0)],
                            [A.GridDropout(ratio=0.2, p=1.0)],
                            [A.Erasing(scale=(0.2, 0.5), ratio=(0.5, 2.0), p=1.0)]]

apply_transformations(Bruits_Particules_Objets, img)