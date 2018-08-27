"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'horse2zebra_train': 1000,
    'horse2zebra_test': 90
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'horse2zebra_train': '.png',
    'horse2zebra_test': '.png',
}

"""The path to the output csv file."""
PATH_TO_CSV = {
    'horse2zebra_train': './input/horse2zebra/horse2zebra_train.csv',
    'horse2zebra_test': './input/horse2zebra/horse2zebra_test.csv',
}
