_FACTORS_IN_ORDER = [
    'floor_hue',
    'wall_hue',
    'object_hue',
    'scale',
    'shape',
    'orientation'
]

_NUM_VALUES_PER_FACTOR = {
    'floor_hue': 10,
    'wall_hue': 10,
    'object_hue': 10,
    'scale': 8,
    'shape': 4,
    'orientation': 15
}

FACTORS = [
    'floor', 'wall', 'object', 'scale', 'shape', 'orient'
]

URL = "https://storage.googleapis.com/storage/v1/b/3d-shapes/o/" \
      "3dshapes.h5?alt=media"
H5 = "3dshapes.h5"
SERIES = "3d_shapes.pkl"
