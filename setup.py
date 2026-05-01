from setuptools import setup, find_packages

setup(
    name='KCD',
    packages=find_packages(exclude=['test', 'tests']),
    install_requires=[
        'torch', 'torchvision',
        'numpy', 'pandas', 'scipy',
        'scikit-image', 'scikit-learn',
        'matplotlib',
        'nibabel', 'simpleitk', 'pydicom', 'dicom2nifti', 'rt_utils',
        'opencv-python', 'numpy-stl', 'pymeshfix',
        'gecatsim',
        'pyyaml',
    ],
    python_requires='>=3.10',
    description='Automated small kidney cancer detection in non-contrast CT',
    version='1.0',
    url='https://github.com/McGoughBill/KCD',
    author='Bill McGough',
    author_email='billy.mcgough1@hotmail.com',
    keywords=['medical-imaging', 'kidney-cancer', 'ct', 'deep-learning', 'pytorch'],
)