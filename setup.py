from setuptools import find_packages, setup

if __name__ == '__main__':
    setup(
        name='lasanet',
        version='1.0',
        description='LaSA-Net: A Language-Guided Network for Outdoor Generalized 3D Referring Expression Segmentation',
        author='Anonymous',
        author_email='',
        packages=find_packages(include=['lasanet', 'lasanet.*'], exclude=['lasanet.torch', 'lasanet.torch.*']),
    )
