from setuptools import setup, find_packages

setup(
    name="quadrotor_gym",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,  # Important to include non-Python files
    package_data={
        'quadrotor_gym': ['assets/*.urdf'],  # Explicitly include URDF files
    },
    install_requires=[
        'numpy',
        'matplotlib',
        'gymnasium',
        'pybullet',
    ],
)