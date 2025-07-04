from setuptools import setup, find_packages

setup(
    name='orbitalchime',
    version='0.1.9',
    description='Calculating irradiance for Earth for various astronomical parameters.',
    author='Bryan C. Lougheed',
    author_email='bryan.lougheed@outlook.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'numba',
    ],
)
