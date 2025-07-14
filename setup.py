from setuptools import setup, find_packages

setup(
    name='nuclearai',
    version='0.1.0',
    description='AI-Guided Nuclear Waste Transmutation Network',
    author='amir hossein rasti',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'torch-geometric>=2.3.0',
        'numpy>=1.24.0',
        'networkx>=3.0',
        'matplotlib>=3.7.0',
        'scipy>=1.10.0',
    ],
    entry_points={
        'console_scripts': [
            'nuclearai=nuclearai.cli:main',
        ],
    },
    python_requires='>=3.8',
) 