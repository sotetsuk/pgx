from setuptools import find_packages, setup
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text()

setup(
    name="pgx",
    version="0.5.1",
    description="GPU/TPU-accelerated parallel game simulators for reinforcement learning (RL)",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/sotetsuk/pgx",
    author="Sotetsu KOYAMADA",
    author_email="sotetsu.koyamada@gmail.com",
    keywords="",
    packages=find_packages(),
    package_data={"": ["LICENSE", "*.svg"]},
    include_package_data=True,
    install_requires=[
        "jax>=0.3.25",  # JAX version on Colab (TPU)
        "svgwrite",
        "typing_extensions"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
