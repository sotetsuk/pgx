from setuptools import find_packages, setup

setup(
    name="pgx",
    version="0.0.8",
    long_description_content_type="text/markdown",
    description="",
    url="",
    author="Sotetsu KOYAMADA",
    author_email="sotetsu.koyamada@gmail.com",
    keywords="",
    packages=find_packages(),
    install_requires=["jax", "flax", "svgwrite"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
)
