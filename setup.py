from setuptools import find_packages, setup

setup(
    name="pgx",
    version="0.1.9",
    long_description_content_type="text/markdown",
    description="",
    url="",
    author="Sotetsu KOYAMADA",
    author_email="sotetsu.koyamada@gmail.com",
    keywords="",
    packages=find_packages(),
    package_data={"": ["LICENSE", "*.svg"]},
    include_package_data=True,
    install_requires=["jax", "svgwrite", "ipywidgets", "msgpack"],
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
)
