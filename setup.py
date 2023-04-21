from setuptools import find_packages, setup

setup(
    name="pgx",
    version="0.4.0",
    long_description_content_type="text/markdown",
    description="",
    url="",
    author="Sotetsu KOYAMADA",
    author_email="sotetsu.koyamada@gmail.com",
    keywords="",
    packages=find_packages(),
    package_data={"": ["LICENSE", "*.svg"]},
    include_package_data=True,
    install_requires=[
        "jax>=0.3.25",  # JAX version on Colab (TPU)
        "chex>=0.1.6",
        "svgwrite",
        "msgpack",
        "typing_extensions"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
