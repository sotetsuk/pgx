from setuptools import find_packages, setup
from pathlib import Path

curr_dir = Path(__file__).parent
long_description = Path(curr_dir / "README.md").read_text()


def _read_requirements(fname):
    with open(Path(curr_dir / "requirements" / fname)) as f:
        return [l.strip() for l in f if not (l.isspace() or l.startswith('#'))]


setup(
    name="pgx",
    description="GPU/TPU-accelerated parallel game simulators for reinforcement learning (RL)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sotetsuk/pgx",
    author="Sotetsu KOYAMADA",
    author_email="sotetsu.koyamada@gmail.com",
    keywords="",
    packages=find_packages(),
    package_data={
        "": ["LICENSE", "*.svg", "_src/assets/*.npy"]
    },
    include_package_data=True,
    install_requires=_read_requirements("requirements.txt"),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
