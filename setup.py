from setuptools import find_packages, setup

setup(
    name="soft_snake_cpg_rl",
    version="0.1.0",
    description="Hierarchical CPG-Regulated Option-Critic RL for a Continuum Soft Snake Robot",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyyaml>=6.0",
        "gymnasium>=0.29.0",
        "torch>=2.1.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.14.0",
        "pytest>=7.4.0",
    ],
)

