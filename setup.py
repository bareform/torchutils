import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="torchutils",
        version="0.1.0",
        description="A PyTorch library with helpful utility APIs",
        install_requires=[
            "numpy",
            "torch",
        ],
        extras_require={"dev": ["pytest", "tomli-w"]},
        packages=["torchutils"],
        python_requires=">=3.10",
    )
