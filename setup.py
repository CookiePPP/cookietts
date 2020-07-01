import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CookieTTS", # Replace with your own username
    version="0.0.1",
    author="cookie",
    author_email="cookietriplep@gmail.com",
    description="A messy package of Text to Speech models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CookiePPP/cookietts",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)