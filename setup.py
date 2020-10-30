import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as fin:
    lines = fin.readlines()
    lines = [o.strip() for o in lines]
    lines = [o for o in lines if len(o) > 0]
    req = [o for o in lines if not o.startswith('#')]

setuptools.setup(

    name="outside",  # Replace with your username

    version="1.0.0",

    author="Hung",

    author_email="thanhhungqb@gmail.com",

    description="Code snip from many sources",

    long_description=long_description,

    install_requires=req,

    long_description_content_type="text/markdown",

    url="<https://github.com/authorname/templatepackage>",

    packages=setuptools.find_packages(),

    python_requires='>=3.6',
)
