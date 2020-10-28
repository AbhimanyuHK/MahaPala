from setuptools import setup
import glob

data_files = [f.replace("\\", "/") for f in glob.glob("data/*/*")]
print(data_files)

with open("README.md", "r", encoding="utf8") as readme:
    description = readme.read()

with open("version.txt", "r") as version:
    version_x = version.read()
    version_x = version_x.strip()

with open("requirements.txt", "r") as requirements:
    install_requires = requirements.read()

install_requires = [x.strip() for x in install_requires.split("\n") if x]

keys = [x.strip(x.split("/")[-1]).strip("/").strip() for x in data_files]

setup(
    name='MahaPala',
    version=version_x,
    packages=['mahapala'],
    url='https://github.com/AbhimanyuHK/MahaPala',
    license='',
    author='Abhimanyu HK',
    author_email='manyu1994@hotmail.com',
    description='Detection of fruits disease by using Machine learning',
    long_description=description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    data_files=[(keys[0], data_files)],
)
