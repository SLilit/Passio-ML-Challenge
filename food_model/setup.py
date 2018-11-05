from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
]

setup(
    name='food_model',
    version='0.1',
    author = 'L Sargsyan',
    author_email = 'lak@cloud.google.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Food prediction in Cloud ML',
    requires=[]
)
