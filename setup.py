#! /usr/bin/env python

import os
from distutils.core import setup

BIN = 'bin'
PACKAGE_NAME = ''
PACKAGES = []
VERSION = 0.01
MAINTAINER='Mohamed Leila',
MAINTAINER_EMAIL='mohamed.leila.1989@gmail.com',
DESCRIPTION='',
LICENSE=''


def setup_package():
    scripts = [os.path.join(BIN, f) for f in os.listdir(f'./{BIN}')]
    setup(
        name=PACKAGE_NAME,
        description=DESCRIPTION,
        version=VERSION,
        scripts=scripts,
        packages=PACKAGES,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        license=LICENSE
        )


if __name__ == "__main__":
    setup_package()
