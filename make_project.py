import os
import shutil
import pathlib
from pathlib import Path
from subprocess import check_output

TEST = True

HOME = os.environ['HOME']
here = pathlib.Path(__file__).parent.resolve()
package_name = here.stem

package_root = Path(f'./src/{package_name}')
archive_root = Path('./.setup_archive')

classifiers_default = [
            # How mature is this project? Common values are
            #   3 - Alpha
            #   4 - Beta
            #   5 - Production/Stable
            'Development Status :: 3 - Alpha',

            # Indicate who your project is intended for
            'Intended Audience :: Developers',
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate you support Python 3. These classifiers are *not*
            # checked by 'pip install'. See instead 'python_requires' below.
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3 :: Only',
        ]

def make_setup():
    author_name = input('Author name: ')
    author_email = input('Author email: ')
    description = input('Short description (~1 sentence): ')
    repo_url = check_output('git config --get remote.origin.url', shell=True).decode('utf-8').strip()
    python_requires = input('Minimum required python version (e.g. "3.7"): ')
    keywords = input('Keywords (comma separated, with space. E.g. "fun, stuff, apteryx": ')
    project_website = input('Project website (leave blank if none): ')

    setup_template = f'''from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install
import pathlib
from subprocess import check_call

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')
install_requires = (here / 'install_requires').read_text(encoding='utf-8').split('\\n')

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        check_call('python install_scripts.py'.split())

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        check_call('python install_scripts.py'.split())

setup(
    name='{package_name}',
    version='0.0.1',
    author='{author_name}',
    author_email='{author_email}',
    desctiption='{description}',
    long_description = long_description,
    long_description_content_type='text/markdown',
    url='{repo_url}',
    python_requires='>={python_requires}',
    keywords='{keywords}',
    project_urls = {{
            'Website' : '{project_website}'
        }},
    package_dir = {{'': 'src'}},
    packages = find_packages(where='src'),
    install_requires = install_requires,
    cmdclass={{
            'develop': PostDevelopCommand,
            'install': PostInstallCommand,
        }}
)
    '''
    with open('setup.py', 'w') as f:
        f.write(setup_template)

def make_install_scripts():
    content = f'''\'''
Runs install steps on first import.
\'''

import os


from src.{package_name}.GLOBALS import (MESSAGE)

def say_hi():
    print(MESSAGE)

if __name__ == '__main__':
    print('Running post-install scripts!')
    say_hi()
'''
    with open('./install_scripts.py', 'w') as f:
        f.write(content)

    globals = f'''MESSAGE='Hello, world!'
'''
    with open(package_root / 'GLOBALS.py', 'w') as f:
        f.write(globals)


def init_empty(path):
    with open(path, 'w') as f:
        f.write('')

def init_w_txt(path, txt):
    with open(path, 'w') as f:
        f.write(txt)

if __name__ == '__main__':

    if TEST:
        if os.path.exists(package_root):
            shutil.rmtree(package_root)
            os.remove('./setup.py')
            os.remove('./setup.cfg')
            os.remove('./')

    ok = input(f'Will make a project with the following name:\n"{package_name}"\nOK? (y/n): ')
    proceed = ok.upper() in 'Y'
    if proceed:
        os.makedirs(package_root)
        make_setup()
        make_install_scripts()

        #Make package __init__.py
        init_empty(package_root / '__init__.py')

        #Make install_requires
        init_w_txt('./install_requires', '# Newline separated package dependencies.')

        #Archive setup files
        os.mkdir(archive_root)
        shutil.move('./make_project.py', './.setup_archive')

    else:
        print('Package creation aborted.')
