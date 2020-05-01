"""
speedrun - A toolkit for quick-and-clean machine learning experiments with Pytorch and beyond.
"""

import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install

import os
import sys
import site


# https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        try:
            choice = input().lower()
        except EOFError:
            choice = ''
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


# https://stackoverflow.com/questions/36187264/how-to-get-installation-directory-using-setuptools-and-pkg-ressources
def binaries_directory():
    """Return the installation directory, or None"""
    if '--user' in sys.argv:
        paths = (site.getusersitepackages(),)
    else:
        py_version = '%s.%s' % (sys.version_info[0], sys.version_info[1])
        paths = (s % (py_version) for s in (
            sys.prefix + '/lib/python%s/site-packages/',
            sys.prefix + '/local/lib/python%s/site-packages/',
            sys.prefix + '/lib/python%s/dist-packages/',
            sys.prefix + '/local/lib/python%s/dist-packages/',
            '/Library/Python/%s/site-packages/',
        ))

    for path in paths:
        if os.path.exists(path):
            return path
    print('no installation path found', file=sys.stderr)
    return None


def setup_shell_init(rcfile, init_lines):
    with open(rcfile, 'r') as file:
        lines = file.readlines()
    start_marker = '# >>> speedrun >>>\n'
    end_marker = '# <<< speedrun <<<\n'
    start, end = None, None
    if start_marker in lines:
        start = lines.index(start_marker)
    if end_marker in lines:
        end = lines.index(end_marker)
    override_previous = (start is not None) and (end is not None) and (end > start)
    if override_previous is True:
        print(f'Overriding previous speedrun init in {rcfile}')
        new_lines = lines[:start]
    else:
        print(f'Adding new speedrun init to {rcfile}')
        new_lines = lines
    new_lines.append(start_marker)
    new_lines.extend(init_lines)
    new_lines.append(end_marker)
    if override_previous:
        new_lines.extend(lines[end+1:])
    with open(rcfile, 'w') as file:
        file.writelines(new_lines)


def setup_shell_inits(speedrun_path):
    init_lines = [
        "# !! Contents within this block are managed by speedrun's install script !!\n"
        '# Enable command line completion for config keys\n',
        f'source {os.path.join(speedrun_path, "autocomplete", "init_completion")}\n',
    ]
    for file in [os.path.expanduser(f) for f in ['~/.bashrc', '~/.zshrc']]:
        if os.path.isfile(file) and query_yes_no(
                f'Should custom command line completion be enabled in {file} ?'):
            setup_shell_init(file, init_lines)


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        setup_shell_inits(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'speedrun'))


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        setup_shell_inits(os.path.join(binaries_directory(), 'speedrun'))


setuptools.setup(
    name="speedrun",
    author="Nasim Rahaman",
    author_email="nasim.rahaman@iwr.uni-heidelberg.de",
    license='GPL-v3',
    description="Toolkit for machine learning experiment management.",
    version="0.1",
    install_requires=['pyyaml>=3.12'],
    packages=setuptools.find_packages(),
    package_data={
        '': ['init_completion'],
    },
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
