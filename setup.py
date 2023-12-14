from setuptools import setup, find_packages

setup(
    name='repo-tree',
    version='0.1.0',
    author='Yotam Nahum',
    author_email='yotam@samplead.co',
    description='A simple python package to display a repository tree structure. '
                'pythonic alternative to the linux tree command.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yotamnahum/repo-tree',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=open('requirements.txt').read().splitlines(),
    entry_points={
        'console_scripts': [
            'repo_tree=repo_tree.repository_tree:main',  # Assuming main() is your entry function
        ],
    },
)
