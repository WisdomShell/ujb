from setuptools import setup, find_packages

setup(
    name="code_ujb",
    version="0.1.0",
    description="A brief description of your package",
    author="ZhengranZeng",
    author_email="zhengranzeng@gmail.com",
    packages=find_packages(),
    install_requires=[
        'transformers',
        'datasets',
        'openai<=0.28.0',
        'text_generation',
        'fschat',
        'psutil',
        'torch',
        'accelerate',
        'chardet',
        'javalang'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ],
)
