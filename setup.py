import setuptools

setuptools.setup(
    name="meta_aad",
    version='1.0.0',
    author="Daochen Zha",
    author_email="daochen.zha@tamu.edu",
    description="Code for ICDM 2020 paper Meta-AAD: Active Anomaly Detection with Deep Reinforcement Learning",
    url="https://github.com/daochenzha/Meta-AAD",
    keywords=["Anomaly Detection", "Active Learning", "Reinforcement Learning"],
    packages=setuptools.find_packages(exclude=('tests',)),
    install_requires=[
        "scikit-learn",
        "gym",
        'tensorflow',
        'stable-baselines==2.10.1',
    ],
    requires_python='>=3.5',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
