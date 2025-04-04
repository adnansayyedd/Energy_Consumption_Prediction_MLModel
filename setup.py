from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

    setup(
        name='Energy Consumption Prediction',
        version="1.0.0",
        description="This is a Energy consumpation Prediciton Model.",
        author="Adnan Sayyed"
    )
    
