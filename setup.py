from setuptools import find_packages,setup
setup(
    name = 'mcqgenerator',
    version='0.0.1',
    author='murugan mari',
    author_email='muruganmack98@gmail.com',
    install_requires=['langchain_google_genai','langchain','streamlit','python-dotenv','pyPDF2','langchain-community'],
    packages=find_packages()
)