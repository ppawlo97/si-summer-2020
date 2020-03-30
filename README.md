[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
![Python][python-shield]



<br />
<p align="center">
  <h3 align="center">Final project for Artificial Intelligence course at SGH.</h3>

  <p align="center">
    Below find the information on setup, reference links and creator notes.
    <br />
    <br />
  </p>
</p>



## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)



## About the Project

This project serves as a complementary work to the report on Facial Emotion Recognition I have written for the A.I. class at SGH. Built web application allows the user to assess the performance of several machine learning models either on preloaded images or video stream via a web camera.

All models have been trained from scratch on the [FER+](https://github.com/microsoft/FERPlus) dataset.

## Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

1. Make sure that you have [Anaconda](https://www.anaconda.com/) installed.

2. In order to use OpenCV's pretrained CascadeClassifier as one of the detectors, [download](https://github.com/opencv/opencv/tree/master/data/haarcascades) `haarcascade_frontalface_default.xml` from OpenCV's GitHub page. 

### Installation
 
1. Clone the repository.
```sh
git clone https://github.com/ppawlo97/si-summer-2020.git
```
2. Create separate virtual environment for the project.
```sh
conda create --name=si_project python=3.7
```
3. Switch to the created environment.
```sh
conda activate si_project
```
4. Install the dependencies.
```sh
pip install -r requirements.txt
```



## Usage

1. Checkout to the `master` branch, if you are not already on it.
```sh
git checkout master
```
2. Remember to always switch to the right virtual environment.
```sh
conda activate si_project
```
3. Export the path to the pretrained CascadeClassifier as a global variable.
```sh
export PRETRAINED_CASCLAS=/absolute/path/to/haarcascade_frontalface_default.xml
```
4. Export the main application file as a global variable.
```sh
export FLASK_APP=fer_app.py
```
5. Run the application from the root directory on localhost.
```sh
flask run --port=X.X.X.X
```



## License

Distributed under the MIT License. See `LICENSE` for more information.



[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/ppawlo97/si-summer-2020/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://pl.linkedin.com/in/piotr-paw%C5%82owski-64390917a
[python-shield]: https://img.shields.io/badge/python-3.7.6-blue?style=flat-square&logo=python
