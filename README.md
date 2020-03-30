[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
![Python][python-shield]



<br />
<p align="center">
  <h3 align="center">Final project for Artificial Intelligence course at SGH.</h3>

  <p align="center">
    Below find the information on usage, reference links and creator notes.
    <br />
    <br />
  </p>
</p>



## Table of Contents

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)



## Getting Started

To get a local copy up and running follow these steps.

### Prerequisites

1. Make sure that you have [Anaconda](https://www.anaconda.com/) installed.

2. In order to use OpenCV's pretrained CascadeClassifier as one of the detectors, [download](https://github.com/opencv/opencv/tree/master/data/haarcascades) `haarcascade_frontalface_default.xml` from OpenCV's GitHub page. 

### Installation
 
1. Clone the repository.
```sh
git clone https://github.com/ppawlo97/si-summer-2020.git
```
2. Create virtual environment for the project.
```sh
conda create --name=si_project python=3.7
```
3. Install the dependencies.
```sh
pip install -r requirements.txt
```



## Usage

1. Remember to switch to the right virtual environment.
```sh
conda activate si_project
```
2. Export path to the pretrained CascadeClassifier as a global variable.
```sh
export PRETRAINED_CASCLAS=/absolute/path/to/haarcascade_frontalface_default.xml
```
3. Export application file as a global variable.
```sh
export FLASK_APP=fer_app.py
```
4. Run the application from the root directory as localhost on the specified port.
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
