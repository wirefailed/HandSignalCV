# HandSignalCV
Hand Signal Computer Vision
# collegeprofessorswebscrap

<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

HandSignalCV is a project designed to detect American Sign Language (ASL) from a camera feed and convert it into a spoken sentence using an audio system. The project will utilize OpenCV and Mediapipe to capture frames and retrieve an ASL dataset. This dataset will then be divided into training, cross-validation, and test sets.

Using this data, the project will create a Deep Neural Network (DNN) with convolutional layers and ReLU activation functions, trained using the Adam optimizer. An AI API will be integrated to verify the constructed sentences, ensuring accuracy. Additionally, Text-to-Speech technology will be implemented to facilitate communication between users who know sign language and those who do not.

The system will be housed on a Raspberry Pi board, equipped with an external camera and audio system, making it portable and convenient for real-world use.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

Python, Rasberry pi
Libraries: OpenCV, Mediapipe, numpy

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites



### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```zsh
   git clone https://github.com/wirefailed/HandSignalCV.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Install OpenCV and mediapipe
- [x] Using both libraries, set it to detect hand and display handlandmarks
- [x] Create a rectangle around the hand using cv2.rectangle
- [x] Crop the rectangle seperately
- [x] Use its size to get lmList and get resized_lmList that fits into white square 300 by 300
- [] Capture hand signals from a to z 200 each and split data samples 6:2:2 ratio
- [] Train the model and optimize it to have >85% accuracy
- [] Deploy the system and connect into AGI api (if not possible, skip this step)
- [] Deploy Text-to-speech translation
- [] Get it working on the computer
- [] Connect Camera and Audio System to Rasberry Pi
- [] Deploy the system
- [] 3D print the case

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Junsoo Kim - junsooki@usc.edu

Project Link: [https://github.com/wirefailed/HandSignalCV.git](https://github.com/wirefailed/HandSignalCV.git)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

<p align="right">(<a href="#readme-top">back to top</a>)</p>



