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
    <li><a href="#process">Process</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

HandSignalCV is a project designed to detect American Sign Language (ASL) from a camera feed and convert it into a spoken sentence using an audio system. The project will utilize OpenCV and Mediapipe to capture frames and retrieve an ASL dataset. This dataset will then be divided into training, validation, and test sets.

Using this data, the project will create a Deep Neural Network (DNN) with convolutional layers and ReLU activation functions, trained using the Adam optimizer. An AI API will be integrated to verify the constructed sentences, ensuring accuracy. Additionally, Text-to-Speech technology will be implemented to facilitate communication between users who know sign language and those who do not.


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

Python, Jupyter Notebook
Libraries: OpenCV, Mediapipe, numpy, tensorflow, sk-learn, h5py

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

pip install tables
pip install h5py pillow
pip install jupyter 
pip install tensorflow
python -m pip install cohere --upgrade
pip install python-dotenv

### Installation

1. Clone the repo
   ```zsh
   git clone https://github.com/wirefailed/HandSignalCV.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Process

1. Run capturingSignals.py
    '''zsh
    python capturingSignals.py
    '''
    Each iterations of run, write alphabets and press 's' to save the photo. Hold it and it automatically will close when it has enough data or the program crash.

2. Run python hdf5_conversion.py
    '''zsh
    python hdf5_conversion.py
    '''
    Simply run the code and it will automatically create hdf5_file for you.

3. Get Cohere API Key and create .env file
    '''zsh
    echo "CO_API_KEY=your_actual_api_key_here" > .env
    '''
    This will echo CO_API_KEY=your_actual_api_key_here in .env. It will automatically create .env file
    if it does not exist. Simply change 'your_actual_api_key_here' to Cohere API Key
  
4. Run detectingSignal.py
    '''zsh
    python detectingSignal.py
    '''
    Run the following code and it will start displaying the letter on the monitor
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage
For individuals who don't understand American Sign Language (ASL), this tool allows ASL users to communicate by using hand signals in front of a computer, which then converts these signals into written sentences. This makes it easier for non-ASL users to read and understand hand signals conveniently.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Install OpenCV and mediapipe
- [x] Using both libraries, set it to detect hand and display handlandmarks
- [x] Create a rectangle around the hand using cv2.rectangle
- [x] Crop the rectangle seperately
- [x] Use its size to get lmList and get resized_lmList that fits into white square 300 by 300
- [x] Capture hand signals from a to z 200 each and split data samples 6:2:2 ratio
- [x] Train the model and optimize it to have >85% accuracy
- [x] Deploy the system and connect into AGI api (if not possible, skip this step)

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

https://github.com/cohere-ai/cohere-python
https://github.com/chetan-mehta707/Machine-Learning-Projects/tree/master/Computer%20Vision%20Problem
https://github.com/Devansh-47/Sign-Language-To-Text-and-Speech-Conversion
https://github.com/kinivi/hand-gesture-recognition-mediapipe
https://youtube.com/playlist?list=PL0FM467k5KSyt5o3ro2fyQGt-6zRkHXRv&si=E9pz2R6eCOkGhc_I
https://github.com/cvzone/cvzone

<p align="right">(<a href="#readme-top">back to top</a>)</p>



