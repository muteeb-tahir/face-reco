# FaceRecognition

This is a machine learning project designed to recognize people from images, similar to Facebook's face recognition. It leverages [dlib](http://dlib.net/) for state-of-the-art face recognition, achieving an accuracy of 99.38% on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark.

## Dependencies

- Python 3.x
- Numpy
- Scipy
- [Scikit-learn](http://scikit-learn.org/stable/install.html)
- [dlib](http://dlib.net/)

  **Tip**: Installing dlib can be complex. For macOS or Linux, follow [this guide](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf).

- **Extras**:

  - **OpenCV** (required for `webcam.py` to capture frames from the webcam)
  - For using `./demo-python-files/projecting_faces.py`, install [Openface](https://cmusatyalab.github.io/openface/setup/):
    ```bash
    $ git clone https://github.com/cmusatyalab/openface.git
    $ cd openface
    $ pip install -r requirements.txt
    $ sudo python setup.py install
    ```

## Result

![Result Gif](https://user-images.githubusercontent.com/17249362/28241776-a45a5eb0-69b8-11e7-9024-2a7a776914e6.gif)

## Procedure

### Cloning the Repository

Clone this repository to your local machine:
```bash
git clone git@github.com:muteeb-tahir/FaceRecognition.git
