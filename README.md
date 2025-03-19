# Driver Drowsiness Detection

This project implements a **Driver Drowsiness Detection System** using **Convolutional Neural Networks (CNN)** and **OpenCV** for real-time face and eye tracking. The system detects drowsiness in drivers and alerts them to prevent accidents.

## Features
- **Face & Eye Detection** using Haar Cascade Classifiers
- **Real-time Monitoring** using a laptop camera
- **CNN Model** to classify open & closed eyes
- **Alert System** for drowsiness detection

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries.

### Clone the Repository
```sh
git clone https://github.com/manasis5/Mainprojrct.git
cd Mainprojrct
```

### Install Dependencies
opencv-python
tensorflow
keras
numpy
matplotlib
pandas
scikit-learn
imutils
pygame


pip install your_dependency_name
eg:pip install keras



## Project Files
- `driver.ipynb` - Jupyter Notebook for testing and training the model
- `haarcascades/` - Haar cascade XML files for face & eye detection
-`main.ipynb` - used for live detection and gives the result.

## Haarcascade Files
This project uses OpenCV’s Haar cascade classifiers for:
- **Frontal Face Detection**: `haarcascade_frontalface_alt.xml`
- **Left Eye Detection**: `haarcascade_lefteye_2splits.xml`
- **Right Eye Detection**: `haarcascade_righteye_2splits.xml`

## Contributing
Contributions are welcome! Feel free to fork the repo and submit pull requests.

## License
This project is open-source and available under the **MIT License**.

## Brief Description of the code to be used in VScode
-create a new file and with name driver with extension.ipny(jupyter notebook)
-then save the file in a folder in the desktop
-in the folder upload your haarcascadeclassifier files
-now upload the alarm.mp3 in the same folder
-also upload the dataset in the same folder 
-now in vs code open a new file with main.ipynb name
-execute the main.ipynb
-you can download the dataset from the folder dataset there is a link attached copy the link and paste it then you can download in the zip format
-now if you want to build a front end with flask i have given the code app.py
-create a new file with app.py and in the terminal run python app.py else just execute the main.ipynb

---
_Developed by Manasi_

