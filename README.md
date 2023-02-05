# Python-Digit-Classification

This project explores the main classification algorithms to train the best possible model for the classification of handwritten digits. The dataset used is the Keras digit dataset, which contains 60,000 training images and 10,000 test images of handwritten digits. However, for computational reasons, only 10% of the training images and 10% of the test images are used.

## Requirements

-   Python 3
-   Packages from `requirements.txt`

## Installation

1.  Clone the repository

```bash
git clone https://github.com/dan-koller/Python-Digit-Classification
```

2. Create a virtual environment\*

```bash
python3 -m venv venv
```

3. Activate the virtual environment

```bash
source venv/bin/activate
```

4. Install the requirements\*

```bash
pip3 install -r requirements.txt
```

5. Run the app\*

```
python3 main.py
```

_\*) You might need to use `python` and `pip` instead of `python3` and `pip3` depending on your system._

## Usage

The main file `main.py` contains the code to train and test the models. In its current state, the script is set up to train and test the models on the Keras digit dataset and output the best model estimators.

Output:

```bash
K-nearest neighbors algorithm
best estimator: KNeighborsClassifier()
Accuracy: 0.958

Random forest algorithm
best estimator: RandomForestClassifier()
Accuracy: 0.945
```

The script can be easily modified to analyze other datasets or parameters. Expect the script to take a while to run, depending on the dataset, parameters, and hardware.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
