# Sign Language Translator (Alphabet Recognizer)

This project is a sign language alphabet recognizer using Python, openCV and tensorflow for training InceptionV3 model, a convolutional neural network model for classification.
The framework used for the CNN implementation can be found here:

#### Requirements

This project uses python 3 and above and the PIP following packages:
* opencv
* tensorflow
* matplotlib
* numpy

See requirements.txt and Dockerfile for versions and required APT packages

Using Docker
```
docker build -t hands-classifier .
docker run -it hands-classifier bash
```
Install using PIP
```
pip3 install -r requirements.txt
```
To train the model, use the following command (see framework github link for more command options):
```
python train.py \
  --bottleneck_dir=logs/bottlenecks \
  --how_many_training_steps=2000 \
  --model_dir=inception \
  --summaries_dir=logs/training_summaries/basic \
  --output_graph=logs/trained_graph.pb \
  --output_labels=logs/trained_labels.txt \
  --image_dir=./dataset
```

  
To test classification, use the following command:
```
python classify.py path/to/image.jpg
```

To use webcam, use the following command:
```
python classify_webcam.py
```
