# Mountain Named Entity Recognition and Satellite Image Feature Matching Projects 👩‍💻

This repository contains two projects:

1. **Mountain Named Entity Recognition (NER) Project**: A project that creates a dataset of sentences related to famous mountains from Wikipedia, preprocesses the data, and trains a BERT model for Named Entity Recognition to identify mountain names within sentences.

2. **Satellite Image Feature Matching with SuperPoint and SuperGlue**: This project performs feature matching between pairs of satellite images using the SuperPoint and SuperGlue models. It detects and matches keypoints between images, useful for monitoring environmental changes like deforestation.


## Repository Structure 🏝️

```
├── mountain/
│   ├── Mountain.ipynb         # Jupyter notebook for dataset creation, training, inference, and demo
│   ├── train_model.py         # Python script for training the Mountain NER model
│   ├── inference_model.py     # Python script for inference using the trained Mountain NER model
├── keypoints-match/
│   ├── keypoints-match.ipynb  # Jupyter notebook for Satellite Image Keypoints Matching
│   ├── model_training_2.py    # Python script for setting up and running the feature matching
│   ├── model_inference_2.py   # Python script for inference on new images
│   ├── matches_example.png    # Example image showing keypoint matches
├── requirements.txt           # List of required libraries for both tasks
└── README.md                  # This README file
```

---

## Mountain Named Entity Recognition (NER) Project 🏔️

### Description

This project involves:

- **Data Collection**: Scraping Wikipedia articles for sentences containing specific mountain names.
- **Data Preprocessing**: Tokenizing sentences and labeling mountain names for NER.
- **Model Training**: Training a BERT-based model to recognize mountain names in sentences.
- **Model Hosting**: The trained model is hosted on Hugging Face for easy access.
- **Inference**: Performing inference on new sentences to identify mountain names.

### Model Link

The trained NER model is available on Hugging Face:

**[NER Mountain Model on Hugging Face](https://huggingface.co/victoriapvo/ner-mountain-model)**

### Files

- **Mountain.ipynb**: Jupyter notebook that includes all steps—data collection, preprocessing, model training, inference, and demonstration.
- **train_model.py**: Python script for training the Mountain NER model.
- **inference_model.py**: Python script for performing inference using the trained model hosted on Hugging Face.

### Usage

#### 1. Navigate to the Mountain Project Directory

```bash
cd mountain
```

#### 2. Install Required Libraries

Install the necessary packages listed in `requirements.txt`:

```bash
pip install -r ../requirements.txt
```

#### 3. Run the Jupyter Notebook

Launch Jupyter Notebook and open `Mountain.ipynb`:

```bash
jupyter notebook Mountain.ipynb
```

- Follow the notebook cells sequentially to execute the entire workflow.

#### 4. Training Using the Python Script

If you prefer running the training script directly:

```bash
python train_model.py
```

- The script will train the model and push it to Hugging Face.

#### 5. Inference

After training, perform inference using:

```bash
python inference_model.py
```

- Enter a sentence when prompted, and the script will identify any mountain names using the model hosted on Hugging Face.

#### 6. Using the Model in Other Projects

You can load the model directly from Hugging Face in your own projects:

```python
from transformers import BertTokenizerFast, BertForTokenClassification

tokenizer = BertTokenizerFast.from_pretrained('victoriapvo/ner-mountain-model')
model = BertForTokenClassification.from_pretrained('victoriapvo/ner-mountain-model')
```


## Satellite Image Feature Matching with SuperPoint and SuperGlue 🛰️

### Description

This project performs:

- **Feature Matching**: Matching keypoints between pairs of satellite images.
- **Models Used**: Utilizes pre-trained SuperPoint and SuperGlue models.
- **Applications**: Useful for monitoring changes over time in satellite imagery, such as deforestation.

### Files

- **keypoints-match.ipynb**: Jupyter notebook that covers dataset handling, model setup, and feature matching demonstration.
- **model_training_2.py**: Python script for setting up and running the feature matching process.
- **model_inference_2.py**: Python script for performing inference on new image pairs.
- **matches_example.png**: An example output image showing keypoint matches between two satellite images.

### Usage

#### 1. Navigate to the Keypoints-Match Project Directory

```bash
cd keypoints-match
```

#### 2. Install Required Libraries

Install the necessary packages:

```bash
pip install -r ../requirements.txt
```

#### 3. Clone the SuperGlue Repository

```bash
git clone https://github.com/magicleap/SuperGluePretrainedNetwork.git
```

#### 4. Add SuperGlue to Python Path

In your scripts or notebooks, add the following lines to include SuperGlue modules:

```python
import sys
sys.path.append('./SuperGluePretrainedNetwork')
```

#### 5. Run the Jupyter Notebook

Launch Jupyter Notebook and open `keypoints-match.ipynb`:

```bash
jupyter notebook keypoints-match.ipynb
```

- Follow the notebook cells sequentially to execute the entire workflow.

#### 6. Running the Feature Matching Script

- Update the `data_dir` variable in `model_training_2.py` to point to your dataset directory.
- Run the script:

  ```bash
  python model_training_2.py
  ```

- The matching visualization will be saved as `matches.png`. The example is `matches_example.png`

#### 7. Inference on New Images

- Update image paths in `model_inference_2.py`.
- Run the script:

  ```bash
  python model_inference_2.py
  ```



## Requirements

All required Python packages are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```


## Additional Notes

- **Data Paths**: Ensure that you update all file paths in the scripts and notebooks to point to your local data directories.
- **Satellite Images**: For the keypoints matching project, you need satellite images in the expected format (e.g., Sentinel-2 bands in `.jp2` format).
- **Model Weights**: The SuperPoint and SuperGlue models are pre-trained and included in the cloned repository. No additional training is required.
- **Results Visualization**: Output images showing the results are saved in the scripts' respective directories.
- **Model Hosting**: The NER Mountain model is hosted on Hugging Face, making it easy to load and use in various applications.

