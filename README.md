# Mountain NER and Satellite Keypoints Matching Project

This project contains two tasks in separate folders:

1. **Mountain NER (Named Entity Recognition)**: Using a BERT-based model to identify mountain names in text by creating an NER dataset and training a model.
2. **Satellite Keypoints Matching**: Using SuperPoint and SuperGlue models to match keypoints between satellite image pairs to monitor environmental changes.



## Project Structure

```
├── mountain/
│   ├── mountain.ipynb          # Jupyter notebook for Mountain NER dataset creation, training, inference, and demo
├── keypoints-match/
│   ├── keypoints-match.ipynb    # Jupyter notebook for Satellite Image Keypoints Matching dataset creation, training, inference, and demo
└── requirements.txt             # List of required libraries for both tasks
```



## Task 1: Mountain NER Model

### Overview
The `mountain/mountain.ipynb` notebook covers:
- **Dataset Creation**: Scrapes Wikipedia to collect sentences with mountain names, labels them, and saves as a CSV.
- **Model Training**: Configures and trains a BERT model for NER on the mountain dataset, saving model weights.
- **Inference and Demo**: Shows predictions on sample sentences using the trained model.



## Task 2: Satellite Keypoints Matching

### Overview
The `keypoints-match/keypoints-match.ipynb` notebook covers:
- **Dataset Creation**: Organizes and pairs satellite images based on date and region, focusing on selected bands (B02, B03, B04).
- **Model Training**: Configures and trains SuperPoint and SuperGlue models for matching keypoints on the image pairs.
- **Inference and Demo**: Demonstrates keypoint matching on selected image pairs, including visualizations.


## Installation

1. Clone this repository.
2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Open and run the notebooks (`mountain/mountain.ipynb` and `keypoints-match/keypoints-match.ipynb`) in Jupyter to follow each task step-by-step.

