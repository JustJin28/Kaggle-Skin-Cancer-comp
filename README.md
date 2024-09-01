# ISIC 2024 Skin Lesion Classification Challenge

## Project Overview
This project addresses the ISIC 2024 Challenge, focusing on the classification of skin lesions using machine learning techniques. The goal is to develop a model that can accurately identify malignant skin lesions from images and associated metadata.

## Key Features
- Data preprocessing pipeline for both image and tabular data
- Custom data generator for efficient batch processing
- Implementation of a hybrid CNN model combining image and tabular inputs
- Use of focal loss to address class imbalance
- K-fold cross-validation for robust model evaluation
- Ensemble predictions from multiple folds

## Technical Stack
- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- OpenCV

## Project Structure
- `load_and_preprocess_data()`: Handles data loading and preprocessing
- `ISICDataGenerator`: Custom data generator for batch processing
- `create_model()`: Defines the neural network architecture
- `focal_loss()`: Implements focal loss for handling class imbalance
- `train_model()`: Manages the training process including k-fold cross-validation

## Getting Started
1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Prepare your data in the specified format
4. Run the main script: `python main.py`

## Data Preprocessing
The project includes robust preprocessing steps:
- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Image preprocessing and augmentation

## Model Architecture
The model combines convolutional layers for image processing with dense layers for tabular data, merging them for final classification.

## Training Process
- Utilizes k-fold cross-validation
- Implements early stopping and learning rate reduction
- Uses class weighting to handle imbalanced datasets

## Prediction
The final prediction is an ensemble of models from different folds, improving robustness and accuracy.

## Results
[Include any notable results, metrics, or comparisons here]

## Future Improvements
- Hyperparameter tuning
- Experimenting with different model architectures
- Implementing more advanced data augmentation techniques

## Contributing
Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License
[Specify your license here]

## Acknowledgements
- ISIC for providing the dataset
- [Any other acknowledgements]
