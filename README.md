# Fruit Classification with CNN

An image-classification project that trains a custom convolutional neural network (TensorFlow/Keras) to recognise four fruits — apple, banana, mango, and orange — and then investigates and fixes the model's mistakes. The notebook covers the full workflow: exploring the data, training the CNN, evaluating it with a confusion matrix, diagnosing a systematic misclassification, expanding the dataset to correct it, and retraining.

## Key Features

- **Custom CNN** — three Conv2D/MaxPooling blocks (32, 64, 128 filters), Flatten, Dense(128) → Dense(64) → Dropout(0.3) → 4-way softmax; trained with Adam and sparse categorical cross-entropy.
- **Dataset included** — `fruit_dataset.zip` with `Train/` and `Test/` folders, 4 classes.
- **Error analysis** — the notebook inspects the confusion matrix and finds that green fruits are systematically misclassified as apple (the apple class contains green apples, while the other classes contain only ripe-coloured fruit).
- **Dataset correction and retraining** — 50 greenish images per class are added to the training set and 14 per class to the test set, an L2 regulariser is added, and the model is retrained with checkpointing of the best model (`best_model.h5`).
- **Single-image prediction** — a `predict_and_plot_image` helper loads the saved model and classifies one image file, showing the actual vs. predicted label.

## Dataset

- Source: `fruit_dataset.zip` (included in the repo), extracted by the notebook.
- **Original split:** 800 images — 640 train / 160 test. A 10% validation split of the training data gives 576 train / 64 validation / 160 test.
- **After adding green-fruit images:** 840 train files / 216 test files → 776 train / 64 validation / 216 test.
- Images are 256 × 256, batch size 32, pixel values normalised to [0, 1].

## Results

Both runs are recorded in the notebook (test-set evaluation after training):

| Run | Test accuracy | Test loss |
| --- | --- | --- |
| First model (up to 40 epochs, early stopping patience 5) | 0.775 | 2.036 |
| Retrained model (up to 100 epochs, + L2 regularisation, checkpointed best model) | 0.625 | 2.694 |

The retrained model scores lower on the expanded test set, which now includes the harder green-fruit cases; the notebook notes remaining confusion, mainly between mango and banana, and suggests more data and augmentation as the next improvement.

## Tech Stack

- **Python**, **Jupyter Notebook** (Google Colab)
- **TensorFlow / Keras** — custom CNN, `image_dataset_from_directory`, EarlyStopping, ModelCheckpoint
- **NumPy, Pandas, OpenCV**
- **Matplotlib, Seaborn** — data distribution plots, training curves, confusion matrix
- **scikit-learn** — confusion matrix

## How to Run

```bash
git clone https://github.com/majidhanif230/Fruit_Classification_CNN.git
cd Fruit_Classification_CNN
pip install -r requirements.txt
```

1. Open `Fruits_Classifier.ipynb` in Jupyter (or Google Colab) and run the cells in order. The notebook unzips `fruit_dataset.zip` itself.
2. Follow the notebook to train the model, evaluate it, and run the single-image predictor on `best_model.h5`.

## Project Structure

```
Fruit_Classification_CNN/
├── Fruits_Classifier.ipynb   # EDA, custom CNN training, confusion matrix,
│                             # error analysis, dataset expansion, retraining,
│                             # single-image prediction
├── fruit_dataset.zip         # Fruit image dataset (Train/ and Test/)
├── requirements.txt          # TensorFlow, Keras, NumPy, Pandas, OpenCV
└── README.md
```

## Contributors

Project given by **Machine Learning 1 Pvt Ltd** and created by:

- Majid Hanif
- Usama Shafeeq (Team Lead)
- Ammar Rizwan
- Muhammad Shahzaib

## Author

**Majid Hanif** — AI Engineer · BS Artificial Intelligence, The Islamia University of Bahawalpur (CGPA 3.81/4.0)

- Portfolio: https://snap-portfolio-me.lovable.app/portfolio
- LinkedIn: https://www.linkedin.com/in/majid-hanif-23720b2a2/
- GitHub: https://github.com/majidhanif230
