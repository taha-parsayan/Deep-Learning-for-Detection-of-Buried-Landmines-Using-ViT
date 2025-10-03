![Static Badge](https://img.shields.io/badge/Classification-FF0000)
![Static Badge](https://img.shields.io/badge/Computer%20Vision-FF0000)
![Static Badge](https://img.shields.io/badge/PyTorch-8A2BE2)
![Static Badge](https://img.shields.io/badge/Torch%20Vision-8A2BE2)
![Static Badge](https://img.shields.io/badge/Python-4CAF50)

# Classification of landmines using computer vision

---
## (Publication in IEEE)[https://ieeexplore.ieee.org/abstract/document/11166148]
---

This repository contains code and resources for detecting and classifying land mines. The primary goal is to analyze and classify data derived from ground-penetrating radar and other sensors. The codebase supports machine learning techniques to ensure accurate and efficient classification.
This project utilizes a Vision Transformer (ViT) model for classifying land mines using image data. The ViT model, specifically the vit_b_16 variant from the PyTorch torchvision library, is employed with pretrained weights as a feature extractor. The method involves:
1. Preprocessing: Sensor data is preprocessed and resized to match the ViT input dimensions (e.g., 224x224 pixels for images).
2. Feature Extraction: The pretrained ViT processes the input data to extract high-dimensional features using its transformer-based architecture, which excels at capturing long-range dependencies and spatial patterns.
3. Classifier Training: A custom classification head is trained on top of the extracted features to classify the presence of land mines in a binary manner.
4. Evaluation: The model is evaluated using standard performance metrics such as accuracy, precision, and recall, ensuring its robustness and reliability for land mine detection tasks.

Data: Free zone, 0cm depth mine, 1cm depth mine, 5cm depth mine, 10 cm depth mine

![image](https://github.com/user-attachments/assets/8a4b839c-c252-4ce4-b8d7-8db7c6239b52)


## Results

- Phase 1 of classification: Free zone vs. Land mine (all mine classes)

**Train accuracy: 93%**
  
**Test accuracy: 84%**

- Phase 2 of classification: Surface mine (0cm & 1cm depth) vs. Deep mine (5cm & 10cm depth)

**Train accuracy: 86%**
  
**Test accuracy: 75%**

## Dataset

The data used in this project is sourced from the following publication:

- **Title**: Land Mine Detection Dataset
- **Source**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340923005437#refdata001)

The dataset contains sensor readings and annotations that are used for training and validating machine learning models.

### Accessing the Dataset
To access the dataset:
1. Visit the [dataset link](https://www.sciencedirect.com/science/article/pii/S2352340923005437#refdata001).
2. Follow the instructions on the website to download the data.

## Classification model:
The pre-trained ViT classifier was used. Through transfer-learning, all the layers were frozen. Then the head layer was modified for 2 classes and a drop-out with 20% probability.

![image](https://github.com/user-attachments/assets/c61b99b1-b8e0-4149-9563-da2c1f3f65e4)


## Project Structure

- `Land_mine_main.ipynb`: Main notebook containing the code for data preprocessing, model training, evaluation, and visualization.
- `data/`: Directory where raw and processed data files should be placed (not provided in this repo).
- `Land_mine_data_management_functions.py/`: Python functions for data management for this project. Data automatically is copied into train and test folders.
- `going_modular/data_setup.py/`: Python functions to pre-process the images.
- `going_modular/engin.py/`: Python functions to run the classification model (train and test).


## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/land-mine-detection.git
   cd land-mine-detection
   ```

2. Place the downloaded dataset in the `data/` folder.

3. Open `Land_mine_main.ipynb` in Jupyter Notebook or any compatible environment.

4. Run the notebook cells step by step to preprocess the data, train the model, and evaluate its performance.

## Results

- Model accuracy, precision, and recall metrics are reported in the notebook.
- Visualization of classification results is provided for better interpretability.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- Thanks to the authors of the [dataset publication](https://www.sciencedirect.com/science/article/pii/S2352340923005437#refdata001) for providing the data.
- Inspired by advancements in ground-penetrating radar technology and machine learning.

