# CropClassifier

CropClassifier is a deep learning model I created for potato plant disease detection. It can accurately classify potato plant images as either healthy or unhealthy, aiding in the early detection and management of crop diseases. The model achieved an impressive accuracy of 97% after training for 10 epochs.

## Usage Examples

To use the CropClassifier, follow these steps:

1. Load the trained model:
   ```python
   from tensorflow.keras.models import load_model

   model = load_model('crop_classifier_model.h5')
   ```

2. Preprocess an image for prediction:
   ```python
   from tensorflow.keras.preprocessing import image
   import numpy as np

   img_path = 'path/to/image.jpg'
   img = image.load_img(img_path, target_size=(256, 256))
   img_array = image.img_to_array(img)
   img_array = np.expand_dims(img_array, axis=0)
   preprocessed_img = img_array / 255.0
   ```

3. Make a prediction:
   ```python
   prediction = model.predict(preprocessed_img)
   if prediction[0][0] > 0.5:
       print("Unhealthy plant")
   else:
       print("Healthy plant")
   ```

## Results and Evaluation

The CropClassifier achieved a remarkable accuracy of 97% in identifying healthy and unhealthy plants. This accuracy indicates the model's ability to accurately classify plant images and distinguish between healthy and diseased potato crops. In addition to accuracy, other metrics such as precision, recall, and F1 score were evaluated to assess the model's performance comprehensively.

## Requirements

The following dependencies are required to run the CropClassifier:

- [TensorFlow](https://www.tensorflow.org/) (Version 2.5.0)
- [NumPy](https://numpy.org/) (Version 1.19.5)

Please refer to the documentation of the respective libraries for installation instructions.

## License

The CropClassifier project is licensed under the [MIT License](LICENSE).

Feel free to contribute to the project by submitting issues or pull requests. Your contributions are greatly appreciated!

## Acknowledgments

Thank you to [spMohnaty](https://github.com/spMohanty/PlantVillage-Dataset/tree/master/raw/color), a GitHub user, for generously providing the open-source dataset of plant leaves. Their extensive collection of healthy and diseased plant leaf images served as the foundation for my project.

Thank you to the open-source community and the developers of TensorFlow, NumPy, and other libraries used in this project.
