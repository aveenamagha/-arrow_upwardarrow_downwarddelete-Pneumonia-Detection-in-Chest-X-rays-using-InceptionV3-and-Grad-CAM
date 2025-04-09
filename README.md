# Pneumonia Detection in Chest X-rays using InceptionV3 and Grad-CAM

**My Kaggle Notebook**: [Pneumonia Detection in Chest X-rays](https://www.kaggle.com/code/aveenamagha/pneumonia-detection-in-chest-x-rays)

Pneumonia is a potentially life-threatening respiratory illness, particularly dangerous for young children, elderly individuals, and those with weakened immune systems. While chest X-rays are the standard imaging modality for diagnosing pneumonia, interpreting them manually can be both subjective and time-consuming. This project leverages deep learning to build an automated pipeline that accurately detects pneumonia in chest X-ray images.

I used **InceptionV3**, a powerful convolutional neural network pre-trained on ImageNet, and fine-tuned it for binary classification of chest X-rays into `NORMAL` and `PNEUMONIA`. In addition to high classification accuracy, I also incorporated **Grad-CAM** to generate visual explanations of the model’s predictions—crucial for clinical trust and interpretability.

---

## Dataset

The dataset used is the **Chest X-ray Pneumonia Dataset (Kaggle v3)**, created by the National Institutes of Health (NIH) in collaboration with academic partners. It consists of 5,856 pediatric chest X-ray images, collected from children aged 1–5 at Guangzhou Women and Children’s Medical Center.

Each image is labeled as either `NORMAL` or `PNEUMONIA`. The pneumonia images include both bacterial and viral infections, but for this project, I merged them into a single `PNEUMONIA` class to simplify the problem to binary classification. All images were resized to 224×224 pixels to ensure uniformity during training.

---

## Model Architecture

The model is based on the **InceptionV3** architecture. I removed the top layers of the base model and added custom layers including Global Average Pooling, Dropout, and Dense layers with ReLU activation, followed by a final Dense layer with sigmoid activation for binary classification.

To make the model task-specific, I fine-tuned the top 40 layers of InceptionV3 after training the custom head. I also used `Adam` as the optimizer and `binary_crossentropy` as the loss function.

Training was stabilized using **EarlyStopping** and **ModelCheckpoint**, ensuring that the model did not overfit and that the best version was saved.

---

## Training and Evaluation

The model was trained for 10 epochs on the preprocessed dataset.

- **Train Accuracy**: 98.82%  
- **Validation Accuracy**: 95.56%  
- **Test Accuracy**: 94.54%

To further evaluate the model's performance, I generated a **confusion matrix** and **classification report**. These metrics revealed strong precision and recall values, confirming the model's effectiveness in distinguishing between pneumonia and normal cases.

---

## Grad-CAM Visualizations

To interpret the model’s predictions, I applied **Grad-CAM (Gradient-weighted Class Activation Mapping)** on test images. This allowed me to visualize the areas of the X-rays the model considered most important during prediction.

Each visualization consists of two rows:

- The first row shows the original chest X-ray from the test set.
- The second row shows the corresponding Grad-CAM heatmap overlaid on the image.

Red and yellow regions in the heatmaps highlight the areas the model focused on when making a classification. Transparent or faint regions indicate areas with less influence. These visualizations offer insight into the model's decision-making process and help confirm whether it focuses on clinically relevant parts of the image.

---

## Conclusion

This project demonstrates that transfer learning with **InceptionV3**, combined with interpretability via **Grad-CAM**, can produce a highly accurate and interpretable model for pneumonia detection in chest X-rays. The approach offers not only strong predictive performance but also visual transparency, which is essential for real-world deployment in healthcare settings.

---

## Future Work

There are several exciting directions to explore in future iterations of this project:

- Extend the model to **multi-class classification** to distinguish between normal, bacterial, and viral pneumonia cases.
- Train and evaluate the model on **adult chest X-ray datasets** to generalize the solution to a broader population.
- **Deploy the model as a web application** or diagnostic tool for clinicians and researchers.

---

## References

- [Kaggle Chest X-ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
- [InceptionV3 - Google Research](https://arxiv.org/abs/1512.00567)  
- [Grad-CAM: Visual Explanations](https://arxiv.org/abs/1610.02391)

---

Thank you for checking out my project! You can explore the full code, training pipeline, and visualizations in my Kaggle notebook linked above.
