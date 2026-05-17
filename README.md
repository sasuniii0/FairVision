# FairVision: Age Group Classification System 👀

FairVision is a machine learning application designed to predict age groups from face images, with a strong focus on auditing and mitigating bias in AI systems. It features a custom Convolutional Neural Network (CNN) built from scratch using PyTorch and trained on the balanced **FairFace** dataset.

## Features

- **Age Group Prediction**: Classifies uploaded face images into one of 9 age groups (0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+).
- **Custom CNN Architecture**: Utilizes a 3-layer CNN without relying on pre-trained models (no transfer learning).
- **Bias Awareness**: The project investigates and applies mitigation techniques for bias across different demographic groups.
- **Interactive UI**: Built with Streamlit for a simple, user-friendly web interface to test model predictions.

## Project Structure

- `app.py`: The main Streamlit web application.
- `models/`: Directory containing the trained PyTorch model weights (`baseline_model.pth` or `mitigated_model.pth`).
- `FairVision_Technical_Appendix.ipynb`: Jupyter notebook containing the exploratory data analysis, model training, evaluation, and fairness audits.
- `requirements.txt`: Python dependencies required to run the project.

## Installation and Setup

1. **Clone or Download the Repository**
2. **Create a Virtual Environment** (Recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Ensure Models are Present**:
   Make sure you have either `baseline_model.pth` or `mitigated_model.pth` inside the `models/` directory.

## Running the Application

To launch the Streamlit interface, run:
```bash
streamlit run app.py
```
This will open the application in your default web browser (usually at `http://localhost:8501`).

## Usage

1. Open the application in your browser.
2. Click "Browse files" to upload a face image (`.jpg`, `.jpeg`, or `.png`).
3. The system will automatically process the image and display the predicted age group along with the confidence score.
4. You can view the top 3 predicted classes for additional context.

## Ethical Note

This application is a prototype developed for educational purposes to demonstrate bias auditing and mitigation techniques in computer vision. It may still exhibit bias despite mitigation attempts and should not be used for critical decision-making. Please use it responsibly.
