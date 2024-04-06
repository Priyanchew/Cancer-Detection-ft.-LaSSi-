import streamlit as st
import pickle as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tcp_latency import measure_latency
import tensorflow as tf
import time
import cv2
from tensorflow import keras
import numpy as np
import pandas as pd
from PIL import Image

# Measure Latency

#35.201.127.49:443
#192.168.18.6:8501

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=r"C:\Users\jaska\AppData\Local\GitHubDesktop\app-3.3.5\CANCER-ft.-LaSSi\InceptionResNetV2Skripsi.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a function to resize the input image
def resize_image(image):
    # Resize the image to 150x150 pixels
    resized_image = tf.image.resize(image, [150, 150])
    return resized_image.numpy()

# Define a function to run inference on the TensorFlow Lite model

def brain_image(image):
  labels2 = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
  image = np.array(Image.open(image).convert("RGB"))
  image=cv2.resize(image,(150,150))
  reshaped_image = image.reshape((1, 150, 150, 3)) 
  loaded_model = keras.models.load_model(r"C:\Users\jaska\AppData\Local\GitHubDesktop\app-3.3.5\CANCER-ft.-LaSSi\effnet.keras")
  new_pred=loaded_model.predict(reshaped_image)
  max_index = np.argmax(new_pred)
  column_number = max_index % new_pred.shape[1]
  return(labels2[column_number])
def classify_image(image):
    # Pre-process the input image
    resized_image = resize_image(image)
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    with st.spinner('Classifying...'):
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()

    # Calculate the classification duration
    classifying_duration = end_time - start_time

    # Get the output probabilities
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data[0], classifying_duration

# Define the labels for the 7 classes
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

from PIL import Image

def get_clean_data():
  data = pd.read_csv(r"C:\Users\jaska\AppData\Local\GitHubDesktop\app-3.3.5\CANCER-ft.-LaSSi\data\data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data


def add_sidebar_breast():
  st.sidebar.header("Cell Nuclei Measurements")
  
  data = get_clean_data()
  
  slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict


def get_scaled_values(input_dict):
  data = get_clean_data()
  
  X = data.drop(['diagnosis'], axis=1)
  
  scaled_dict = {}
  
  for key, value in input_dict.items():
    max_val = X[key].max()
    min_val = X[key].min()
    scaled_value = (value - min_val) / (max_val - min_val)
    scaled_dict[key] = scaled_value
  
  return scaled_dict
  

def get_radar_chart(input_data):
  
  input_data = get_scaled_values(input_data)
  
  categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                'Smoothness', 'Compactness', 
                'Concavity', 'Concave Points',
                'Symmetry', 'Fractal Dimension']

  fig = go.Figure()

  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
          input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
          input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
          input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
          input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
          input_data['concave points_se'], input_data['symmetry_se'],input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
  ))
  fig.add_trace(go.Scatterpolar(
        r=[
          input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
          input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
          input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
          input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
  ))

  fig.update_layout(
    polar=dict(
      radialaxis=dict(
        visible=True,
        range=[0, 1]
      )),
    showlegend=True
  )
  
  return fig


def add_predictions(input_data):
  model = pickle.load(open(r"C:\Users\jaska\AppData\Local\GitHubDesktop\app-3.3.5\CANCER-ft.-LaSSi\model\model.pkl", "rb"))
  scaler = pickle.load(open(r"C:\Users\jaska\AppData\Local\GitHubDesktop\app-3.3.5\CANCER-ft.-LaSSi\model\scaler.pkl", "rb"))
  
  input_array = np.array(list(input_data.values())).reshape(1, -1)
  
  input_array_scaled = scaler.transform(input_array)
  
  prediction = model.predict(input_array_scaled)
  
  st.subheader("Cell cluster prediction")
  st.write("The cell cluster is:")
  
  if prediction[0] == 0:
    st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
  else:
    st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
    
  
  st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
  st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
  
  st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")



def main():
    st.set_page_config(
            page_title="Breast Cancer Predictor",
            page_icon=":female-doctor:",
            layout="wide",
            initial_sidebar_state="expanded"
    )
    tabs = st.tabs(["Breast", "Lung", "Skin", "Brain", "Blood"])

    with open(
            r"C:\Users\jaska\AppData\Local\GitHubDesktop\app-3.3.5\CANCER-ft.-LaSSi\assets\style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    with tabs[0]:
      input_data = add_sidebar_breast()

      with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

      col1, col2 = st.columns([4,1])

      with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
      with col2:
        add_predictions(input_data)

    with tabs[1]:  # Lung Cancer
        with st.container():
            st.title("Lung Cancer Predictor")
        st.write("Information about lung cancer, including symptoms, diagnosis, and treatment options.")
        st.markdown("[Learn more about Lung Cancer](Lung Cancer Resource URL)")  # Replace with actual URL

        
    with tabs[2]:  # Skin Cancer
        st.title("Skin Cancer Classification")

        st.write(
            "The application allows users to quickly and easily check for signs of skin cancer from the comfort of their own homes. It's important to note that the application is not a substitute for medical advice, but rather a tool to help users identify potential skin issues and seek professional help if necessary. Heres the overview:")
        st.write("1.	The user is prompted to upload an image of a skin lesion.")
        st.write(
            "2.	Once the user uploads an image, it is displayed on the screen and the application runs the image through a machine learning model.")
        st.write(
            "3.	The model outputs the top 3 predictions for the type of skin cancer (out of 7 classes) that the lesion may be, along with the confidence level for each prediction.")
        st.write("4.	The user is shown the top 3 predictions and their confidence levels.")
        st.write(
            "5.	If the model's confidence level is less than 70%, the application informs the user that the skin is either healthy or the input image is unclear. It is also recommended that the user consults a dermatologist for any skin concerns.")
        st.write(
            "6.	The application also measures the latency or response time for the model to classify the image, and displays it to the user.")
        st.write(
            "Please note that this model still has room for academic revision as it can only classify the following 7 classes")
        st.write(
            "- ['akiec'](https://en.wikipedia.org/wiki/Squamous-cell_carcinoma) - squamous cell carcinoma (actinic keratoses dan intraepithelial carcinoma),")
        st.write("- ['bcc'](https://en.wikipedia.org/wiki/Basal-cell_carcinoma) - basal cell carcinoma,")
        st.write(
            "- ['bkl'](https://en.wikipedia.org/wiki/Seborrheic_keratosis) - benign keratosis (serborrheic keratosis),")
        st.write("- ['df'](https://en.wikipedia.org/wiki/Dermatofibroma) - dermatofibroma, ")
        st.write("- ['nv'](https://en.wikipedia.org/wiki/Melanocytic_nevus) - melanocytic nevus, ")
        st.write("- ['mel'](https://en.wikipedia.org/wiki/Melanoma) - melanoma,")
        st.write(
            "- ['vasc'](https://en.wikipedia.org/wiki/Vascular_anomaly) - vascular skin (Cherry Angiomas, Angiokeratomas, Pyogenic Granulomas.)")
        st.write(
            "Due to imperfection of the model and a room of improvement for the future, if the probabilities shown are less than 70%, the skin is either healthy or the input image is unclear. This means that the model can be the first diagnostic of your skin illness. As precautions for your skin illness, it is better to do consultation with dermatologist. ")

        # Get the input image from the user
        image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        # Show the input image
        if image is not None:
            image = np.array(Image.open(image).convert("RGB"))
            st.image(image, width=150)

            # Run inference on the input image
            probs, classifying_duration = classify_image(image)

            # Display the classification duration
            st.write(f"Classification duration: {classifying_duration:.4f} seconds")

            # Display the top 3 predictions
            top_3_indices = np.argsort(probs)[::-1][:3]
            st.write("Top 3 predictions:")
            for i in range(3):
                st.write("%d. %s (%.2f%%)" % (i + 1, labels[top_3_indices[i]], probs[top_3_indices[i]] * 100))

            latency = measure_latency(host='35.201.127.49', port=443)
            st.write("Network Latency:")
            st.write(latency[0])


    with tabs[3]:  # Brain Cancer
        st.title("Brain Cancer Classification")

        st.write(
            "The application allows users to quickly and easily check for signs of Brain cancer from the comfort of their own homes. It's important to note that the application is not a substitute for medical advice, but rather a tool to help users identify potential issues and seek professional help if necessary. Heres the overview:")
        st.write("1.	The user is prompted to upload an image of a MRI Scan.")
        st.write(
            "2.	Once the user uploads an image, it is displayed on the screen and the application runs the image through a machine learning model.")
        st.write(
            "3.	The model outputs the top 3 predictions for the type of skin cancer (out of 7 classes) that the lesion may be, along with the confidence level for each prediction.")
        st.write("4.	The user is shown the top 3 predictions and their confidence levels.")
        st.write(
            "5.	If the model's confidence level is less than 70%, the application informs the user that the skin is either healthy or the input image is unclear. It is also recommended that the user consults a dermatologist for any skin concerns.")
        st.write(
            "6.	The application also measures the latency or response time for the model to classify the image, and displays it to the user.")
        st.write(
            "Please note that this model still has room for academic revision as it can only classify the following 7 classes")
        st.write(
            "- ['pituitary_tumor'](https://en.wikipedia.org/wiki/Pituitary_adenoma) - Pituitary Adenoma,")
        st.write("- ['meningioma_tumor'](https://en.wikipedia.org/wiki/Meningioma) - Meningioma,")
        st.write(
            "- ['glioma_tumor'](https://en.wikipedia.org/wiki/Glioma) - Glioma,")
        st.write("- No tumour- You are Safe ")
        st.write(
            "Due to imperfection of the model and a room of improvement for the future, if the probabilities shown are less than 70%, the skin is either healthy or the input image is unclear. This means that the model can be the first diagnostic of your skin illness. As precautions for your skin illness, it is better to do consultation with dermatologist. ")

        # Get the input image from the user
        image1 = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"],key="file_uploader_1")
        # Show the input image
        if image1 is not None:
          st.write(brain_image(image1))

        st.write()



    with tabs[4]:  # Blood Cancer
        st.subheader("Blood Cancer")
        st.write("Information about blood cancer, including symptoms, diagnosis, and treatment options.")
        st.markdown("[Learn more about Blood Cancer](Blood Cancer Resource URL)")


 
if __name__ == '__main__':
  main()