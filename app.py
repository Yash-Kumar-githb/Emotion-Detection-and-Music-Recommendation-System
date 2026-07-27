import streamlit as st          # Creates the web UI.
import torch                    # PyTorch library.
import torch.nn as nn           # Neural network layers.
from torchvision import transforms   # Image preprocessing.
from PIL import Image           # Opens uploaded images.

# ----------------------------------
# Image Transform
# ----------------------------------

transform = transforms.Compose([

    transforms.Resize((48,48)),      # Resize image to 48×48 (same as training).
    transforms.ToTensor(),           # Convert image to tensor and scale pixels to 0–1.
    transforms.Normalize(            # Normalize image using training values.
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])


# ----------------------------------
# CNN Model
# ----------------------------------

class CNN(nn.Module):                 # Same CNN architecture used during training.

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)

        )

        self.fc = nn.Sequential(

            nn.Flatten(),              # Convert feature maps into a 1D vector.
            nn.Linear(64*6*6,512),     # Fully connected layer.
            nn.ReLU(),
            nn.Linear(512,7)           # 7 output neurons (7 emotions).

        )

    def forward(self,x):               # Defines how data flows through the network.

        x = self.conv_layers(x)
        x = self.fc(x)

        return x


    
# ----------------------------------
# Load Model
# ----------------------------------

@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load("mood_model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()






# ----------------------------------
# Class Names
# ----------------------------------

classes = [       # Maps output index (0–6) to emotion name.
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]
# ----------------------------------
## Youtube videos 
# ----------------------------------

import requests
import re


def get_youtube_videos(query):

    
    url = f"https://www.youtube.com/results?search_query=bollywood {query} song"

    response = requests.get(url, timeout=10)

    video_ids = re.findall(
        r"watch\?v=(\S{11})",
        response.text
    )

    
    return video_ids[:1]    # return the list of  1 video IDs found


def show_music(emotion):

    if emotion == "Happy":
        query = "happy hindi songs"

    elif emotion == "Sad":
        query = "sad hindi songs"

    elif emotion == "Angry":
        query = "energetic hindi songs"

    elif emotion == "Fear":
        query = "calm relaxing songs"

    elif emotion == "Disgust":
        query = "feel good songs"

    elif emotion == "Neutral":
        query = "lofi songs"

    elif emotion == "Surprise":
        query = "party songs"
        
    st.subheader("🎵 Recommended Song")


    videos = get_youtube_videos(query)


    for vid in videos:

        url = f"https://www.youtube.com/watch?v={vid}"

        st.video(url)     



# ----------------------------------
# Streamlit UI
# ----------------------------------

st.title("😊 Emotion Detection")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")              # Open uploaded image using Pillow.[PIL]



    st.image(image, caption="Uploaded Image", width=250)            #displays the uploaded image on the webpage.

    if st.button("Predict"):

        img = transform(image)
        img = img.unsqueeze(0)                                       # Add batch dimension: (3,48,48) → (1,3,48,48)

        with torch.no_grad():

            output = model(img)

            _, prediction = torch.max(output, 1)    ## axis =1
            prediction.item()

            emotion = classes[prediction]

        st.success(f"Predicted Emotion : {emotion}")

        
        show_music(emotion)






