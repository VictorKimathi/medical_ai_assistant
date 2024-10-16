# Import necessary libraries
import streamlit as st
from pathlib import Path
import google.generativeai as genAI
from api_key import api_KEY

# Configure the generative AI with the provided API key
genAI.configure(api_key=api_KEY)

# Configuration for generation settings such as temperature, top_p, and top_k.
# These parameters control the randomness and creativity of the model's responses.
GENERATION_CONFIG = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the generative model with specified name and configuration.
# Adjust the safety settings as necessary by referring to documentation at:
# https://ai.google.dev/gemini-api/docs/safety-settings
MODEL = genAI.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=GENERATION_CONFIG,
)

# Define a function to set up the Streamlit interface
def setup_streamlit():
    st.set_page_config(page_title="Vital Image Analytics", page_icon=":robot:")
    st.image("logo.jpeg", width=200)
    st.title("🧠📊 MedivizAI - Smarter Image Insights")
    st.subheader("Application that helps users analyze medical images")

# Define a function to handle file upload
def upload_image():
    return st.file_uploader("Upload the medical image for analysis", type=["png", "jpg", "jpeg"])

# Define a function to start the chat session with the generative model
def start_chat_session(image_data):
    files = [upload_to_gemini("", mime_type="image/jpeg")]
    image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
    prompt_parts = [image_parts[0], SYSTEM_PROMPT]

    # Start a chat session with the model, providing context with a conversation history
    chat_session = MODEL.start_chat(
        history=[
            {
                "role": "user",
                "parts": [files[0], "What is going on in this image?"],
            },
            {
                "role": "model",
                "parts": [
                    "The image shows two different types of mannequins. On the left, a mannequin wearing a full-body suit, designed to resemble a human. This type of mannequin is often used in fashion displays to show off clothing and accessories. On the right, a more traditional, articulated mannequin with visible joints and a black plastic body. This type of mannequin is commonly used in retail settings to display clothing and accessories.",
                ],
            },
        ]
    )
    return chat_session

# Define a function to get a response from the model
def analyze_image(chat_session):
    response = chat_session.send_message("Analyze this image and provide insights.")
    return response.text

# Define the system prompt
SYSTEM_PROMPT = """
As a highly skilled medical practitioner specializing in image analysis, you are tasked with providing accurate and detailed insights into the medical images submitted by users. Your role is crucial in aiding healthcare providers with valuable information that may assist in diagnosis, treatment planning, and further clinical decisions.

Your Responsibilities:

1. **Detailed Analysis**: Carefully examine each image, focusing on identifying any abnormal findings or irregularities. Consider variations in anatomy, signs of pathology, and other clinically significant features that may be present.

2. **Findings Report**: Document all observed anomalies or signs of disease. Clearly articulate the findings with medical terminology and ensure that all relevant observations are included in the report.

3. **Recommendations and Next Steps**: Based on your analysis, suggest potential next steps. This may include recommending further diagnostic tests, such as MRI, CT scans, or specific blood tests, to clarify the diagnosis or rule out other conditions.

4. **Treatment Suggestions**: If appropriate, recommend possible treatment options or interventions that may benefit the patient, such as medication, physical therapy, or referrals to specialists.

Important Notes:

- **Scope of Response**: Only respond if the image pertains to human health issues. If the image is outside this domain, state that you cannot assist.

- **Clarity of Image**: In cases where the image quality impedes clear analysis, note the limitations and suggest the need for a higher-quality image for accurate assessment.

- **Disclaimer**: Always accompany your analysis with the following disclaimer: 
  "Consult with a licensed medical professional before making any clinical decisions based on this analysis."

- **Tone and Professionalism**: Maintain a professional tone in all responses, ensuring that the insights you provide are easy to understand for both medical practitioners and patients.

Your insights are invaluable in guiding clinical decisions. Please proceed with the analysis based on the image provided and ensure that all responses adhere to the above guidelines.
"""

# Main function to execute the Streamlit app logic
def main():
    setup_streamlit()
    uploaded_file = upload_image()

    if uploaded_file and st.button("Generate the Analysis"):
        image_data = uploaded_file.getvalue()
        chat_session = start_chat_session(image_data)
        response_text = analyze_image(chat_session)
        st.write(response_text)
        st.success("Analysis complete. Consult with a licensed medical professional before taking any action.")

# Run the main function
if __name__ == "__main__":
    main()
