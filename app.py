# Import necessary libraries
import streamlit as st
from pathlib import Path
import google.generativeai as genAI
from api_key import api_key  # Ensure your API key is stored securely

# Configure the generative AI with the provided API key
genAI.configure(api_key=api_key)

# Define generation configuration settings
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,  # Max tokens in the response
    "response_mime_type": "text/plain",  # Response format
}

# Initialize the generative model
model = genAI.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Set up the Streamlit page with a custom title and icon
st.set_page_config(page_title="Vital Image Analytics", page_icon=":robot:")

# Display the logo
st.image("logo.jpeg", width=200)

# Set the title and subtitle of the app
st.title("🧠📊 MedivizAI - Smarter Image Insights")
st.subheader("An application to help users analyze medical images")

# File uploader for users to upload medical images
uploaded_file = st.file_uploader("Upload the medical image for analysis", type=["png", "jpg", "jpeg"])

# Button to submit the uploaded file for analysis
submit_button = st.button("Generate the Analysis")

# Define the system prompt for the generative model
system_prompt = """
As a highly skilled medical practitioner specializing in image analysis, you are tasked with providing accurate and detailed insights into the medical images submitted by users. Your role is crucial in aiding healthcare providers with valuable information that may assist in diagnosis, treatment planning, and further clinical decisions.

Your Responsibilities:

1. **Detailed Analysis**: Carefully examine each image, focusing on identifying any abnormal findings or irregularities. Consider variations in anatomy, signs of pathology, and other clinically significant features that may be present.
2. **Findings Report**: Document all observed anomalies or signs of disease. Clearly articulate the findings with medical terminology.
3. **Recommendations and Next Steps**: Suggest further diagnostic tests (e.g., MRI, CT scans) to clarify the diagnosis or rule out other conditions.
4. **Treatment Suggestions**: If appropriate, recommend treatment options or interventions, such as medication, therapy, or specialist referrals.

Important Notes:
- **Scope of Response**: Only respond if the image pertains to human health issues.
- **Clarity of Image**: If image quality impedes analysis, note the limitations and recommend a higher-quality image.
- **Disclaimer**: Accompany your analysis with this disclaimer: "Consult with a licensed medical professional before making any clinical decisions."
- **Tone and Professionalism**: Maintain a professional tone in all responses.
"""

# Function to handle file analysis
def analyze_image(image_data):
    """Analyze the uploaded image using the generative AI model."""
    # Prepare the image for analysis
    files = [{"mime_type": "image/jpeg", "data": image_data}]

    # Define the prompt structure
    prompt_parts = [
        files[0],  # User's uploaded image
        system_prompt,  # System prompt for the analysis
    ]

    # Start a chat session with the generative model
    chat_session = model.start_chat(history=[
        {
            "role": "user",
            "parts": [
                files[0],  # Image file part
                "What is going on in this image?",
            ],
        }
    ])

    # Send a message to the chat session with the provided query or instructions
    response = chat_session.send_message("Please analyze the image and provide insights.")

    # Return the AI model's response
    return response.text

# Process the uploaded image when the user clicks the submit button
if submit_button and uploaded_file:
    image_data = uploaded_file.getvalue()
    
    # Perform the analysis
    analysis_result = analyze_image(image_data)
    
    # Display the result in the Streamlit app
    st.write("### Analysis Results:")
    st.write(analysis_result)
else:
    st.write("Please upload an image and click 'Generate the Analysis' to proceed.")
