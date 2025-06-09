import streamlit as st
import easyocr
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from PIL import Image
import numpy as np

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# OCR Function
def extract_text(image, output_file):
    # Convert uploaded image to a NumPy array (EasyOCR-compatible format)
    image = Image.open(image).convert("RGB")  # Ensure RGB format
    image_np = np.array(image)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_np, detail=0)  # Perform OCR
    extracted_text = " ".join(results)

    # Save extracted text to a file
    with open(output_file, 'w') as file:
        file.write(extracted_text)
    return extracted_text, output_file

# Text preprocessing and summarization functions
def preprocess_text(text):
    sentences = sent_tokenize(text)
    words = [word.lower() for sentence in sentences for word in word_tokenize(sentence)]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    return words, sentences

def get_pos_tags(words):
    return nltk.pos_tag(words)

def get_key_sentences(sentences, pos_tags):
    nouns_verbs_adjectives = [word for word, tag in pos_tags if tag.startswith('N') or tag.startswith('V') or tag.startswith('J')]
    word_freq = Counter(nouns_verbs_adjectives)
    key_sentences = [(sentence, sum(word_freq[word] for word in word_tokenize(sentence.lower()) if word in word_freq)) for sentence in sentences]
    key_sentences.sort(key=lambda x: x[1], reverse=True)
    return [sentence for sentence, score in key_sentences[:3]]  # Top 3 sentences

def summarize_text(text):
    words, sentences = preprocess_text(text)
    pos_tags = get_pos_tags(words)
    key_sentences = get_key_sentences(sentences, pos_tags)
    return ' '.join(key_sentences)

# Streamlit UI
st.title("Legal Text Summarization System")

# Option to perform OCR or direct summarization
option = st.radio("Choose an action:", ("Upload Image and Summarize", "Upload Text File and Summarize"))

if option == "Upload Image and Summarize":
    # Upload JPG Image
    uploaded_image = st.file_uploader("Upload a JPG image", type=["jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        output_txt_file = "ocr_output.txt"

        # Perform OCR
        with st.spinner("Performing OCR..."):
            extracted_text, file_path = extract_text(uploaded_image, output_txt_file)
            st.success("OCR Completed!")
            st.write("Extracted Text:")
            st.text(extracted_text)

        # Option to download the generated text file
        st.download_button(
            label="Download Extracted Text File",
            data=extracted_text,
            file_name="ocr_output.txt",
            mime="text/plain",
        )

        # Summarization Section
        st.subheader("Text Summarization")
        if st.button("Summarize Extracted Text"):
            with st.spinner("Summarizing Text..."):
                summary = summarize_text(extracted_text)
                st.success("Summarization Completed!")
                st.write("Summary:")
                st.text(summary)

                # Option to download the summarized text
                st.download_button(
                    label="Download Summarized Text File",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain",
                )

elif option == "Upload Text File and Summarize":
    # Upload a Text File
    uploaded_text_file = st.file_uploader("Upload a Text File", type=["txt"])
    if uploaded_text_file:
        text = uploaded_text_file.read().decode("utf-8")
        st.write("Uploaded Text:")
        st.text(text)

        # Summarization Section
        if st.button("Summarize Uploaded Text"):
            with st.spinner("Summarizing Text..."):
                summary = summarize_text(text)
                st.success("Summarization Completed!")
                st.write("Summary:")
                st.text(summary)

                # Option to download the summarized text
                st.download_button(
                    label="Download Summarized Text File",
                    data=summary,
                    file_name="summary.txt",
                    mime="text/plain",
                )
