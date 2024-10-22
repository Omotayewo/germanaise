import streamlit as st
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from googletrans import Translator
from transformers import pipeline

# UI
st.title("GermanAIse")
prompt = st.text_area("Input a prompt") # Obtain user input

# Translate input to german
def translate_to_german(text):
  translator = Translator()
  return translator.translate(text, src='en', dest='de').text

if "german_text" not in st.session_state:
    st.session_state.german_text = ""

if "translation" not in st.session_state:
    st.session_state.translation = ""

if "translated_candidate" not in st.session_state:
    st.session_state.translated_candidate = ""

if st.button("Click to generate text"):
  generator = pipeline("text-generation", model="dbmdz/german-gpt2")
  st.session_state.german_text = generator(translate_to_german(prompt), max_length=50, num_return_sequences=1)[0]["generated_text"]
  
# Display the generated german text
st.write(st.session_state.german_text)

# User input for translation
st.session_state.translation = st.text_area("Enter your translation:")

# Submit button
if st.button("Submit Translation"):
    reference = st.session_state.german_text
    #candidate = st.session_state.translation

    try:
      translator = Translator()
      translated = translator.translate(st.session_state.translation, src='en', dest='de')
      if translated is not None and hasattr(translated, 'text'):
          #st.session_state.translated_candidate = translated.text
          scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # Create a scorer instance
          score = scorer.score(reference, translated.text) # Calculate scores
          st.write(f"Your translation score is: {score['rougeL'].fmeasure:.2f}") # Display feedback
      else:
          st.write("Translation failed or returned None.")
    except Exception as e:
        st.write(f"An error occurred during translation: {e}")
