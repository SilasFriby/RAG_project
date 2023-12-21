import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")

st.title('Question Answering with Starling-LM-7B-alpha')

user_input = st.text_input("Enter your question:")

if user_input:
    inputs = tokenizer(user_input, return_tensors='pt')
    reply = model.generate(**inputs)
    answer = tokenizer.decode(reply[0], skip_special_tokens=True)

    st.write(answer)
