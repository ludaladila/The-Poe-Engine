import streamlit as st
import torch
import os
from unsloth import FastLanguageModel
from PIL import Image
import base64
def load_model(model_path):
    """
    Load the saved fine-tuned model
    
    Args:
        model_path: Path to the saved model
            
    Returns:
        model, tokenizer
    """
    # Show loading message
    with st.spinner("Loading model... This may take a minute..."):
        # Load the model with same parameters used during training
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            dtype=None,  # Auto-detect
            load_in_4bit=True
        )
    
    return model, tokenizer

def inference(prompt, model, tokenizer, max_new_tokens=2048, temperature=0.7, top_p=0.92, top_k=50, repetition_penalty=1.05):
    """
    Run inference with the fine-tuned model
    
    Args:
        prompt: Input text prompt
        model: The fine-tuned model
        tokenizer: The tokenizer
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
        
    Returns:
        Generated text
    """
    # Format the prompt properly
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Display a spinner while generating
    with st.spinner("Channeling the spirit of Edgar Allan Poe..."):
        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate the response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=False
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part (after "### Response:")
        try:
            response = generated_text.split("### Response:")[1].strip()
        except IndexError:
            response = generated_text  # Fallback if the split doesn't work
    
    return response

def add_bg_from_base64(base64_string):
    """
    Adds a background image in base64 format to the Streamlit app
    """
    bin_str = base64.b64decode(base64_string)
    return bin_str

def main():
    # Set page config
    st.set_page_config(
        page_title="Edgar Allan Poe Text Generator",
        page_icon="🪦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #1a1a1a;
        color: #d0d0d0;
    }
    
    .stButton>button {
        color: #1a1a1a;
        background-color: #9c7c38;
        border: 2px solid #9c7c38;
    }
    .stButton>button:hover {
        color: #9c7c38;
        background-color: #1a1a1a;
        border: 2px solid #9c7c38;
    }
    h1, h2, h3 {
        color: #9c7c38;
    }
    .story-container {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 5px;
        border-left: 5px solid #9c7c38;
        font-family: 'Times New Roman', Times, serif;
        line-height: 1.6;
    }
    .sidebar .sidebar-content {
        background-color: #1a1a1a;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("The Raven's Quill")
    st.subheader("Generate Edgar Allan Poe-style Stories with AI")
    

    # Set model path
    model_path = "model"
    
    # Generation parameters
    # Sidebar introduction
    st.sidebar.markdown("## 🪶 About This App")
    st.sidebar.markdown(
    """
    **The Raven's Quill** lets you generate short stories in the style of **Edgar Allan Poe**.
    📝 **How to use:**
    - Select the genre, atmosphere, themes, and character details
    - Add special elements like ravens or unreliable narrators
    - Click **Generate Poe Story** to receive your tale of terror
  
    """
    )

    max_new_tokens = st.sidebar.slider("Max Tokens", 500, 4096, 2048, 100,
                                     help="Maximum length of generated text")
    # Story characteristics input
    st.markdown("## Craft Your Tale of Terror")
    
    
    genre = st.selectbox("Genre", [
            "Choose options",
            "Horror", "Mystery", "Gothic", "Psychological", 
            "Supernatural", "Thriller", "Satire", "Essay", "Detective Fiction"
        ])
        
    atmosphere = st.selectbox("Atmosphere", [
            "Choose options",
            "Gloomy", "Tense", "Mysterious", "Oppressive", "Dream-like",
            "Eerie", "Foreboding", "Melancholic", "Sinister", "Suspenseful"
        ])
        
    themes = st.multiselect("Themes", [
            "Choose options",
            "Isolation", "Madness", "Death", "Fear", "Revenge", 
            "Guilt", "Supernatural", "Obsession", "Doppelganger", "Loss",
            "Premature burial", "Unreliable narrator", "Cosmic horror"
        ])

    mood = st.selectbox("Mood", [
            "Choose options",
            "Dread", "Mystery", "Paranoia", "Melancholy", "Terror",
            "Psychological Tension", "Despair", "Anxiety", "Resignation"
        ])
    
    
    characters = st.text_input(
            "Character?",
        value="A lonely scholar",
        help="Describe the protagonist (e.g. 'A lonely scholar obsessed with ancient texts')"
        )

    setting = st.text_input(
        "Scene",
        value="An abandoned mansion on the outskirts of a desolate village",
        help="Describe the main setting or location of the story"
        )

        
    
    # Additional options
    st.markdown("## Additional Elements")
    
    col3, col4 = st.columns(2)
    
    with col3:
        include_raven = st.checkbox("Include a raven", value=False)
        include_heart = st.checkbox("Include a beating heart motif", value=False)
        include_burial = st.checkbox("Include premature burial elements", value=False)
    
    with col4:
        first_person = st.checkbox("First-person narration", value=True)
        unreliable_narrator = st.checkbox("Unreliable narrator", value=False)
        twist_ending = st.checkbox("Twist ending", value=False)
    
    # Construct the prompt
    prompt_elements = [
        f"Write a short story in the style of Edgar Allan Poe with the following characteristics:",
        f"- Genre: {genre}",
        f"- Characters: {characters}",
        f"- Setting: {setting}",
        f"- Atmosphere: {atmosphere}",
        f"- Mood: {mood}",
        f"- Themes: {', '.join(themes)}"
    ]
    
    # Add optional elements to the prompt
    optional_elements = []
    if include_raven:
        optional_elements.append("- Include a raven as a symbolic element")
    if include_heart:
        optional_elements.append("- Include a beating heart motif")
    if include_burial:
        optional_elements.append("- Include elements of premature burial or being trapped")
    if first_person:
        optional_elements.append("- Use first-person narration")
    if unreliable_narrator:
        optional_elements.append("- The narrator should be unreliable")
    if twist_ending:
        optional_elements.append("- End with a twist or revelation")
    
    if optional_elements:
        prompt_elements.extend(optional_elements)
    
    prompt = "\n".join(prompt_elements)
    
    # Show the constructed prompt
    with st.expander("View Prompt"):
        st.code(prompt)
    
    # Generate button
    if st.button("Generate Poe Story", key="generate"):
        if not os.path.exists(model_path):
            st.error(f"Model path '{model_path}' does not exist. Please provide a valid path.")
        else:
            # Load model
            model, tokenizer = load_model(model_path)
            
            # Generate text
            generated_text = inference(
                prompt, model, tokenizer, 
                max_new_tokens=max_new_tokens,
                temperature = 0.7,
                top_p = 0.9,
                top_k = 40,
                repetition_penalty = 1.05
            )
            
            # Display the generated story
            st.markdown("## Your Tale of Terror")
            st.markdown(f"<div class='story-container'>{generated_text}</div>", unsafe_allow_html=True)
            
            # Add download button for the story
            st.download_button(
                label="Download Story",
                data=generated_text,
                file_name=f"poe_story_{genre.lower().replace(' ', '_')}.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Created with LLaMA-3.1 fine-tuned on Edgar Allan Poe's works. "
        "*Once upon a midnight dreary, while you pondered, weak and weary...*"
    )

if __name__ == "__main__":
    main()