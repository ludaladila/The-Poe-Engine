import pandas as pd
import time
import requests
import json
import os

def load_dataset(file_path):
    """
    Load the CSV dataset into a pandas DataFrame.
    """
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)# Ensure the file exists
    print(f"Loaded {len(df)} records.")
    return df


def create_analysis_prompt(title, text):
    """
    Create a structured prompt for Gemini API to extract characters, locations, and scene.
    """
    # Truncate text to a maximum length to avoid API limits
    max_text_length = 10000
    truncated_text = text[:max_text_length] 
    # Warn if text was truncated
    if len(text) > max_text_length:
        print(f"Warning: Text for '{title}' was truncated from {len(text)} to {max_text_length} characters")

    prompt = f"""
    Analyze the following Edgar Allan Poe story titled "{title}" and extract the following information:
    
    TEXT:
    {truncated_text}
    
    Please provide the following information in a structured format:
    1. CHARACTERS: A comma-separated list of the main characters mentioned in the story.
    2. LOCATIONS: A comma-separated list of the geographic or narrative settings (cities, mansions, castles, forests, etc.).
    3. SCENE: A brief description of the situational backdrop or atmosphere (e.g., "storm-tossed ship interior," "moonlit garden," "deserted crypt," etc.).
    
    Format your response exactly as follows, with just the requested information:
    CHARACTERS: [list of characters]
    LOCATIONS: [list of locations]
    SCENE: [brief scene description]
    """
    return prompt.strip()


def get_gemini_response(prompt):
    """
    Send a prompt to the Gemini API and return the text response.
    """
    headers = {
        "Content-Type": "application/json"
    }
    # Ensure the API key is set
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        headers=headers,
        data=json.dumps(data)
    )
    # Check if the response is successful
    if response.status_code == 200:
        response_json = response.json()
        if "candidates" in response_json and len(response_json["candidates"]) > 0:
            candidate = response_json["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    if "text" in part:
                        return part["text"]
        return "API response did not contain expected text"
    else:
        print(f"API Error: {response.status_code}")
        print(response.text)
        return f"Error: {response.status_code}"



def parse_llm_response(response):
    """
    Parse the LLM output to extract characters, locations, and scene description.
    """
    lines = response.split('\n')
    characters, locations, scene = "", "", ""
    # Initialize empty strings for characters, locations, and scene
    for line in lines:
        line = line.strip()#
        if line.startswith("CHARACTERS:"):
            characters = line.replace("CHARACTERS:", "").strip()
        elif line.startswith("LOCATIONS:"):
            locations = line.replace("LOCATIONS:", "").strip()
        elif line.startswith("SCENE:"):
            scene = line.replace("SCENE:", "").strip()# Ensure we only keep the relevant parts
    
    return characters, locations, scene


def enhance_dataset(df, max_entries=None):
    """
    For each story in the dataset, send to Gemini, parse the response, and save the extracted metadata.
    """
    df['characters'] = ""
    df['locations'] = ""
    df['scene'] = ""
    # Ensure the DataFrame has the necessary columns
    num_entries = len(df) if max_entries is None else min(max_entries, len(df))
    # Print the number of entries to process
    for index in range(num_entries):
        row = df.iloc[index]
        print(f"Processing {index + 1}/{num_entries}: {row['title']}")
        # Skip if the text is empty or NaN
        if pd.isna(row['text']) or row['text'].strip() == "":
            print(f"Skipping {row['title']} - No text available")
            continue
       # Create the analysis prompt
        prompt = create_analysis_prompt(row['title'], row['text'])

        try:
            response = get_gemini_response(prompt)
            characters, locations, scene = parse_llm_response(response)

            # Update the DataFrame
            df.at[index, 'characters'] = characters
            df.at[index, 'locations'] = locations
            df.at[index, 'scene'] = scene

            print(f"✓ Analyzed {row['title']}")
            print(f"Characters: {characters}")
            print(f"Locations: {locations}")
            print(f"Scene: {scene}")
            print("-" * 50)

        except Exception as e:
            print(f"Error analyzing '{row['title']}': {e}")

        # Prevent hitting API rate limits
        time.sleep(2)

    return df


def main():
    """
    Load, enhance, and save the Poe story dataset with extracted metadata.
    """
    # Define input and output file paths
    GEMINI_API_KEY = "..."  
    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

    input_file = 'preprocessed_data.csv'
    output_file = 'enhanced_poe_dataset.csv'

    # Ensure the input file exists
    df = load_dataset(input_file)
    enhanced_df = enhance_dataset(df)
    enhanced_df.to_csv(output_file, index=False)

    print(f"\n✅ Dataset enhancement complete. Saved to '{output_file}'")


# Run the script
if __name__ == "__main__":
    main()
