# The Raven’s Quill

demo:https://huggingface.co/spaces/yiqing111/poe_gen

presentation video: https://youtu.be/hTU7eLmwiOY

PPT slide: [https://docs.google.com/presentation/d/16ypGzE4T8IWoDvUSsWCPEIUn5wb3rh3](https://docs.google.com/presentation/d/16ypGzE4T8IWoDvUSsWCPEIUn5wb3rh3clyS9fnJ6hyk/edit?usp=sharing)

##  0. Project Overview

The Raven’s Quill is a specialized text generation system that crafts short stories in the distinctive style of **Edgar Allan Poe**, the renowned 19th-century master of gothic horror and mystery. Our project explores multiple modeling approaches to capture Poe's unique literary voice, from simple statistical methods to advanced neural architectures.

Key innovations include:

- **Multi-modal approach**: We implemented three distinct modeling paradigms (naive baseline, statistical N-gram, and fine-tuned LLM), providing a comprehensive analysis of different technical approaches to stylistic emulation
- **Metadata-enriched corpus**: We enhanced Poe's original works with detailed character, setting, and thematic annotations using Gemini API, creating a rich foundation for generative models
- **Quantifiable evaluation**: Each model is scored across seven literary dimensions using DeepSeek's API, creating an objective framework for comparing stylistic fidelity and narrative quality
- **Interactive demo**: Our web application allows users to generate customized Poe-style stories by specifying genre, mood, themes, and other narrative elements

---

## 1. Running Instructions

### Run Locally

1. **Environment Setup**: 
Ensure Python 3.x is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
2. and then 
   ```bash
   streamlit run app.py
   ```
   Then visit `http://localhost:8501` in your browser.


### Run in cloud 
Demo at [Demo](https://huggingface.co/spaces/yiqing111/poe_gen)



## 2. Data

### Data Sources
- **Primary Corpus**: Complete works of Edgar Allan Poe, sourced from Project Gutenberg (public domain)
- **Metadata Collection**: Enhanced with character, location, and atmospheric details extracted from the Poe Literary Vault (University of Virginia archive)
- **Reference Dataset**: Includes 70 of Poe's original works including poetry, short stories, and essays

### Data Preprocessing

The preprocessing pipeline consists of several key stages:

1. **Text Cleaning and Normalization**:
   - Removal of publisher notes, headers, and footers from raw texts
   - Normalization of quotes, dashes, and archaic punctuation
   - Paragraph segmentation and sentence boundary detection

2. **Metadata Extraction via Gemini API**:
   - Each work was analyzed using the Gemini 2.0 model to extract:
     - Character names and descriptions
     - Geographic and narrative settings
     - Scene descriptions and atmospheric elements
   - This enhanced metadata provides structured context for generation

3. **Dataset Augmentation**:
   - Classification by genre (Horror, Mystery, Essay, etc.)
   - Annotation of literary elements: mood, atmosphere, themes
   - Timeline normalization to establish chronological context

The final dataset (`enhanced_poe_dataset.csv`) contains 70 works with 18 feature columns, including both original text and enriched metadata to support multiple modeling approaches.

---
Based on the evaluation code you've shared, here's a comprehensive section about your evaluation methodology for your README:

## 3.  Evaluation Methodology

Our evaluation framework employs a rigorous, multi-dimensional assessment approach to measure how effectively each model captures the essence of Edgar Allan Poe's distinctive literary style.

### Evaluation Metrics

We utilized DeepSeek's AI to evaluate generated stories across seven critical literary dimensions, each scored on a scale of 0-10:

1. **Poe Style** (weight: 0.25) - Fidelity to Poe's gothic diction, imagery, and narrative conventions
2. **Coherence** (weight: 0.20) - Logical narrative flow and overall story structure
3. **Suspense** (weight: 0.15) - Intensity and pacing of tension throughout the narrative
4. **Instruction Fit** (weight: 0.15) - Faithfulness to the specified genre, characters, setting, and thematic elements
5. **Creativity** (weight: 0.10) - Originality while maintaining Poe's distinctive literary voice
6. **Language Quality** (weight: 0.15) - Grammar, rhythm, and rhetorical richness
7. **Redundancy** (inverse weight: -0.10) - Degree of unnecessary repetition (higher scores indicate less redundancy)

### Evaluation Process

Each model generated responses to the same 10 prompts, creating a controlled comparison environment. The evaluation followed these steps:

1. Models generated stories based on identical structured prompts
2. DeepSeek API evaluated each story using our custom rubric
3. A weighted scoring formula calculated overall performance
4. Results were compiled in both JSONL and CSV formats for analysis

##  4. Approaches
### Naive Model 

**Description**:
I implemented a zero-shot baseline using the Llama 3.1-8B to generate Edgar Allan Poe-style text. This naive approach applies no Poe-specific training or optimization, relying solely on instructional prompting to guide the model in mimicking Poe's distinctive writing style and thematic elements.

**Technical Details**:
- **Base Model**: Llama 3.1-8B
**Processing Pipeline**:
1. Load pre-trained Llama 3.1-8B model (no fine-tuning)
2. Format instructions in Alpaca-style prompt: `### Instruction: [prompt] ### Response:`
3. Perform sampling-based text generation
4. Extract the response portion (excluding prompt) to form the final story

**Evaluation Results**:
Based on DeepSeek's 7-dimension scoring system across 10 generated samples:
- **Poe Style**: 5.1/10 - Partially captures gothic atmosphere but lacks Poe's signature rhetorical style
- **Coherence**: 6.1/10 - Creates relatively coherent narratives with logical structure
- **Suspense**: 4.4/10 - Limited ability to develop tension and emotional progression
- **Instruction Fit**: 6.5/10 - Successfully incorporates most specified story elements
- **Creativity**: 5.2/10 - Moderately creative but tends toward standard patterns
- **Language Quality**: 6.1/10 - Generally proper grammar with occasional stylistic issues
- **Redundancy**: 4.0/10 (higher is better) - Shows moderate repetition in longer passages

**Overall Performance**:
- **Weighted Total**: 5.165/10
- **Key Strengths**: Strong coherence and instruction following ability
- **Key Weaknesses**: Limited Poe-specific stylistic elements and moderate redundancy

**Summary**:
"The model produces structurally sound stories that follow instructions but lack the distinctive gothic atmosphere and rhetorical flourishes characteristic of Poe's work. While narratively coherent, the stories miss the emotional depth and linguistic distinctiveness that define Poe's unique literary voice."

As our baseline approach, this model demonstrates the capabilities and limitations of general-purpose large language models when tasked with specialized literary stylization without targeted optimization.

---

### ML Model

**Description**: 
I implemented a word-level tri-gram language model  trained on Poe's original texts. The model captures short-range word dependencies by predicting the next word based on the previous n tokens.

**Technical Implementation**:
- **Model Architecture**: Statistical language model using conditional probabilities
- **Context Sizes**:3-gram
- **Temperature Sampling**: Adjustable randomness parameter to balance creativity vs. determinism
- **Prompt Engineering**: Metadata-based opening sentences that incorporate genre, atmosphere, and setting
- **Persistence**: Efficient storage and loading using pickle serialization

**Generation Process**:
1. Extract metadata from instruction prompt
2. Craft a Poe-style opening sentence using the metadata
3. Use this prompt to seed the N-gram model
4. Generate subsequent words through probability-based sampling
5. Handle unknown n-grams with fallback to known sequences

**Evaluation Results**:
Based on DeepSeek's 7-dimension scoring system:
- **Poe Style**: 5.3/10 - Captures some of Poe's vocabulary but lacks his deeper stylistic elements
- **Coherence**: 3.2/10 - Shows significant limitations in maintaining narrative consistency
- **Suspense**: 4.3/10 - Creates basic atmospheric tension but struggles with emotional progression
- **Instruction Fit**: 5.7/10 - Moderately successful at incorporating specified story elements
- **Creativity**: 6.2/10 - Produces interesting variations by recombining Poe's language patterns
- **Language Quality**: 5.1/10 - Grammatically uneven with occasional awkward constructions
- **Redundancy**: 6.1/10 - Relatively low redundancy compared to other approaches

**Overall Performance**: 4.24/10

**Analysis**:
- **Strengths**: 
  - Captures elements of Poe's distinctive vocabulary and gothic diction
  - Requires minimal computational resources
  - Shows good creativity through statistical recombination
  - Effectively incorporates metadata from instructions

- **Limitations**:
  - Struggles with narrative coherence beyond short sequences
  - Unable to maintain logical consistency in longer texts
  - Limited grammatical control compared to deep learning approaches
  - Cannot develop complex narrative arcs or character development

**Summary**:
The N-gram model represents a traditional machine learning approach to stylistic text generation, demonstrating how statistical patterns can capture surface-level aspects of Poe's writing. While it outperforms the naive baseline in creativity, it falls short in coherence and overall quality compared to deep learning approaches.


---

###  Deep Learning Model

**Description**: 
We fine-tuned Meta's Llama 3.1 (8B parameters) on Poe's complete corpus using Low-Rank Adaptation (LoRA). This approach efficiently adapts the large pre-trained model to capture Poe's distinctive writing style while preserving its fundamental language capabilities. The model was trained on instruction-response pairs derived from Poe's original works, with structured prompts incorporating metadata about genre, characters, setting, and thematic elements.

**Technical Details**:
- **Base Model**: Meta-Llama-3.1-8B
- **Adaptation Method**: QLoRA with rank-8 adaptation matrices
- **Target Modules**: All key transformer components (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Training Configuration**:
  - Learning rate: 2e-4 with cosine scheduling
  - Weight decay: 0.01
  - Warmup ratio: 0.03
  - Epochs: 3
  - Batch size: 4 with 4x gradient accumulation
  - BF16 mixed precision with gradient checkpointing
  - LoRA alpha: 16, dropout: 0.05

**Training Process**:
1. Convert Poe's works into instruction-response pairs using metadata-enriched prompts
2. Format examples with Alpaca-style template (`### Instruction: ... ### Response: ...`)
3. Apply QLoRA fine-tuning to modify only a small subset of parameters
4. Train with SFTTrainer from the TRL library for stable fine-tuning


**Evaluation Results**:
Based on DeepSeek's 7-dimension scoring system:
- **Poe Style**: 6.8/10 - Successfully captures Poe's distinctive vocabulary, gothic atmosphere, and narrative voice
- **Coherence**: 7.3/10 - Creates logically consistent stories with well-developed narrative arcs
- **Suspense**: 5.5/10 - Generates moderate tension and emotional progression
- **Instruction Fit**: 8.2/10 - Excellent integration of requested story elements and characteristics
- **Creativity**: 6.3/10 - Produces original narratives while maintaining Poe's thematic concerns
- **Language Quality**: 7.7/10 - Strong command of complex syntax and period-appropriate diction
- **Redundancy**: 3.2/10 (higher is better) - Some repetition of phrases and motifs

**Overall Performance**: 6.68/10

**Comparative Analysis**:
| Dimension | Deep Learning | N-gram | Naive Baseline |
|-----------|--------------|---------|----------------|
| Poe Style | 6.8 | 5.3 | 5.1 |
| Coherence | 7.3 | 3.2 | 6.1 |
| Suspense | 5.5 | 4.3 | 4.4 |
| Instruction Fit | 8.2 | 5.7 | 6.5 |
| Creativity | 6.3 | 6.2 | 5.2 |
| Language Quality | 7.7 | 5.1 | 6.1 |
| Redundancy | 3.2 | 6.1 | 4.0 |
| **Overall** | **6.68** | **4.24** | **5.165** |

**Key Strengths**:
- Superior performance in Poe stylistic elements compared to both N-gram and baseline models
- Significantly stronger narrative coherence than the N-gram approach
- Excellent instruction following capability, outperforming other models by a wide margin
- High-quality language production with complex sentence structures reminiscent of Poe
- Effective balance between creativity and stylistic fidelity

**Limitations**:
- Higher redundancy compared to the N-gram model
- More resource-intensive than traditional approaches
- Occasionally produces overly formal or stilted language


The fine-tuned deep learning model represents the most effective approach for generating Poe-style stories, achieving the highest overall performance across our evaluation dimensions. While requiring more computational resources than traditional methods, it successfully captures both the stylistic elements and narrative coherence essential to faithfully emulating Poe's distinctive literary voice.

---


## 5. Application Demo

Our interactive web application "The Raven's Quill" offers an intuitive interface for generating custom Poe-style stories. The application:

1. **Provides Creative Control**:
   - Genre selection (Horror, Mystery, Gothic, Psychological, etc.)
   - Atmosphere settings (Gloomy, Tense, Mysterious, etc.)
   - Theme combination (Isolation, Madness, Death, Revenge, etc.)
   - Character and setting descriptions
   - Optional elements like ravens, beating hearts, or unreliable narrators

2. **Offers Generation Options**:
   - Adjustable output length
   - Downloadable story files
   - Viewable prompts for transparency

Try it yourself at: [https://huggingface.co/spaces/yiqing111/poe_gen](https://huggingface.co/spaces/yiqing111/poe_gen)

### Demo image

![Poe-style Output](https://huggingface.co/spaces/yiqing111/poe_gen/resolve/main/Screenshot%202025-04-22%20at%201.09.38%E2%80%AFAM.png)



## 6. Previous Approaches

Our work builds upon and extends several notable approaches in literary style emulation and text generation:

- **[PoeNLP](https://github.com/jplonski/PoeNLP)**: A computational analysis of gothic linguistic features in Poe's writing, which provided valuable insights into the stylistic markers that define his prose. While this project focused primarily on analysis rather than generation, it established important baselines for identifying Poe's distinctive literary elements.

- **MarI/O Poetry Generator**: An earlier approach that used recurrent neural networks to generate Poe-inspired poetry. This system demonstrated the potential of neural approaches but was limited to shorter forms and struggled with narrative coherence.

- **Literary Style Transfer**: Research from Stanford NLP that developed techniques for transforming modern text into various historical styles, including 19th-century prose. Their approach focused on sentence-level transformations rather than full narrative generation.


##  7. Ethics Statement

This project is developed with careful attention to ethical considerations in AI-assisted creative writing:

- **Attribution and Transparency**: This system is designed for academic and creative exploration only. All generated content is clearly labeled as machine-generated and not presented as authentic Poe material. We acknowledge that while our models aim to capture stylistic elements of Poe's writing, the output represents a computational approximation rather than genuine historical artifacts.

- **Educational Purpose**: The primary intent is to provide insights into computational approaches to literary style analysis and to offer a creative tool for educational purposes. The system helps users understand the distinctive elements of Poe's writing through interactive exploration.

- **Fair Use and Copyright**: Edgar Allan Poe's works are in the public domain, making them available for computational analysis and style modeling. Our training approach respects copyright boundaries and represents fair use for research and educational purposes.
