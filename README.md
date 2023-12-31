 <div align="center">
  <h1>
   ChappyAI (GPT, Pinecone, Web Scraping)</br>
   Simple RAG (Can be improved, see below how)
 </h1>
 </div>

ChappyAI is an intelligent question-and-answer system leveraging state-of-the-art language models and vector search. It utilizes OpenAI's embeddings and Pinecone's vector search to deliver fast and accurate answers.

<div align="center">
  <h2>Talk with Chappy AI: https://chappyai.streamlit.app/</h2>
  <h4> (WARNING: May be down due to Free Tier Inactivity) </h4>
  <img src="chappyai.png" alt="Aisha Logo" width="300" height="300">
</div>

## Table of Contents

- [Introduction](#introduction)
- [HowtoImprove](#HowToImprove)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

ChappyAI is designed to provide efficient retrieval-based question answering by integrating powerful language models with a robust indexing system. The project is modular and built on well-structured components.

## HowToImprove

- Enhanced Retrieval Algorithms:
  1. Advanced Query Understanding: Implement Natural Language Processing (NLP) techniques to better understand the intent and nuances of the user's query, allowing for more accurate and relevant retrieval.
  2. Semantic Search: Move beyond keyword-based retrieval to semantic search, which understands the meaning behind the query and retrieves documents based on conceptual relevance, not just keyword matching.
  3. Machine Learning Optimized Retrieval: Utilize machine learning algorithms to continually learn and improve from past retrieval successes and failures, refining the retrieval process over time.
  4. Context-Aware Retrieval: Develop systems that take into account the broader context of the conversation or document, ensuring that the retrieved information is not just relevant to the query but also to the overall context.
  5. Personalized Retrieval: Tailor the retrieval process to individual users based on their past interactions, preferences, or the specific task at hand, making the results more personalized and relevant.
- Dynamic Knowledge Updating: Integrate systems for real-time updating of the knowledge base to keep information current.
- Contextual Understanding: Improve context understanding to enhance the relevance of retrieved documents to the query.
- Diverse Data Sources: Expand and diversify the range of data sources to enhance the model's coverage and accuracy.
- Efficient Indexing: Optimize indexing strategies for faster and more efficient retrieval of information.
- Better Re-ranking Mechanisms: Implement sophisticated re-ranking algorithms to ensure the best documents are chosen.
- Cross-Lingual Capabilities: Enhance cross-lingual retrieval to support a wider range of languages and dialects.
- User Feedback Loop: Integrate a system for user feedback to continually refine and improve retrieval relevance.
- Robustness to Noise: Improve the model's robustness to noisy or irrelevant data in the retrieval process.

## Requirements

- Python 3.7 or higher
- OpenAI API key
- Pinecone API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dimitri-sky/ChappyAI.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables for OpenAI and Pinecone API keys.

## Usage

1. Run `app.py` to start the Streamlit server:
   ```bash
   streamlit run app.py
   ```

2. Navigate to the URL provided in the terminal to interact with the ChappyAI interface.

## Contributing

If you would like to contribute to this project, please fork the repository, create a new branch for your feature, and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
