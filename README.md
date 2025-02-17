# SemNovel

A New Approach to Detecting Semantic Novelty of Biomedical Publications using Embeddings of Large Language Models

## Visualization

The prototype of our proposed interface as well as a sample dataset are available at https://clinicalnlp.org/SemNovel.

## Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/BIDS-Xu-Lab/SemNovel.git
cd SemNovel
pip install -r requirements.txt
```

## File Structure

```bash
SemNovel/
├── 1_dataset_prepare/         # Prepare datasets and pre-processing
├── 2_semantic_space/          # Construct semantic space from given papers
├── 3_semnovel/                # Calcuate SemNovel score
├── requirements.txt           # Python dependencies
```

For questions or support, please open [an issue](./issues).
