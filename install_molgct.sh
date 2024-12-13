#!/bin/bash

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Starting MolGCT Installation Process...${NC}"

# Step 1: Install Anaconda
echo -e "${YELLOW}Step 1: Installing Anaconda...${NC}"
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh -b -p /content/anaconda3
rm Anaconda3-2023.09-0-Linux-x86_64.sh

# Add conda to path and initialize
export PATH="/content/anaconda3/bin:$PATH"
eval "$(/content/anaconda3/bin/conda shell.bash hook)"

# Verify conda installation
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Conda installation failed. Exiting.${NC}"
    exit 1
fi
echo -e "${GREEN}Conda successfully installed and initialized${NC}"

# Step 2: Create environment file
echo -e "${YELLOW}Step 2: Creating environment file...${NC}"
cat > molgct_env_linux.yaml << 'ENVEOF'
name: molgct
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.7
  - pip=20.2.4
  - numpy=1.16.0
  - pandas=1.1.3
  - scipy=1.3.1
  - scikit-learn=0.21.3
  - matplotlib
  - seaborn
  - tqdm
  - dill
  - joblib
  - nltk
  - pip:
    - torch==1.6.0
    - torchvision==0.7.0
    - torchtext==0.6.0
    - rdkit==2020.09.1
    - selfies==1.0.4
    - SmilesPE==0.0.3
    - moses
    - spacy==3.1.1
    - spacy-legacy==3.0.8
    - blis==0.7.4
    - catalogue==2.0.4
    - cymem==2.0.5
    - murmurhash==1.0.5
    - pathy==0.6.0
    - preshed==3.0.5
    - srsly==2.4.1
    - thinc==8.0.8
    - typer==0.3.2
    - wasabi==0.8.2
ENVEOF

# Step 3: Create conda environment
echo -e "${YELLOW}Step 3: Creating conda environment...${NC}"
conda env create -f molgct_env_linux.yaml

if [ $? -ne 0 ]; then
    echo -e "${RED}Environment creation failed. Exiting.${NC}"
    exit 1
fi

# Step 4: Activate environment and verify packages
echo -e "${YELLOW}Step 4: Activating environment and verifying packages...${NC}"
source activate molgct
if [ $? -ne 0 ]; then
    echo -e "${RED}Environment activation failed. Exiting.${NC}"
    exit 1
fi

# Additional pip installs to ensure correct versions
pip install --no-deps torchtext==0.6.0
pip install --no-deps torch==1.6.0
pip install --no-deps torchvision==0.7.0

# Step 5: Verify installation
echo -e "${YELLOW}Step 5: Verifying installation...${NC}"
python -c "
import sys
print('Python version:', sys.version)
import torch
print('PyTorch version:', torch.__version__)
import torchtext
print('torchtext version:', torchtext.__version__)
import rdkit
print('RDKit version:', rdkit.__version__)
import selfies
print('SELFIES version:', selfies.__version__)
import SmilesPE
print('SmilesPE imported successfully')
import spacy
print('spaCy version:', spacy.__version__)
print('All key packages imported successfully!')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Package verification failed. Please check the error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}To activate the environment, run:${NC}"
echo -e "${GREEN}source activate molgct${NC}"
