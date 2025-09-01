# Comparative Parser Setup

This guide will help you set up and run the comparative parser using **spaCy** and **Stanza**.

```bash
# 1. Save the Script
# Save your code into a file, for example:
comparative_parser.py

# 2. Create a Virtual Environment (optional but recommended)
# Creating a virtual environment avoids conflicts with other Python projects.

# Create virtual environment
python3 -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

# 3. Install Dependencies
# Run the following commands in your terminal:

pip install spacy stanza pandas

# Download spaCy English model
python -m spacy download en_core_web_sm

# Download Stanza English model
python -c "import stanza; stanza.download('en')"

# 4. Run the Script
python comparative_parser.py
