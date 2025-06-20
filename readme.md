### Prerequisites

- Python 3.10+
- Google Gemini API keys (3 recommended for rotation)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/WebSockets_FastApi.git
cd WebSockets_FastApi/medium_article
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Create a .env file in the root directory with your Gemini API keys:
```bash
GEMINI_API_KEY_1=your_first_api_key
GEMINI_API_KEY_2=your_second_api_key
GEMINI_API_KEY_3=your_third_api_key
```

### Running the Application

1. Open a terminal and run:
```bash
uvicorn app:app --reload --port 8000
```

2. Open another terminal and run:
```bash
streamlit run streamlit_app.py
```

3. Launch the index.html or index_nolandmarks.html



