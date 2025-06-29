# 💼 LinkedIn AI Career Assistant

## 🚀 Overview

**LinkedIn AI Career Assistant** is a powerful AI-powered tool designed to analyze LinkedIn profiles and offer personalized career insights using cutting-edge LLM technology. From profile optimization to job fit analysis and career path forecasting, this tool helps professionals take control of their career trajectory.

---

## ✨ Key Features

### 🔍 Profile Analysis

* Full LinkedIn profile scraping and assessment
* Evaluate professional positioning and trajectory
* Identify skill gaps and content improvement areas
* Score profile quality and relevance

### 🎯 Job Fit Analysis

* Match profiles to job descriptions
* Identify strengths, gaps, and competitive edges
* Suggest application strategies

### 📈 Career Path Prediction

* Map personalized short-, mid-, and long-term goals
* Offer upskilling roadmaps
* Align with industry trends
* Provide executive-level planning

### 🧰 Profile Optimization

* Improve headline and About section
* Suggest relevant skills and content
* Provide ATS optimization tips
* Recommend content strategies

### 💬 AI-Powered Career Chat

* Natural language Q\&A for career help
* Real-time, personalized advice
* Industry-specific coaching
* Analyze job roles or career directions interactively

---

## 🛠️ Tech Stack

| Layer           | Technologies                 |
| --------------- | ---------------------------- |
| **Frontend**    | Streamlit                    |
| **Backend**     | Python, LangChain, LangGraph |
| **AI/LLM**      | Google Gemini API            |
| **Scraping**    | Apify LinkedIn Actor         |
| **Data/Memory** | SQLite, Pydantic, dotenv     |

---

## 📋 Prerequisites

* Python 3.8+
* **API Keys Required:**

  * `GEMINI_API_KEY`
  * *(Optional)* `APIFY_API_TOKEN`, `LINKEDIN_COOKIE` for real scraping

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/linkedin-ai-career-assistant.git
cd linkedin-ai-career-assistant
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
APIFY_API_TOKEN=your_apify_token_here  # Optional
LINKEDIN_COOKIE=your_linkedin_cookie_here  # Optional
```

---

## 📦 Dependencies

`requirements.txt`:

```
streamlit>=1.28.0
sqlite3
python-dotenv>=1.0.0
requests>=2.31.0
langchain>=0.1.0
langchain-google-genai>=1.0.0
langchain-openai>=0.1.0
langgraph>=0.1.0
pydantic>=2.0.0
asyncio
pathlib>=1.0.0
```

---

## 🚀 Usage

### Start the App

```bash
streamlit run cc.py
```

### Open in Browser

Visit: `https://linkedin-enhance-ai-assistant-jsh3a8hqgn49wqunddvcsa.streamlit.app/`

### Analyze a LinkedIn Profile

* Paste profile URL → Get insights in 30–60s

### Use AI Chat Assistant

* Ask career-related questions
* Get resume tips, job fit analysis, skill suggestions

---

## 💡 Usage Examples

### ✔️ Profile Analysis

```
Input: https://linkedin.com/in/john-doe
Output: Full analysis, score, and improvement suggestions
```

### ✔️ Job Fit Analysis

```
Chat: "How do I fit this role? [paste job description]"
Output: Fit score, key strengths/gaps, strategy
```

### ✔️ Career Pathing

```
Chat: "What's my ideal next career move?"
Output: Personalized roadmap (skills, roles, timeline)
```

---

## 🤖 Multi-Agent Architecture

1. **Profile Analyzer Agent** – Assesses strengths, weaknesses
2. **Job Fit Agent** – Compares against job descriptions
3. **Content Enhancer Agent** – Suggests profile improvements
4. **Career Coach Agent** – Offers tailored career guidance

---

## 📊 Analytics & Insights

* Profile strength: Score 1–10
* Relevance of experience
* Engagement metrics
* AI-generated improvement ideas

---

### Apify Scraper (optional)

```python
APIFY_API_TOKEN = "your_token"
ACTOR_ID = "2SyF0bVxmgGr8IVCZ"
```

### SQLite DB

```python
DB_PATH = "data.db"
```


## 📝 License

Licensed under [MIT](LICENSE)

---

## 🙏 Acknowledgments

* **Google Gemini API** – AI model
* **LangChain/LangGraph** – Agent orchestration
* **Streamlit** – Frontend interface
* **Apify** – Web scraping LinkedIn data

---

**🛠 Built with passion by Ashu Pabreja**
**⏳ Note**: Please allow 30–60 seconds for analysis and 5–15 seconds for AI responses.
