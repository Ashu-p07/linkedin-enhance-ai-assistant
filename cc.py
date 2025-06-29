# LinkedIn Profile AI Assistant - Enhanced Version

import streamlit as st
import sqlite3
import json
import asyncio
import requests
from datetime import datetime
from typing import Dict, List, Optional, TypedDict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import time
from pathlib import Path

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict,field_validator

load_dotenv()

# Set page config
st.set_page_config(
    page_title="LinkedIn AI Career Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'profile_data' not in st.session_state:
    st.session_state.profile_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Enhanced Data Models



class EnhancedProfileData(BaseModel):
    # Basic Info
    full_name: Optional[str] = ""
    first_name: Optional[str] = ""
    last_name: Optional[str] = ""
    headline: Optional[str] = ""
    about: Optional[str] = ""
    location: Optional[str] = ""
    connections: Optional[int] = 0
    followers: Optional[int] = 0

    # Contact & URLs
    email: Optional[str] = None
    mobile_number: Optional[str] = None
    linkedin_url: Optional[str] = ""
    profile_pic: Optional[str] = ""

    # Current Position
    current_job_title: Optional[str] = ""
    current_company: Optional[str] = ""
    current_company_industry: Optional[str] = ""
    current_company_website: Optional[str] = ""
    current_job_duration: Optional[str] = ""
    current_job_duration_years: Optional[float] = 0.0

    # Detailed Experience
    experiences: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []

    # Skills
    skills: List[Dict[str, Any]] = []
    top_skills: List[str] = []

    # Certifications & Licenses
    certifications: List[Dict[str, Any]] = []

    # Recent Activity
    recent_posts: List[Dict[str, Any]] = []

    # Additional sections
    honors_awards: List[Dict[str, Any]] = []
    languages: List[Dict[str, Any]] = []
    volunteer_work: List[Dict[str, Any]] = []
    interests: List[Dict[str, Any]] = []

    # Raw data
    raw_data: Dict[str, Any] = {}

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True
    )

    # Convert None to empty string
    @field_validator(
        "about", "headline", "location", "current_job_title", "current_company",
        "current_company_industry", "current_company_website", "current_job_duration",
        "linkedin_url", "profile_pic", "full_name", "first_name", "last_name",
        mode="before"
    )
    @classmethod
    def none_to_empty_str(cls, v):
        return "" if v is None else str(v)

    # Convert None to zero
    @field_validator("connections", "followers", "current_job_duration_years", mode="before")
    @classmethod
    def none_to_zero(cls, v):
        if v is None:
            return 0
        try:
            return float(v) if "." in str(v) else int(v)
        except (TypeError, ValueError):
            return 0

    # Convert None to list/dict
    @field_validator(
        "experiences", "education", "skills", "top_skills", "certifications",
        "recent_posts", "honors_awards", "languages", "volunteer_work", "interests", "raw_data",
        mode="before"
    )
    @classmethod
    def none_to_collection(cls, v, info):
        if v is None:
            return {} if info.field_name == "raw_data" else []
        return v


class JobFitAnalysis(BaseModel):
    """Job fit analysis results"""
    overall_score: float = Field(description="Overall fit score 0-100")
    strengths: List[str] = Field(description="Key strengths for the role")
    gaps: List[str] = Field(description="Areas needing improvement")
    recommendations: List[str] = Field(description="Specific recommendations")
    skill_match_percentage: float = Field(description="Skill match percentage")
    experience_relevance: str = Field(description="Experience relevance assessment")

class ProfileEnhancement(BaseModel):
    """Profile enhancement suggestions"""
    headline_suggestions: List[str] = Field(description="Improved headline options")
    about_rewrite: str = Field(description="Enhanced about section")
    skill_additions: List[str] = Field(description="Recommended skills to add")
    experience_improvements: Dict = Field(description="Experience section enhancements")

class AgentState(TypedDict):
    """State shared between agents"""
    profile: EnhancedProfileData
    job_description: Optional[str]
    analysis: Optional[JobFitAnalysis]
    enhancements: Optional[ProfileEnhancement]
    messages: List[str]
    current_step: str

class ProfileDataProcessor:
    """Process raw LinkedIn data into structured format"""

    @staticmethod
    def safe_get(data: Dict, key: str, default=""):
        """Safely get value from dict, handling None cases"""
        value = data.get(key, default)
        if value is None:
            return default if isinstance(default, (str, int, float, list, dict)) else ""
        return value

    @staticmethod
    def safe_int(value, default=0):
        """Safely convert to int"""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def safe_float(value, default=0.0):
        """Safely convert to float"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def process_raw_profile(raw_data: Dict[str, Any]) -> EnhancedProfileData:
        """Convert raw scrapped data to structured profile"""

        if not isinstance(raw_data, dict):
            raw_data = {}

        # Extract skills with context
        skills = []
        skills_list = ProfileDataProcessor.safe_get(raw_data, "skills", [])
        if isinstance(skills_list, list):
            for skill in skills_list:
                if isinstance(skill, dict):
                    skills.append({
                        "title": ProfileDataProcessor.safe_get(skill, "title", ""),
                        "endorsements": ProfileDataProcessor.safe_get(skill, "subComponents", []),
                        "context": ProfileDataProcessor.safe_get(skill, "subComponents", [])
                    })

        # Extract top skills
        top_skills = []
        if isinstance(skills_list, list):
            for skill in skills_list[:10]:
                if isinstance(skill, dict) and skill.get("title"):
                    top_skills.append(str(skill.get("title", "")))

        # Process experiences
        experiences = []
        for exp in ProfileDataProcessor.safe_get(raw_data, "experiences", []):
            if isinstance(exp, dict):
                subtitle = ProfileDataProcessor.safe_get(exp, "subtitle", "")
                company = subtitle.split("·")[0].strip() if subtitle else ""
                experiences.append({
                    "title": ProfileDataProcessor.safe_get(exp, "title", ""),
                    "company": company,
                    "duration": ProfileDataProcessor.safe_get(exp, "caption", ""),
                    "location": ProfileDataProcessor.safe_get(exp, "metadata", ""),
                    "description": ProfileDataProcessor.safe_get(exp, "subComponents", [{}])[0].get("description", []) if exp.get("subComponents") else [],
                    "company_logo": ProfileDataProcessor.safe_get(exp, "logo", ""),
                    "company_id": ProfileDataProcessor.safe_get(exp, "companyId", "")
                })

        # Process education
        education = []
        for edu in ProfileDataProcessor.safe_get(raw_data, "educations", []):
            if isinstance(edu, dict):
                education.append({
                    "institution": ProfileDataProcessor.safe_get(edu, "title", ""),
                    "degree": ProfileDataProcessor.safe_get(edu, "subtitle", ""),
                    "duration": ProfileDataProcessor.safe_get(edu, "caption", ""),
                    "logo": ProfileDataProcessor.safe_get(edu, "logo", ""),
                    "activities": ProfileDataProcessor.safe_get(edu, "subComponents", [{}])[0].get("description", []) if edu.get("subComponents") else []
                })

        # Process certifications
        certifications = []
        for cert in ProfileDataProcessor.safe_get(raw_data, "licenseAndCertificates", []):
            if isinstance(cert, dict):
                certifications.append({
                    "title": ProfileDataProcessor.safe_get(cert, "title", ""),
                    "issuer": ProfileDataProcessor.safe_get(cert, "subtitle", ""),
                    "issue_date": ProfileDataProcessor.safe_get(cert, "caption", ""),
                    "credential_id": ProfileDataProcessor.safe_get(cert, "metadata", ""),
                    "logo": ProfileDataProcessor.safe_get(cert, "logo", "")
                })

        # Process recent posts
        recent_posts = []
        for post in ProfileDataProcessor.safe_get(raw_data, "updates", []):
            if isinstance(post, dict):
                recent_posts.append({
                    "text": ProfileDataProcessor.safe_get(post, "postText", ""),
                    "image": ProfileDataProcessor.safe_get(post, "image", ""),
                    "likes": ProfileDataProcessor.safe_int(post.get("numLikes"), 0),
                    "comments": ProfileDataProcessor.safe_int(post.get("numComments"), 0),
                    "post_link": ProfileDataProcessor.safe_get(post, "postLink", "")
                })

        # Create EnhancedProfileData object
        return EnhancedProfileData(
            full_name=ProfileDataProcessor.safe_get(raw_data, "fullName", ""),
            first_name=ProfileDataProcessor.safe_get(raw_data, "firstName", ""),
            last_name=ProfileDataProcessor.safe_get(raw_data, "lastName", ""),
            headline=ProfileDataProcessor.safe_get(raw_data, "headline", ""),
            about=ProfileDataProcessor.safe_get(raw_data, "about", ""),
            location=ProfileDataProcessor.safe_get(raw_data, "addressWithCountry", ""),
            connections=ProfileDataProcessor.safe_int(raw_data.get("connections"), 0),
            followers=ProfileDataProcessor.safe_int(raw_data.get("followers"), 0),

            email=ProfileDataProcessor.safe_get(raw_data, "email", None),
            mobile_number=ProfileDataProcessor.safe_get(raw_data, "mobileNumber", None),
            linkedin_url=ProfileDataProcessor.safe_get(raw_data, "linkedinUrl", ""),
            profile_pic=ProfileDataProcessor.safe_get(raw_data, "profilePicHighQuality") or ProfileDataProcessor.safe_get(raw_data, "profilePic", ""),

            current_job_title=ProfileDataProcessor.safe_get(raw_data, "jobTitle", ""),
            current_company=ProfileDataProcessor.safe_get(raw_data, "companyName", ""),
            current_company_industry=ProfileDataProcessor.safe_get(raw_data, "companyIndustry", ""),
            current_company_website=ProfileDataProcessor.safe_get(raw_data, "companyWebsite", ""),
            current_job_duration=ProfileDataProcessor.safe_get(raw_data, "currentJobDuration", ""),
            current_job_duration_years=ProfileDataProcessor.safe_float(raw_data.get("currentJobDurationInYrs"), 0.0),

            experiences=experiences,
            education=education,
            skills=skills,
            top_skills=top_skills,
            certifications=certifications,
            recent_posts=recent_posts,
            honors_awards=ProfileDataProcessor.safe_get(raw_data, "honorsAndAwards", []),
            languages=ProfileDataProcessor.safe_get(raw_data, "languages", []),
            volunteer_work=ProfileDataProcessor.safe_get(raw_data, "volunteerAndAwards", []),
            interests=ProfileDataProcessor.safe_get(raw_data, "interests", []),
            raw_data=raw_data
        )

class LinkedInScraper:
    def __init__(self, apify_token: Optional[str] = None, linkedin_cookie: Optional[str] = None):
        self.apify_token = apify_token or os.getenv("APIFY_API_TOKEN")
        self.linkedin_cookie = linkedin_cookie or os.getenv("LINKEDIN_COOKIE")
        self.base_url = "https://api.apify.com/v2"

    def scrape_profile(self, linkedin_url: str) -> Dict[str, Any]:
        """Scrape LinkedIn profile using Apify actor with polling"""

        if not self.apify_token:
            print("[WARN] No Apify token found, returning mock data.")
            return self._get_mock_profile_data()

        actor_id = "2SyF0bVxmgGr8IVCZ"  

        headers = {
            "Authorization": f"Bearer {self.apify_token}",
            "Content-Type": "application/json",
        }

        # Actor input uses 'profileUrls' array instead of 'startUrls'
        run_input = {
            "profileUrls": [linkedin_url]
        }

        # Add cookie if available
        if self.linkedin_cookie:
            run_input["cookie"] = self.linkedin_cookie

        try:
            # 1. Start the actor run
            start_url = f"{self.base_url}/acts/{actor_id}/runs"
            response = requests.post(start_url, headers=headers, json=run_input)

            if response.status_code != 201:
                raise Exception(f"Failed to start actor: {response.text}")

            run_data = response.json()["data"]
            run_id = run_data["id"]
            print(f"[INFO] Actor run started. Run ID: {run_id}")

            # 2. Poll for status until done
            status_url = f"{self.base_url}/acts/{actor_id}/runs/{run_id}"
            while True:
                status_response = requests.get(status_url, headers=headers)
                status = status_response.json()["data"]["status"]
                print(f"[INFO] Run status: {status}")
                if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
                    break
                time.sleep(5)

            if status != "SUCCEEDED":
                raise Exception(f"Actor run failed or was aborted. Status: {status}")

            # 3. Get dataset ID
            dataset_id = status_response.json()["data"]["defaultDatasetId"]
            data_url = f"{self.base_url}/datasets/{dataset_id}/items?clean=true"

            # 4. Fetch data from dataset
            data_response = requests.get(data_url, headers=headers)

            if data_response.status_code != 200:
                raise Exception(f"Failed to fetch dataset items: {data_response.text}")

            data = data_response.json()
            if isinstance(data, list) and data:
                return data[0]  # Return first profile's data
            return {}

        except Exception as e:
            print(f"[ERROR] Scraping failed: {str(e)}")
            return self._get_mock_profile_data()

    def _get_mock_profile_data(self) -> Dict[str, Any]:
        """Mock profile data for demonstration - matches JSON structure"""
        return {
            "linkedinUrl": "https://www.linkedin.com/in/demo-profile/",
            "firstName": "John",
            "lastName": "Doe",
            "fullName": "John Doe",
            "headline": "Senior Software Engineer | Full-Stack Developer | Tech Lead",
            "connections": 500,
            "followers": 600,
            "email": None,
            "mobileNumber": None,
            "jobTitle": "Senior Software Engineer",
            "companyName": "Tech Corp",
            "companyIndustry": "Information Technology And Services",
            "companyWebsite": "techcorp.com",
            "currentJobDuration": "2 yrs",
            "currentJobDurationInYrs": 2.0,
            "addressWithCountry": "San Francisco, CA, USA",
            "about": "Experienced software engineer with 8+ years in full-stack development. Passionate about building scalable applications using modern technologies.",
            "experiences": [
                {
                    "title": "Senior Software Engineer",
                    "subtitle": "Tech Corp · Full-time",
                    "caption": "2022 - Present · 2 yrs",
                    "metadata": "San Francisco, CA · Remote",
                    "subComponents": [{"description": ["Led development of microservices architecture", "Built React/Node.js applications"]}]
                }
            ],
            "skills": [
                {"title": "Python", "subComponents": []},
                {"title": "JavaScript", "subComponents": []},
                {"title": "React", "subComponents": []},
                {"title": "Node.js", "subComponents": []},
                {"title": "AWS", "subComponents": []},
                {"title": "Docker", "subComponents": []}
            ],
            "educations": [
                {
                    "title": "University of Technology",
                    "subtitle": "Bachelor of Science - Computer Science",
                    "caption": "2015 - 2019"
                }
            ],
            "licenseAndCertificates": [],
            "updates": [],
            "honorsAndAwards": [],
            "languages": [],
            "volunteerAndAwards": [],
            "interests": []
        }

class LLMManager:
    def __init__(self, api_key: str = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("API key is required. Set GEMINI_API_KEY in .env.")

        self.llm = ChatGoogleGenerativeAI(
            google_api_key=self.api_key,
            model=self.model,
            temperature=0.7,
        )

    def get_llm(self) -> BaseChatModel:
        return self.llm

class MemoryManager:
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.init_database()
        self.checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    
    def init_database(self):
        """Initialize SQLite database for memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp DATETIME,
                message_type TEXT,
                content TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                profile_data TEXT,
                created_at DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, session_id: str, message_type: str, content: str, metadata: Dict = None):
        """Save conversation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (session_id, timestamp, message_type, content, metadata)
            VALUES (?, ?, ?, ?, ?)
        ''', (session_id, datetime.now(), message_type, content, json.dumps(metadata or {})))
        
        conn.commit()
        conn.close()
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Retrieve conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM conversations WHERE session_id = ? ORDER BY timestamp
        ''', (session_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [{"type": row[3], "content": row[4], "timestamp": row[2]} for row in results]

# Enhanced Multi-Agent System using LangGraph
class LinkedInAssistantAgents:
    def __init__(self, llm_manager: LLMManager, memory_manager: MemoryManager):
        self.llm = llm_manager.get_llm()
        self.memory = memory_manager
        self.graph = self._create_agent_graph()
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the multi-agent workflow graph"""
        graph = StateGraph(AgentState)
        
        # Add nodes (agents)
        graph.add_node("profile_analyzer", self.profile_analyzer_agent)
        graph.add_node("job_fit_agent", self.job_fit_agent)
        graph.add_node("content_enhancer", self.content_enhancer_agent)
        graph.add_node("career_coach", self.career_coach_agent)
        
        # Define the workflow
        graph.set_entry_point("profile_analyzer")
        graph.add_edge("profile_analyzer", "job_fit_agent")
        graph.add_edge("job_fit_agent", "content_enhancer")
        graph.add_edge("content_enhancer", "career_coach")
        graph.add_edge("career_coach", END)
        
        return graph.compile(checkpointer=self.memory.checkpointer)
    def _format_profile_for_prompt(self, profile: EnhancedProfileData) -> str:
        """Format comprehensive profile data for LLM prompts (robust and structured)"""
        def safe_get_list(lst, fallback=[]):
            return lst if isinstance(lst, list) else fallback
        def safe_get_str(value):
            return str(value) if value not in [None, "", [], {}] else "N/A"
        exp_text = ""
        for i, exp in enumerate(safe_get_list(profile.experiences)[:5], 1):
            title = safe_get_str(exp.get("title"))
            company = safe_get_str(exp.get("company") or exp.get("subtitle"))
            duration = safe_get_str(exp.get("caption"))
            location = safe_get_str(exp.get("metadata"))

            descriptions = []
            for sub in safe_get_list(exp.get("subComponents")):
                 for desc_item in safe_get_list(sub.get("description")):
                     descriptions.append(safe_get_str(desc_item.get("text")))
            desc_text = " | ".join(descriptions) if descriptions else "No description available"
            exp_text += (
            f"\n{i}. {title} at {company} ({duration})"
            f"\n   Location: {location}"
            f"\n   Details: {desc_text}\n"
        )
                
           

        
        # === EDUCATION ===
        edu_text = ""
        for i, edu in enumerate(safe_get_list(profile.education)[:5], 1):
            degree = safe_get_str(edu.get("subtitle"))
            institution = safe_get_str(edu.get("title"))
            duration = safe_get_str(edu.get("caption"))
            edu_text += f"\n{i}. {degree} from {institution} ({duration})\n"

            
           

        # === SKILLS ===
        skills_list = safe_get_list(profile.skills)
        skill_titles = [safe_get_str(s.get("title")) for s in skills_list if isinstance(s, dict)]
        skills_text = ", ".join(skill_titles[:15]) if skill_titles else "N/A"

        # === CERTIFICATIONS ===
        cert_text = ""
        for i, cert in enumerate(safe_get_list(profile.certifications)[:5], 1):
            title = safe_get_str(cert.get("title"))
            issuer = safe_get_str(cert.get("subtitle"))
            issue_date = safe_get_str(cert.get("caption"))
            cert_text += f"\n{i}. {title} from {issuer} ({issue_date})\n"
        activity_text = ""
        for i, post in enumerate(safe_get_list(profile.recent_posts)[:3], 1):
            post_text = post.get("postText", "")
            preview = (post_text[:200] + "...") if len(post_text) > 200 else post_text
            likes = post.get("numLikes", 0)
            activity_text += f"\n{i}. Post with {likes} likes: {preview}\n"

        

           
        
        # === FINAL PROFILE STRING ===
        formatted_profile = f"""
COMPREHENSIVE PROFILE ANALYSIS:

=== BASIC INFORMATION ===
Name: {safe_get_str(profile.full_name)}
Headline: {safe_get_str(profile.headline)}
Location: {safe_get_str(profile.location)}
Connections: {safe_get_str(profile.connections)} | Followers: {safe_get_str(profile.followers)}
LinkedIn: {safe_get_str(profile.linkedin_url)}

=== CURRENT POSITION ===
Title: {safe_get_str(profile.current_job_title)}
Company: {safe_get_str(profile.current_company)} ({safe_get_str(profile.current_company_industry)})
Duration: {safe_get_str(profile.current_job_duration)} ({safe_get_str(profile.current_job_duration_years)} years)
Company Website: {safe_get_str(profile.current_company_website)}


=== ABOUT SECTION ===
{safe_get_str(profile.about)}
=== PROFESSIONAL EXPERIENCE ==={exp_text}

=== EDUCATION ==={edu_text}

=== TOP SKILLS ===
{skills_text}

=== CERTIFICATIONS ==={cert_text}
=== RECENT ACTIVITY ==={activity_text}

=== ADDITIONAL CONTEXT ===
Total Experience Positions: {len(safe_get_list(profile.experiences))}
Total Certifications: {len(safe_get_list(profile.certifications))}
Total Skills Listed: {len(skills_list)}
Profile Completeness: High (includes experience, education, skills, certifications)
"""
        return formatted_profile.strip()
             
    
    async def profile_analyzer_agent(self, state: AgentState) -> AgentState:
        """Enhanced agent that analyzes LinkedIn profile comprehensively"""
        profile = state["profile"]
        formatted_profile = self._format_profile_for_prompt(profile)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a senior LinkedIn profile analysis expert with deep expertise in career development and professional branding.
        
        Analyze this comprehensive LinkedIn profile and provide detailed insights(but make sure your resposes are not too large , practical and as per market):
        NOTE: THE RESPONSE IS SHORT NOT TOO LARGE AND TOO THE POINT IF USER ASK TO EXPLAIN THEN ONLY EXPLAIN , ELSE JUST GIVE THE TOO THE POINT ANSWER

        {profile_data}

📌 **Instructions**:
- Keep the total output under **100 words**.
- Be specific, actionable, and market-aligned.
- Focus only on key improvements.
                                                  
        Provide a thorough analysis covering:

        1. **Profile Strength Assessment** (Score 1-10 with explanation)
        2. **Professional Positioning Analysis**
           - Industry positioning
           - Unique value proposition
        3. **Career Progression Evaluation**
        4. **Content Quality Assessment**
        5. **Network & Engagement Analysis**
        6. **Key Strengths & Differentiators**
        7. **Critical Improvement Areas**
        8. **Strategic Recommendations**
        **Constraints:**
          - DO NOT write essays or explanations
n         - Keep suggestions brief and skimmable
          - No numbered sub-points unless critical
          - concise and to the point


        Be specific, actionable, and professional. Focus on both technical skills and soft skills evident from the profile.
        """)
        
        response = await self.llm.ainvoke(prompt.format(profile_data=formatted_profile))
        
        state["messages"].append(f"📊 **Profile Analysis Complete**\n\n{response.content}")
        state["current_step"] = "profile_analyzed"
        return state
    
    async def job_fit_agent(self, state: AgentState) -> AgentState:
        """Enhanced agent that evaluates job fit against job descriptions"""
        if not state.get("job_description"):
            state["messages"].append("⚠️ No job description provided - skipping job fit analysis")
            return state
        
        profile = state["profile"]
        job_desc = state["job_description"]
        formatted_profile = self._format_profile_for_prompt(profile)
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert job fit analyzer and career strategist.
        NOTE: THE RESPONSE IS SHORT NOT TOO LARGE AND TOO THE POINT IF USER ASK TO EXPLAIN THEN ONLY EXPLAIN , ELSE JUST GIVE THE TOO THE POINT ANSWER
        Compare this comprehensive LinkedIn profile against the job description:

        === CANDIDATE PROFILE ===
        {profile_data}

        === TARGET JOB DESCRIPTION ===
        {job_description}

        Provide a detailed job fit analysis(but make sure your resposes are not too large , practical and as per market):

        1. **Overall Fit Score** (0-100 with detailed breakdown)
        2. **Skills Alignment Analysis**
           - Technical skills match (%)
          
        3. **Experience Relevance Assessment**
           - Industry experience alignment
           
        4. **Key Competitive Advantages**
           - Unique strengths for this role
    
        5. **Gap Analysis**
           - Critical missing qualifications
           -
        6. **Application Strategy Recommendations**
           - How to position candidacy
           
        7. **Interview Preparation Insights**
           - Likely interview focus areas
           - Gap mitigation strategies
          **Constraints:**
          - DO NOT write essays or explanations
         - Keep suggestions brief and skimmable
          - No numbered sub-points unless critical
          - concise and to the point

        Be specific about how the candidate's background aligns with job requirements.
        """)
        
        response = await self.llm.ainvoke(
            prompt.format(
                profile_data=formatted_profile,
                job_description=job_desc
            )
        )
        
        state["messages"].append(f"🎯 **Job Fit Analysis Complete**\n\n{response.content}")
        state["current_step"] = "job_fit_analyzed"
        return state
    
    async def content_enhancer_agent(self, state: AgentState) -> AgentState:
        """Enhanced agent that suggests comprehensive profile improvements"""
        profile = state["profile"]
        formatted_profile = self._format_profile_for_prompt(profile)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a LinkedIn optimization expert specializing in profile enhancement and personal branding.
        NOTE: THE RESPONSE IS SHORT NOT TOO LARGE AND TOO THE POINT IF USER ASK TO EXPLAIN THEN ONLY EXPLAIN , ELSE JUST GIVE THE TOO THE POINT ANSWER
        Analyze this profile and provide comprehensive enhancement recommendations(but make sure your resposes are not too large , practical and as per market):

        {profile_data}
                                                  
📌 **Instructions**:
- Keep the total output under **100 words**.
- Be specific, actionable, and market-aligned.
- Focus only on key improvements.

        Provide enhancement suggestions:

        1. **Headline Optimization**
           - 3-5 improved headline options
           

        2. **About Section Rewrite**
           - Compelling narrative structure
          
       
        3. **Content & Engagement Strategy**
           - Post content recommendations
           

        4. **Profile Completeness**
           - Missing sections to add
           
        5. **ATS Optimization**
           - Keyword density recommendations
           - Formatting improvements
           - Searchability enhancements
        **Constraints:**
          - DO NOT write essays or explanations
n         - Keep suggestions brief and skimmable
          - No numbered sub-points unless critical
          - concise and to the point

        Focus on measurable improvements and industry best practices.
        """)
        
        response = await self.llm.ainvoke(prompt.format(profile_data=formatted_profile))
        
        state["messages"].append(f"✨ **Profile Enhancement Recommendations**\n\n{response.content}")
        state["current_step"] = "content_enhanced"
        return state
    
    async def career_coach_agent(self, state: AgentState) -> AgentState:
        """Enhanced agent that provides strategic career guidance"""
        profile = state["profile"]
        formatted_profile = self._format_profile_for_prompt(profile)
        
        prompt = ChatPromptTemplate.from_template("""
        You are a senior career strategist and executive coach with expertise across multiple industries.
        NOTE: THE RESPONSE IS SHORT NOT TOO LARGE AND TOO THE POINT IF USER ASK TO EXPLAIN THEN ONLY EXPLAIN , ELSE JUST GIVE THE TOO THE POINT ANSWER
        Based on this comprehensive professional profile, provide strategic career guidance(but make sure your resposes are not too large , practical and as per market):

        {profile_data}
                                                  
📌 **Instructions**:
- Keep the total output under **100 words**.
- Be specific, actionable, and market-aligned.
- Focus only on key improvements.

        Provide comprehensive career coaching:

        1. **Career Trajectory Analysis**
           
        2. **Strategic Career Path Options**
          

        3. **Skill Development Roadmap**
         

        4. **Professional Branding Strategy**
           

        5. **Networking & Relationship Building**
          
        6. **Market Intelligence**
           
        7. **Action Plan & Timeline**
          
        Be specific, actionable, and aligned with current market realities.
        """)
        
        response = await self.llm.ainvoke(prompt.format(profile_data=formatted_profile))
        
        state["messages"].append(f"🎓 **Career Strategy & Coaching**\n\n{response.content}")
        state["current_step"] = "coaching_complete"
        return state
    
    async def run_analysis(self, profile: EnhancedProfileData, job_description: str = None, session_id: str = "default") -> List[str]:
        """Run the complete multi-agent analysis"""
        initial_state = AgentState(
            profile=profile,
            job_description=job_description,
            analysis=None,
            enhancements=None,
            messages=[],
            current_step="starting"
        )
        
        config = {"configurable": {"thread_id": session_id}}
        
        try:
            final_state = await self.graph.ainvoke(initial_state, config)
            return final_state["messages"]
        except Exception as e:
            return [f"❌ Analysis error: {str(e)}"]

# Enhanced Streamlit Application
class LinkedInAssistantApp:
    def __init__(self):
        self.scraper = LinkedInScraper()
        self.llm_manager = LLMManager()
        self.memory_manager = MemoryManager()
        self.agents = LinkedInAssistantAgents(self.llm_manager, self.memory_manager)
        self.processor = ProfileDataProcessor()

        if 'session_id' not in st.session_state:
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def render_sidebar(self):
        st.sidebar.title("🔧 Enhanced Options")
        st.sidebar.markdown("Comprehensive LinkedIn profile analysis with advanced AI insights.")
        
        include_job_fit = st.sidebar.checkbox(  
            "Include Job Fit Analysis",
            value=True,
            help="Analyze profile against a job description for better career insights"
        )
            
        
        if st.session_state.profile_data:
            st.sidebar.markdown("---")
            if st.sidebar.button("🔁 Analyze Another Profile"):
                st.session_state.profile_data = None
                st.session_state.messages = []
                st.session_state.analysis_complete = False
                st.rerun()
        
        return include_job_fit

    def render_main_interface(self):
        """Render the main application interface"""
        st.title("💼 LinkedIn AI Career Assistant")
        st.markdown("*Advanced AI-powered LinkedIn profile analysis with comprehensive career insights*")
        
        if not st.session_state.profile_data:
            self.render_profile_input()
        else:
            self.render_chat_interface()

    def render_profile_input(self):
        """Render the profile input section"""
        st.subheader("🔍 Professional Profile Analysis")
        
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                linkedin_url = st.text_input(
                    "LinkedIn Profile URL",
                    placeholder="https://linkedin.com/in/username",
                    help="Enter the full LinkedIn profile URL for comprehensive analysis"
                )
            
            with col2:
                st.markdown("<div style='padding-top: 1.8em'></div>", unsafe_allow_html=True)
                analyze_clicked = st.button("🚀 Analyze Profile", use_container_width=True)
            
            if analyze_clicked:
                if linkedin_url:
                    with st.spinner("🔄 Scraping and processing profile data..."):
                        try:
                            # Scrape profile data
                            raw_profile_data = self.scraper.scrape_profile(linkedin_url)
                            
                            # Process into enhanced structure
                            enhanced_profile = self.processor.process_raw_profile(raw_profile_data)
                            
                            st.session_state.profile_data = enhanced_profile
                            st.success("✅ Profile analysis complete! You can now chat with your AI career assistant.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error analyzing profile: {str(e)}")
                else:
                    st.error("⚠️ Please enter a valid LinkedIn URL")

    def render_chat_interface(self):
        """Render the enhanced chat interface with profile summary"""
        st.subheader("💬 AI Career Assistant Chat")
        
        profile = st.session_state.profile_data
        
        # Enhanced Profile Summary
        with st.expander("📋 Comprehensive Profile Summary", expanded=False):
            self.render_enhanced_profile_summary(profile)
        
        # Action Buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧭 Generate Career Path", use_container_width=True):
                self.generate_career_path(profile)
        
        with col2:
            if st.button("✨ Optimize Profile", use_container_width=True):
                self.optimize_profile_content(profile)
        
        # Chat Interface
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat Input
        if prompt := st.chat_input("Ask about career advice, profile improvements, job fit analysis, or any career-related questions..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            self.memory_manager.save_conversation(st.session_state.session_id, "user", prompt)
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🤔 Analyzing your request..."):
                    response = self.generate_enhanced_response(prompt, profile)
                    st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            self.memory_manager.save_conversation(st.session_state.session_id, "assistant", response)

    def render_enhanced_profile_summary(self, profile: EnhancedProfileData):
        """Render comprehensive profile summary"""
        st.markdown("### 👤 Professional Profile Overview")
        
        # Basic Info Section
        cols = st.columns([1, 3])
        with cols[0]:
            if profile.profile_pic:
                st.image(profile.profile_pic, width=120, caption="Profile Photo")
            else:
                st.markdown("📷 No profile picture")
        
        with cols[1]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Name:** {profile.full_name}")
                st.markdown(f"**Location:** {profile.location}")
                st.markdown(f"**Connections:** {profile.connections:,}")
                st.markdown(f"**Followers:** {profile.followers:,}")
            
            with col2:
                st.markdown(f"**Current Role:** {profile.current_job_title}")
                st.markdown(f"**Company:** {profile.current_company}")
                st.markdown(f"**Industry:** {profile.current_company_industry}")
                st.markdown(f"**Duration:** {profile.current_job_duration}")
        
        # Professional Headline
        st.markdown("### 🎯 Professional Headline")
        st.info(profile.headline)
        
        # About Section
        if profile.about:
            st.markdown("### 📝 About")
            st.markdown(profile.about)
        
        # Skills Overview
        if profile.top_skills:
            st.markdown("### 🛠️ Top Skills")
            skills_cols = st.columns(3)
            for i, skill in enumerate(profile.top_skills[:15]):
                with skills_cols[i % 3]:
                    st.markdown(f"• {skill}")
        
        # Experience Summary
        if profile.experiences:
            st.markdown("### 💼 Experience Summary")
            for exp in profile.experiences[:3]:  # Show top 3
                with st.container():
                    st.markdown(f"**{exp.get('title', 'N/A')}** at *{exp.get('company', 'N/A')}*")
                    st.markdown(f"📅 {exp.get('duration', 'N/A')} | 📍 {exp.get('location', 'N/A')}")
                    st.markdown("---")

    def generate_career_path(self, profile: EnhancedProfileData):
        """Generate personalized career path prediction"""
        with st.spinner("🔮 Predicting your personalized career trajectory..."):
            try:
                formatted_profile = self.agents._format_profile_for_prompt(profile)
                
                prompt = ChatPromptTemplate.from_template("""
                You are a senior career strategist and industry expert. Based on this comprehensive professional profile, predict and design an optimal career path progression.
                make sure it is medium length not too large, specific, clear and market-aligned (practical). 

                {profile_data}
                
📌 **Instructions**:
- Keep the total output under **100 words**.
- Be specific, actionable, and market-aligned.
- Focus only on key improvements.                                          
                                                          

                Create a career roadmap with:

                1. **Current Position Assessment**
                   - Career stage evaluation
                   - Strengths and market position

                2. **Short-term Progression (1-2 years)**
                   - Next logical role
                   - Skills to develop
                   

                3. **Medium-term Goals (3-5 years)**
                   - Target senior positions
                   

                4. **Long-term Vision (5-10 years)**
                   - Executive/specialist tracks
                   - Entrepreneurial opportunities
                   - Industry thought leadership

                5. **Skill Development Roadmap**
                   - Technical skills priorities
                   
                   - Industry certifications

                **Constraints:**
          - DO NOT write essays or explanations
n         - Keep suggestions brief and skimmable
          - No numbered sub-points unless critical
          - concise and to the point

                Make recommendations specific, actionable, and aligned with current market trends.
                """)
                
                response = self.llm_manager.get_llm().invoke(
                    prompt.format(profile_data=formatted_profile)
                )
                
                st.markdown("#### 📈 Your Personalized Career Path")
                st.success(response.content)
                
            except Exception as e:
                st.error(f"Error generating career path: {str(e)}")

    

    def optimize_profile_content(self, profile: EnhancedProfileData):
        """Generate profile optimization suggestions"""
        with st.spinner("✨ Generating profile optimization recommendations..."):
            try:
                formatted_profile = self.agents._format_profile_for_prompt(profile)
                
                prompt = ChatPromptTemplate.from_template("""
                You are a LinkedIn optimization expert. Analyze this profile and provide specific,clear, market-aligned actionable optimization recommendations(not too large):
                
📌 **Instructions**:
- Keep the total output under **100 words**.
- Be specific, actionable, and market-aligned.
- Focus only on key improvements.

                {profile_data}

                Provide optimization suggestions:

                1. **Headline Optimization**
                   - 3 improved headline variations
                   - Industry positioning

                2. **About Section Enhancement**
                   - Value proposition clarity
                   - Call-to-action recommendations

                3. **Skills Strategy**
                   - Priority skills to add
                   - Skills to remove/replace
                   

                4. **Content Strategy**
                   - Post content ideas
                   - Engagement tactics
                   

                5. **Profile Completeness**
                   - Missing sections to add
                   - Visual content suggestions
                  **Constraints:**
          - DO NOT write essays or explanations
n         - Keep suggestions brief and skimmable
          - No numbered sub-points unless critical
          - concise and to the point

                Make all recommendations specific and actionable.
                """)
                
                response = self.llm_manager.get_llm().invoke(
                    prompt.format(profile_data=formatted_profile)
                )
                
                st.markdown("#### ✨ Profile Optimization Recommendations")
                st.success(response.content)
                
            except Exception as e:
                st.error(f"Error generating optimization suggestions: {str(e)}")

    def generate_enhanced_response(self, user_input: str, profile: EnhancedProfileData) -> str:
        """Generate enhanced AI responses based on user input"""
        try:
            # Check for specific query types
            job_keywords = ["job", "position", "role", "fit", "match", "application", "interview"]
            career_keywords = ["career", "path", "progression", "growth", "promotion"]
            skill_keywords = ["skill", "learn", "develop", "training", "certification"]
            
            is_job_query = any(k in user_input.lower() for k in job_keywords)
            is_career_query = any(k in user_input.lower() for k in career_keywords)
            is_skill_query = any(k in user_input.lower() for k in skill_keywords)
            
            # Format profile for context
            formatted_profile = self.agents._format_profile_for_prompt(profile)
            
            if is_job_query and "job description" in user_input.lower():
                # Job fit analysis
                return self.analyze_job_fit(user_input, profile)
            elif is_career_query:
                # Career guidance
                return self.provide_career_guidance(user_input, formatted_profile)
            elif is_skill_query:
                # Skill development
                return self.suggest_skill_development(user_input, formatted_profile)
            else:
                # General career advice
                return self.provide_general_advice(user_input, formatted_profile)
                
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try rephrasing your question."

    def analyze_job_fit(self, user_input: str, profile: EnhancedProfileData) -> str:
        """Analyze job fit against provided job description"""
        try:
            # Extract job description from user input
            job_desc = user_input.replace("job description", "").strip()
            
            # Run job fit analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            messages = loop.run_until_complete(
                self.agents.run_analysis(profile, job_desc, st.session_state.session_id)
            )
            loop.close()
            
            return "\n\n".join(messages)
            
        except Exception as e:
            return f"Error analyzing job fit: {str(e)}"

    def provide_career_guidance(self, user_input: str, formatted_profile: str) -> str:
        """Provide career guidance"""
        prompt = ChatPromptTemplate.from_template("""
        You are a senior career strategist. The user is asking about career guidance: "{user_question}"

        Based on this comprehensive profile:
        {profile_data}

        Provide specific, actionable career guidance that addresses their question.
        Focus on practical steps, industry insights, and strategic recommendations.
        """)
        
        response = self.llm_manager.get_llm().invoke(
            prompt.format(
                user_question=user_input,
                profile_data=formatted_profile
            )
        )
        return response.content

    def suggest_skill_development(self, user_input: str, formatted_profile: str) -> str:
        """Suggest skill development strategies"""
        prompt = ChatPromptTemplate.from_template("""
        You are a professional development expert. The user is asking about skill development: "{user_question}"

📌 **Instructions**:
- Keep the total output under **100 words**.
- Be specific, actionable, and market-aligned.
- Focus only on key improvements.
        Based on this profile:
        {profile_data}

        Provide specific skill development recommendations including:
        - Priority skills to develop
        - Learning resources and strategies
        - Certification recommendations
        - Timeline for skill acquisition
        - How to demonstrate these skills
        - make concise response to the point

        Make recommendations specific to their current role and career goals.
        """)
        
        response = self.llm_manager.get_llm().invoke(
            prompt.format(
                user_question=user_input,
                profile_data=formatted_profile
            )
        )
        return response.content

    def provide_general_advice(self, user_input: str, formatted_profile: str) -> str:
        """Provide general career advice"""
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful LinkedIn career assistant. The user is asking: "{user_question}"

📌 **Instructions**:
- Keep the total output under **300 words**.
- Be specific, actionable, and market-aligned and practical.
- Focus only on maximumpractical answer if need then eloborate.
        Based on their comprehensive profile:
        {profile_data}

        Provide helpful, specific, and actionable advice that addresses their question.
        Keep responses conversational, professional, and focused on practical solutions.
        Draw insights from their profile to make recommendations personal and relevant.
        """)
        
        response = self.llm_manager.get_llm().invoke(
            prompt.format(
                user_question=user_input,
                profile_data=formatted_profile
            )
        )
        return response.content

    def run(self):
        """Main application runner"""
        include_job_fit = self.render_sidebar()
        self.render_main_interface()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        ### 🚀 **How to Use This Advanced AI Career Assistant:**
        
        1. **Profile Analysis** - Paste your LinkedIn profile URL for comprehensive analysis
        2. **AI Chat** - Ask specific questions about career development, job fit, or profile optimization
        3. **Career Path** - Get personalized career progression predictions
        4. **Complete Analysis** - Run full multi-agent analysis with job fit scoring
        5. **Profile Optimization** - Get specific recommendations to enhance your LinkedIn presence
        
        
        
        *Created by Ashu Pabreja*
        """)

# Application Entry Point
if __name__ == "__main__":
    try:
        app = LinkedInAssistantApp()
        app.run()
    except Exception as e:
        st.error(f"Application startup error: {str(e)}")
        st.markdown("Please check your environment variables and API keys.")