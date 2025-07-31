#!/usr/bin/env python3

import pandas as pd
import os
import re
import sys
import time
import signal
import json
import glob
import csv
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import random

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted
import groq 


# Define the columns we want for outreach
OUTREACH_COLUMNS = [
    # Essential
    'firstName', 'lastName', 'fullName', 'email', 'linkedinUrl',
    'companyName', 'companyIndustry', 'companySize', 'jobTitle', 'headline', 'about',

    # High Value
    'currentJobDuration', 'currentJobDurationInYrs', 'addressWithCountry',
    'companyFoundedIn', 'companyWebsite', 'connections',

    # All Experiences (up to 5)
    'experiences/0/title', 'experiences/0/subtitle', 'experiences/0/caption',
    'experiences/0/subComponents/0/title', 'experiences/0/subComponents/0/subtitle',
    'experiences/0/subComponents/0/description/0/text', 'experiences/0/subComponents/0/description/1/text',
    'experiences/0/subComponents/1/title', 'experiences/0/subComponents/1/subtitle',
    'experiences/0/subComponents/1/description/0/text',
    'experiences/0/subComponents/2/title', 'experiences/0/subComponents/2/subtitle',
    'experiences/0/subComponents/2/description/0/text',

    'experiences/1/title', 'experiences/1/subtitle', 'experiences/1/caption',
    'experiences/1/subComponents/0/title', 'experiences/1/subComponents/0/subtitle',
    'experiences/1/subComponents/0/description/0/text', 'experiences/1/subComponents/0/description/1/text',
    'experiences/1/subComponents/1/title', 'experiences/1/subComponents/1/subtitle',
    'experiences/1/subComponents/1/description/0/text',
    'experiences/1/subComponents/2/title', 'experiences/1/subComponents/2/subtitle',
    'experiences/1/subComponents/2/description/0/text',

    'experiences/2/title', 'experiences/2/subtitle', 'experiences/2/caption',
    'experiences/2/subComponents/0/title', 'experiences/2/subComponents/0/subtitle',
    'experiences/2/subComponents/0/description/0/text',

    'experiences/3/title', 'experiences/3/subtitle', 'experiences/3/caption',
    'experiences/3/subComponents/0/title', 'experiences/3/subComponents/0/subtitle',
    'experiences/3/subComponents/0/description/0/text',

    'experiences/4/title', 'experiences/4/subtitle', 'experiences/4/caption',
    'experiences/4/subComponents/0/title', 'experiences/4/subComponents/0/subtitle',
    'experiences/4/subComponents/0/description/0/text',

    # Education
    'educations/0/title', 'educations/0/subtitle', 'educations/0/caption',
    'educations/0/subComponents/0/description/0/text', 'educations/0/subComponents/0/description/1/text',
    'educations/1/title', 'educations/1/subtitle', 'educations/1/caption',

    # Skills (up to 20)
    'skills/0/title', 'skills/1/title', 'skills/2/title', 'skills/3/title', 'skills/4/title',
    'skills/5/title', 'skills/6/title', 'skills/7/title', 'skills/8/title', 'skills/9/title',
    'skills/10/title', 'skills/11/title', 'skills/12/title', 'skills/13/title', 'skills/14/title',
    'skills/15/title', 'skills/16/title', 'skills/17/title', 'skills/18/title', 'skills/19/title',
    'topSkillsByEndorsements',

    # Projects
    'projects/0/title', 'projects/0/subtitle',
    'projects/0/subComponents/0/description/0/text', 'projects/0/subComponents/0/description/1/text',
    'projects/1/title', 'projects/1/subtitle',
    'projects/1/subComponents/0/description/0/text', 'projects/1/subComponents/0/description/1/text',

    # Recent Activity/Updates (up to 5)
    'updates/0/postText', 'updates/0/numLikes', 'updates/0/numComments',
    'updates/1/postText', 'updates/1/numLikes', 'updates/1/numComments',
    'updates/2/postText', 'updates/2/numLikes', 'updates/2/numComments',
    'updates/3/postText', 'updates/3/numLikes', 'updates/3/numComments',
    'updates/4/postText', 'updates/4/numLikes', 'updates/4/numComments',

    # Achievements & Recognition
    'honorsAndAwards/0/title', 'honorsAndAwards/0/subtitle',
    'honorsAndAwards/0/subComponents/0/description/0/text',
    'honorsAndAwards/1/title', 'honorsAndAwards/1/subtitle',

    # Certifications
    'licenseAndCertificates/0/title', 'licenseAndCertificates/0/subtitle',
    'licenseAndCertificates/1/title', 'licenseAndCertificates/1/subtitle',

    # Publications
    'publications/0/title', 'publications/0/subtitle',
    'publications/0/subComponents/0/description/0/text',
    'publications/1/title', 'publications/1/subtitle',

    # Patents
    'patents/0/title', 'patents/0/subtitle',
    'patents/0/subComponents/0/description/0/text',

    # Volunteer Work
    'volunteerAndAwards/0/title', 'volunteerAndAwards/0/subtitle',
    'volunteerAndAwards/0/subComponents/0/description/0/text',
    'volunteerAndAwards/1/title', 'volunteerAndAwards/1/subtitle',

    # Languages
    'languages/0/title', 'languages/0/caption',
    'languages/1/title', 'languages/1/caption',

    # Organizations
    'organizations/0/title', 'organizations/0/subtitle',
    'organizations/0/subComponents/0/description/0/text',
    'organizations/1/title', 'organizations/1/subtitle',

    # Courses
    'courses/0/title', 'courses/0/subtitle',
    'courses/0/subComponents/0/description/0/text',
    'courses/1/title', 'courses/1/subtitle',

    # Test Scores
    'testScores/0/title', 'testScores/0/subtitle',

    # Highlights
    'highlights/0/title', 'highlights/0/subtitle',
    'highlights/0/subComponents/0/description/0/text',

    # Creator Website
    'creatorWebsite/name', 'creatorWebsite/link'
]

# Global settings
DATASETS_DIR = "./datasets"
OUTPUTS_DIR = "./outputs"
LOGS_DIR = "./logs"
STATE_FILE = "./logs/processing_state.json"
LOG_FILE = "./logs/outreach.log"
POLL_INTERVAL = 10  # seconds
RATE_LIMIT_RETRY_HOURS = 2  # hours to wait on rate limit

class ProcessingStatus(Enum):
    SUCCESS = "success"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class ProcessingResult:
    status: ProcessingStatus
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_after: Optional[datetime] = None

@dataclass
class FileState:
    last_processed: int
    total_rows: int
    completed: bool
    output_file: str
    completed_at: Optional[str] = None

class ProcessingState:
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = self._load()

    def _load(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except:
                logging.warning("Could not load processing state, starting fresh")
        return {"processed_files": {}, "rate_limited_until": None}

    def save(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def get_file_state(self, file_key: str) -> Optional[FileState]:
        file_data = self.state["processed_files"].get(file_key)
        if file_data:
            return FileState(**file_data)
        return None

    def update_file_state(self, file_key: str, file_state: FileState):
        self.state["processed_files"][file_key] = asdict(file_state)
        self.save()

    def is_rate_limited(self) -> bool:
        """Check if we're globally rate limited"""
        if self.state.get("rate_limited_until"):
            return datetime.now() < datetime.fromisoformat(self.state["rate_limited_until"])
        return False

    def set_rate_limit(self, until: datetime):
        """Set global rate limit"""
        self.state["rate_limited_until"] = until.isoformat()
        self.save()

    def clear_rate_limit(self):
        """Clear global rate limit"""
        self.state["rate_limited_until"] = None
        self.save()

class LLMProvider(Enum):
    GEMINI = "gemini"
    OPENAI = "openai" 
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    QWEN = "qwen"

class LLMRotationManager:
    def __init__(self):
        self.providers = []
        self.current_provider_index = 0
        self.rate_limits = {}
        self.consecutive_uses = {}  # Track consecutive uses per provider
        self.max_consecutive_uses = 50  # Use same provider for 50 requests before switching
        
        self._initialize_providers()
        
        # Sort providers by preference (web search first, then by cost)
        self.providers.sort(key=lambda p: (
            0 if p['has_web_search'] else 1,  # Web search providers first
            p['cost_per_1k']  # Then by cost
        ))
    
    def _initialize_providers(self):
        """Initialize all available LLM providers"""
        
        # Gemini (with Google Search)
        if os.getenv('GEMINI_API_KEY'):
            self.providers.append({
                'type': LLMProvider.GEMINI,
                'client': genai.Client(api_key=os.getenv('GEMINI_API_KEY')),
                'model': 'gemini-2.0-flash',
                'has_web_search': True,
                'cost_per_1k': 0.002  # Approximate cost
            })
        
        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            self.providers.append({
                'type': LLMProvider.OPENAI,
                'client': OpenAI(api_key=os.getenv('OPENAI_API_KEY')),
                'model': 'gpt-4o',
                'has_web_search': False,
                'cost_per_1k': 0.005
            })
        
        # Groq
        if os.getenv('GROQ_API_KEY'):
            self.providers.append({
                'type': LLMProvider.GROQ,
                'client': groq.Groq(api_key=os.getenv('GROQ_API_KEY')),
                'model': 'llama-3.3-70b-versatile',
                'has_web_search': False,
                'cost_per_1k': 0.001
            })
        
        # DeepSeek via OpenRouter
        if os.getenv('OPENROUTER_API_KEY'):
            self.providers.append({
                'type': LLMProvider.DEEPSEEK,
                'client': OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv('OPENROUTER_API_KEY')
                ),
                'model': 'deepseek/deepseek-chat-v3-0324:free',
                'has_web_search': False,
                'cost_per_1k': 0.0  # Free tier
            })
        
        # Qwen via OpenRouter  
        if os.getenv('OPENROUTER_API_KEY'):
            self.providers.append({
                'type': LLMProvider.QWEN,
                'client': OpenAI(
                    base_url="https://openrouter.ai/api/v1", 
                    api_key=os.getenv('OPENROUTER_API_KEY')
                ),
                'model': 'qwen/qwen3-coder:free',
                'has_web_search': False,
                'cost_per_1k': 0.0  # Free tier
            })
        
        logging.info(f"🤖 Initialized {len(self.providers)} LLM providers")
        for provider in self.providers:
            logging.info(f"   ✅ {provider['type'].value}: {provider['model']}")
    
    def get_next_available_provider(self):
        """Get next available provider, preferring to stick with current one"""
        if not self.providers:
            raise Exception("No LLM providers available")
        
        current_provider = self.providers[self.current_provider_index]
        
        # Check if we can continue with current provider
        if not self._is_provider_rate_limited(current_provider['type']):
            consecutive_count = self.consecutive_uses.get(current_provider['type'], 0)
            
            # Use same provider until rate limited or hit max consecutive uses
            if consecutive_count < self.max_consecutive_uses:
                self.consecutive_uses[current_provider['type']] = consecutive_count + 1
                return current_provider
        
        # Need to switch provider
        self.consecutive_uses[current_provider['type']] = 0
        
        # Find next available provider
        for i in range(1, len(self.providers)):
            provider_index = (self.current_provider_index + i) % len(self.providers)
            provider = self.providers[provider_index]
            
            if not self._is_provider_rate_limited(provider['type']):
                self.current_provider_index = provider_index
                self.consecutive_uses[provider['type']] = 1
                return provider
        
        # All providers are rate limited
        raise Exception("All LLM providers are currently rate limited")

    def _is_provider_rate_limited(self, provider_type: LLMProvider) -> bool:
        """Check if a specific provider is rate limited"""
        rate_limit_info = self.rate_limits.get(provider_type)
        if not rate_limit_info:
            return False
        
        return datetime.now() < rate_limit_info['retry_after']
    
    def set_provider_rate_limit(self, provider_type: LLMProvider, retry_after: datetime):
        """Set rate limit for a specific provider"""
        self.rate_limits[provider_type] = {
            'retry_after': retry_after,
            'count': self.rate_limits.get(provider_type, {}).get('count', 0) + 1
        }
        logging.warning(f"⏳ {provider_type.value} rate limited until {retry_after}")

class OutreachGenerator:
    def __init__(self):
        self.llm_manager = LLMRotationManager()
        self.shutdown_requested = False

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            logging.info("🛑 Graceful shutdown requested")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Windows compatibility
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    def _create_standardized_prompt(self, person_data: Dict) -> str:
        """Create a standardized prompt optimized for all LLM providers"""
        
        # Extract key information
        full_name = f"{person_data.get('firstName', '')} {person_data.get('lastName', '')}"
        company = person_data.get('companyName', '')
        title = person_data.get('jobTitle', person_data.get('headline', ''))
        industry = person_data.get('companyIndustry', '')
        
        # Build comprehensive context
        context_parts = []
        
        # Basic info
        context_parts.append(f"=== PERSON & COMPANY ===")
        context_parts.append(f"Name: {full_name}")
        context_parts.append(f"Title: {title}")
        context_parts.append(f"Company: {company}")
        context_parts.append(f"Industry: {industry}")
        context_parts.append(f"Email: {person_data.get('email', 'N/A')}")
        
        # About section (key for personalization)
        if person_data.get('about'):
            context_parts.append(f"\n=== ABOUT ===")
            context_parts.append(person_data['about'][:500])  # Limit length
        
        # Recent activity (very important for personalization)
        recent_posts = []
        for i in range(3):  # Only first 3 posts
            post_text = person_data.get(f'updates/{i}/postText')
            if post_text:
                recent_posts.append(f"• {post_text[:200]}...")
        
        if recent_posts:
            context_parts.append(f"\n=== RECENT ACTIVITY ===")
            context_parts.extend(recent_posts[:2])  # Max 2 posts
        
        # Key experience
        current_exp = person_data.get('experiences/0/subComponents/0/description/0/text')
        if current_exp:
            context_parts.append(f"\n=== CURRENT ROLE DETAILS ===")
            context_parts.append(current_exp[:300])
        
        # Skills (condensed)
        skills = []
        for i in range(10):  # Only top 10 skills
            skill = person_data.get(f'skills/{i}/title')
            if skill:
                skills.append(skill)
        
        if skills:
            context_parts.append(f"\n=== KEY SKILLS ===")
            context_parts.append(", ".join(skills))
        
        context = "\n".join(context_parts)
        
        # Create optimized prompt for all LLM providers
        prompt = f"""You are Grant, CEO of Sleft Payments, writing a highly personalized LinkedIn outreach email.

GRANT'S STYLE:
- Casual, friendly, and conversational tone
- Shows genuine interest in their business
- References specific details from their profile/company
- Focuses on partnership and revenue opportunities
- Mentions South Florida location when relevant
- Never sounds templated or salesy

TARGET PERSON:
{context}

SLEFT PAYMENTS FOCUS:
- Payment processing solutions for businesses
- Revenue-share partnerships with banks/credit unions
- Cost reduction and efficiency improvements
- Based in South Florida

CRITICAL INSTRUCTIONS:
1. Research current information about {company} and {industry} trends
2. Reference specific details from their profile (experience, posts, achievements)
3. Connect their role/challenges to payment processing benefits
4. If they're at a bank/credit union, focus on partnership opportunities
5. Mention revenue-share when appropriate
6. Keep it under 120 words
7. Use Grant's casual, friendly style
8. Include specific call-to-action
9. NO template language - make it feel personal and genuine

REQUIRED FORMAT:
Subject: [Compelling 2-4 word subject]

Hi [First name],

[Personalized opening that references something specific about them/their company]
[Ask a genuinly insightful  question about their recent activity or a personal question that gets them to open up, unrelated to business]
[Connect their situation to Sleft's solutions - be specific about benefits]

[Soft call-to-action that feels natural]

Best regards,
Grant
CEO, Sleft Payments
grant@sleftpayments.com
(215) 595-6671

Write the email for {full_name} at {company}:"""

        return prompt

    def generate_outreach(self, person_data: Dict) -> ProcessingResult:
        """Generate outreach with consistent quality across providers"""
        prompt = self._create_standardized_prompt(person_data)  # Updated method name
        
        max_retries = 3  # Reduced retries for faster processing
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                provider = self.llm_manager.get_next_available_provider()
                logging.info(f"🤖 Using {provider['type'].value} ({provider['model']})")
                
                result = self._generate_with_provider(provider, prompt)
                
                if result.status == ProcessingStatus.SUCCESS:
                    return result
                elif result.status == ProcessingStatus.RATE_LIMITED:
                    # Set rate limit for this specific provider
                    self.llm_manager.set_provider_rate_limit(
                        provider['type'], 
                        result.retry_after
                    )
                    retry_count += 1
                    continue
                else:
                    # Other error, try next provider
                    retry_count += 1
                    continue
                    
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    retry_after = datetime.now() + timedelta(hours=1)
                    return ProcessingResult(
                        status=ProcessingStatus.RATE_LIMITED,
                        error_message=str(e),
                        retry_after=retry_after
                    )
                
                logging.error(f"❌ Provider error: {e}")
                retry_count += 1
                continue
        
        # All providers failed
        return ProcessingResult(
            status=ProcessingStatus.ERROR,
            error_message="All LLM providers failed or are rate limited"
        )
    
    def _generate_with_provider(self, provider, prompt) -> ProcessingResult:
        """Generate content with a specific provider"""
        try:
            if provider['type'] == LLMProvider.GEMINI:
                return self._generate_with_gemini(provider, prompt)
            elif provider['type'] in [LLMProvider.OPENAI, LLMProvider.GROQ]:
                return self._generate_with_openai_compatible(provider, prompt)
            elif provider['type'] in [LLMProvider.DEEPSEEK, LLMProvider.QWEN]:
                return self._generate_with_openrouter(provider, prompt)
            
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                retry_after = datetime.now() + timedelta(hours=2)
                return ProcessingResult(
                    status=ProcessingStatus.RATE_LIMITED,
                    error_message=str(e),
                    retry_after=retry_after
                )
            else:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=str(e)
                )
    
    def _generate_with_gemini(self, provider, prompt) -> ProcessingResult:
        """Generate with Gemini - enhanced for consistency"""
        try:
            grounding_tool = types.Tool(google_search=types.GoogleSearch())
            cfg = types.GenerateContentConfig(
                tools=[grounding_tool],
                temperature=0.3,  # Lower temperature for consistency
                max_output_tokens=300
            )
            
            response = provider['client'].models.generate_content(
                model=provider['model'],
                contents=prompt,
                config=cfg,
            )
            
            candidate = response.candidates[0]
            full_text = response.text
            
            # Extract citations
            citations = []
            gm = getattr(candidate, "grounding_metadata", None)
            if gm and hasattr(gm, "grounding_chunks") and gm.grounding_chunks:
                for chunk in gm.grounding_chunks:
                    if hasattr(chunk, "web") and hasattr(chunk.web, "uri"):
                        citations.append(chunk.web.uri)
            
            ok, subject, body = self._parse_email_content(full_text)
            if not ok:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=subject
                )
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data={
                    "subject": subject,
                    "body": body,
                    "full_text": full_text,
                    "citations": citations,
                    "provider": provider['type'].value
                }
            )
            
        except Exception as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                retry_after = datetime.now() + timedelta(hours=1)
                return ProcessingResult(
                    status=ProcessingStatus.RATE_LIMITED,
                    error_message=str(e),
                    retry_after=retry_after
                )
            else:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=str(e)
                )

    def _generate_with_openai_compatible(self, provider, prompt) -> ProcessingResult:
        """Generate with OpenAI-compatible APIs with consistent parameters"""
        try:
            response = provider['client'].chat.completions.create(
                model=provider['model'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Consistent low temperature
                max_tokens=300,   # Consistent token limit
                top_p=0.9
            )
            
            full_text = response.choices[0].message.content
            
            ok, subject, body = self._parse_email_content(full_text)
            if not ok:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=subject
                )
            
            return ProcessingResult(
                status=ProcessingStatus.SUCCESS,
                data={
                    "subject": subject,
                    "body": body,
                    "full_text": full_text,
                    "citations": [],
                    "provider": provider['type'].value
                }
            )
            
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                retry_after = datetime.now() + timedelta(hours=1)
                return ProcessingResult(
                    status=ProcessingStatus.RATE_LIMITED,
                    error_message=str(e),
                    retry_after=retry_after
                )
            else:
                return ProcessingResult(
                    status=ProcessingStatus.ERROR,
                    error_message=str(e)
                )

    def _generate_with_openrouter(self, provider, prompt) -> ProcessingResult:
        """Generate with OpenRouter models (DeepSeek, Qwen)"""
        response = provider['client'].chat.completions.create(
            extra_headers={
                "HTTP-Referer": os.getenv('SITE_URL', 'https://localhost'),
                "X-Title": os.getenv('SITE_NAME', 'Outreach Generator'),
            },
            model=provider['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        full_text = response.choices[0].message.content
        
        ok, subject, body = self._parse_email_content(full_text)
        if not ok:
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=subject
            )
        
        return ProcessingResult(
            status=ProcessingStatus.SUCCESS,
            data={
                "subject": subject,
                "body": body,
                "full_text": full_text,
                "citations": [],  # No web search
                "provider": provider['type'].value
            }
        )

    def _create_outreach_prompt(self, person_data: Dict) -> str:
        """Create the prompt for OpenAI with all available information"""
        # Build comprehensive context about the person
        context_parts = []

        # Basic info
        name = person_data.get('firstName', person_data.get('fullName', 'there'))
        company = person_data.get('companyName', 'their company')
        title = person_data.get('jobTitle', person_data.get('headline', ''))

        context_parts.append(f"=== BASIC INFORMATION ===")
        context_parts.append(f"Name: {name}")
        context_parts.append(f"Full Name: {person_data.get('fullName', 'N/A')}")
        context_parts.append(f"Email: {person_data.get('email', 'N/A')}")
        context_parts.append(f"LinkedIn: {person_data.get('linkedinUrl', 'N/A')}")

        # Current Position
        context_parts.append(f"\\n=== CURRENT POSITION ===")
        if company:
            context_parts.append(f"Company: {company}")
        if title:
            context_parts.append(f"Title: {title}")
        if 'headline' in person_data:
            context_parts.append(f"Headline: {person_data['headline']}")
        if 'currentJobDuration' in person_data:
            context_parts.append(f"Duration: {person_data['currentJobDuration']}")
        if 'currentJobDurationInYrs' in person_data:
            context_parts.append(f"Years: {person_data['currentJobDurationInYrs']}")

        # Company Details
        context_parts.append(f"\\n=== COMPANY DETAILS ===")
        if 'companyIndustry' in person_data:
            context_parts.append(f"Industry: {person_data['companyIndustry']}")
        if 'companySize' in person_data:
            context_parts.append(f"Company Size: {person_data['companySize']}")
        if 'companyFoundedIn' in person_data:
            context_parts.append(f"Founded: {person_data['companyFoundedIn']}")
        if 'companyWebsite' in person_data:
            context_parts.append(f"Website: {person_data['companyWebsite']}")

        # About Section
        if 'about' in person_data:
            context_parts.append(f"\\n=== ABOUT ===")
            context_parts.append(f"{person_data['about']}")

        # Location
        if 'addressWithCountry' in person_data:
            context_parts.append(f"\\n=== LOCATION ===")
            context_parts.append(f"Location: {person_data['addressWithCountry']}")

        # Work Experience
        experiences = []
        for i in range(5):
            exp_title = person_data.get(f'experiences/{i}/title')
            if exp_title:
                exp_detail = f"• {exp_title}"
                if person_data.get(f'experiences/{i}/subtitle'):
                    exp_detail += f" at {person_data[f'experiences/{i}/subtitle']}"
                if person_data.get(f'experiences/{i}/caption'):
                    exp_detail += f" ({person_data[f'experiences/{i}/caption']})"
                experiences.append(exp_detail)

                # Add sub-experiences
                for j in range(3):
                    sub_title = person_data.get(f'experiences/{i}/subComponents/{j}/title')
                    if sub_title:
                        sub_detail = f"  - {sub_title}"
                        if person_data.get(f'experiences/{i}/subComponents/{j}/subtitle'):
                            sub_detail += f": {person_data[f'experiences/{i}/subComponents/{j}/subtitle']}"
                        experiences.append(sub_detail)

                        # Add descriptions
                        for k in range(2):
                            desc = person_data.get(f'experiences/{i}/subComponents/{j}/description/{k}/text')
                            if desc:
                                experiences.append(f"    {desc[:200]}...")

        if experiences:
            context_parts.append(f"\\n=== WORK EXPERIENCE ===")
            context_parts.extend(experiences)

        # Education
        education = []
        for i in range(2):
            edu_title = person_data.get(f'educations/{i}/title')
            if edu_title:
                edu_detail = f"• {edu_title}"
                if person_data.get(f'educations/{i}/subtitle'):
                    edu_detail += f" - {person_data[f'educations/{i}/subtitle']}"
                if person_data.get(f'educations/{i}/caption'):
                    edu_detail += f" ({person_data[f'educations/{i}/caption']})"
                education.append(edu_detail)

        if education:
            context_parts.append(f"\\n=== EDUCATION ===")
            context_parts.extend(education)

        # Skills
        skills = []
        for i in range(20):
            skill = person_data.get(f'skills/{i}/title')
            if skill:
                skills.append(skill)

        if skills:
            context_parts.append(f"\\n=== SKILLS ===")
            context_parts.append(f"Skills: {', '.join(skills[:15])}")  # Limit to 15 for readability
            if len(skills) > 15:
                context_parts.append(f"Additional skills: {', '.join(skills[15:])}")

        if 'topSkillsByEndorsements' in person_data:
            context_parts.append(f"Top Endorsed Skills: {person_data['topSkillsByEndorsements']}")

        # Recent Activity/Posts
        recent_posts = []
        for i in range(5):
            post_text = person_data.get(f'updates/{i}/postText')
            if post_text:
                post_info = f"• Post {i+1}: {post_text[:300]}..."

        if recent_posts:
            context_parts.append(f"\\n=== RECENT LINKEDIN ACTIVITY ===")
            context_parts.extend(recent_posts)

        # Projects
        projects = []
        for i in range(2):
            proj_title = person_data.get(f'projects/{i}/title')
            if proj_title:
                proj_detail = f"• {proj_title}"
                if person_data.get(f'projects/{i}/subtitle'):
                    proj_detail += f" - {person_data[f'projects/{i}/subtitle']}"
                projects.append(proj_detail)

                # Add project descriptions
                for j in range(2):
                    desc = person_data.get(f'projects/{i}/subComponents/0/description/{j}/text')
                    if desc:
                        projects.append(f"  {desc[:200]}...")

        if projects:
            context_parts.append(f"\\n=== PROJECTS ===")
            context_parts.extend(projects)

        # Patents
        if person_data.get('patents/0/title'):
            context_parts.append(f"\\n=== PATENTS ===")
            context_parts.append(f"• {person_data['patents/0/title']}")

        # Organizations
        orgs = []
        for i in range(2):
            org = person_data.get(f'organizations/{i}/title')
            if org:
                orgs.append(f"• {org}")

        if orgs:
            context_parts.append(f"\\n=== ORGANIZATIONS ===")
            context_parts.extend(orgs)

        # Creator Website
        if person_data.get('creatorWebsite/name'):
            context_parts.append(f"\\n=== PERSONAL WEBSITE ===")
            context_parts.append(f"{person_data['creatorWebsite/name']}: {person_data.get('creatorWebsite/link', '')}")

        context = "\\n".join(context_parts)

        prompt = f"""You are an expert at writing personalized cold outreach emails with access to current web information.

You are writing a personalized cold outreach message for Sleft, a payment processing company that helps businesses streamline their payment operations and reduce transaction costs.

COMPREHENSIVE PROFILE DATA:
{context}

CRITICAL: You MUST search the web for current information about:
1. {company} - recent news, developments, funding, product launches, challenges they might be facing
2. Reviews and feedback about {company} - what customers are saying, pain points mentioned
3. {company}'s competitors and their reviews - to understand the competitive landscape and common industry challenges
4. Recent trends in the {person_data.get('companyIndustry', 'their')} industry
5. Any recent professional updates about {name} - new roles, speaking engagements, published content, etc.

IMPORTANT: Analyze ALL the profile data provided above to find unique angles for personalization. Look for:
- Specific projects or achievements that relate to business operations or scaling
- Recent posts or updates that indicate business priorities or challenges
- Skills or experience that suggest they understand the importance of efficient payment systems
- Career trajectory that shows growth focus
- Any hints about their company's expansion plans or operational challenges

Use this current web information AND the detailed profile data to write a highly personalized cold outreach message that:
1. References specific recent developments about their company or industry, but do NOT include URLs
2. Connects to their specific experience, projects, or recent activity early on in the email
3. Demonstrates deep understanding of their role and likely challenges do not mention things like " [Based on your LinkedIn.]"
Do not mention things like "[Based on the provided profile data]" or "[About]" as the message should feel genuine and not reference any pre-templated data.
4. ABSOLUTELY DO NOT INCLUDE ANYTHING TEMPLATED! There should be no "[Company]" or "[Recent projects]"
4. Shows how Sleft's solutions align with their specific situation
5. References something unique from their profile that shows you've done thorough research
6. Includes a soft call-to-action that feels natural
7. Keeps it concise but impactful (under 150 words)
8. Uses a casual and friendly tone that doesn't feel like a mass email, it should feel very personalized and chill and witty,very friendly and do not sound templated or automatic.
9. If the business is clearly a bank or credit union, the focus should be to partner with them to be their in-house payment processor. We are not providing payment procesing solutions for the bank or credit union but rather, their clients.  The goal should be to gain their trust with our due dilligence in this email.
10. When appropriate, there should be a gentle reminder that we offer a revenue-share to banks and credit unions for any clients of theirs that we process payments for.
11. When relevant, mention that we are local to South Florida and would welcome a meeting in person to discuss further but don't sound salesy.
REQUIRED FORMAT - You must respond with EXACTLY this structure:

Subject: [Your compelling subject line here - do not include the word "Subject:" in the actual subject]

[Email body content here with proper greeting and professional close]

Best regards,
Grant
CEO, Sleft Payments
grant@sleftpayments.com
(215) 595-6671

The message should feel like it's written by someone who genuinely understands their business and current market conditions based on fresh web research and deep profile analysis."""

        return prompt

    def _parse_email_content(self, text: str) -> Tuple[bool, str, Optional[str]]:
        """Parse email text to separate subject from body"""
        lines = text.strip().split('\n')

        if not lines:
            return False, "Empty response text", None

        subject = ""
        body_lines = []
        found_subject = False

        for i, line in enumerate(lines):
            line = line.strip()

            # Look for subject line
            if line.lower().startswith('subject:'):
                subject = line[8:].strip()
                found_subject = True
                continue

            # Skip empty lines after subject
            if found_subject and not line:
                continue

            # Start collecting body after subject
            if found_subject:
                body_lines.extend(lines[i:])
                break

        if not found_subject:
            return False, f"No 'Subject:' line found in response\n{text}", None

        if not subject:
            return False, "Empty subject line", None

        body = '\\n'.join(body_lines).strip()

        body = body.replace('\\n', '\n')
        if not body:
            logging.error(f"{text}")
            return False, "Empty email body", None

        # Validate that body contains key elements
        if 'grant' not in body.lower() or 'sleft' not in body.lower():
            return False, "Email body missing required signature elements", None

        return True, subject, body

class CSVProcessor:
    # Essential fields that must be present
    ESSENTIAL_FIELDS = [
        'firstName', 'lastName', 'fullName', 'email',
        'linkedinUrl', 'companyName'
    ]

    def __init__(self, outreach_generator: OutreachGenerator):
        self.generator = outreach_generator
        self.processing_delay = 1  # 1 second delay between requests

    def validate_person_data(self, person_data: Dict) -> bool:
        """Validate that person has all essential fields"""
        for field in self.ESSENTIAL_FIELDS:
            if field not in person_data or not person_data[field]:
                return False
        return True

    def process_person(self, person_data: Dict, row_num: int, total_rows: int) -> ProcessingResult:
        """Process a single person with shutdown checking"""
        full_name = f"{person_data.get('firstName', '')} {person_data.get('lastName', '')}".strip()
        company = person_data.get('companyName', 'Unknown Company')
        
        logging.info(f"[{row_num + 1}/{total_rows}] Processing person")
        logging.info(f"🚀 Processing: {full_name} at {company}")

        # Check for shutdown before starting
        if self.generator.shutdown_requested:
            logging.info("🛑 Shutdown requested, stopping person processing")
            return ProcessingResult(
                status=ProcessingStatus.SHUTDOWN,
                error_message="Shutdown requested"
            )

        try:
            result = self.generator.generate_outreach(person_data)
            
            if result.status == ProcessingStatus.SUCCESS:
                logging.info(f"✅ Successfully generated email for {full_name}")
                return result
            elif result.status == ProcessingStatus.RATE_LIMITED:
                logging.warning(f"⏳ Rate limited while processing {full_name}")
                return result
            else:
                logging.error(f"❌ Failed to generate email for {full_name}: {result.error_message}")
                return result

        except Exception as e:
            logging.error(f"❌ Error processing {full_name}: {e}")
            return ProcessingResult(
                status=ProcessingStatus.ERROR,
                error_message=str(e)
            )
        finally:
            # Only sleep if not shutting down
            if not self.generator.shutdown_requested:
                time.sleep(self.processing_delay)

    def validate_email_content(self, email_subject: str, email_body: str) -> bool:
        """Validate that email content doesn't contain error messages, invalid content, or template placeholders"""
        # Combine subject and body for checking
        full_content = (email_subject + " " + email_body).lower()

        # List of phrases that indicate an error or invalid response
        invalid_phrases = [
            "error",
            "would you like me",
            " api ",
            "openai",
            "http",
            "rate limit",
            "i cannot",
            "i can't",
            "i'm unable",
            "i am unable",
            "i was unable",
            "as an ai",
            "as a language model",
            "i don't have access",
            "i do not have access",
            "web search failed",
            "failed to retrieve",
            "insert"
        ]

        # Check for any invalid phrases
        for phrase in invalid_phrases:
            if phrase in full_content:
                logging.warning(f"❌ Invalid email content detected: contains '{phrase}'")
                return False

        # Check for template placeholders in original content (not lowercased)
        combined_original = email_subject + " " + email_body

        # Check for square brackets - these are almost always placeholders
        if re.search(r'\[[^\]]+\]', combined_original):
            logging.warning(f"❌ Invalid email content: contains square bracket placeholder")
            return False

        # Check for curly braces - these are almost always placeholders
        if re.search(r'\{[^}]+\}', combined_original):
            logging.warning(f"❌ Invalid email content: contains curly brace placeholder")
            return False

        # Check for angle brackets, but exclude email addresses
        angle_matches = re.findall(r'<[^>]+>', combined_original)
        for match in angle_matches:
            # If it's not an email address, it's likely a placeholder
            if not re.match(r'<[^@\s]+@[^@\s]+\.[^@\s]+>$', match):
                logging.warning(f"❌ Invalid email content: contains angle bracket placeholder '{match}'")
                return False

        # Check parentheses more carefully
        paren_matches = re.findall(r'\([^)]+\)', combined_original)
        for match in paren_matches:
            inner = match[1:-1]
            # Allow phone numbers like (215) or area codes
            if re.match(r'^\d{3,4}$', inner):
                continue
            # Allow short abbreviations like (CEO), (USA), (CA)
            if len(inner) <= 4 and inner.isupper():
                continue
            # Check for placeholder-like content
            if any(keyword in inner.lower() for keyword in ['profile', 'name', 'company', 'insert', 'placeholder']):
                logging.warning(f"❌ Invalid email content: contains parenthesis placeholder '{match}'")
                return False

        # Check if email is too short (likely an error)
        if len(email_body) < 50:
            logging.warning("❌ Invalid email content: too short")
            return False

        # Check if subject is too short
        if len(email_subject) < 5:
            logging.warning("❌ Invalid email content: subject too short")
            return False

        return True

    def extract_person_data(self, row: pd.Series, available_columns: List[str]) -> Dict:
        """Extract relevant data for a person"""
        person_data = {}

        for col in available_columns:
            value = row[col]
            if pd.notna(value) and str(value).strip():
                person_data[col] = str(value).strip()

        return person_data

    def load_and_analyze_csv(self, file_path: str) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        """Load CSV and analyze column availability"""
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
            logging.info(f"📊 Loaded {len(df)} rows with {len(df.columns)} columns from {os.path.basename(file_path)}")

            # Check which of our desired columns exist
            available_columns = [col for col in OUTREACH_COLUMNS if col in df.columns]

            logging.info(f"✓ Found {len(available_columns)}/{len(OUTREACH_COLUMNS)} desired columns")
            return df, available_columns

        except Exception as e:
            logging.error(f"✗ Error loading CSV {file_path}: {e}")
            return None, None

    def create_output_csv(self, input_file_path: str) -> str:
        """Create output CSV file with proper headers"""
        input_name = Path(input_file_path).stem
        output_file = os.path.join(OUTPUTS_DIR, f"{input_name}_outreach.csv")

        # Prepare columns for output CSV
        output_columns = [
            'firstName', 'lastName', 'email', 'linkedinUrl', 'companyName', 'jobTitle',
            'emailSubject', 'emailBody', 'citations'
        ]

        # Create empty output file with headers if it doesn't exist
        if not os.path.exists(output_file):
            df_output = pd.DataFrame(columns=output_columns)
            df_output.to_csv(output_file, index=False)

        return output_file

    def append_to_output_csv(self, output_file: str, person_data: Dict, outreach_result: Dict):
        """Append a single result to the output CSV"""
        row_data = {
            'firstName': person_data.get('firstName', ''),
            'lastName': person_data.get('lastName', ''),
            'email': person_data.get('email', ''),
            'linkedinUrl': person_data.get('linkedinUrl', ''),
            'companyName': person_data.get('companyName', ''),
            'jobTitle': person_data.get('jobTitle', ''),
            'emailSubject': outreach_result['subject'],
            'emailBody': outreach_result['body'],
            'citations': ', '.join(outreach_result.get('citations', []))
        }

        # Append to CSV
        df_row = pd.DataFrame([row_data])
        df_row.to_csv(output_file, mode='a', header=False, index=False)

class DaemonController:
    def __init__(self):
        self.state = ProcessingState(STATE_FILE)
        self.generator = None
        self.processor = None
        self.shutdown_requested = False
        self.processing_delay = 1

    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        # Create formatter - avoid emojis on Windows for file logging
        if os.name == 'nt':  # Windows
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        else:  # Unix/Linux/macOS
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        # File handler with UTF-8 encoding
        file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # Console handler - handle both Unix/Linux and Windows
        console_handler = None
        
        if os.name == 'nt':  # Windows
            # On Windows, we'll check if we're being run interactively
            # by checking if stdout is connected to a terminal
            import sys
            if sys.stdout.isatty():
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                console_handler.setLevel(logging.INFO)
        else:  # Unix/Linux/macOS
            try:
                # Check if we're the process group leader (foreground process)
                if os.getpid() == os.getpgrp():
                    console_handler = logging.StreamHandler()
                    console_handler.setFormatter(formatter)
                    console_handler.setLevel(logging.INFO)
            except AttributeError:
                # Fallback if getpgrp is not available on some Unix variants
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                console_handler.setLevel(logging.INFO)
        
        # Setup root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()  # Clear any existing handlers
        logger.addHandler(file_handler)
        
        if console_handler:
            logger.addHandler(console_handler)
        
        # Test logging - use text instead of emoji on Windows
        if os.name == 'nt':
            logger.info(">> Grant Outreach Generator daemon starting...")
        else:
            logger.info("🚀 Grant Outreach Generator daemon starting...")

    def get_csv_files(self) -> List[str]:
        """Get list of CSV files to process"""
        pattern = os.path.join(DATASETS_DIR, "*.csv")
        csv_files = glob.glob(pattern)
        return [f for f in csv_files if os.path.isfile(f)]

    def setup_signal_handlers(self):
        """Setup signal handlers for the daemon controller"""
        def signal_handler(sig, frame):
            logging.info(f"🛑 Signal {sig} received - initiating shutdown")
            self.shutdown_requested = True
            if self.generator:
                self.generator.shutdown_requested = True
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Windows compatibility
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    def initialize(self):
        """Initialize the daemon components"""
        # Create directories
        os.makedirs(DATASETS_DIR, exist_ok=True)
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Setup signal handlers for the controller
        self.setup_signal_handlers()

        # Initialize LLM rotation manager
        self.generator = OutreachGenerator()
        self.generator.setup_signal_handlers()
        self.processor = CSVProcessor(self.generator)

    def process_file(self, file_path: str, test_mode=False, test_run_count=3) -> ProcessingStatus:
        """Process a single CSV file with proper shutdown handling"""
        file_key = os.path.basename(file_path)
        logging.info(f"🚀 Starting processing: {file_key}")

        # Check for shutdown at the beginning
        if self.shutdown_requested:
            logging.info("🛑 Shutdown requested, skipping file processing")
            return ProcessingStatus.SHUTDOWN

        # Get or create file state
        file_state = self.state.get_file_state(file_key)
        if not file_state:
            file_state = FileState(
                last_processed=0,
                total_rows=0,
                completed=False,
                output_file=""
            )

        # Load CSV
        df, available_columns = self.processor.load_and_analyze_csv(file_path)
        if df is None:
            return ProcessingStatus.ERROR

        # Create output file
        output_file = self.processor.create_output_csv(file_path)
        file_state.output_file = output_file
        file_state.total_rows = len(df)

        # Process rows starting from last position
        start_row = file_state.last_processed
        logging.info(f"📋 Processing rows {start_row} to {file_state.total_rows-1}")

        skipped_count = 0
        processed_count = 0

        for idx in range(start_row, file_state.total_rows):
            # Check for shutdown before each person
            if self.shutdown_requested:
                logging.info(f"🛑 Shutdown requested at row {idx}, saving progress")
                file_state.last_processed = idx
                self.state.update_file_state(file_key, file_state)
                return ProcessingStatus.SHUTDOWN

            if test_mode and idx >= start_row + test_run_count:
                sys.exit(0)

            # Process person
            row = df.iloc[idx]
            person_data = self.processor.extract_person_data(row, available_columns)

            # Validate essential fields
            if not self.processor.validate_person_data(person_data):
                skipped_count += 1
                logging.debug(f"[{idx + 1}/{file_state.total_rows}] Skipping row - missing essential fields")
                file_state.last_processed = idx + 1
                continue

            # Use the corrected method signature
            result = self.processor.process_person(person_data, idx, file_state.total_rows)

            if result.status == ProcessingStatus.RATE_LIMITED:
                file_state.last_processed = idx
                self.state.update_file_state(file_key, file_state)
                self.state.set_rate_limit(result.retry_after)
                logging.warning(f"Rate limited until {result.retry_after}")
                return ProcessingStatus.RATE_LIMITED

            elif result.status == ProcessingStatus.ERROR or result.status != ProcessingStatus.SUCCESS:
                logging.error(f"❌ Critical error processing row {idx}: {result.error_message}")
                logging.error("Exiting due to processing error")
                sys.exit(1)

            if not self.processor.validate_email_content(result.data['subject'], result.data['body']):
                logging.error(f"Invalid email found! Dropping it! {result.data}")
                continue

            if test_mode:
                print(f"Subject:\n\t{result.data['subject']}\n\nBody:\n\t{result.data['body']}")

            self.processor.append_to_output_csv(output_file, person_data, result.data)
            file_state.last_processed = idx + 1
            processed_count += 1
            self.state.update_file_state(file_key, file_state)

        # Mark as completed only if we finished normally (not shutdown)
        if not self.shutdown_requested:
            file_state.completed = True
            file_state.completed_at = datetime.now().isoformat()
            self.state.update_file_state(file_key, file_state)

            logging.info(f"🎉 Completed processing {file_key}!")
            logging.info(f"📊 Processed: {processed_count}, Skipped: {skipped_count}")
            logging.info(f"📁 Output saved to: {output_file}")

        return ProcessingStatus.SUCCESS

    def run_daemon_loop(self):
        """Main daemon loop with proper shutdown handling"""
        logging.info("🚀 Grant Outreach Generator daemon started")
        logging.info(f"📁 Monitoring directory: {DATASETS_DIR}")
        logging.info(f"📊 Polling every 10 seconds")

        try:
            while not self.shutdown_requested:
                try:
                    # Check for shutdown at the beginning of each loop
                    if self.shutdown_requested:
                        logging.info("🛑 Shutdown requested, breaking main loop")
                        break
                    
                    # Check for CSV files
                    csv_files = self.get_csv_files()

                    if csv_files:
                        logging.info(f"📂 Found {len(csv_files)} CSV file(s)")

                        for file_path in csv_files:
                            # Check for shutdown before processing each file
                            if self.shutdown_requested:
                                logging.info("🛑 Shutdown requested, stopping file processing")
                                break

                            file_key = os.path.basename(file_path)

                            # Check if already completed
                            file_state = self.state.get_file_state(file_key)
                            if file_state and file_state.completed:
                                logging.debug(f"⏭️  Skipping completed file: {file_key}")
                                continue

                            # Check if we're rate limited
                            if self.state.is_rate_limited():
                                rate_limit_until = datetime.fromisoformat(self.state.state["rate_limited_until"])
                                logging.info(f"⏳ Rate limited until {rate_limit_until}")
                                # Instead of sleeping 60 seconds, sleep in smaller chunks and check for shutdown
                                for _ in range(60):
                                    if self.shutdown_requested:
                                        logging.info("🛑 Shutdown requested during rate limit wait")
                                        break
                                    time.sleep(1)
                                continue

                            # Process the file
                            status = self.process_file(file_path)

                            if status == ProcessingStatus.SHUTDOWN:
                                logging.info("🛑 Received shutdown status from file processing")
                                break
                            elif status == ProcessingStatus.RATE_LIMITED:
                                logging.info(f"⏳ Rate limited, pausing all processing")
                                # Sleep in smaller chunks and check for shutdown
                                for _ in range(60):
                                    if self.shutdown_requested:
                                        logging.info("🛑 Shutdown requested during rate limit wait")
                                        break
                                    time.sleep(1)
                                continue
                        
                        # After processing files, break if shutdown requested
                        if self.shutdown_requested:
                            logging.info("🛑 Shutdown requested after file processing")
                            break
                    else:
                        # No files to process, wait before checking again
                        # Sleep in smaller chunks so we can respond to shutdown quickly
                        for i in range(POLL_INTERVAL):
                            if self.shutdown_requested:
                                logging.info("🛑 Shutdown requested during polling wait")
                                break
                            time.sleep(1)

                except KeyboardInterrupt:
                    logging.info("🛑 Keyboard interrupt received in daemon loop")
                    self.shutdown_requested = True
                    break
                except Exception as e:
                    logging.error(f"❌ Error in daemon loop: {e}")
                    if self.shutdown_requested:
                        logging.info("🛑 Shutdown requested, exiting due to error")
                        break
                    import traceback
                    traceback.print_exc()
                    time.sleep(10)

        except KeyboardInterrupt:
            logging.info("🛑 Keyboard interrupt in main loop")
            self.shutdown_requested = True

        finally:
            logging.info("🛑 Daemon shutdown complete")
            # Clean up PID file
            if os.path.exists("./logs/daemon.pid"):
                try:
                    os.remove("./logs/daemon.pid")
                except:
                    pass

def run_as_daemon():
    """Fork and run as background daemon (Unix/Linux/macOS only)"""
    controller = DaemonController()
    controller.initialize()

    # Check if we're on Windows (can't fork)
    if os.name == 'nt':
        logging.info("🚀 Running in foreground mode on Windows")
        controller.run_daemon_loop()
        return

    try:
        pid = os.fork()
        if pid > 0:
            # Parent process
            print(f"🚀 Started outreach generator daemon (PID: {pid})")
            print(f"📁 Monitoring: {DATASETS_DIR}")
            print(f"📊 Outputs: {OUTPUTS_DIR}")
            print(f"📋 Logs: {LOG_FILE}")
            print(f"🛑 Stop with: kill {pid}")

            # Save PID for easy killing
            os.makedirs(LOGS_DIR, exist_ok=True)
            with open("./logs/daemon.pid", "w") as f:
                f.write(str(pid))

            sys.exit(0)
        else:
            # Child process (daemon)
            # Detach from terminal
            os.setsid()

            # Redirect stdout/stderr to /dev/null (Unix/macOS)
            null_fd = os.open('/dev/null', os.O_RDWR)
            os.dup2(null_fd, sys.stdout.fileno())
            os.dup2(null_fd, sys.stderr.fileno())
            os.close(null_fd)

            # Run the daemon
            controller.run_daemon_loop()

    except OSError as e:
        logging.error(f"❌ Could not fork daemon process: {e}")
        sys.exit(1)

def main():
    controller = DaemonController()
    controller.initialize()
    controller.run_daemon_loop()

if __name__ == "__main__":
    main()
