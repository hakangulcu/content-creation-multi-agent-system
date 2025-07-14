#!/usr/bin/env python3
"""
Content Creation Multi-Agent System
AAIDC Module 2 Project

This script demonstrates a sophisticated multi-agent system for automated content creation.
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# Additional imports for tools
import requests
import re
from urllib.parse import urlparse
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentType(Enum):
    BLOG_POST = "blog_post"
    ARTICLE = "article"
    SOCIAL_MEDIA = "social_media"
    NEWSLETTER = "newsletter"
    MARKETING_COPY = "marketing_copy"

@dataclass
class ContentRequest:
    topic: str
    content_type: ContentType
    target_audience: str
    word_count: int
    tone: str = "professional"
    keywords: List[str] = None
    special_requirements: str = ""

@dataclass
class ResearchData:
    sources: List[str]
    key_facts: List[str]
    statistics: List[str]
    quotes: List[str]
    related_topics: List[str]

@dataclass
class ContentPlan:
    title: str
    outline: List[str]
    key_points: List[str]
    target_keywords: List[str]
    estimated_length: int

@dataclass
class ContentDraft:
    title: str
    content: str
    word_count: int
    reading_time: int

@dataclass
class ContentAnalysis:
    readability_score: float
    grade_level: float
    keyword_density: Dict[str, float]
    suggestions: List[str]

class ContentCreationState(TypedDict):
    """State object that flows through the multi-agent pipeline"""
    request: Optional[ContentRequest]
    research_data: Optional[ResearchData]
    content_plan: Optional[ContentPlan]
    draft: Optional[ContentDraft]
    analysis: Optional[ContentAnalysis]
    final_content: Optional[str]
    feedback_history: List[str]
    revision_count: int
    metadata: Dict[str, Any]

# =============================================================================
# TOOLS DEFINITION
# =============================================================================

@tool
def web_search_tool(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs web search to gather information for content research.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and snippet
    """
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        
        # Parse results (simplified parsing)
        parsed_results = []
        lines = results.split('\n')
        for i, line in enumerate(lines[:max_results]):
            if line.strip():
                parsed_results.append({
                    "title": f"Result {i+1}",
                    "url": "https://example.com",  # Placeholder
                    "snippet": line.strip()
                })
        
        return parsed_results
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return [{"title": "Error", "url": "", "snippet": f"Search failed: {str(e)}"}]

@tool
def content_analysis_tool(content: str) -> Dict[str, Any]:
    """
    Analyzes content for readability, SEO, and quality metrics.
    
    Args:
        content: The content text to analyze
        
    Returns:
        Dictionary containing analysis metrics
    """
    try:
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Calculate readability metrics
        readability = flesch_reading_ease(content)
        grade_level = flesch_kincaid_grade(content)
        
        # Word count and reading time
        word_count = len(content.split())
        reading_time = max(1, word_count // 200)  # ~200 words per minute
        
        # Basic keyword density (simplified)
        words = content.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Only count words longer than 3 characters
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Calculate density for top words
        keyword_density = {}
        total_words = len(words)
        for word, count in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            keyword_density[word] = round((count / total_words) * 100, 2)
        
        return {
            "readability_score": readability,
            "grade_level": grade_level,
            "word_count": word_count,
            "reading_time": reading_time,
            "keyword_density": keyword_density,
            "analysis_timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Content analysis error: {e}")
        return {"error": str(e)}

@tool
def seo_optimization_tool(content: str, target_keywords: List[str]) -> Dict[str, Any]:
    """
    Provides SEO optimization suggestions for content.
    
    Args:
        content: The content to optimize
        target_keywords: List of target keywords
        
    Returns:
        SEO analysis and suggestions
    """
    try:
        suggestions = []
        content_lower = content.lower()
        
        # Check keyword presence
        keyword_analysis = {}
        for keyword in target_keywords:
            count = content_lower.count(keyword.lower())
            keyword_analysis[keyword] = count
            
            if count == 0:
                suggestions.append(f"Consider adding the keyword '{keyword}' to your content")
            elif count > 10:
                suggestions.append(f"Keyword '{keyword}' may be overused ({count} times)")
        
        # Check title and headings
        lines = content.split('\n')
        has_title = any(line.startswith('#') for line in lines)
        if not has_title:
            suggestions.append("Add a compelling title using # markdown")
        
        # Check content length
        word_count = len(content.split())
        if word_count < 300:
            suggestions.append("Content is quite short for SEO - consider expanding")
        elif word_count > 3000:
            suggestions.append("Content is very long - consider breaking into sections")
        
        return {
            "keyword_analysis": keyword_analysis,
            "suggestions": suggestions,
            "word_count": word_count,
            "seo_score": min(100, max(0, 70 - len(suggestions) * 5))  # Simple scoring
        }
    except Exception as e:
        logger.error(f"SEO analysis error: {e}")
        return {"error": str(e)}

@tool
def save_content_tool(content: str, filename: str) -> Dict[str, str]:
    """
    Saves content to a file.
    
    Args:
        content: Content to save
        filename: Name of the file
        
    Returns:
        Save operation result
    """
    try:
        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)
        
        filepath = os.path.join("outputs", filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "status": "success",
            "filepath": filepath,
            "message": f"Content saved to {filepath}"
        }
    except Exception as e:
        logger.error(f"Save error: {e}")
        return {
            "status": "error",
            "message": f"Failed to save: {str(e)}"
        }

# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

class ResearchAgent:
    """Agent responsible for gathering information and research data"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [web_search_tool]
    
    async def research(self, state: ContentCreationState) -> ContentCreationState:
        """Conduct research based on the content request"""
        request = state["request"]
        
        # Create research queries
        queries = [
            request.topic,
            f"{request.topic} statistics",
            f"{request.topic} trends {datetime.now().year}",
            f"{request.topic} expert opinions"
        ]
        
        all_sources = []
        key_facts = []
        statistics = []
        
        # Perform searches
        for query in queries:
            results = web_search_tool.invoke({"query": query, "max_results": 3})
            for result in results:
                if result.get("snippet"):
                    all_sources.append(result["snippet"])
                    
                    # Extract facts and statistics (simplified)
                    snippet = result["snippet"]
                    if any(word in snippet.lower() for word in ["percent", "%", "million", "billion", "study"]):
                        statistics.append(snippet)
                    else:
                        key_facts.append(snippet)
        
        # Create research data
        research_data = ResearchData(
            sources=all_sources[:10],  # Limit to top 10
            key_facts=key_facts[:5],
            statistics=statistics[:3],
            quotes=[],  # Would be extracted with more sophisticated parsing
            related_topics=[]
        )
        
        state["research_data"] = research_data
        state["metadata"]["research_completed"] = datetime.now().isoformat()
        
        logger.info(f"Research completed: {len(all_sources)} sources gathered")
        return state

class PlanningAgent:
    """Agent responsible for creating content structure and plan"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def plan_content(self, state: ContentCreationState) -> ContentCreationState:
        """Create a detailed content plan"""
        request = state["request"]
        research = state["research_data"]
        
        # Create planning prompt
        planning_prompt = f"""
        Create a detailed content plan for a {request.content_type.value} about "{request.topic}".
        
        Content Requirements:
        - Target audience: {request.target_audience}
        - Word count: {request.word_count}
        - Tone: {request.tone}
        - Keywords: {request.keywords or 'None specified'}
        
        Available Research:
        Key Facts: {research.key_facts[:3] if research.key_facts else 'None'}
        Statistics: {research.statistics[:2] if research.statistics else 'None'}
        
        Create a content plan with:
        1. Compelling title
        2. Detailed outline (5-7 main sections)
        3. Key points for each section
        4. Target keywords to include
        5. Estimated word count distribution
        
        Format as JSON with keys: title, outline, key_points, target_keywords, estimated_length
        """
        
        messages = [SystemMessage(content="You are an expert content strategist."), 
                   HumanMessage(content=planning_prompt)]
        
        response = await self.llm.ainvoke(messages)
        
        # Parse response (simplified - in real implementation, use structured output)
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            else:
                json_str = content
            
            plan_data = json.loads(json_str)
        except:
            # Fallback plan
            plan_data = {
                "title": f"Complete Guide to {request.topic}",
                "outline": [
                    "Introduction",
                    "Background and Context", 
                    "Key Concepts",
                    "Practical Applications",
                    "Best Practices",
                    "Future Trends",
                    "Conclusion"
                ],
                "key_points": [
                    "Engage reader with hook",
                    "Provide comprehensive overview",
                    "Include actionable insights",
                    "Support with data and examples"
                ],
                "target_keywords": request.keywords or [request.topic],
                "estimated_length": request.word_count
            }
        
        content_plan = ContentPlan(
            title=plan_data.get("title", f"Guide to {request.topic}"),
            outline=plan_data.get("outline", []),
            key_points=plan_data.get("key_points", []),
            target_keywords=plan_data.get("target_keywords", []),
            estimated_length=plan_data.get("estimated_length", request.word_count)
        )
        
        state["content_plan"] = content_plan
        state["metadata"]["planning_completed"] = datetime.now().isoformat()
        
        logger.info(f"Content plan created: {content_plan.title}")
        return state

class WriterAgent:
    """Agent responsible for creating the initial content draft"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def write_content(self, state: ContentCreationState) -> ContentCreationState:
        """Generate content based on the plan and research"""
        request = state["request"]
        plan = state["content_plan"]
        research = state["research_data"]
        
        # Create writing prompt
        writing_prompt = f"""
        Write a high-quality {request.content_type.value} based on the following plan:
        
        Title: {plan.title}
        Target Length: {request.word_count} words
        Target Audience: {request.target_audience}
        Tone: {request.tone}
        
        Outline:
        {chr(10).join([f"- {section}" for section in plan.outline])}
        
        Key Points to Include:
        {chr(10).join([f"- {point}" for point in plan.key_points])}
        
        Research Data to Incorporate:
        {chr(10).join([f"- {fact}" for fact in research.key_facts[:5]])}
        
        Statistics to Include:
        {chr(10).join([f"- {stat}" for stat in research.statistics[:3]])}
        
        Requirements:
        1. Write engaging, well-structured content
        2. Include all sections from the outline
        3. Incorporate research data naturally
        4. Maintain consistent tone throughout
        5. Use markdown formatting for headings
        6. Target approximately {request.word_count} words
        
        Write the complete article now:
        """
        
        messages = [
            SystemMessage(content="You are an expert content writer known for creating engaging, informative content."),
            HumanMessage(content=writing_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        content = response.content
        
        # Calculate metrics
        word_count = len(content.split())
        reading_time = max(1, word_count // 200)
        
        draft = ContentDraft(
            title=plan.title,
            content=content,
            word_count=word_count,
            reading_time=reading_time
        )
        
        state["draft"] = draft
        state["metadata"]["writing_completed"] = datetime.now().isoformat()
        
        logger.info(f"Content draft created: {word_count} words")
        return state

class EditorAgent:
    """Agent responsible for editing and improving content quality"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [content_analysis_tool]
    
    async def edit_content(self, state: ContentCreationState) -> ContentCreationState:
        """Edit and improve the content draft"""
        draft = state["draft"]
        request = state["request"]
        
        # Analyze current content
        analysis_result = content_analysis_tool.invoke({"content": draft.content})
        
        # Create editing prompt
        editing_prompt = f"""
        Please edit and improve the following content:
        
        Original Content:
        {draft.content}
        
        Current Analysis:
        - Word count: {analysis_result.get('word_count', 'Unknown')}
        - Readability score: {analysis_result.get('readability_score', 'Unknown')}
        - Grade level: {analysis_result.get('grade_level', 'Unknown')}
        
        Target Requirements:
        - Word count: {request.word_count}
        - Tone: {request.tone}
        - Audience: {request.target_audience}
        
        Please improve the content by:
        1. Enhancing clarity and flow
        2. Improving sentence structure
        3. Adding transitions between sections
        4. Ensuring consistent tone
        5. Optimizing for readability
        6. Adjusting length to meet target word count
        
        Return the edited content:
        """
        
        messages = [
            SystemMessage(content="You are an expert editor focused on clarity, engagement, and quality."),
            HumanMessage(content=editing_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        edited_content = response.content
        
        # Update draft
        word_count = len(edited_content.split())
        reading_time = max(1, word_count // 200)
        
        state["draft"] = ContentDraft(
            title=draft.title,
            content=edited_content,
            word_count=word_count,
            reading_time=reading_time
        )
        
        # Store analysis
        final_analysis = content_analysis_tool.invoke({"content": edited_content})
        state["analysis"] = ContentAnalysis(
            readability_score=final_analysis.get('readability_score', 0),
            grade_level=final_analysis.get('grade_level', 0),
            keyword_density=final_analysis.get('keyword_density', {}),
            suggestions=[]
        )
        
        state["metadata"]["editing_completed"] = datetime.now().isoformat()
        
        logger.info(f"Content edited: {word_count} words, readability: {final_analysis.get('readability_score', 'N/A')}")
        return state

class SEOAgent:
    """Agent responsible for SEO optimization"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [seo_optimization_tool]
    
    async def optimize_seo(self, state: ContentCreationState) -> ContentCreationState:
        """Optimize content for SEO"""
        draft = state["draft"]
        plan = state["content_plan"]
        
        # Perform SEO analysis
        seo_result = seo_optimization_tool.invoke({
            "content": draft.content,
            "target_keywords": plan.target_keywords
        })
        
        # If there are suggestions, apply them
        if seo_result.get("suggestions"):
            optimization_prompt = f"""
            Optimize the following content for SEO based on these suggestions:
            
            Content:
            {draft.content}
            
            SEO Suggestions:
            {chr(10).join([f"- {suggestion}" for suggestion in seo_result['suggestions']])}
            
            Target Keywords: {plan.target_keywords}
            
            Please apply the suggestions while maintaining content quality and readability.
            Return the optimized content:
            """
            
            messages = [
                SystemMessage(content="You are an SEO expert who optimizes content while maintaining quality."),
                HumanMessage(content=optimization_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            optimized_content = response.content
            
            # Update draft
            word_count = len(optimized_content.split())
            reading_time = max(1, word_count // 200)
            
            state["draft"] = ContentDraft(
                title=draft.title,
                content=optimized_content,
                word_count=word_count,
                reading_time=reading_time
            )
        
        state["metadata"]["seo_optimization_completed"] = datetime.now().isoformat()
        state["metadata"]["seo_score"] = seo_result.get("seo_score", 0)
        
        logger.info(f"SEO optimization completed. Score: {seo_result.get('seo_score', 'N/A')}")
        return state

class QualityAssuranceAgent:
    """Agent responsible for final quality check and delivery"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [save_content_tool]
    
    async def finalize_content(self, state: ContentCreationState) -> ContentCreationState:
        """Perform final quality check and save content"""
        draft = state["draft"]
        request = state["request"]
        
        # Final quality check
        quality_prompt = f"""
        Perform a final quality check on this content:
        
        Content:
        {draft.content}
        
        Requirements Check:
        - Target word count: {request.word_count} (Actual: {draft.word_count})
        - Target audience: {request.target_audience}
        - Tone: {request.tone}
        - Content type: {request.content_type.value}
        
        Please review and provide:
        1. Overall quality score (1-10)
        2. Whether it meets requirements
        3. Any final suggestions
        4. Approval status (APPROVED/NEEDS_REVISION)
        
        Format as: SCORE: X/10, STATUS: [APPROVED/NEEDS_REVISION], NOTES: [your notes]
        """
        
        messages = [
            SystemMessage(content="You are a quality assurance specialist ensuring content meets all requirements."),
            HumanMessage(content=quality_prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        qa_feedback = response.content
        
        # Save the final content
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{request.topic.replace(' ', '_')}_{timestamp}.md"
        
        # Create final content with metadata
        final_content = f"""# {draft.title}

**Content Type:** {request.content_type.value}
**Target Audience:** {request.target_audience}
**Word Count:** {draft.word_count}
**Reading Time:** {draft.reading_time} minutes
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

{draft.content}

---

**Quality Assurance Report:**
{qa_feedback}

**Generation Metadata:**
- Research completed: {state["metadata"].get('research_completed', 'N/A')}
- Planning completed: {state["metadata"].get('planning_completed', 'N/A')}
- Writing completed: {state["metadata"].get('writing_completed', 'N/A')}
- Editing completed: {state["metadata"].get('editing_completed', 'N/A')}
- SEO optimization completed: {state["metadata"].get('seo_optimization_completed', 'N/A')}
- SEO Score: {state["metadata"].get('seo_score', 'N/A')}
"""
        
        # Save content
        save_result = save_content_tool.invoke({
            "content": final_content,
            "filename": filename
        })
        
        state["final_content"] = final_content
        state["metadata"]["qa_completed"] = datetime.now().isoformat()
        state["metadata"]["output_file"] = save_result.get("filepath", "")
        state["feedback_history"].append(qa_feedback)
        
        logger.info(f"Content finalized and saved: {save_result.get('filepath', 'N/A')}")
        return state

# =============================================================================
# LANGGRAPH WORKFLOW DEFINITION
# =============================================================================

class ContentCreationWorkflow:
    """Main workflow orchestrator using LangGraph"""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        # Initialize local Ollama LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
            base_url=base_url,
            # Additional parameters for better performance
            num_predict=4096,  # Max tokens to generate
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1
        )
        
        # Initialize agents
        self.research_agent = ResearchAgent(self.llm)
        self.planning_agent = PlanningAgent(self.llm)
        self.writer_agent = WriterAgent(self.llm)
        self.editor_agent = EditorAgent(self.llm)
        self.seo_agent = SEOAgent(self.llm)
        self.qa_agent = QualityAssuranceAgent(self.llm)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(ContentCreationState)
        
        # Add nodes (agents)
        workflow.add_node("research", self.research_agent.research)
        workflow.add_node("planning", self.planning_agent.plan_content)
        workflow.add_node("writing", self.writer_agent.write_content)
        workflow.add_node("editing", self.editor_agent.edit_content)
        workflow.add_node("seo_optimization", self.seo_agent.optimize_seo)
        workflow.add_node("quality_assurance", self.qa_agent.finalize_content)
        
        # Define the workflow edges
        workflow.add_edge("research", "planning")
        workflow.add_edge("planning", "writing")
        workflow.add_edge("writing", "editing")
        workflow.add_edge("editing", "seo_optimization")
        workflow.add_edge("seo_optimization", "quality_assurance")
        workflow.add_edge("quality_assurance", END)
        
        # Set entry point
        workflow.set_entry_point("research")
        
        # Compile the workflow
        return workflow.compile()
    
    async def create_content(self, content_request: ContentRequest) -> ContentCreationState:
        """Execute the complete content creation workflow"""
        
        # Initialize state as a dictionary
        state: ContentCreationState = {
            "request": content_request,
            "research_data": None,
            "content_plan": None,
            "draft": None,
            "analysis": None,
            "final_content": None,
            "feedback_history": [],
            "revision_count": 0,
            "metadata": {}
        }
        
        logger.info(f"Starting content creation for: {content_request.topic}")
        
        # Execute workflow
        final_state = await self.workflow.ainvoke(state)
        
        logger.info("Content creation workflow completed successfully")
        return final_state

# =============================================================================
# DEMO AND TESTING
# =============================================================================

async def demo_content_creation():
    """Demo function to showcase the multi-agent system"""
    
    # Get Ollama configuration from environment (with defaults)
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"🤖 Using Ollama model: {model_name}")
    print(f"🌐 Ollama server: {base_url}")
    
    # Create workflow
    try:
        workflow = ContentCreationWorkflow(model_name=model_name, base_url=base_url)
        print("✅ Ollama connection established")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        print(f"💡 And the model is installed: ollama pull {model_name}")
        return
    
    # Define a content request
    request = ContentRequest(
        topic="Artificial Intelligence in Healthcare",
        content_type=ContentType.BLOG_POST,
        target_audience="Healthcare professionals and technology leaders",
        word_count=1500,
        tone="professional yet accessible",
        keywords=["AI in healthcare", "medical AI", "healthcare technology", "patient care"],
        special_requirements="Include recent statistics and real-world examples"
    )
    
    print("🚀 Starting Multi-Agent Content Creation System")
    print(f"📝 Topic: {request.topic}")
    print(f"🎯 Target: {request.target_audience}")
    print(f"📊 Length: {request.word_count} words")
    print("-" * 60)
    
    try:
        # Execute workflow
        result = await workflow.create_content(request)
        
        print("\n✅ Content Creation Completed Successfully!")
        print(f"📄 Final word count: {result['draft'].word_count}")
        print(f"⏱️ Reading time: {result['draft'].reading_time} minutes")
        print(f"📁 Saved to: {result['metadata'].get('output_file', 'N/A')}")
        print(f"🔍 SEO Score: {result['metadata'].get('seo_score', 'N/A')}")
        
        # Display content preview
        if result["final_content"]:
            print("\n📖 Content Preview:")
            print("-" * 40)
            preview = result["final_content"][:500] + "..." if len(result["final_content"]) > 500 else result["final_content"]
            print(preview)
        
    except Exception as e:
        print(f"❌ Error during content creation: {e}")
        logger.error(f"Content creation failed: {e}")

def main():
    """Main entry point"""
    print("Content Creation Multi-Agent System")
    print("AAIDC Module 2 Project")
    print("=" * 50)
    
    # Run demo
    asyncio.run(demo_content_creation())

if __name__ == "__main__":
    main()