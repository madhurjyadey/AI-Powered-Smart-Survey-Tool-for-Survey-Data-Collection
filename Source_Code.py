import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import re
from collections import Counter
import json


# Configure page
st.set_page_config(
    page_title="AI-Powered Smart Survey Tool",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = {}
if 'current_survey' not in st.session_state:
    st.session_state.current_survey = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'responses' not in st.session_state:
    st.session_state.responses = {}
if 'survey_completed' not in st.session_state:
    st.session_state.survey_completed = False
if 'ai_analysis' not in st.session_state:
    st.session_state.ai_analysis = None

# Survey templates
SURVEY_TEMPLATES = {
    "customer_satisfaction": {
        "title": "Customer Experience Survey",
        "description": "AI-enhanced survey to understand customer satisfaction patterns",
        "questions": [
            {
                "id": 1,
                "type": "rating",
                "text": "How satisfied are you with our service overall?",
                "scale": 5,
                "ai_context": "Primary satisfaction metric",
                "required": True
            },
            {
                "id": 2,
                "type": "multiple_choice",
                "text": "What's the primary reason for your rating?",
                "options": ["Product Quality", "Customer Service", "Pricing", "User Experience", "Other"],
                "ai_context": "Follow-up based on rating",
                "required": True
            },
            {
                "id": 3,
                "type": "text",
                "text": "What specific improvements would you like to see?",
                "ai_analysis": True,
                "ai_context": "Sentiment and theme analysis",
                "required": False
            },
            {
                "id": 4,
                "type": "rating",
                "text": "How likely are you to recommend us to others? (1-10)",
                "scale": 10,
                "ai_context": "NPS calculation",
                "required": True
            }
        ]
    },
    "employee_feedback": {
        "title": "Employee Engagement Survey",
        "description": "Smart survey with adaptive questioning based on responses",
        "questions": [
            {
                "id": 1,
                "type": "rating",
                "text": "How engaged do you feel at work?",
                "scale": 5,
                "ai_context": "Core engagement metric",
                "required": True
            },
            {
                "id": 2,
                "type": "multiple_choice",
                "text": "What motivates you most at work?",
                "options": ["Career Growth", "Recognition", "Work-Life Balance", "Compensation", "Team Collaboration"],
                "ai_context": "Motivation analysis",
                "required": True
            },
            {
                "id": 3,
                "type": "text",
                "text": "What changes would improve your work experience?",
                "ai_analysis": True,
                "ai_context": "Theme extraction and categorization",
                "required": False
            },
            {
                "id": 4,
                "type": "rating",
                "text": "How would you rate your work-life balance?",
                "scale": 5,
                "ai_context": "Work-life balance assessment",
                "required": True
            }
        ]
    }
}

# AI Analysis Functions
class AIAnalyzer:
    @staticmethod
    def analyze_sentiment(text):
        """Simple sentiment analysis based on keywords"""
        if not text or len(text.strip()) < 3:
            return "neutral"
        
        text = text.lower()
        positive_words = ['good', 'great', 'excellent', 'amazing', 'fantastic', 'love', 'perfect', 'wonderful', 'outstanding', 'satisfied', 'happy', 'pleased']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'disappointing', 'frustrated', 'angry', 'unsatisfied', 'poor', 'worst']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    @staticmethod
    def extract_themes(text_responses):
        """Extract common themes from text responses"""
        if not text_responses:
            return []
        
        all_text = " ".join(text_responses).lower()
        
        # Common themes and keywords
        theme_keywords = {
            "Product Quality": ["quality", "product", "item", "goods", "material"],
            "Customer Service": ["service", "support", "staff", "help", "representative", "agent"],
            "Pricing": ["price", "cost", "expensive", "cheap", "value", "money", "fee"],
            "User Experience": ["experience", "interface", "usability", "easy", "difficult", "navigation"],
            "Delivery": ["delivery", "shipping", "fast", "slow", "package", "arrive"],
            "Communication": ["communication", "contact", "response", "email", "phone", "chat"]
        }
        
        themes = []
        for theme, keywords in theme_keywords.items():
            mentions = sum(1 for keyword in keywords if keyword in all_text)
            if mentions > 0:
                # Analyze sentiment for this theme
                theme_text = " ".join([resp for resp in text_responses if any(kw in resp.lower() for kw in keywords)])
                sentiment = AIAnalyzer.analyze_sentiment(theme_text)
                themes.append({
                    "theme": theme,
                    "mentions": mentions,
                    "sentiment": sentiment
                })
        
        return sorted(themes, key=lambda x: x["mentions"], reverse=True)[:6]
    
    @staticmethod
    def generate_recommendations(responses, survey_type="customer_satisfaction"):
        """Generate AI-powered recommendations based on responses"""
        recommendations = []
        
        # Analyze ratings
        ratings = [v for v in responses.values() if isinstance(v, (int, float))]
        avg_rating = np.mean(ratings) if ratings else 0
        
        # Analyze text responses
        text_responses = [v for v in responses.values() if isinstance(v, str) and len(v.strip()) > 10]
        themes = AIAnalyzer.extract_themes(text_responses)
        
        # Generate recommendations based on analysis
        if avg_rating < 3:
            recommendations.append("⚠️ Low satisfaction scores detected. Immediate attention required to address customer concerns.")
        
        if themes:
            negative_themes = [t for t in themes if t["sentiment"] == "negative"]
            if negative_themes:
                top_issue = negative_themes[0]["theme"]
                recommendations.append(f"🔍 Focus on improving {top_issue} - identified as the primary concern area.")
            
            positive_themes = [t for t in themes if t["sentiment"] == "positive"]
            if positive_themes:
                strength = positive_themes[0]["theme"]
                recommendations.append(f"✅ Leverage your strength in {strength} for marketing and competitive advantage.")
        
        # NPS-specific recommendations
        if survey_type == "customer_satisfaction":
            nps_responses = [v for k, v in responses.items() if "recommend" in str(k).lower()]
            if nps_responses:
                nps_score = np.mean(nps_responses)
                if nps_score >= 9:
                    recommendations.append("🌟 High NPS score! Consider implementing a referral program.")
                elif nps_score <= 6:
                    recommendations.append("📉 Low NPS score indicates risk of customer churn. Implement retention strategies.")
        
        if not recommendations:
            recommendations.append("📊 Continue monitoring feedback patterns and maintain current service quality.")
        
        return recommendations

def create_survey_page():
    """Survey creation and selection page"""
    st.title("🧠 AI-Powered Smart Survey Tool")
    st.markdown("### Create intelligent surveys with adaptive questioning and AI-powered analysis")
    
    col1, col2 = st.columns(2)
    
    for idx, (key, template) in enumerate(SURVEY_TEMPLATES.items()):
        col = col1 if idx % 2 == 0 else col2
        
        with col:
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin: 10px 0; background: white;">
                    <h3 style="color: #1f77b4; margin-top: 0;">{template['title']}</h3>
                    <p style="color: #666; margin: 10px 0;">{template['description']}</p>
                    <p style="font-size: 14px; color: #888;">
                        📝 {len(template['questions'])} questions | 🧠 AI-Enhanced
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Start {template['title']}", key=f"start_{key}", type="primary"):
                    st.session_state.current_survey = template
                    st.session_state.current_question = 0
                    st.session_state.responses = {}
                    st.session_state.survey_completed = False
                    st.session_state.ai_analysis = None
                    st.rerun()
    
    # AI Features Section
    st.markdown("---")
    st.markdown("### 🚀 AI Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        **📈 Adaptive Questioning**
        - Questions adapt based on previous responses
        - Smart conditional logic
        """)
    
    with feature_col2:
        st.markdown("""
        **💬 Sentiment Analysis**
        - AI analyzes emotional tone in text
        - Automatic theme extraction
        """)
    
    with feature_col3:
        st.markdown("""
        **📊 Smart Insights**
        - Automated analysis and recommendations
        - Visual data representation
        """)

def take_survey_page():
    """Survey taking interface"""
    survey = st.session_state.current_survey
    current_q_idx = st.session_state.current_question
    
    if current_q_idx >= len(survey["questions"]):
        st.session_state.survey_completed = True
        st.rerun()
        return
    
    question = survey["questions"][current_q_idx]
    
    # Progress bar
    progress = (current_q_idx + 1) / len(survey["questions"])
    st.progress(progress, text=f"Question {current_q_idx + 1} of {len(survey['questions'])}")
    
    st.title(survey["title"])
    
    # Question container
    with st.container():
        st.markdown("---")
        st.markdown(f"### {question['text']}")
        
        # AI Context indicator
        if question.get("ai_context"):
            st.info(f"🧠 AI Context: {question['ai_context']}")
        
        response = None
        
        # Different question types
        if question["type"] == "rating":
            response = st.slider(
                "Your rating:",
                min_value=1,
                max_value=question["scale"],
                value=st.session_state.responses.get(question["id"], 1),
                key=f"q_{question['id']}"
            )
            
            # Show rating labels
            if question["scale"] == 5:
                st.markdown("1 = Very Poor | 2 = Poor | 3 = Average | 4 = Good | 5 = Excellent")
            elif question["scale"] == 10:
                st.markdown("1 = Not at all likely | 10 = Extremely likely")
        
        elif question["type"] == "multiple_choice":
            response = st.radio(
                "Select your answer:",
                options=question["options"],
                index=question["options"].index(st.session_state.responses.get(question["id"], question["options"][0])) if st.session_state.responses.get(question["id"]) in question["options"] else 0,
                key=f"q_{question['id']}"
            )
        
        elif question["type"] == "text":
            response = st.text_area(
                "Your response:",
                value=st.session_state.responses.get(question["id"], ""),
                height=100,
                key=f"q_{question['id']}",
                placeholder="Share your thoughts..."
            )
            
            if question.get("ai_analysis"):
                st.success("🧠 AI will analyze sentiment and themes from this response")
        
        # Store response
        if response is not None:
            st.session_state.responses[question["id"]] = response
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_q_idx > 0:
            if st.button("⬅️ Previous", type="secondary"):
                st.session_state.current_question -= 1
                st.rerun()
    
    with col3:
        if current_q_idx < len(survey["questions"]) - 1:
            if st.button("Next ➡️", type="primary"):
                if question.get("required") and not response:
                    st.error("This question is required!")
                else:
                    st.session_state.current_question += 1
                    st.rerun()
        else:
            if st.button("Complete Survey ✅", type="primary"):
                if question.get("required") and not response:
                    st.error("This question is required!")
                else:
                    st.session_state.survey_completed = True
                    st.rerun()

def analyze_responses():
    """Perform AI analysis on survey responses"""
    if st.session_state.ai_analysis is None:
        with st.spinner("🧠 AI is analyzing responses..."):
            time.sleep(2)  # Simulate processing time
            
            responses = st.session_state.responses
            survey = st.session_state.current_survey
            
            # Extract text responses for analysis
            text_responses = []
            for q_id, response in responses.items():
                question = next((q for q in survey["questions"] if q["id"] == q_id), None)
                if question and question.get("ai_analysis") and isinstance(response, str):
                    text_responses.append(response)
            
            # Perform analysis
            themes = AIAnalyzer.extract_themes(text_responses)
            
            # Calculate sentiment distribution
            sentiments = [AIAnalyzer.analyze_sentiment(text) for text in text_responses]
            sentiment_counts = Counter(sentiments)
            total_sentiments = len(sentiments) if sentiments else 1
            
            sentiment_dist = {
                "positive": (sentiment_counts.get("positive", 0) / total_sentiments) * 100,
                "neutral": (sentiment_counts.get("neutral", 0) / total_sentiments) * 100,
                "negative": (sentiment_counts.get("negative", 0) / total_sentiments) * 100
            }
            
            # Calculate metrics
            ratings = [v for v in responses.values() if isinstance(v, (int, float))]
            avg_satisfaction = np.mean(ratings) if ratings else 0
            
            # Calculate NPS (assuming last rating question is NPS)
            nps_responses = []
            for q_id, response in responses.items():
                question = next((q for q in survey["questions"] if q["id"] == q_id), None)
                if question and "recommend" in question["text"].lower() and isinstance(response, (int, float)):
                    nps_responses.append(response)
            
            nps_score = 0
            if nps_responses:
                promoters = sum(1 for score in nps_responses if score >= 9)
                detractors = sum(1 for score in nps_responses if score <= 6)
                nps_score = ((promoters - detractors) / len(nps_responses)) * 100
            
            # Generate recommendations
            survey_type = "customer_satisfaction" if "customer" in survey["title"].lower() else "employee_feedback"
            recommendations = AIAnalyzer.generate_recommendations(responses, survey_type)
            
            st.session_state.ai_analysis = {
                "sentiment_distribution": sentiment_dist,
                "themes": themes,
                "avg_satisfaction": avg_satisfaction,
                "nps_score": nps_score,
                "recommendations": recommendations,
                "total_responses": len(responses)
            }

def analysis_page():
    """Analysis results page"""
    st.title("📊 AI Analysis Results")
    st.markdown("### Intelligent insights from survey responses")
    
    analyze_responses()
    analysis = st.session_state.ai_analysis
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Satisfaction", f"{analysis['avg_satisfaction']:.1f}/5" if analysis['avg_satisfaction'] <= 5 else f"{analysis['avg_satisfaction']:.1f}/10")
    
    with col2:
        st.metric("NPS Score", f"{analysis['nps_score']:.0f}")
    
    with col3:
        st.metric("Total Responses", analysis['total_responses'])
    
    with col4:
        st.metric("Key Themes", len(analysis['themes']))
    
    # Charts section
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("🧠 Sentiment Analysis")
        if any(analysis['sentiment_distribution'].values()):
            fig_sentiment = px.pie(
                values=list(analysis['sentiment_distribution'].values()),
                names=list(analysis['sentiment_distribution'].keys()),
                title="Response Sentiment Distribution",
                color_discrete_map={
                    'positive': '#28a745',
                    'neutral': '#6c757d',
                    'negative': '#dc3545'
                }
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        else:
            st.info("No text responses available for sentiment analysis")
    
    with chart_col2:
        st.subheader("💬 Key Themes")
        if analysis['themes']:
            theme_df = pd.DataFrame(analysis['themes'])
            fig_themes = px.bar(
                theme_df,
                x='mentions',
                y='theme',
                color='sentiment',
                orientation='h',
                title="Most Mentioned Themes",
                color_discrete_map={
                    'positive': '#28a745',
                    'neutral': '#6c757d',
                    'negative': '#dc3545'
                }
            )
            st.plotly_chart(fig_themes, use_container_width=True)
        else:
            st.info("No themes identified from responses")
    
    # AI Recommendations
    st.subheader("🎯 AI Recommendations")
    for i, recommendation in enumerate(analysis['recommendations'], 1):
        st.info(f"{i}. {recommendation}")
    
    # Raw Data Section
    with st.expander("📋 View Raw Response Data"):
        responses_df = pd.DataFrame([
            {"Question ID": k, "Response": v} 
            for k, v in st.session_state.responses.items()
        ])
        st.dataframe(responses_df, use_container_width=True)
    
    # Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Analysis", type="secondary"):
            # Create exportable data
            export_data = {
                "survey_title": st.session_state.current_survey["title"],
                "analysis_date": datetime.now().isoformat(),
                "metrics": {
                    "avg_satisfaction": analysis['avg_satisfaction'],
                    "nps_score": analysis['nps_score'],
                    "total_responses": analysis['total_responses']
                },
                "sentiment_distribution": analysis['sentiment_distribution'],
                "themes": analysis['themes'],
                "recommendations": analysis['recommendations'],
                "responses": st.session_state.responses
            }
            
            st.download_button(
                label="Download JSON Report",
                data=json.dumps(export_data, indent=2),
                file_name=f"survey_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("🔄 Analyze Again", type="secondary"):
            st.session_state.ai_analysis = None
            st.rerun()
    
    with col3:
        if st.button("🏠 Create New Survey", type="primary"):
            # Reset all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

def main():
    """Main application logic"""
    # Sidebar
    with st.sidebar:
        st.markdown("### 🧠 AI Survey Tool")
        st.markdown("---")
        
        if st.session_state.current_survey is None:
            st.info("Select a survey template to get started")
        elif not st.session_state.survey_completed:
            st.success(f"Taking: {st.session_state.current_survey['title']}")
            st.progress((st.session_state.current_question + 1) / len(st.session_state.current_survey['questions']))
        else:
            st.success("Survey completed! Viewing analysis.")
        
        st.markdown("---")
        st.markdown("""
        **Features:**
        - 🎯 Adaptive questioning
        - 🧠 AI sentiment analysis
        - 📊 Theme extraction
        - 💡 Smart recommendations
        - 📈 Visual analytics
        """)
        
        if st.button("🔄 Reset All", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    if st.session_state.current_survey is None:
        create_survey_page()
    elif not st.session_state.survey_completed:
        take_survey_page()
    else:
        analysis_page()

if __name__ == "__main__":
    main()
