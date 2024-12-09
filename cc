import streamlit as st
import time
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import threading

class CodeAutocompleteApp:
    def __init__(self):
        # Configure Azure OpenAI settings
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        # Initialize the LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_key=self.azure_api_key,
            temperature=0.2,
            max_tokens=200
        )
        
        # Suggestion tracking
        self.current_suggestion = ""
        self.suggestion_thread = None
        self.last_input_time = 0
        
    def generate_completion(self, partial_code):
        """
        Generate code completion using Azure OpenAI
        
        Args:
            partial_code (str): Partial Python code to complete
        
        Returns:
            str: Suggested code completion
        """
        # Prompt template for code completion
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Python code completion assistant. "
             "Complete the following Python code snippet precisely and concisely. "
             "Focus on writing clean, efficient code."),
            ("human", "Complete this Python code:\n{code}")
        ])
        
        # Create chain and invoke
        chain = prompt | self.llm
        response = chain.invoke({"code": partial_code})
        return response.content.strip()
    
    def start_suggestion_timer(self, input_text):
        """
        Start a timer to trigger code completion suggestion
        
        Args:
            input_text (str): Current input text
        """
        # Cancel previous thread if exists
        if self.suggestion_thread and self.suggestion_thread.is_alive():
            return
        
        def suggestion_worker():
            # Wait for 2 seconds of inactivity
            time.sleep(2)
            
            # Generate suggestion
            try:
                self.current_suggestion = self.generate_completion(input_text)
            except Exception as e:
                st.error(f"Error generating suggestion: {e}")
                self.current_suggestion = ""
        
        # Start new suggestion thread
        self.suggestion_thread = threading.Thread(target=suggestion_worker)
        self.suggestion_thread.start()
    
    def render_app(self):
        """
        Render the Streamlit application UI
        """
        st.title("🐍 Python Code Autocomplete")
        
        # Input text area for code
        input_text = st.text_area(
            "Write your Python code", 
            height=300, 
            key="code_input"
        )
        
        # Trigger suggestion generation
        if input_text:
            self.start_suggestion_timer(input_text)
        
        # Display suggestion in light grey if available
        if self.current_suggestion:
            st.markdown(
                f"**Suggestion:**\n```python\n{self.current_suggestion}```", 
                unsafe_allow_html=True
            )
        
        # Optional: Add a button to manually trigger completion
        if st.button("Generate Completion"):
            self.current_suggestion = self.generate_completion(input_text)

def main():
    app = CodeAutocompleteApp()
    app.render_app()

if __name__ == "__main__":
    main()

# Requirements (requirements.txt):
# streamlit
# azure-identity
# langchain
# langchain-openai
