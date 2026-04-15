import asyncio
import streamlit as st
from browser_use import Agent, SystemPrompt
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
import re

async def generate_meme(query: str, model_choice: str, api_key: str) -> None:
    # Initialize the appropriate LLM based on user selection
    if model_choice == "Claude":
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=api_key
        )
    elif model_choice == "Deepseek":
        llm = ChatOpenAI(
            base_url='https://api.deepseek.com/v1',
            model='deepseek-chat',
            api_key=api_key,
            temperature=0.3
        )
    else:  # OpenAI
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=0.0
        )

    task_description = (
        "You are a meme generator. Generate a meme for this topic: '{0}'\n\n"
        "Steps:\n"
        "1. Go to https://imgflip.com/memetemplates\n"
        "2. In the search box, type a single keyword related to '{0}' and press Enter\n"
        "3. Click the 'Add Caption' button under any meme template that fits the topic\n"
        "4. Fill in the Top Text with a setup line and Bottom Text with a punchline about '{0}'\n"
        "5. Click the 'Generate Meme' button\n"
        "6. After the meme is generated, find the meme image URL on the page.\n"
        "   The URL will look like: https://i.imgflip.com/XXXXX.jpg or https://imgflip.com/i/XXXXX\n"
        "7. Return ONLY the final meme URL as your answer. Example output: https://i.imgflip.com/abc123.jpg\n\n"
        "If the search returns no results, go back and try a simpler or different keyword.\n"
        "If the meme generation page does not load, try clicking 'Add Caption' on a popular template.\n"
    ).format(query)

    agent = Agent(
        task=task_description,
        llm=llm,
        max_actions_per_step=5,
        max_failures=25,
        use_vision=(model_choice != "Deepseek")
    )

    history = await agent.run()

    try:
        # Extract final result from agent history
        final_result = history.final_result()

        # Log the full agent result for debugging
        print(f"[DEBUG] Agent final_result: {final_result!r}")

        if final_result is None:
            print("[ERROR] Agent returned no result - check API key and model availability")
            return None

        # Try multiple URL patterns to handle different formats the agent may return

        # Pattern 1: Direct image URL (https://i.imgflip.com/XXXXX.jpg/.png/.gif)
        match = re.search(r'https?://i\.imgflip\.com/[\w/-]+(?:\.\w+)?', final_result)
        if match:
            url = match.group(0)
            print(f"[DEBUG] Extracted image URL (pattern 1): {url}")
            return url

        # Pattern 2: Imgflip share page URL (https://imgflip.com/i/XXXXX)
        match = re.search(r'https?://(?:www\.)?imgflip\.com/i/([\w-]+)', final_result)
        if match:
            meme_id = match.group(1)
            url = f"https://i.imgflip.com/{meme_id}.jpg"
            print(f"[DEBUG] Extracted share URL (pattern 2), built image URL: {url}")
            return url

        # Pattern 3: Any imgflip.com URL as a fallback (stops before whitespace or sentence punctuation)
        match = re.search(r'https?://(?:www\.)?imgflip\.com/[^\s,)>"\']+', final_result)
        if match:
            url = match.group(0).rstrip('.')
            print(f"[DEBUG] Extracted imgflip URL (pattern 3): {url}")
            return url

        # Pattern 4: Any http/https URL in the result (last-resort fallback)
        match = re.search(r'https?://[^\s,)>"\']+', final_result)
        if match:
            url = match.group(0).rstrip('.')
            print(f"[DEBUG] Extracted generic URL (pattern 4 fallback): {url}")
            return url

        print(f"[ERROR] Could not extract any URL from agent result: {final_result!r}")
        return None

    except Exception as e:
        print(f"[ERROR] Error extracting meme URL: {str(e)}")
        return None

def main():
    # Custom CSS styling


    st.title("🥸 AI Meme Generator Agent - Browser Use")
    st.info("This AI browser agent does browser automation to generate memes based on your input with browser use. Please enter your API key and describe the meme you want to generate.")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown('<p class="sidebar-header">⚙️ Model Configuration</p>', unsafe_allow_html=True)
        
        # Model selection
        model_choice = st.selectbox(
            "Select AI Model",
            ["Claude", "Deepseek", "OpenAI"],
            index=0,
            help="Choose which LLM to use for meme generation"
        )
        
        # API key input based on model selection
        api_key = ""
        if model_choice == "Claude":
            api_key = st.text_input("Claude API Key", type="password", 
                                  help="Get your API key from https://console.anthropic.com")
        elif model_choice == "Deepseek":
            api_key = st.text_input("Deepseek API Key", type="password",
                                  help="Get your API key from https://platform.deepseek.com")
        else:
            api_key = st.text_input("OpenAI API Key", type="password",
                                  help="Get your API key from https://platform.openai.com")

    # Main content area
    st.markdown('<p class="header-text">🎨 Describe Your Meme Concept</p>', unsafe_allow_html=True)
    
    query = st.text_input(
        "Meme Idea Input",
        placeholder="Example: 'Ilya's SSI quietly looking at the OpenAI vs Deepseek debate while diligently working on ASI'",
        label_visibility="collapsed"
    )

    if st.button("Generate Meme 🚀"):
        if not api_key:
            st.warning(f"Please provide the {model_choice} API key")
            st.stop()
        if not query:
            st.warning("Please enter a meme idea")
            st.stop()

        with st.spinner(f"🧠 {model_choice} is generating your meme..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                st.info("🤖 Browser agent is navigating imgflip.com... This may take a minute.")
                meme_url = loop.run_until_complete(generate_meme(query, model_choice, api_key))
                
                if meme_url:
                    st.success("✅ Meme Generated Successfully!")
                    st.image(meme_url, caption="Generated Meme Preview", use_container_width=True)
                    st.markdown(f"""
                        **Direct Link:** [Open in ImgFlip]({meme_url})  
                        **Embed URL:** `{meme_url}`
                    """)
                else:
                    st.error("❌ Failed to generate meme. The agent could not extract a valid image URL.")
                    st.warning(
                        "💡 **Troubleshooting tips:**\n"
                        "- Check your API key is valid and has sufficient credits\n"
                        "- Try a simpler or different meme prompt\n"
                        "- If using Deepseek, note that vision is disabled — results may vary\n"
                        "- Check the terminal/console for `[DEBUG]` and `[ERROR]` logs to see what the agent returned"
                    )
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("💡 If using OpenAI, ensure your account has GPT-4o access")
            finally:
                loop.close()

if __name__ == '__main__':
    main()