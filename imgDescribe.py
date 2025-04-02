import streamlit as st
import tempfile
import os
from pyvisionai import (
    describe_image_openai,
    describe_image_claude,
    describe_image_ollama
)
from deep_translator import (
    GoogleTranslator, 
    BaiduTranslator,
    DeeplTranslator
)

# Set up page configuration
st.set_page_config(
    page_title="Image Description Tool",
    page_icon="🖼️",
    layout="wide"
)

# Function to translate text to Chinese based on selected service
def translate_to_chinese(text, service, api_key=None, app_id=None):
    try:
        if service == "Baidu":
            # Validate Baidu credentials
            if not api_key or not app_id:
                st.warning("Baidu translation requires both API key and App ID")
                return "Translation failed: Missing Baidu API credentials"
            
            try:
                # 正确使用BaiduTranslator而不是回退到Google
                import os
                # 设置环境变量
                os.environ["BAIDU_APPID"] = str(app_id)
                os.environ["BAIDU_APPKEY"] = str(api_key)
                
                # 调试信息
                st.info(f"Using Baidu Translation API with App ID: {app_id[:4]}...")
                
                # 直接使用BaiduTranslator
                translator = BaiduTranslator(
                    source="en",
                    target="zh"
                )
                return translator.translate(text)
            except Exception as e:
                # 只有在Baidu失败时才回退到Google
                st.error(f"Baidu translation error: {e}")
                st.info("Falling back to Google translation")
                return GoogleTranslator(source='en', target='zh-CN').translate(text)
            
        elif service == "Google":
            translator = GoogleTranslator(source='en', target='zh-CN')
        elif service == "DeepL":
            if not api_key:
                return "DeepL translation requires an API key"
            translator = DeeplTranslator(
                api_key=api_key,
                source="en",
                target="zh"
            )
        
        # Split long text into chunks if needed (some APIs have character limits)
        if len(text) > 4000:
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            translated = ""
            for chunk in chunks:
                translated += translator.translate(chunk) + " "
            return translated.strip()
        else:
            return translator.translate(text)
            
    except Exception as e:
        st.error(f"Translation error: {e}")
        return "Translation failed"

# Title and description
st.title("🖼️ Image Description Tool")
st.markdown("Upload an image and get AI-powered descriptions in English and Chinese")

# Create a 2-column layout for the main interface
left_col, right_col = st.columns([3, 4])  # Left column slightly narrower than right

with left_col:
    # File uploader for image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Create columns for model selection and prompt input
    model_option = st.selectbox(
        "Select AI Model",
        ["Ollama (Local)", "OpenAI (GPT-4 Vision)", "Claude Vision"],
        index=0  # Set default to Ollama
    )
    
    # Model-specific configurations
    if model_option == "OpenAI (GPT-4 Vision)":
        openai_model = st.selectbox("Select OpenAI Model", ["gpt-4o", "gpt-4o-mini"])
        openai_api_key = st.text_input("OpenAI API Key (optional)", type="password")
    
    elif model_option == "Claude Vision":
        claude_api_key = st.text_input("Anthropic API Key (optional)", type="password")
    
    elif model_option == "Ollama (Local)":
        ollama_model = st.selectbox("Select Ollama Model", ["llama3.2-vision", "bakllava"])

    # Prompt input
    prompt = st.text_input("Prompt (optional)", value="Describe this image in detail")
    max_tokens = st.slider("Max Tokens", 100, 1000, 300)

    # Translation service options (with option to disable)
    translation_options = ["Baidu", "Google", "DeepL", "None"]
    translation_service = st.selectbox(
        "Select Translation Service",
        translation_options,
        index=0  # Default to Baidu now
    )

    # Show relevant API settings based on the selected translation service
    if translation_service == "Baidu":
        st.warning("Note: Baidu Translation API requires registration at Baidu Developer")
        
        # Get credentials from environment variables if available
        env_app_id = os.environ.get("BAIDU_APPID", "")
        env_app_key = os.environ.get("BAIDU_APPKEY", "")
        
        # Display message if environment variables are found
        if env_app_id and env_app_key:
            st.success("Baidu API credentials found in environment variables")
        
        # Show credentials in input fields with password visibility toggle
        col_baidu1, col_baidu2 = st.columns(2)
        with col_baidu1:
            # Fill App ID input with environment variable value if available
            credential_placeholder = "From environment" if env_app_id else ""
            password = st.checkbox("Hide App ID", value=True, key="hide_app_id")
            input_type = "password" if password else "default"
            baidu_app_id = st.text_input(
                "Baidu App ID", 
                value=env_app_id,
                placeholder=credential_placeholder,
                type=input_type,
                help="Required for Baidu translation"
            )
                
        with col_baidu2:
            # Fill API Key input with environment variable value if available
            credential_placeholder = "From environment" if env_app_key else ""
            password = st.checkbox("Hide API Key", value=True, key="hide_api_key")
            input_type = "password" if password else "default"
            baidu_api_key = st.text_input(
                "Baidu API Key", 
                value=env_app_key,
                placeholder=credential_placeholder,
                type=input_type,
                help="Required for Baidu translation"
            )
    
    elif translation_service == "DeepL":
        deepl_api_key = st.text_input("DeepL API Key", type="password")
    
    # Generate button at the bottom of the left column
    generate_button = st.button("Generate Description", use_container_width=True)

with right_col:
    # Image preview area in the right column
    if uploaded_file is not None:
        st.subheader("Image Preview")
        
        # Image display options
        image_size = st.slider("Preview Size (%)", 10, 100, 70)
        
        # Display the image with the selected size
        # Use a container to keep the image within the column
        image_container = st.container()
        with image_container:
            # Display image with the specified width within the container
            st.image(uploaded_file, use_container_width=True)
        
        # Results display area (will be filled in when generate button is clicked)
        if generate_button:
            with st.spinner("Generating description..."):
                # Save uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name
                    
                try:
                    # Call the appropriate function based on the selected model
                    if model_option == "OpenAI (GPT-4 Vision)":
                        description = describe_image_openai(
                            temp_path,
                            model=openai_model,
                            api_key=openai_api_key if openai_api_key else None,
                            max_tokens=max_tokens,
                            prompt=prompt
                        )
                        
                    elif model_option == "Claude Vision":
                        description = describe_image_claude(
                            temp_path,
                            api_key=claude_api_key if claude_api_key else None,
                            prompt=prompt
                        )
                        
                    elif model_option == "Ollama (Local)":
                        description = describe_image_ollama(
                            temp_path,
                            model=ollama_model,
                            prompt=prompt
                        )
                    
                    # Display results in the right column
                    st.subheader("English Description")
                    st.write(description)
                    
                    # Only translate if translation service is not "None"
                    if translation_service != "None":
                        st.subheader("中文描述")
                        
                        # Get translation API details
                        api_key = None
                        app_id = None
                        
                        if translation_service == "Baidu":
                            # Try to get credentials from environment variables first
                            api_key = os.environ.get("BAIDU_APPKEY", "")
                            app_id = os.environ.get("BAIDU_APPID", "")
                            
                            # If not found in environment, use the UI inputs
                            if not api_key:
                                api_key = baidu_api_key
                            if not app_id:
                                app_id = baidu_app_id
                                
                            # Show information about where credentials are coming from
                            if os.environ.get("BAIDU_APPID") and os.environ.get("BAIDU_APPKEY"):
                                st.info("Using Baidu credentials from environment variables")
                            elif baidu_app_id and baidu_api_key:
                                st.info("Using Baidu credentials from form inputs")
                            else:
                                st.warning("Baidu credentials not found in environment variables or form inputs")
                            
                            # Validate before proceeding
                            if not api_key or not app_id:
                                st.error("Baidu translation requires API key and App ID in environment variables or form inputs")
                                st.markdown("""
                                **To use Baidu translation:**
                                1. Register at [Baidu Developer](https://fanyi-api.baidu.com/api/trans/product/desktop)
                                2. Either:
                                   * Set BAIDU_APPID and BAIDU_APPKEY environment variables, or
                                   * Enter your App ID and API Key in the form above
                                """)
                                chinese_description = "Translation failed: Missing Baidu API credentials"
                            else:
                                # Set environment variables to ensure they're available in other functions
                                os.environ["BAIDU_APPID"] = str(app_id)
                                os.environ["BAIDU_APPKEY"] = str(api_key)
                                
                                # Translate to Chinese
                                chinese_description = translate_to_chinese(
                                    description, 
                                    translation_service, 
                                    api_key=api_key,
                                    app_id=app_id
                                )
                        elif translation_service == "DeepL":
                            api_key = deepl_api_key
                            
                            # Validate before proceeding
                            if not api_key:
                                st.error("DeepL translation requires an API key")
                                chinese_description = "Translation failed: Missing DeepL API key"
                            else:
                                # Translate to Chinese
                                chinese_description = translate_to_chinese(
                                    description, 
                                    translation_service, 
                                    api_key=api_key
                                )
                        else:
                            # For services that don't require API keys
                            chinese_description = translate_to_chinese(
                                description, 
                                translation_service
                            )
                        
                        # Check if translation failed
                        if "Translation error:" in chinese_description or chinese_description == "Translation failed":
                            st.warning(f"{translation_service} translation failed. Trying Google Translate...")
                            
                            try:
                                # Fallback to Google Translate
                                fallback_translator = GoogleTranslator(source='en', target='zh-CN')
                                chinese_description = fallback_translator.translate(description)
                                st.write(chinese_description)
                            except Exception as fallback_error:
                                st.error(f"Fallback translation also failed: {fallback_error}")
                                
                                # Very basic translation dictionary for common image description terms
                                translation_dict = {
                                    "image": "图像",
                                    "depicts": "描绘",
                                    "shows": "显示",
                                    "contains": "包含",
                                    "person": "人",
                                    "people": "人们",
                                    "man": "男人",
                                    "woman": "女人",
                                    "child": "孩子",
                                    "children": "孩子们",
                                    "sky": "天空",
                                    "water": "水",
                                    "tree": "树",
                                    "trees": "树木",
                                    "building": "建筑",
                                    "car": "汽车",
                                    "background": "背景",
                                    "foreground": "前景",
                                    "left": "左边",
                                    "right": "右边",
                                    "top": "顶部",
                                    "bottom": "底部",
                                    "color": "颜色",
                                    "red": "红色",
                                    "blue": "蓝色",
                                    "green": "绿色",
                                    "yellow": "黄色",
                                    "black": "黑色",
                                    "white": "白色"
                                }
                                
                                # Simple word-by-word translation
                                simple_translation = description
                                for eng, chn in translation_dict.items():
                                    simple_translation = simple_translation.replace(" " + eng + " ", " " + chn + " ")
                                
                                st.write("Note: This is a very simplified translation.")
                                st.write(simple_translation)
                        else:
                            st.write(chinese_description)
                    
                except Exception as e:
                    st.error(f"Error generating description: {e}")
                
                finally:
                    # Delete the temporary file
                    os.unlink(temp_path)
    else:
        # Placeholder when no image is uploaded
        st.info("Upload an image to see preview and generate descriptions")

# Footer at the bottom of the page
st.markdown("---")
st.markdown("<div style='text-align: center'>Built with Streamlit and pyvisionai</div>", unsafe_allow_html=True)