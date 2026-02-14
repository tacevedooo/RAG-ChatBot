"""
Setup and Testing Guide for Medical RAG Chatbot
Run this script to verify your environment is set up correctly
"""

import sys

def check_dependencies():
    """Check if all required packages are installed"""
    print("Checking dependencies...\n")
    
    required_packages = {
        'streamlit': 'streamlit',
        'anthropic': 'anthropic',
        'PyPDF2': 'PyPDF2',
        'sentence_transformers': 'sentence-transformers',
        'faiss': 'faiss-cpu',
        'numpy': 'numpy',
        'torch': 'torch'
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            missing.append(pip_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    else:
        print("\n✅ All dependencies are installed!")
        return True

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible\n")
        return True
    else:
        print("❌ Python 3.8 or higher is required\n")
        return False

def api_key_guide():
    """Print guide for getting Anthropic API key"""
    print("\n" + "="*60)
    print("HOW TO GET YOUR ANTHROPIC API KEY")
    print("="*60)
    print("""
1. Go to: https://console.anthropic.com/
2. Sign up or log in to your account
3. Navigate to "API Keys" section
4. Click "Create Key"
5. Copy your API key
6. Paste it in the Streamlit sidebar when running the app

⚠️  Keep your API key secure and never share it publicly!
""")

def usage_guide():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("HOW TO RUN THE CHATBOT")
    print("="*60)
    print("""
1. Open terminal in the project directory
2. Run: streamlit run medical_rag_chatbot.py
3. Your browser will open automatically
4. If not, go to: http://localhost:8501

Then:
- Enter your API key in the sidebar
- Upload a medical PDF
- Click "Process PDF"
- Start asking questions!
""")

def main():
    print("\n" + "="*60)
    print("MEDICAL RAG CHATBOT - SETUP VERIFICATION")
    print("="*60 + "\n")
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Show guides
    api_key_guide()
    usage_guide()
    
    # Final status
    print("="*60)
    if python_ok and deps_ok:
        print("✅ Your environment is ready!")
        print("\nRun: streamlit run medical_rag_chatbot.py")
    else:
        print("⚠️  Please fix the issues above before running the chatbot")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
