import streamlit as st
import requests

st.title("🧮 LangGraph Math Assistant")

# Handle user input
if user_input := st.chat_input("What would you like me to calculate?"):
    with st.chat_message("user"):
        st.write(user_input)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Send the request to your FastAPI backend
            backend_url = "http://127.0.0.1:8000/chat"
            payload = {"message": user_input}
            
            try:
                response = requests.post(backend_url, json=payload)
                data = response.json()
                
                # Display the response from FastAPI
                st.write(data["response"])
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")