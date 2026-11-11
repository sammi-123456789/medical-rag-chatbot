import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/ask"


def main():
    st.title("Ask Chatbot!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(API_URL, json={"question": prompt}, timeout=60)
            response.raise_for_status()
            data = response.json()

            result_to_show = data["answer"]
            if data["sources"]:
                result_to_show += "\n\n**Sources:** " + ", ".join(data["sources"])

            st.chat_message("assistant").markdown(result_to_show)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Start it with: uvicorn app.main:app --reload")
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()