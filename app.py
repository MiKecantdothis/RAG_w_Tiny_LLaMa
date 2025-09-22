import streamlit as st
from rag import EnhancedRAGAgent
from dotenv import load_dotenv

def main():
    
    st.set_page_config(page_title="RAG Agent with Tiny Llama", page_icon="🐒", layout="wide")
    st.title("🐒 RAG Agent with Tiny Llama")
    st.markdown("Upload PDF documents and chat with them using advanced AI!")
    @st.cache_resource
    def load_ragagent():
        return EnhancedRAGAgent()
    if 'rag_agent' not in st.session_state:
        st.session_state.rag_agent = load_ragagent()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.header("📚 Document Management")
        uploaded_files = st.file_uploader("Upload PDF documents", type=['pdf'], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        success = st.session_state.rag_agent.add_pdf_document(uploaded_file, uploaded_file.name)
                        if success:
                            st.success(f"✅ Added {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed to process {uploaded_file.name}")

        st.subheader("📄 Loaded Documents")
        for doc in st.session_state.rag_agent.documents:
            with st.expander(f"{doc['filename']}"):
                st.write(f"**Chunks:** {doc['chunk_count']}")
                st.write(f"**Added:** {doc['added_at']}")
                st.write(f"**Text length:** {len(doc['text'])} characters")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Chat Interface")
        chat_container = st.container()
        with chat_container:
            for user_msg, ai_msg, sources in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(user_msg)
                with st.chat_message("assistant"):
                    st.write(ai_msg)
                    if sources:
                        with st.expander("📖 Sources"):
                            for j, (chunk, score, metadata) in enumerate(sources):
                                st.write(f"**Source {j+1}** (Score: {score:.3f}) - {metadata['filename']}")
                                st.write(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                                st.divider()

        if query := st.chat_input("Ask a question about your documents..."):
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, sources = st.session_state.rag_agent.chat(query)
                st.write(response)
                if sources:
                    with st.expander("📖 Sources"):
                        for j, (chunk, score, metadata) in enumerate(sources):
                            st.write(f"**Source {j+1}** (Score: {score:.3f}) - {metadata['filename']}")
                            st.write(chunk[:200] + "..." if len(chunk) > 200 else chunk)
                            st.divider()
            st.session_state.chat_history.append((query, response, sources))

    with col2:
        st.header("⚙️ Settings")
        with st.expander("🔧 Model Information"):
            st.write("**LLM Model:** meta-llama/Llama-3.2-3B-Instruct")
            st.write("**Embedding Model:** all-MiniLM-L6-v2")
            st.write("**Vector Database:** FAISS")
            st.write("**Similarity:** Cosine Similarity")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        if st.session_state.rag_agent.documents:
            st.subheader("📊 Statistics")
            total_chunks = sum(doc['chunk_count'] for doc in st.session_state.rag_agent.documents)
            st.metric("Total Documents", len(st.session_state.rag_agent.documents))
            st.metric("Total Chunks", total_chunks)
            st.metric("Chat Messages", len(st.session_state.chat_history))


if __name__ == "__main__":

    main()



