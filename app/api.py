import os
import asyncio
import tempfile
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from app.services.loader import Doc_loader
from app.utils.chunks import chunk_texts
from app.services.storage import store_chunks
from app.RAG.graph import rag_app, GraphState

app = Flask(__name__)
CORS(app)

initialized = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/initialize', methods=['POST'])
def initialize():
    global initialized
    try:
        # Check if it's a file upload
        if 'file' in request.files:
            file = request.files['file']
            if not file.filename:
                return jsonify({'error': 'No file selected'}), 400
            
            extension = file.filename.split('.')[-1].lower()
            if extension not in ['pdf', 'txt']:
                return jsonify({'error': 'Unsupported file type'}), 400

            # Save file to a temporary location for processing
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp_file:
                file.save(tmp_file.name)
                tmp_path = tmp_file.name

            try:
                docs = Doc_loader(tmp_path, source_type=extension)
                chunks = chunk_texts(docs)
                store_chunks(chunks)
                initialized = True
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
            return jsonify({'status': 'success', 'source': file.filename})

        # Handle URL source
        else:
            url = request.json.get('doc_url')
            if not url:
                return jsonify({'error': 'No URL or file provided'}), 400
            docs = Doc_loader(url, source_type="url")
            chunks = chunk_texts(docs)
            store_chunks(chunks)
            initialized = True
            return jsonify({'status': 'success', 'source': url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query():
    if not initialized:
        return jsonify({'error': 'Load docs first'}), 400
    
    user_prompt = request.json.get('prompt')
    inputs = GraphState(
            question=user_prompt,
            category="",
            rewritten_queries=[],   
            sub_queries=[],         
            context="",             
            answer=""               
        )
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        final_state = loop.run_until_complete(rag_app.ainvoke(inputs))
        return jsonify({'answer': final_state['answer']})
    finally:
        loop.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)