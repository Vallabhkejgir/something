import asyncio
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
        url = request.json.get('doc_url')
        docs = Doc_loader(url)
        chunks = chunk_texts(docs)
        store_chunks(chunks)
        initialized = True
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query():
    if not initialized:
        return jsonify({'error': 'Load docs first'}), 400
    
    user_prompt = request.json.get('prompt')
    inputs = GraphState(
            question=user_prompt,
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