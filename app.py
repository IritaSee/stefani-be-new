
from flask import Flask, request, jsonify
from flask_cors import CORS
import llm_handler

app = Flask(__name__)
CORS(app)

@app.route('/api/ask', methods=['POST'])
def ask_llm():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    question = data.get('question')
    conversation_id = data.get('conversation_id')
    if not question or not conversation_id:
        return jsonify({'error': 'Both question and conversation_id are required'}), 400

    # Prepare input for LLM handler (adapting to expected input format)
    llm_input = {
        'anon_identifier': conversation_id,
        'context_text': '',
        'feedback_text': question
    }
    try:
        llm_response = llm_handler.summarize_text_with_llm(llm_input, model_provider="openai")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'result': llm_response}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False, port=5001)

