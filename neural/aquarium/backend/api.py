"""
Flask API for Neural Aquarium AI Assistant

Provides REST API endpoints for the React frontend to interact with
the Neural DSL AI Assistant.
"""

from flask import Flask, request, jsonify
import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from neural.ai.ai_assistant import NeuralAIAssistant
from neural.ai.multi_language import MultiLanguageSupport
from neural.security import (
    load_security_config,
    create_basic_auth,
    create_jwt_auth,
    require_auth,
    apply_security_middleware,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load security configuration
security_config = load_security_config()

app = Flask(__name__)

# Apply security middleware
apply_security_middleware(
    app,
    cors_enabled=security_config.cors_enabled,
    cors_origins=security_config.cors_origins,
    cors_methods=security_config.cors_methods,
    cors_allow_headers=security_config.cors_allow_headers,
    cors_allow_credentials=security_config.cors_allow_credentials,
    rate_limit_enabled=security_config.rate_limit_enabled,
    rate_limit_requests=security_config.rate_limit_requests,
    rate_limit_window_seconds=security_config.rate_limit_window_seconds,
    security_headers_enabled=security_config.security_headers_enabled,
)

# Setup authentication if enabled
auth_middleware = None
if security_config.auth_enabled:
    if security_config.auth_type == 'jwt' and security_config.jwt_secret_key:
        auth_middleware = create_jwt_auth(
            security_config.jwt_secret_key,
            security_config.jwt_algorithm,
            security_config.jwt_expiration_hours
        )
    elif security_config.auth_type == 'basic':
        auth_middleware = create_basic_auth(
            security_config.basic_auth_username,
            security_config.basic_auth_password
        )

ai_assistant = None
multi_lang_support = MultiLanguageSupport()


def get_assistant():
    """Get or create AI assistant instance."""
    global ai_assistant
    if ai_assistant is None:
        ai_assistant = NeuralAIAssistant(use_llm=True)
    return ai_assistant


@app.route('/api/ai/chat', methods=['POST'])
def chat():
    """
    Handle chat messages from the frontend.
    
    Request body:
    {
        "user_input": str,
        "context": dict (optional),
        "language": str (optional, default: "en")
    }
    
    Returns:
    {
        "response": str,
        "dsl_code": str (optional),
        "intent": str,
        "success": bool,
        "language": str
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'user_input' not in data:
            return jsonify({
                'error': 'Missing user_input in request body'
            }), 400
        
        user_input = data['user_input']
        context = data.get('context', {})
        language = data.get('language', 'en')
        
        assistant = get_assistant()
        
        if language != 'en':
            lang_result = multi_lang_support.process(user_input, target_lang='en')
            user_input = lang_result['final']
        
        result = assistant.chat(user_input, context)
        
        if language != 'en' and result.get('response'):
            try:
                translated = multi_lang_support.translator.translate(
                    result['response'], 
                    target_lang=language,
                    source_lang='en'
                )
                result['response'] = translated
            except Exception as e:
                logger.warning(f"Translation failed: {e}")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'response': 'An error occurred processing your request.',
            'success': False,
            'intent': 'error',
            'language': 'en'
        }), 500


@app.route('/api/ai/current-model', methods=['GET'])
def get_current_model():
    """
    Get the current model DSL.
    
    Returns:
    {
        "model": str
    }
    """
    try:
        assistant = get_assistant()
        model_dsl = assistant.get_current_model()
        return jsonify({'model': model_dsl}), 200
    except Exception as e:
        logger.error(f"Error getting current model: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/reset', methods=['POST'])
def reset_assistant():
    """
    Reset the AI assistant state.
    
    Returns:
    {
        "success": bool,
        "message": str
    }
    """
    try:
        assistant = get_assistant()
        assistant.reset()
        return jsonify({
            'success': True,
            'message': 'AI assistant reset successfully'
        }), 200
    except Exception as e:
        logger.error(f"Error resetting assistant: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ai/translate', methods=['POST'])
def translate():
    """
    Translate text between languages.
    
    Request body:
    {
        "text": str,
        "target_lang": str,
        "source_lang": str (optional)
    }
    
    Returns:
    {
        "translated_text": str,
        "source_lang": str,
        "target_lang": str
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data or 'target_lang' not in data:
            return jsonify({
                'error': 'Missing required fields: text, target_lang'
            }), 400
        
        text = data['text']
        target_lang = data['target_lang']
        source_lang = data.get('source_lang')
        
        translated = multi_lang_support.translator.translate(
            text,
            target_lang=target_lang,
            source_lang=source_lang
        )
        
        return jsonify({
            'translated_text': translated,
            'source_lang': source_lang or 'auto',
            'target_lang': target_lang
        }), 200
        
    except Exception as e:
        logger.error(f"Error in translate endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/detect-language', methods=['POST'])
def detect_language():
    """
    Detect the language of input text.
    
    Request body:
    {
        "text": str
    }
    
    Returns:
    {
        "language": str,
        "language_name": str
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text in request body'
            }), 400
        
        text = data['text']
        detected_lang = multi_lang_support.detector.detect(text)
        lang_name = multi_lang_support.get_language_name(detected_lang)
        
        return jsonify({
            'language': detected_lang,
            'language_name': lang_name
        }), 200
        
    except Exception as e:
        logger.error(f"Error in detect-language endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/supported-languages', methods=['GET'])
def get_supported_languages():
    """
    Get list of supported languages.
    
    Returns:
    {
        "languages": {
            "code": "name",
            ...
        }
    }
    """
    return jsonify({
        'languages': multi_lang_support.SUPPORTED_LANGUAGES
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'neural-aquarium-api',
        'version': '0.3.0'
    }), 200


@app.route('/api/health', methods=['GET'])
def api_health_check():
    """API health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'neural-aquarium-api',
        'version': '0.3.0'
    }), 200


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Neural Aquarium AI Assistant API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Neural Aquarium AI Assistant API on {args.host}:{args.port}")
    logger.info(f"Debug mode: {args.debug}")
    
    app.run(host=args.host, port=args.port, debug=args.debug)
