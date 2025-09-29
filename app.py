# Enhanced CDSC Chatbot with Semantic Understanding
# Using Flask + Sentence Transformers for better rephrasing handling

from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CDSCChatbot:
    def __init__(self, intents_file='intents.json'):
        """Initialize the chatbot with semantic understanding capabilities"""
        
        # Load pre-trained sentence transformer model
        logger.info("Loading Sentence Transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load intents and prepare embeddings
        self.intents = self.load_intents(intents_file)
        self.intent_embeddings = self.prepare_intent_embeddings()
        
        logger.info(f"Chatbot initialized with {len(self.intents)} intents")
    
    def load_intents(self, filename):
        """Load intents from JSON file"""
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                return data['intents']
        except FileNotFoundError:
            logger.error(f"Intents file {filename} not found")
            return []
    
    def prepare_intent_embeddings(self):
        """Pre-compute embeddings for all intent patterns"""
        intent_embeddings = {}
        
        for intent in self.intents:
            tag = intent['tag']
            patterns = intent['patterns']
            
            if patterns:  # Skip empty patterns
                # Compute embeddings for all patterns in this intent
                embeddings = self.model.encode(patterns)
                intent_embeddings[tag] = {
                    'embeddings': embeddings,
                    'patterns': patterns,
                    'responses': intent['responses']
                }
        
        return intent_embeddings
    
    def find_best_intent(self, user_message, similarity_threshold=0.5):
        """
        Find the best matching intent using semantic similarity
        
        Args:
            user_message (str): User's input message
            similarity_threshold (float): Minimum similarity score to consider a match
            
        Returns:
            dict: Best matching intent or fallback intent
        """
        if not user_message.strip():
            return self.get_fallback_intent()
        
        # Encode user message
        user_embedding = self.model.encode([user_message])
        
        best_intent = None
        best_similarity = 0.0
        best_response = None
        
        # Compare with all intent patterns
        for tag, intent_data in self.intent_embeddings.items():
            if tag == 'fallback':  # Skip fallback for comparison
                continue
                
            # Calculate similarity with all patterns in this intent
            similarities = cosine_similarity(user_embedding, intent_data['embeddings'])[0]
            max_similarity = np.max(similarities)
            
            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_intent = tag
                best_response = intent_data['responses']
        
        # Check if best match meets threshold
        if best_similarity >= similarity_threshold:
            logger.info(f"Matched intent '{best_intent}' with similarity {best_similarity:.3f}")
            return {
                'tag': best_intent,
                'confidence': best_similarity,
                'response': np.random.choice(best_response)
            }
        else:
            logger.info(f"No good match found. Best similarity: {best_similarity:.3f}")
            return self.get_fallback_intent()
    
    def get_fallback_intent(self):
        """Return fallback response when no good match is found"""
        fallback = next((intent for intent in self.intents if intent['tag'] == 'fallback'), None)
        if fallback:
            return {
                'tag': 'fallback',
                'confidence': 0.0,
                'response': np.random.choice(fallback['responses'])
            }
        return {
            'tag': 'fallback',
            'confidence': 0.0,
            'response': "I'm sorry, I don't understand that. Please contact our team for assistance."
        }
    
    def add_training_example(self, user_message, correct_intent):
        """
        Add a new training example and update embeddings
        This allows the bot to learn from interactions
        """
        # Find the intent and add the new pattern
        for intent in self.intents:
            if intent['tag'] == correct_intent:
                intent['patterns'].append(user_message)
                # Re-compute embeddings for this intent
                self.intent_embeddings[correct_intent] = {
                    'embeddings': self.model.encode(intent['patterns']),
                    'patterns': intent['patterns'],
                    'responses': intent['responses']
                }
                logger.info(f"Added new training example: '{user_message}' -> {correct_intent}")
                break

# Initialize the chatbot
chatbot = CDSCChatbot()

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    API endpoint to handle user messages with semantic understanding
    Returns JSON response with bot reply and confidence score
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'response': 'Please ask me something!', 'confidence': 0.0})
        
        # Get response using semantic matching
        result = chatbot.find_best_intent(user_message)
        
        return jsonify({
            'response': result['response'],
            'confidence': float(result['confidence']),
            'intent': result['tag']
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': 'Sorry, I encountered an error. Please try again.',
            'confidence': 0.0,
            'intent': 'error'
        }), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Endpoint for collecting feedback to improve the bot
    This can be used by your team to continuously improve responses
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        correct_intent = data.get('correct_intent', '')
        
        if user_message and correct_intent:
            chatbot.add_training_example(user_message, correct_intent)
            return jsonify({'status': 'success', 'message': 'Feedback recorded'})
        
        return jsonify({'status': 'error', 'message': 'Invalid feedback data'})
        
    except Exception as e:
        logger.error(f"Error in feedback endpoint: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Failed to record feedback'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for deployment monitoring"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': chatbot.model is not None,
        'intents_count': len(chatbot.intents)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
