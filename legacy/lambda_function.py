import json
import pickle
import base64

# Load model at cold start
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def lambda_handler(event, context):
    try:
        # Parse input
        body = json.loads(event['body']) if isinstance(event.get('body'), str) else event
        features = body['features']
        
        # Predict
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0].tolist()
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'prediction': int(prediction),
                'probabilities': probability
            })
        }
    except Exception as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': str(e)})
        }
