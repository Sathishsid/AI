import re
responses = {
    'hello': 'Hello! How can I help you today?',
    'hi': 'Hi there! How can I assist you?',
    'how are you': 'I am just a bot, but I am here to help you!',
    'what is your name': 'I am a simple chatbot created by a user.',
    'bye': 'Goodbye! Have a great day!',
    'default': 'I am not sure how to respond to that. Can you please rephrase?'
}

def get_response(user_input):
    user_input = user_input.lower()
    for keyword in responses:
        if re.search(r'\b' + re.escape(keyword) + r'\b', user_input):
            return responses[keyword]
    
    return responses['default']
    
print("Welcome to the simple chatbot! Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'bye':
        print("Chatbot: " + responses['bye'])
        break
    response = get_response(user_input)
    print("Chatbot: " + response)
