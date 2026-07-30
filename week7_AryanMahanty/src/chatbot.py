import cohere
import uuid

class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())
        self.co = cohere.Client(cohere_api_key)

    def respond(self, user_message: str):
        retrieved_docs = self.vectorstore.retrieve(user_message)

        if retrieved_docs:
            response = self.co.chat_stream(
                message=user_message,
                model="command-a-03-2025",
                documents=[{"text": doc} for doc in retrieved_docs],
                conversation_id=self.conversation_id,
            )
        else:
            response = self.co.chat_stream(
                message=user_message,
                model="command-a-03-2025",
                conversation_id=self.conversation_id,
            )
        return response, retrieved_docs
