
# PDFChatbot

Pdf Chatbot is a RAG-based system, a context-aware system trained to answer queries regarding the Document that the user gives. It can Trace the Chat from the encoding to generating the output using the powerful Langsmith.

The Document need to be fed to the model in order to train the model for the question and answer session.
![Inital_Upload](https://github.com/user-attachments/assets/f80b5a07-4e63-45a2-8b92-84bd6273e99f)

**HuggingFace Embeddings:**

The application uses the HuggingFace BAAI/llm-embedder model to convert text data into embeddings, which are then used to facilitate efficient and accurate search within the PDF content.

**FAISS Vector Store:**

FAISS (Facebook AI Similarity Search) is employed to store and search text embeddings. This allows the application to quickly retrieve relevant sections of the PDF based on user queries.

**Processing the PDF: Extracting and Embedding Text**

The core functionality starts with processing the uploaded PDF. Using PyPDFLoader fromLangChain, the PDF content is extracted and split into manageable chunks. These chunks are then transformed into embeddings using the HuggingFace model. The embeddings are stored in a FAISS vector store, allowing for quick and efficient retrieval during user interactions.

**Handling User Queries**

The application is designed to handle user queries seamlessly and efficiently. When a user enters a question, the system retrieves relevant document sections using FAISS, applies a custom prompt template, and generates a response using the Groq LLM model(mixtral-8x7b-32768).

**Storing the History of the Chat and displaying it using Interactive UI**
![image](https://github.com/user-attachments/assets/8ef44055-06b1-4d2c-b347-6e6d37c9d27d)

The user interface is built using Streamlit, providing an easy-to-use platform for interacting with the PDF. Users can upload their PDFs, enter queries, and view previous interactions all within the same interface. The chat history is stored in the session state, ensuring that the conversation flows smoothly and previous questions and answers are easily accessible.

**Tracing Using Langsmith**
![image](https://github.com/user-attachments/assets/2cb8cc76-dc63-4700-89eb-8a06d2071870)

Detailed Tracing:

LangSmith provides detailed tracing of the entire workflow, helping you understand how data flows through the system. This is especially useful when you’re dealing with multiple components like document loaders, embedding models, and vector databases.

Debugging:

By tracing each step in the process, LangSmith makes it easier to identify where something might be going wrong. For instance, if the responses to user queries are not as expected, you can trace back through the chain to see if the issue lies in the retrieval process, the embedding model, or the prompt generation.

Performance Monitoring:

LangSmith allows you to monitor the performance of each component, giving insights into which parts of the process are taking the most time or resources. This is crucial for optimizing the application, especially as it scales.

## API Reference

#### Get all items

```http
  GET /api/items
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `api_key` | `GroqAPI` | **Required**. Your API key |
| `api_key` | `LangsmithAPI` |  Your API key |

#### Required items For Feeding the Data

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `PDF`      | `Document` | **Required**. path of item to fetch |







- [@Github](https://www.github.com/JohnPrabhasith)
- [@HuggingFace](https://www.huggingface.co/BLJohnPrabhasith)
- [@Medium](https://medium.com/@johnprabhasith)
## Tech Stack

**LLM** : GroqApi(mixtral-8x7b-32768)

**Embedding** : HuggingFaceBgeEmbeddings

**Langchain** 

**Langsmith**




## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`GROQ_API`

`LANGSMITH_API`


