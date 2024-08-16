from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from SimplerLLM.tools.generic_loader import load_content

load = load_content("https://learnwithhasan.com/how-to-build-a-semantic-plagiarism-detector/")

# Use 'utf-8' encoding to open the file
with open("state_of_the_union.txt", encoding='utf-8') as f:
    state_of_the_union = f.read()

text_splitter = SemanticChunker(
    OpenAIEmbeddings()
)
# Process the text to create documents
docs = text_splitter.create_documents([state_of_the_union])
print(docs)
