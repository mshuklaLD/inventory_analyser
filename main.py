from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Load LLM
model = OllamaLLM(model="deepseek-r1:1.5b")
# model = OllamaLLM(model="gemma3:4b")

# Prompt template
template = """
You are an expert in analyzing diamond inventory.

Here are some matching diamonds: {inventory}

Following is a description of some of the columns:
Aging - How long the stone has been in my inventory, measured in days. 
Rank - How does my stone rank compared to other similar stones on RapNet platform based on the price per carat. Rank 1 is the best, and the higher the number, the worse the rank.
Count - Total number of similar stones are available on RapNet platform
My Stock - Number of stones similar to this that I have in my inventory. If it says 1, it means there are no other stones like this in my inventory
My Sales	- How many similar items have I sold in the last 2 months
My YTD Sales - How many similar items have I sold from the beginning of 2025
Rap $/Ct	- This is the price per carat of the diamond according to the Rapaport price list
Rap Total - This is the total price of the diamond according to the Rapaport price list
Rap % - The discount percentage of the Rapaport price that I am asking for this stone. For example, if the Rap % is 0, I am asking for the full Rapaport price. If it is -40, I am asking for 40% less than the Rapaport price.
$/Ct - This is the price per carat of the diamond according to my asking price
Total Price - This is the total price of the diamond i.e. Carat X $/Ct
Answer the question based on this data: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    if question.lower() == "q":
        break

    docs = retriever.invoke(question)

    inventory = "\n".join([doc.page_content for doc in docs])
    result = chain.invoke({"inventory": inventory, "question": question})
    print(result)
