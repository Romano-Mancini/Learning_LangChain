from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

def analyzePro(description):
    template = ChatPromptTemplate.from_messages(
    [
        ('system',"""You are an helpful assistant which receives a product description. 
                     You will give all the pros of the product as a bullet list.
                     Don't give anything other then the bullets."""),
        ('human',"{description}.")
    ]
    )
    
    return template.format_prompt(description=description)

def analyzeCons(description):
    template = ChatPromptTemplate.from_messages(
    [
        ('system',"""You are an helpful assistant which receives a product description. 
                     You will give all the cons of the product as a bullet list.
                     Don't give anything other then the bullets."""),
        ('human',"{description}.")
    ]
    )
    
    return template.format_prompt(description=description)



load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
first_template = ChatPromptTemplate.from_messages(
    [
        ('system',"You are an helpful {field} assistant. You answer the user's questions."),
        ('human',"Tell me everything that I should know before buying {product}.")
    ]
)


get_pros_chain =  RunnableLambda(lambda x : analyzePro(x)) | model | StrOutputParser()
get_cons_chain =  RunnableLambda(lambda x : analyzeCons(x)) | model | StrOutputParser()

whole_chain = (first_template 
               | model
               | StrOutputParser()
               | RunnableParallel(branches={'pros': get_pros_chain, 'cons': get_cons_chain})
               | RunnableLambda(lambda x : print(f'Pros:\n{x["branches"]["pros"]}\n----------\nCons:\n{x["branches"]["cons"]}')))

whole_chain.invoke({"field":"tech", "product": "iPhone 16 Pro Max"})