from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

def add_one(x: int) -> int:
    return 1 + x

def mult_two(y: int) -> int:
    return y * 2

def divide_two(z: int) -> int:
    return z / 2

my_runnable = RunnableLambda(add_one)
my_runnable2 = RunnableLambda(mult_two)
my_runnable3 = RunnableLambda(divide_two)

sequence = my_runnable | my_runnable2

resposta_sequece = sequence.invoke(3)

# parallel = my_runnable | {
#     "mult_two": my_runnable2,
#     "divide_two": my_runnable3,
# }

parallel = my_runnable | RunnableParallel(mult_two=my_runnable2, divide_two=my_runnable3)| RunnablePassthrough()
resposta_parallel = parallel.invoke(2)

print(resposta_parallel)

def transformer_dict(my_dict: dict):
    num: int = len(my_dict["input"]) 
    my_dict.update({"num_caract": num})
    return my_dict

def transforme_entrada(entrada:dict) -> str:
    return entrada["input"] + " conseguiu!"

transformer_dict_function = RunnableLambda(transformer_dict)

def outer_func(entrada:dict):
    entrada['transformer_entrada'] = "outro resultado"
    return {"outer value": entrada['transformer_entrada']}
chain = RunnablePassthrough() | transformer_dict_function |RunnableParallel(transformer_entrada=RunnableLambda(transforme_entrada), passar_para_frente= RunnablePassthrough()) | RunnablePassthrough().assign(context = RunnableLambda(outer_func))

result = chain.invoke({"input": "Parabéns Você"})
print(result)