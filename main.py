import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()  # Load environment variables from .env file


def main():
    print("Hello from langchain-course!")
    print(f"Model Name: {os.getenv('MODEL_NAME')}")

    information1 = """Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021.

Born into a wealthy New York City family, Trump graduated from the University of Pennsylvania in 1968 with a bachelor's degree in economics. He became the president of his family's real estate business in 1971, renamed it the Trump Organization, and began acquiring and building skyscrapers, hotels, casinos, and golf courses. He launched side ventures, many licensing the Trump name, and filed for six business bankruptcies in the 1990s and 2000s. From 2004 to 2015, he hosted the reality television show The Apprentice, bolstering his fame as a billionaire. Presenting himself as a political outsider, Trump won the 2016 presidential election against Democratic Party nominee Hillary Clinton.

During his first presidency, Trump imposed a travel ban on seven Muslim-majority countries, expanded the Mexico–United States border wall, and enforced a family separation policy on the border. He rolled back environmental and business regulations, signed the Tax Cuts and Jobs Act, and appointed three Supreme Court justices. He withdrew the U.S. from agreements on climate, trade, and Iran's nuclear program, and started a trade war with China. In response to the COVID-19 pandemic in 2020, he downplayed its severity, contradicted health officials, and signed the CARES Act. After losing the 2020 presidential election to Joe Biden, Trump attempted to overturn the result, culminating in the January 6 Capitol attack in 2021. He was impeached twice—in 2019 for abuse of power and obstruction of Congress and in 2021 for incitement of insurrection—and acquitted by the Senate both times.

In 2023, Trump was found liable in civil cases for sexual abuse and defamation and for business fraud. In May 2024, he was found guilty on 34 counts of falsifying business records, making him the first U.S. president convicted of a felony. After winning the 2024 presidential election against Vice President Kamala Harris, he was given a no-penalty sentence, and two federal felony indictments against him for retention of classified documents and obstruction of the 2020 election were dismissed without prejudice."""
    
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOllama(temperature=0, model="gemma3:270m")
    llm = ChatGoogleGenerativeAI(temperature=0, model=os.getenv("MODEL_NAME"))
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information1})
    print(response.content)

if __name__ == "__main__":
    main()
