from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, load_prompt
from typing import TypedDict, Annotated, Optional

from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2-Exp',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# schema TypedDict
# class Review(TypedDict):

#     summary: str
#     sentiment: str

class Review(TypedDict):
    key_theme : Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "Return sentiment of the review either negative, positive or neutral"]
    pros : Annotated[Optional[list[str]], "Write down all the pros inside a list"]
    cons : Annotated[Optional[list[str]], "Write down all the cons inside a list"]
    name : Annotated[Optional[list[str]], "Write down the name of reviewer"]


structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
I’ve been using the NovaX Pro 15 Laptop for over three months now, and I have mixed feelings about it. Starting with the build quality, it’s excellent — the aluminum chassis feels premium, the keyboard is very comfortable for long typing sessions, and the hinge is sturdy yet smooth. The display is crisp and vibrant, especially when watching HDR content. Battery life is also impressive — I easily get around 9–10 hours of moderate use, which is great for travel.
However, when it comes to performance, things get complicated. The laptop handles daily productivity tasks well — browsing, spreadsheets, and light photo editing are smooth. But the moment you start gaming or running heavy workloads like video editing, the fans get extremely loud, and the device heats up quickly. The thermal management seems poorly optimized.
The software experience is also not as clean as I’d like. There are too many pre-installed applications that serve no purpose other than consuming storage and running background processes. It gives the whole system a bloated feel. On top of that, the UI design of their custom control center looks dated and unintuitive. I wish they would redesign it to match modern aesthetics.
Another small annoyance is the trackpad — while it’s large, it occasionally registers ghost touches, which interrupts scrolling or dragging.
On the bright side, the speakers are surprisingly loud and clear, and the webcam performs well even in low light. The Wi-Fi 6 support ensures stable connectivity, and I appreciate the wide port selection — no need for extra dongles.
Customer support deserves praise too — I had an issue with the charging adapter, and they replaced it within two days without any hassle. That’s rare these days.
Overall, the NovaX Pro 15 offers great hardware and reliability but is held back by subpar software optimization and some usability quirks. If the company rolls out a major software update to fix these issues, it could easily become one of the best laptops in its segment.
""")

print(result)