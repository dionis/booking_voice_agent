import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli, ChatContext, ChatMessage
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    silero,
    google
)

# from livekit.plugins import noise_cancellation
import supabase_client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from util import load_prompt

logger = logging.getLogger("restaurant-example")
logger.setLevel(logging.INFO)

load_dotenv('config/.env')


EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

voices = {
    "greeter": "794f9389-aac1-45b6-b726-9d9369183238",
    "reservation": "156fb8d2-335b-4950-9cb3-a2d33befec77",
    "takeaway": "6f84f4b8-58a2-430c-8c79-688dad597532",
    "checkout": "39b376fc-488e-4d0c-8b37-e00b72059fdd",
}

SERVICE_NAME = 'Punta Cana Booking platform'
WELCOME_GREETINGS =  (f"Hi there! Welcome to our {SERVICE_NAME}."
                      f" I can speak in multiple languages including Spanish, French, German, and Italian. "
                      f"Just ask me to switch to any of these languages.")

WELCOME_RESERVATION =  (f"Please, tell us your favorite tour features for show our options?.")

NOT_EXIST_EXPERIENCES_TOUR_WITH_YOUR_PREFERENCES = (f"Sorry, not exits tour with your preferences, "
                                                    f"can you give another details about?.")

@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None

    reservation_time: Optional[str] = None

    order: Optional[list[str]] = None

    customer_credit_card: Optional[str] = None
    customer_credit_card_expiry: Optional[str] = None
    customer_credit_card_cvv: Optional[str] = None

    expense: Optional[float] = None
    checked_out: Optional[bool] = None

    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "reservation_time": self.reservation_time or "unknown",
            "order": self.order or "unknown",
            "credit_card": {
                "number": self.customer_credit_card or "unknown",
                "expiry": self.customer_credit_card_expiry or "unknown",
                "cvv": self.customer_credit_card_cvv or "unknown",
            }
            if self.customer_credit_card
            else None,
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        # summarize in yaml performs better than json
        return yaml.dump(data)

@dataclass
class BookingUserData:
    customer_name: Optional[str] = None
    customer_last_name: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_email: Optional[str] = None

    booking_number_of_passengers: Optional[int] = 1
    booking_adults_number: Optional[int] = 1
    booking_children_number: Optional[int] = 0

    booking_amount: Optional[float] = 0.0
    booking_tour_language: Optional[str] = ''

    booking_tour_id: Optional[int] = 0
    booking_payment_type: Optional[str] = ''

    reservation_time: Optional[str] = None

    order: Optional[list[str]] = None

    # customer_credit_card: Optional[str] = None
    # customer_credit_card_expiry: Optional[str] = None
    # customer_credit_card_cvv: Optional[str] = None

    expense: Optional[float] = None
    checked_out: Optional[bool] = None

    agents: dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_last_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "customer_email": self.customer_phone or "unknown",
            "reservation_time": self.reservation_time or "unknown",
            "order": self.order or "unknown",
            "booking_number_of_passengers": self.reservation_time or "unknown",
            "booking_adults_number": self.reservation_time or "unknown",
            "booking_children_number": self.reservation_time or "unknown",
            "booking_amount": self.reservation_time or "unknown",
            "booking_tour_language": self.reservation_time or "unknown",
            "booking_tour_id": self.reservation_time or "unknown",
            "booking_payment_type": self.reservation_time or "unknown",
            # "credit_card": {
            #     "number": self.customer_credit_card or "unknown",
            #     "expiry": self.customer_credit_card_expiry or "unknown",
            #     "cvv": self.customer_credit_card_cvv or "unknown",
            # }
            # if self.customer_credit_card
            # else None,
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        # summarize in yaml performs better than json
        return yaml.dump(data)


RunContext_T = RunContext[UserData]


# common functions


@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_name = name
    return f"The name is updated to {name}"

@function_tool()
async def update_last_name(
    last_name: Annotated[str, Field(description="The customer's last name")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their name.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_last_name = last_name
    return f"The name is updated to {last_name}"


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their phone number.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_phone = phone
    return f"The phone number is updated to {phone}"


@function_tool()
async def update_email(
    email: Annotated[str, Field(description="The customer's email")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their email.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.customer_email = email
    return f"The email is updated to {email}"

@function_tool()
async def update_booking_number_of_passengers(
    booking_number_of_passengers: Annotated[int, Field(description="The customer's booking number of passengers")],
    context: RunContext_T,
) -> str:
    """Called when the user provides their booking number of passengers.
    Confirm the spelling with the user before calling the function."""
    userdata = context.userdata
    userdata.booking_number_of_passengers = booking_number_of_passengers
    return f"The booking number of passengers is updated to {booking_number_of_passengers}"

@function_tool()
async def update_booking_adults_number(
            booking_adults_number: Annotated[
                int, Field(description="The customer's booking adults number of passengers")],
            context: RunContext_T,
    ) -> str:
        """Called when the user provides their booking adults number.
        Confirm the spelling with the user before calling the function."""
        userdata = context.userdata
        userdata.booking_adults_number = booking_adults_number
        return f"The booking adults number is updated to {booking_adults_number}"

@function_tool()
async def update_booking_children_number(
            booking_children_number: Annotated[
                int, Field(description="The customer's booking child number of passengers")],
            context: RunContext_T,
    ) -> str:
        """Called when the user provides their booking children number.
        Confirm the spelling with the user before calling the function."""
        userdata = context.userdata
        userdata.booking_children_number = booking_children_number
        return f"The booking child number is updated to {booking_children_number}"


@function_tool()
async def to_greeter(context: RunContext_T) -> Agent:
    """Called when user asks any unrelated questions or requests
    any other services not in your job description."""
    curr_agent: BaseAgent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)



class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # add the previous agent's chat history to the current agent
        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True, exclude_function_call=False
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in truncated_chat_ctx.items if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # add an instructions including the user data as assistant message
        chat_ctx.add_message(
            role="system",  # role=system works for OpenAI's LLM and Realtime API
            content=f"You are {agent_name} agent. Current user data is {userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transferring to {name}."



##### SEEE EXAMPLE ##########################################
class TriageAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=load_prompt('triage_prompt.yaml'),
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(),
            vad=silero.VAD.load()
        )

    @function_tool
    async def identify_customer(self, first_name: str, last_name: str):
        """
        Identify a customer by their first and last name.

        Args:
            first_name: The customer's first name
            last_name: The customer's last name
        """
        userdata: UserData = self.session.userdata
        userdata.first_name = first_name
        userdata.last_name = last_name
        userdata.customer_id = db.get_or_create_customer(first_name, last_name)

        return f"Thank you, {first_name}. I've found your account."

    @function_tool
    async def transfer_to_sales(self, context: RunContext_T) -> Agent:
        # Create a personalized message if customer is identified
        userdata: UserData = self.session.userdata
        if userdata.is_identified():
            message = f"Thank you, {userdata.first_name}. I'll transfer you to our Sales team who can help you find the perfect product."
        else:
            message = "I'll transfer you to our Sales team who can help you find the perfect product."

        await self.session.say(message)
        return await self._transfer_to_agent("sales", context)

    @function_tool
    async def transfer_to_returns(self, context: RunContext_T) -> Agent:
        # Create a personalized message if customer is identified
        userdata: UserData = self.session.userdata
        if userdata.is_identified():
            message = f"Thank you, {userdata.first_name}. I'll transfer you to our Returns department who can assist with your return or exchange."
        else:
            message = "I'll transfer you to our Returns department who can assist with your return or exchange."

        await self.session.say(message)
        return await self._transfer_to_agent("returns", context)





##############################################################
class Greeter(BaseAgent):
    def __init__(self, menu: str) -> None:

        #Define default language
        self.current_language = "es"

        super().__init__(
            instructions=(
                f"You are a friendly restaurant receptionist. The menu is: {menu}\n"
                "Your jobs are to greet the caller and understand if they want to "
                "make a reservation or order takeaway. Guide them to the right agent using tools."
                "ask to new user if want to a reservation or order takeaway."
                "You can switch to a different language if asked.  Don't use any unpronounceable character"
            ),

            #llm=openai.LLM(parallel_tool_calls=False),

            llm=google.LLM(
                model="gemini-2.0-flash-exp",
                temperature=0.8,
            ),

            tts=cartesia.TTS(
                voice = voices["greeter"],
                language = self.current_language
            ),
        )
        self.menu = menu

        self.language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian"
        }

        self.deepgram_language_codes = {
            "en": "en",
            "es": "es",
            "fr": "fr-CA",
            "de": "de",
            "it": "it"
        }

        self.greetings = {
            "en": "Hello! I'm now speaking in English. How can I help you today?",
            "es": "¡Hola! Ahora estoy hablando en español. ¿Cómo puedo ayudarte hoy?",
            "fr": "Bonjour! Je parle maintenant en français. Comment puis-je vous aider aujourd'hui?",
            "de": "Hallo! Ich spreche jetzt Deutsch. Wie kann ich Ihnen heute helfen?",
            "it": "Ciao! Ora sto parlando in italiano. Come posso aiutarti oggi?"
        }

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when user wants to make or update a reservation.
        This function handles transitioning to the reservation agent
        who will collect the necessary details like reservation time,
        customer name and phone number."""
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to place a takeaway order.
        This includes handling orders for pickup, delivery, or when the user wants to
        proceed to checkout with their existing order."""
        return await self._transfer_to_agent("takeaway", context)

    async def on_enter(self):
        await self.session.say(WELCOME_GREETINGS)

    async def _switch_language(self, language_code: str) -> None:
        """Helper method to switch the language"""
        if language_code == self.current_language:
            await self.session.say(f"I'm already speaking in {self.language_names[language_code]}.")
            return

        if self.tts is not None:
            self.tts.update_options(language=language_code)

        if self.stt is not None:
            deepgram_language = self.deepgram_language_codes.get(language_code, language_code)
            self.stt.update_options(language=deepgram_language)

        self.current_language = language_code

        await self.session.say(self.greetings[language_code])

    @function_tool
    async  def switch_to_english(self):
        """Switch to speaking English"""
        await self._switch_language("en")

    @function_tool
    async def switch_to_spanish(self):
        """Switch to speaking Spanish"""
        await self._switch_language("es")

    @function_tool
    async def switch_to_french(self):
        """Switch to speaking French"""
        await self._switch_language("fr")

    @function_tool
    async def switch_to_german(self):
        """Switch to speaking German"""
        await self._switch_language("de")

    @function_tool
    async def switch_to_italian(self):
        """Switch to speaking Italian"""
        await self._switch_language("it")


###
#   Call remote rag functions
##
async def rag_lookup(query_to_rag: str) -> list[str]:
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Create embeddings for each chunk
    embedding = model.encode(query_to_rag).tolist()
    print(embedding)

    ## Fix error https://github.com/langchain-ai/langchain/issues/10065
    result_rag = supabase_client.search_similar_embedding_experiences(embedding)
    experience_id = -1

    if len(result_rag.data) == 0:
        print(f"Not exist result :{result_rag.data}")
    else:
        experience_id =  result_rag.data[0]['experience_id']
        # for iResult_in_rag in result_rag.data:
        #     print(f"The most related experiences is : {iResult_in_rag['experience_id']}")

    return experience_id


class Reservation(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions="You are a reservation agent at a restaurant. Your jobs are to ask for "
            "the reservation time, then customer's name, and phone number. Then "
            "confirm the reservation details with the customer.",
            tools=[
                update_name,
                update_last_name,
                update_phone,
                update_email,
                update_booking_number_of_passengers,
                update_booking_adults_number,
                update_booking_children_number,
                to_greeter
            ],
            tts = cartesia.TTS(voice=voices["reservation"]),
        )
    async def on_enter(self):
        await self.session.say( WELCOME_RESERVATION)

    ### Wait the user ask for booking reservation options
    async def on_user_turn_completed(
                self, turn_ctx: ChatContext, new_message: ChatMessage,
        ) -> None:
            rag_content = await rag_lookup(new_message.text_content())

            if rag_content < 0 :
                #Say there are not result
                self.session.say(NOT_EXIST_EXPERIENCES_TOUR_WITH_YOUR_PREFERENCES)
            else:
                booking_userdata = turn_ctx.userdata
                booking_userdata.booking_tour_id = rag_content

                #
                # Use for give more LLM context !!!! IMPORTANT !!!!!
                #

                #turn_ctx.add_message(role="assistant", content=rag_content)
                #await self.update_chat_ctx(turn_ctx)

    @function_tool()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="The reservation time")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their reservation time.
        Confirm the time with the user before calling the function."""
        userdata = context.userdata
        userdata.reservation_time = time
        return f"The reservation time is updated to {time}"

    @function_tool()
    async def confirm_reservation(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the reservation."""
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone number first."

        if not userdata.reservation_time:
            return "Please provide reservation time first."

        return await self._transfer_to_agent("greeter", context)


class Takeaway(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"Your are a takeaway agent that takes orders from the customer. "
                f"Our menu is: {menu}\n"
                "Clarify special requests and confirm the order with the customer."
            ),
            tools=[to_greeter],
            tts=cartesia.TTS(voice=voices["takeaway"]),
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[list[str], Field(description="The items of the full order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user create or update their order."""
        userdata = context.userdata
        userdata.order = items
        return f"The order is updated to {items}"

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the order."""
        userdata = context.userdata
        if not userdata.order:
            return "No takeaway order found. Please make an order first."

        return await self._transfer_to_agent("checkout", context)


class Checkout(BaseAgent):
    def __init__(self, menu: str) -> None:
        super().__init__(
            instructions=(
                f"You are a checkout agent at a restaurant. The menu is: {menu}\n"
                "Your are responsible for confirming the expense of the "
                "order and then collecting customer's name, phone number and credit card "
                "information, including the card number, expiry date, and CVV step by step."
            ),
            tools=[update_name, update_phone, to_greeter],
            tts=cartesia.TTS(voice=voices["checkout"]),
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: RunContext_T,
    ) -> str:
        """Called when the user confirms the expense."""
        userdata = context.userdata
        userdata.expense = expense
        return f"The expense is confirmed to be {expense}"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[str, Field(description="The CVV of the credit card")],
        context: RunContext_T,
    ) -> str:
        """Called when the user provides their credit card number, expiry date, and CVV.
        Confirm the spelling with the user before calling the function."""
        userdata = context.userdata
        userdata.customer_credit_card = number
        userdata.customer_credit_card_expiry = expiry
        userdata.customer_credit_card_cvv = cvv
        return f"The credit card number is updated to {number}"

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when the user confirms the checkout."""
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."

        if (
            not userdata.customer_credit_card
            or not userdata.customer_credit_card_expiry
            or not userdata.customer_credit_card_cvv
        ):
            return "Please provide the credit card information first."

        userdata.checked_out = True
        return await to_greeter(context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> tuple[Agent, str]:
        """Called when the user wants to update their order."""
        return await self._transfer_to_agent("takeaway", context)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"

    userdata = UserData()

    booking_userdata = BookingUserData()

    userdata.agents.update(
        {
            "greeter": Greeter(menu),
            "reservation": Reservation(),
            "takeaway": Takeaway(menu),
            "checkout": Checkout(menu),
        }
    )

    booking_userdata.agents.update(
        {
            "greeter": Greeter(menu),
            "reservation": Reservation(),
            "takeaway": Takeaway(menu),
            "checkout": Checkout(menu),
        }
    )

    session = AgentSession[BookingUserData](
        userdata = booking_userdata,


        stt = deepgram.STT(model="nova-3", language="multi"),
        # llm=openai.LLM(model="gpt-4o-mini"),

        llm = google.LLM(
            model="gemini-2.0-flash-exp",
            temperature=0.8,
        ),

        tts = cartesia.TTS(),

        vad = silero.VAD.load(),

        max_tool_steps = 5,
        # to use realtime model, replace the stt, llm, tts and vad with the following
        # llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    await session.start(
        agent = booking_userdata.agents["greeter"],
        room = ctx.room,
        room_input_options = RoomInputOptions(
            # noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    

    # await agent.say("Welcome to our restaurant! How may I assist you today?")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))