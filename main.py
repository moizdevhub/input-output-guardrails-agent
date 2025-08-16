from agents import (Agent, Runner, OutputGuardrail, GuardrailFunctionOutput,OutputGuardrailTripwireTriggered,
                    output_guardrail, InputGuardrailTripwireTriggered,input_guardrail,
                    trace)

import rich 
import asyncio

from connection import model, config
from pydantic import BaseModel





#############   inputput Guardrail ##################



class FinancialInputValidatorOutput(BaseModel):
    isFinancialQuery:bool
    reason:str





financial_input_validator = Agent(
    name="Financial Intent Classifier Agent",
    instructions="""
    "You are an input guardrail agent." 
    Your task is to detect whether the user's query is financial-related or not. 

    Rules:
    1. If the query is about investments, money, banking, stocks, retirement, cryptocurrency, savings, or financial planning, mark it as financial-related.
    2. If the query is unrelated to finance (for example: weather, sports, health, personal life, or random questions), mark it as not financial-related.
    3. Always respond in a structured way: 
    - isFinancialQuery: True or False
    - reason: A short explanation why you classified it this way.
    """,
    model = model,
    output_type= FinancialInputValidatorOutput

)
 


@input_guardrail
async def financial_input_guardrail(ctx,agent,input)-> GuardrailFunctionOutput:
    result = await Runner.run(financial_input_validator,input, run_config= config)
    print("Input Guardrail: ",result.final_output)

    return GuardrailFunctionOutput(
        output_info = result.final_output.reason,
        tripwire_triggered= not result.final_output.isFinancialQuery
    )



#______________________________________________________________________








#############   Output Guardrail ##################


class FinancialAdviceOutput(BaseModel):
    response: str
    isAppropriateAdvice: bool
    containsDisclaimer: bool
    reason: str



financial_output_guardrail_agent = Agent(
    name='Financial Output Guardrail Agent',
    instructions=""" 
    You are an output guardrail agent for financial advice. Your task is to validate that:
    1. The response contains appropriate disclaimers about not being professional financial advice
    2. The advice is general and educational, not specific investment recommendations
    3. The response encourages consulting with licensed financial advisors
    4. No guarantees about returns or specific outcomes are made
    
    Set isAppropriateAdvice to True only if all safety criteria are met.
    Always include containsDisclaimer as True if proper disclaimers are present.
    """,
   
    output_type = FinancialAdviceOutput
)


@output_guardrail
async def financial_output_guardrail(ctx, agent, output)-> GuardrailFunctionOutput:
    result = await Runner.run(financial_output_guardrail_agent, f"Validate this financial advice response: {output}"
             ,run_config=config)
    

    return GuardrailFunctionOutput(
        output_info =  result.final_output.reason,
        tripwire_triggered = not (result.final_output.isAppropriateAdvice and 
                                  result.final_output.containsDisclaimer)
    )



financial_advisor_agent = Agent(
    name="Financial Advisor Agent",
    instructions=""" 
    You are a financial education agent. 
    Provide general financial information and education.
    Always include disclaimers that this is not professional financial advice.
    Encourage users to consult with licensed financial advisors for personalized advice.
    Never guarantee specific returns or outcomes.""",
    output_guardrails=[financial_output_guardrail]
)

#____________________________________________________________________________





triage_agent = Agent(
name="Triage Agent", instructions= "You are a triage agent for financial queries. Delegate financial questions to the financial advisor agent.",
model = model,
input_guardrails = [financial_input_guardrail],
handoffs=[financial_advisor_agent]
)





async def main():

    with trace("Output Guardrail"):
        try:
            result = await Runner.run(triage_agent, input = "Tell me exactly which stocks to buy to guarantee 50% returns in 6 months", run_config=config)
            print(result.final_output)
        except (InputGuardrailTripwireTriggered , OutputGuardrailTripwireTriggered) as e:
            if isinstance(e, InputGuardrailTripwireTriggered):
                print(f"[red]Input guardrail triggered:[/red] {e}")
            elif isinstance(e, OutputGuardrailTripwireTriggered):
                print(f"[red]Output guardrail triggered:[/red] ")


if __name__ == "__main__":
   asyncio.run(main())
