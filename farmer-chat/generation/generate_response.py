import datetime
import asyncio

from django_core.config import Config
from rag_service.openai_service import make_openai_request

# Prefer Servvia healthcare prompt; fall back to env prompt if not available
try:
    from django_core.servvia_prompts import RESPONSE_GEN_PROMPT as SERVVIA_RESPONSE_PROMPT
except Exception:
    SERVVIA_RESPONSE_PROMPT = None


async def setup_prompt(user_name, context_chunks, rephrased_query, system_prompt=Config.RESPONSE_GEN_PROMPT):
    """
    Setup generation response prompt for a rephrased user query with retrieved chunks.
    Prefer Servvia's healthcare prompt if available.
    """
    prompt_name_1 = user_name if user_name else "a person"
    prompt_template = SERVVIA_RESPONSE_PROMPT or system_prompt
    response_prompt = prompt_template.format(
        name_1=prompt_name_1,
        context=context_chunks,
        input=rephrased_query,
    )

    return response_prompt


async def generate_query_response(original_query, user_name, context_chunks, rephrased_query):
    """
    Generate final response for a rephrased user query with the retrieved chunks.
    """
    response_map = {}
    llm_response = None
    response_gen_start = None
    response_gen_end = None

    generation_completion_tokens = 0
    generation_prompt_tokens = 0
    generation_total_tokens = 0

    response_gen_exception = None
    response_gen_retries = 0

    response_map.update(
        {
            "response": llm_response,
            "original_query": original_query,
            "rephrased_query": rephrased_query,
            "generation_start_time": response_gen_start,
            "generation_end_time": response_gen_end,
            "completion_tokens": generation_completion_tokens,
            "prompt_tokens": generation_prompt_tokens,
            "total_tokens": generation_total_tokens,
            "response_gen_exception": response_gen_exception,
            "response_gen_retries": response_gen_retries,
        }
    )

    response_prompt = await setup_prompt(user_name, context_chunks, rephrased_query)

    response_gen_start = datetime.datetime.now()
    generated_response, response_gen_exception, response_gen_retries = await make_openai_request(response_prompt)
    response_gen_end = datetime.datetime.now()

    if generated_response:
        llm_response = generated_response.choices[0].message.content
        usage = getattr(generated_response, "usage", None)
        if usage:
            generation_completion_tokens = getattr(usage, "completion_tokens", 0)
            generation_prompt_tokens = getattr(usage, "prompt_tokens", 0)
            generation_total_tokens = getattr(usage, "total_tokens", 0)

    response_map.update(
        {
            "response": llm_response,
            "original_query": original_query,
            "rephrased_query": rephrased_query,
            "generation_start_time": response_gen_start,
            "generation_end_time": response_gen_end,
            "completion_tokens": generation_completion_tokens,
            "prompt_tokens": generation_prompt_tokens,
            "total_tokens": generation_total_tokens,
            "response_gen_exception": response_gen_exception,
            "response_gen_retries": response_gen_retries,
        }
    )

    return response_map
