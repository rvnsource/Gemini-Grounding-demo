import os
from IPython.display import Markdown, display
import vertexai
from vertexai.generative_models import (
    GenerationResponse,
    GenerativeModel,
    Tool,
    grounding,
)
from vertexai.preview.generative_models import grounding as preview_grounding

#############################################
# Create Gemini 1.5 model using VertexAI sdk.
#############################################
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/ravi/.config/gcloud/genai-443318-637e44e3cf32.json"
PROJECT_ID = "genai-443318"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}
vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-1.5-flash")

def print_grounding_response(response: GenerationResponse):
    """Prints Gemini response with grounding citations."""
    grounding_metadata = response.candidates[0].grounding_metadata

    # Citation indices are in byte units
    ENCODING = "utf-8"
    text_bytes = response.text.encode(ENCODING)

    prev_index = 0
    markdown_text = ""

    for grounding_support in grounding_metadata.grounding_supports:
        text_segment = text_bytes[
            prev_index : grounding_support.segment.end_index
        ].decode(ENCODING)

        footnotes_text = ""
        for grounding_chunk_index in grounding_support.grounding_chunk_indices:
            footnotes_text += f"[{grounding_chunk_index + 1}]"

        markdown_text += f"{text_segment} {footnotes_text}\n"
        prev_index = grounding_support.segment.end_index

    if prev_index < len(text_bytes):
        markdown_text += str(text_bytes[prev_index:], encoding=ENCODING)

    markdown_text += "\n----\n## Grounding Sources\n"

    if grounding_metadata.web_search_queries:
        markdown_text += (
            f"\n**Web Search Queries:** {grounding_metadata.web_search_queries}\n"
        )
        if grounding_metadata.search_entry_point:
            markdown_text += f"\n**Search Entry Point:**\n {grounding_metadata.search_entry_point.rendered_content}\n"
    elif grounding_metadata.retrieval_queries:
        markdown_text += (
            f"\n**Retrieval Queries:** {grounding_metadata.retrieval_queries}\n"
        )

    markdown_text += "### Grounding Chunks\n"

    for index, grounding_chunk in enumerate(
        grounding_metadata.grounding_chunks, start=1
    ):
        context = grounding_chunk.web or grounding_chunk.retrieved_context
        if not context:
            print(f"Skipping Grounding Chunk {grounding_chunk}")
            continue

        markdown_text += f"{index}. [{context.title}]({context.uri})\n"
        print(markdown_text)

    display(Markdown(markdown_text))



if __name__ == "__main__":
    PROMPT = "You are an expert in astronomy. When is the next solar eclipse in the US?"
    tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())
    response = model.generate_content(PROMPT, tools=[tool])
    print_grounding_response(response)

