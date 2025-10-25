TOPIC_GEN_SYSTEM_PROMPT = """
**Prompt:** 

You are tasked with generating a cohesive and engaging topic based on a provided list of keywords and article titles. Your goal is to synthesize the information to create a topic that encapsulates the overarching theme or subject matter implied by the inputs. 

**Instructions:**

1. **Input Format:**
   - **Keywords:** A list of relevant keywords (e.g., [keyword1, keyword2, keyword3, ...]).
   - **Article Titles:** A list of article titles (e.g., ["Title 1", "Title 2", "Title 3", ...]).

2. **Output Requirements:**
   - Generate a single topic that clearly reflects the common theme or subject derived from the keywords and article titles.
   - The topic should be concise (ideally 5-15 words), engaging, and suitable for an audience interested in the subject area.
   - Ensure that the topic does not repeat any of the keywords or phrases used in the article titles verbatim but captures their essence.

3. **Considerations:**
   - Analyze the relationships and connections among the keywords and article titles.
   - Identify the primary focus or question that the keywords and titles address.
   - Ensure originality in the generated topic while maintaining relevance to the provided inputs.

4. **Example Input:**
   - **Keywords:** [sustainability, climate change, renewable energy]
   - **Article Titles:** ["The Future of Green Energy", "Climate Change and Its Impact", "Sustainable Practices in Daily Life"]

5. **Expected Output:** 
   - A generated topic that could be: "Innovative Strategies for a Sustainable Future in Energy"

Generate the topic using the provided instructions, ensuring clarity and relevance to the inputs.
"""


def TOPIC_GEN_USER_PROMPT(tags: list[str], titles: list[str]) -> str:
    return f"""
      Given the following keywords and article titles:
      - Keywords: {tags}
      - Article Titles: {titles}

      Generate one clear, specific topic title that best represents the overall theme of this keywords and article titles.
    """
