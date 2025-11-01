SYSTEM_PROMPT = """
You are tasked with generating a cohesive and engaging topic based on a provided list of keywords and article titles. 
Your goal is to synthesize the information to create a topic that encapsulates the overarching theme or subject matter implied by the inputs. 

**Instructions:**

1. **Input Format:**
   - **Keywords:** A list of relevant keywords (e.g., [keyword1, keyword2, keyword3, ...]).
   - **Article Titles:** A list of article titles (e.g., ["Title 1", "Title 2", "Title 3", ...]).

2. **Output Requirements:**
   - Generate a single topic that clearly reflects the common theme or subject derived from the keywords and article titles.
   - The topic should be concise (ideally 5-15 words), engaging, and suitable for an audience interested in the subject area.
   - Ensure that the topic does not repeat any of the keywords or phrases used in the article titles verbatim but captures their essence.
   - Generate a description of the topic that describes the topic in a way that is engaging and interesting to the audience.
   - The description should be 1-2 sentences long, engaging, and suitable for an audience interested in the subject area.
   - Ensure that the description does not repeat any of the keywords or phrases used in the article titles verbatim but captures their essence.
   - Use the keywords and article titles to generate a score for the topic between 0 and 100. The score should be a measure of how well the topic captures the essence of the keywords and article titles.

3. **Considerations:**
   - Analyze the relationships and connections among the keywords and article titles.
   - Identify the primary focus or question that the keywords and titles address.
   - Ensure originality in the generated topic while maintaining relevance to the provided inputs.

4. **Example Input:**
   - **Keywords:** [sustainability, climate change, renewable energy]
   - **Article Titles:** ["The Future of Green Energy", "Climate Change and Its Impact", "Sustainable Practices in Daily Life"]

5. **Expected Output:** 
   - A generated topic that could be: "Innovative Strategies for a Sustainable Future in Energy"
   - A generated description that could be: "This topic is about the innovative strategies for a sustainable future in energy, how the future of green energy is going to affect the climate change and how we can live a sustainable life."
   - A generated score that could be: 85
   
Generate the topic, description, and score using the provided instructions, ensuring clarity and relevance to the inputs.
"""


def TOPIC_GEN_USER_PROMPT(tags: list[str], titles: list[str]) -> str:
    return f"""
      Given the following keywords and article titles:
      - Keywords: {tags}
      - Article Titles: {titles}

      Generate a topic, description, and score for the keywords and article titles.
    """
