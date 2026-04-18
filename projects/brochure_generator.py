from openai import OpenAI
from website_scraper import scrape_website
import json
from urllib.parse import urljoin
from dotenv import load_dotenv
import os
load_dotenv()

ollama = OpenAI(base_url="https://api.groq.com/openai/v1" , api_key= os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

link_system_prompt = """
You MUST return ONLY valid JSON.

Format EXACTLY like this:
{
    "links": [
        {"type": "about page", "url": "https://example.com/about"}
    ]
}

Rules:
- No explanation
- No extra text
- No markdown
- Only JSON
"""

def get_links_user_prompt(url):
    user_prompt = f"""
    Here is the list of links on the website {url} -
    Please decide which of these are relevant web links for a brochure about the company, 
    respond with the full https URL in JSON format.
    Do not include Terms of Service, Privacy, email links.

    Links (some might be relative links):

    """
    webb = scrape_website(url)
    links = webb.links
    for link in links:
        full_url = urljoin(url, link)
        user_prompt += full_url + "\n"
    return user_prompt

def select_relevant_links(url):
    print(f"Selecting relevant links for {url} by calling model:{MODEL}")

    response = ollama.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(url)}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    result = response.choices[0].message.content

    try:
        links = json.loads(result)

        # safety check (IMPORTANT)
        if "links" not in links:
            print("⚠️ Model returned JSON but missing 'links'")
            print(result)
            return {"links": []}

    except Exception as e:
        print("⚠️ JSON parsing failed:", e)
        print("Raw output:", result)
        return {"links": []}

    print(f"Found {len(links.get('links', []))} relevant links")
    return links
    
def fetch_page_and_relevant_links(url):
    website = scrape_website(url)
    contents = website.text
    relevant_links = select_relevant_links(url)
    relevant_links['links'] = relevant_links['links'][:3]
    result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"
    for link in relevant_links['links']:
        w = scrape_website(urljoin(url, link["url"]))
        result += f"\n\n### Link: {link['type']}\n"
        result += w.text
    return result

brochure_system_prompt = """
    You are an assistant that analyzes the contents of several relevant pages from a company website
    and creates a short brochure about the company for prospective customers, investors and recruits.
    Respond in markdown without code blocks.
    Include details of company culture, customers and careers/jobs if you have the information.
    """

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"""
    You are looking at a company called: {company_name}
    Here are the contents of its landing page and other relevant pages;
    use this information to build a short brochure of the company in markdown without code blocks.\n\n
    """
    user_prompt += fetch_page_and_relevant_links(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt

from IPython.display import Markdown

def create_brochure(company_name, url):
    # stream = ollama.chat.completions.create(
    response = ollama.chat.completions.create(
    
        model = MODEL,
        messages=[
            {"role":"system" , "content":brochure_system_prompt},
            {"role":"user" , "content": get_brochure_user_prompt(company_name,url)}
        ]
        # stream = True
    )
    # response = ""
    # display_handle = display(Markdown(""), display_id=True)
    # for chunk in stream:
    #     response += chunk.choices[0].delta.content or ''
    #     update_display(Markdown(response), display_id=display_handle.display_id)
    return response.choices[0].message.content

def main():
    website = input("Enter the website url: ")
    company_name = input("Enter the company name: ")
    result = create_brochure(company_name, website)
    print(result)

if __name__ == "__main__":
    main()