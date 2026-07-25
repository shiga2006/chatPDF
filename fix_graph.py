"""Fix indentation issues in graph.py"""
with open("backend/agents/graph.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Fix the K-Means except block - remove the orphaned 'except Exception:' without its try
# Find line with just 'except Exception:' and adjust indentation
fixed = []
i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    # Fix the orphaned except at end of K-Means block - but it has a try above... 
    # Actually the issue is the misindented LLM try/except block
    
    # Fix indentation for LLM classification block: lines from 'prompt = (' to 'except'
    if stripped == 'prompt = (':
        # Check the prompt block has proper 4-space indent
        pass
        
    fixed.append(line)
    i += 1

# More targeted: find the misindented try block and fix it
content = "".join(lines)

# The LLM try block has 8-space indented body but 0-space except
# Replace the entire LLM section
old_llm_section = """    # --- LLM Classification (final fallback) ---
    prompt = (
        "You are the Supervisor Agent for an enterprise document knowledge assistant.\n"
        f"Analyze the user query: \"{query}\"\n\n"
        "Decide which of the following agents is best suited to answer it:\n"
        "- 'summary_agent': if the user explicitly wants a summary, recap, list of bullet points from a document or specific pages.\n"
        "- 'comparison_agent': if the user wants to compare multiple documents, find differences, similarities, or contrast rules.\n"
        "- 'retrieval_agent': for all other standard fact-finding, search, and informational questions about document contents.\n\n"
        "Respond ONLY with the name of the agent ('summary_agent', 'comparison_agent', or 'retrieval_agent') and nothing else. No punctuation."
    )
    try:
            llm = get_llm(temperature=0.0)
            response = llm.invoke([HumanMessage(content=prompt)])
            next_agent = response.content.strip().lower()
            
            # Clean up the output
            if "summary" in next_agent:
                next_agent = "summary_agent"
            elif "comparison" in next_agent or "compare" in next_agent:
                next_agent = "comparison_agent"
            else:
                next_agent = "retrieval_agent"
        except Exception as e:"""

new_llm_section = """    # --- LLM Classification (final fallback) ---
    prompt = (
        "You are the Supervisor Agent for an enterprise document knowledge assistant.\n"
        f"Analyze the user query: \"{query}\"\n\n"
        "Decide which of the following agents is best suited to answer it:\n"
        "- 'summary_agent': if the user explicitly wants a summary, recap, list of bullet points from a document or specific pages.\n"
        "- 'comparison_agent': if the user wants to compare multiple documents, find differences, similarities, or contrast rules.\n"
        "- 'retrieval_agent': for all other standard fact-finding, search, and informational questions about document contents.\n\n"
        "Respond ONLY with the name of the agent ('summary_agent', 'comparison_agent', or 'retrieval_agent') and nothing else. No punctuation."
    )
    try:
        llm = get_llm(temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])
        next_agent = response.content.strip().lower()

        if "summary" in next_agent:
            next_agent = "summary_agent"
        elif "comparison" in next_agent or "compare" in next_agent:
            next_agent = "comparison_agent"
        else:
            next_agent = "retrieval_agent"
    except Exception as e:"""

if old_llm_section in content:
    content = content.replace(old_llm_section, new_llm_section, 1)
    with open("backend/agents/graph.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Fixed LLM classification try/except indentation")
else:
    print("Could not find old LLM section. Searching for alternatives...")
    # Find the line with the problem
    for i, line in enumerate(lines):
        if 'get_llm(temperature=0.0)' in line:
            print(f"Found at line {i+1}: {repr(line)}")
            for j in range(max(0,i-3), min(len(lines), i+5)):
                print(f"  L{j+1}: {repr(lines[j])}")
