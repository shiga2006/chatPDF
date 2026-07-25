"""Fix all syntax errors in graph.py - K-Means missing except + LLM except indent"""
with open("backend/agents/graph.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

# Fix 1: Insert 'except Exception:pass' after K-Means try block (line 320 is the return })
# Lines: 299=if, 300=try:, 301-320=block, 322=prompt = (
# We need to insert except after line 320
new_lines = []
inserted_kmeans = False
for i, line in enumerate(lines):
    new_lines.append(line)
    if i == 319 and not inserted_kmeans:  # after '}' at line 320
        new_lines.append('        except Exception:\n')
        new_lines.append('            pass\n')
        inserted_kmeans = True
        print(f"Inserted except after line {i+1}")

    # Fix 2: Fix LLM except indent (line 342)
    if i == 342:
        new_lines[-1] = '    except Exception as e:\n'
        print(f"Fixed LLM except at line {i+1}")

with open("backend/agents/graph.py", "w", encoding="utf-8") as f:
    f.writelines(new_lines)

# Verify syntax
import py_compile
try:
    py_compile.compile("backend/agents/graph.py", doraise=True)
    print("SYNTAX CHECK PASSED!")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
