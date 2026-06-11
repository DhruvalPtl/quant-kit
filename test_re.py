import re

src = open('llama-src/convert_hf_to_gguf.py', encoding='utf-8').read()
# Find class ModelNameForCausalLM(Model) etc.
names = re.findall(r'^class (\w+(?:ForCausalLM|ForConditionalGeneration|Model|ForMaskedLM|ForSequenceClassification|ForTokenClassification))\b', src, re.MULTILINE)
print(len(names))
print(names[:5])

# Let's try finding the strings in @Model.register(...)
registers = re.findall(r'@Model\.register\((.*?)\)', src, re.DOTALL)
all_archs = set()
for r in registers:
    archs = re.findall(r'"([^"]+)"', r)
    all_archs.update(archs)
    archs2 = re.findall(r"'([^']+)'", r)
    all_archs.update(archs2)

print(len(all_archs))
print(list(all_archs)[:5])
