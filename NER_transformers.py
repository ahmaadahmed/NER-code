from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load tokenizer and model from local directory
model_dir = r"E:\NER_model"  # Replace with the actual path
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

# Create NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Example text
text = "علي ذهب إلى المدرسة في بغداد."

# Perform NER on the text
results = ner_pipeline(text)

# Print the named entities
for result in results:
    print(f"Entity: {result['word']}, Label: {result['entity']}, Score: {result['score']}")
