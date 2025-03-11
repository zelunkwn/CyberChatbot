import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

csv_path = "D:/Chatbot CB Discord/cyber_bullying_dataset.csv"

df = pd.read_csv(csv_path, sep=",", header=0, engine="python")
print(df.head())  
print(df.shape)   

df.columns = ["id", "kalimat", "sentimen"]
print(df.head()) 

label_mapping = {"negatif": 0, "positif": 1}
df["sentimen"] = df["sentimen"].map(label_mapping)
df = df.dropna()
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

def preprocess_function(examples):
    return {
        **tokenizer(examples["kalimat"], truncation=True, padding="max_length", max_length=128),
        "labels": examples["sentimen"]
    }

tokenized_dataset = dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", num_labels=2)
training_args = TrainingArguments(
    output_dir="D:/Chatbot CB Discord/trained_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="D:/Chatbot DB Discord/logs",
)
train_dataset = tokenized_dataset["train"].with_format("torch")
eval_dataset = tokenized_dataset["test"].with_format("torch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
trainer.train()

model.save_pretrained("D:/Chatbot CB Discord/trained_model")
tokenizer.save_pretrained("D:/Chatbot CB Discord/trained_model")
