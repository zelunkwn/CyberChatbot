import discord
from discord.ext import commands
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

if not TOKEN:
    raise ValueError("❌ DISCORD_TOKEN is missing from .env file!")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="cb", intents=intents)

model_path = "D:/Chatbot CB Discord/trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
print("✅ Model loaded successfully!")
CYBERBULLYING_THRESHOLD = 0.6

message_buffer = []
token_count = 0

LABEL_MAPPING = { 
    "LABEL_0": "CYBERBULLYING", 
    "LABEL_1": "NON_CYBERBULLYING" 
}

@bot.event
async def on_ready():
    print(f"✅ {bot.user} is now online!")

@bot.event
async def on_message(message):
    global token_count, message_buffer

    if message.author == bot.user:
        return

    tokens = tokenizer.tokenize(message.content)
    message_buffer.append(message.content)
    token_count += len(tokens)

    result = nlp_pipeline(message.content)
    label = LABEL_MAPPING.get(result[0]['label'], "UNKNOWN")
    score = result[0]['score']

    print(f"🔍 Detected: {label} with score: {score}")

    if label == "CYBERBULLYING" and score >= CYBERBULLYING_THRESHOLD:
        await message.channel.send(
            f"⚠️ Pesan ini terdeteksi sebagai cyberbullying:\n> {message.content}\n"
            "Mohon untuk menjaga komunikasi yang sehat."
        )

        if message.guild:
            admin_channel = discord.utils.get(message.guild.channels, name="admin-notifikasi")
            if admin_channel:
                await admin_channel.send(
                    f"🚨 Cyberbullying detected from {message.author.mention}\n"
                    f"Pesan: {message.content}"
                )
            else:
                print("⚠️ 'admin-notifikasi' channel not found.")

    await bot.process_commands(message)

bot.run(TOKEN)
