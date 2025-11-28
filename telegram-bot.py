#pip install python-telegram-bot --upgrade
# Define a command handler (for /start)
import requests
import minirag
#check bot is working
TOKEN = "YourToken"
url = f"https://api.telegram.org/bot{TOKEN}/getMe"
response = requests.get(url)
print(response.json())

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Replace with your bot token from @BotFather
BOT_TOKEN = TOKEN

def start(update, context):
    user = update.message.from_user.first_name
    update.message.reply_text(f"Welcome {user}! Please type /help, /ask <query> to get started.")


# --- Command Handlers ---
async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ask <query> command."""
    if context.args:
        query = " ".join(context.args)
        # For now, just echo back the query. You can plug in RAG/LLM here.
        # get query response from mini-rag
        print (" Received query:", query)
        retrieved, answer = minirag.get_answer(query)
        #await update.message.reply_text(f"Answer:\n{result}")
        await update.message.reply_text(answer)
                                        
    else:
        await update.message.reply_text("Usage: /ask Give me ingredients for a masala dosa or hyderabadi biryani")

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /image command with uploaded photo."""
    if update.message.photo:
        # Get the largest photo (highest resolution)
        photo_file = await update.message.photo[-1].get_file()
        await update.message.reply_text("📷 Image received! (Description logic goes here)")
        # You can download the file if needed:
        # await photo_file.download_to_drive("downloaded_image.jpg")
    else:
        await update.message.reply_text("Please upload an image after using /image.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command."""
    help_text = (
        "Welcome to Recipie bot. Please ask the following Bot Commands:\n"
        "/ask <query> — Ask a text or RAG query\n"
        "/image — Upload an image for description\n"
        "/help — Show this help message"
    )
    await update.message.reply_text(help_text)

# --- Main Bot Setup ---
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("help", help_command))

    # Register image handler (triggered when user sends a photo after /image)
    app.add_handler(MessageHandler(filters.PHOTO & filters.CaptionRegex("^/image"), image_handler))

    print("Bot is running...")
    
    app.run_polling()

if __name__ == "__main__":
    main()
