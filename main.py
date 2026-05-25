import requests
import json
import subprocess
import os
import speech_recognition as sr
import analyse as an
from version_server import start_version_server

from dotenv import find_dotenv, load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from typing import Final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

load_dotenv(find_dotenv())

TOKEN: Final = os.getenv('TELEGRAM_BOT_TOKEN') 
BOT_USERNAME = '@EnghlishCoachBot'
VERSION_SERVER_HOST = os.getenv('VERSION_SERVER_HOST', '0.0.0.0')
VERSION_SERVER_PORT = int(os.getenv('VERSION_SERVER_PORT', '8080'))


#Commands
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Welcome to the English Coach Bot")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("I can help you improve your spoken english. Make sure you reply by recording a voice message")

async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! Welcome to the English Coach Bot")


# Responses
def handle_response(text: str) -> str:    
    processedtext: str = text.lower()

    if 'hello' in processedtext:
        return '''Hey there!, welcome to the English Coach Bot. I can help you improve your spoken english. Make sure you reply by recording a voice message.'''   

    if 'how are you' in processedtext:
        return 'I am good!' 

    
    return 'I didnt understamd what you are saying' 

# Handle incoming messages
async def handle_message(update: Update, context: ContextTypes):
  message_type: str = update.message.chat.type
  text: str = update.message.text

  # Log users
  print(f'User ({update.message.chat.id}) in {message_type}: "{text}" ')

  # Handle message type
  if message_type == 'group':
      if BOT_USERNAME in text:
          new_text: str = text.replace(BOT_USERNAME, '').strip()
          response: str = handle_response(new_text)
      else:
          return
  else:
      response: str = handle_response(text)

  # Reply
  print('Bot:', response)
  await update.message.reply_text(response)

# Voice Note Handler    
async def handle_audio_message(update: Update, context: ContextTypes):
  message_type: str = update.message.chat.type
  chat_id: str = update.message.chat.id
  voice_file_id: str = update.message.voice.file_id
  voice_file_size: str = update.message.voice.file_size
  voice_file_name: str = 'voice-'+voice_file_id+'.oga'
  voice_file_name_wav : str = 'voice-'+voice_file_id+'.wav'

  # Download file from the given endpoint in Update response
  new_file = await context.bot.get_file(update.message.voice.file_id)
  response = requests.get(new_file.file_path)
  
  if response.status_code == 200:
    # Write the content of the response to a file
    with open('voice-'+voice_file_id+'.oga', 'wb') as audio_file:  # change the file name and extension as needed
        audio_file.write(response.content)
    print("Audio file downloaded successfully.")
  else:
    print(f"Failed to download the file. Status code: {response.status_code}")
  
  # Convert .OGA to .WAV for Speech Recognition
  convert_oga_to_wav('voice-'+voice_file_id+'.oga','voice-'+voice_file_id+'.wav')
  
  r = sr.Recognizer()  
  
  with sr.AudioFile('voice-'+voice_file_id+'.wav') as source:
    audio = r.record(source)
  
  try:
      sFinalResult = r.recognize_google(audio,language = 'en')
      print("Text: " + sFinalResult)
  except sr.UnknownValueError as e:
      print(f"Speech Recognition could not understand audio (UnknownValueError): {e}")
  except sr.RequestError as e:
      print(f"Could not request results from Google Web Speech API (RequestError): {e}")
  
  # Log users
  # print(f'Voice message ({update.message.chat.id}) in {message_type}: "file id {voice_file_id} and file size {voice_file_size} in bytes" ')
  
  response: str = 'Thank you for the audio. Here is the analysis based on it'
  
  analysis_json : json = an.get_completion(sFinalResult)
  print('Bot: ', response)
  await update.message.reply_text(response)
  await generate_and_send_pdf(analysis_json,voice_file_id,chat_id)
  await delete_file(voice_file_name)
  await delete_file(voice_file_name_wav)
  

def convert_oga_to_wav(input_oga_file, output_wav_file):
  try:
    # Command to convert OGA to WAV
    command = ['ffmpeg', '-i', input_oga_file, output_wav_file]
    
    # Execute the command
    subprocess.run(command, check=True)
  except subprocess.CalledProcessError as e:
    print(f"An error occurred during conversion: {e}")

async def generate_and_send_pdf(analysis_json : json,file_id,chat_id):
  # Sample JSON data
  # json_data = '{"name": "John", "age": 30, "city": "New York"}'
  data = json.loads(analysis_json)
  print(analysis_json)
  # Create a PDF file
  pdf_file = "output-"+file_id+".pdf"
  c = canvas.Canvas(pdf_file, pagesize=letter)
  width, height = letter

  # Add JSON data to PDF
  y_position = height - 40  # Start position for the first line
  for key, value in data.items():
      text = f"{key}: {value}"
      c.drawString(30, y_position, text)
      y_position -= 20

  # Save the PDF file
  c.save()
  print(f"PDF file '{pdf_file}' created.")  
  await send_pdf(TOKEN,chat_id,pdf_file)
  

async def send_pdf(bot_token, chat_id, pdf_file_path):
  url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
  with open(pdf_file_path, 'rb') as pdf_file:
      files = {'document': pdf_file}
      data = {'chat_id': chat_id}
      response = requests.post(url, files=files, data=data)

  if response.status_code == 200:
      print("File sent successfully.")
  else:
      print(f"Failed to send the file. Status code: {response.status_code}")
  await delete_file(pdf_file_path)

async def delete_file(file_path):
  # Check if file exists
  if os.path.exists(file_path):
      # Delete the file
      os.remove(file_path)
      print(f"File '{file_path}' has been deleted.")
  else:
      print(f"The file '{file_path}' does not exist.")

# Error handler
async def error(update: Update, context: ContextTypes):
    print(f'Update {update} caused error: {context.error}')


def main():
    print('Starting up bot...')
    start_version_server(VERSION_SERVER_HOST, VERSION_SERVER_PORT)
    print(f'Version endpoint running at http://{VERSION_SERVER_HOST}:{VERSION_SERVER_PORT}/version/v2')
    app = Application.builder().token(TOKEN).build()

    # Commands
    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help_command))
    app.add_handler(CommandHandler('custom', custom_command))

    # Messages
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.VOICE, handle_audio_message))

    # Errors
    app.add_error_handler(error)

    # Define a poll interval
    print('Polling...')
    app.run_polling(poll_interval=5)


if __name__ == '__main__':
    main()
