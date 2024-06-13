import logging
import time
import whisper
import spacy
from transformers.pipelines import pipeline
import librosa
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from telegram.ext import filters
from telegram.ext import Updater, CallbackContext
from telegram import Update
from sents import fillers


def get_duration(file_path):
    audio_data, sample_rate = librosa.load(file_path)
    duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    return duration

def get_report(duration, fillers_count, words_count, res, emotions):
    text = 'Отчет о самопрезентации 📄:\n\n'
    overall_score = 0

    duration_text = '<b>Длительность самопрезетации: '
    if duration < 60:
        duration_text += 'слишком короткая 😔</b>\nПостарайся в следующий раз более ' \
        'развернуто рассказать о себе. ' \
        'Если самопрезентация слишком короткая, есть риск что слушатели не узнают ' \
        'о твоих положительных качествах, опыте и навыках. ' \
        'Хорошая самопрезетация длится ' \
        '1-3 минуты.\n' \
        'В следующий раз всё получится 😺'

    if duration > 60*3:
        duration_text += 'слишком длинная 😔</b>\nПостарайся в следующий раз более ' \
        'четко рассказать о себе, без воды и по делу. ' \
        'Если слишком долго рассказывать о себе, внимание слушателей снизится ' \
        'и общее впечатление о тебе смажется. ' \
        'Хорошая самопрезетация длится ' \
        '1-3 минуты.\n' \
        'В следующий раз всё получится 😺'
    
    if 60 <= duration <= 60*3:
        duration_text += 'отлично 😃</b>\nСамопрезентация уложилась в диапазон 1-3 минуты. ' \
        'Все четко и по делу, прекрасная работа!'
        overall_score += 1

    fillerc_coef = round((fillers_count / words_count) * 100, 0)
    fillers_text = '<b>Слова-паразиты: ' + str(fillerc_coef) + '% от всей самопрезентации</b>\n'
    
    if fillerc_coef >= 15:
        fillers_text += '<b>Слов паразитов слишком много 😔</b>\n' \
        'Но не стоит переживать, это поправимо 😃\n' \
        'Запиши текст самопрезентации и читай с листа, ' \
        'постепенно всё реже и реже поглядывай в лист и '\
        'вскоре от слов-паразитов не останется и следа!\n'
    else:
        fillers_text += '<b>Слов паразитов очень мало 😃</b>\n' \
        'Это отличный результат! Продолжай в том же духе!\n'
        overall_score += 1

    if fillers_count > 0:
        fillers_text += 'Найденные слова паразиты: <i>'
        fillers_list = []
        for key, value in res.items():
            if value > 0:
                fillers_list.append(key)
        fillers_text += ', '.join(fillers_list) + '</i>'

    emotions_text = '<b>Эмоциональный окрас речи самопрезентации: '

    max_emotion_label = None
    max_emotion_value = 0
    for emotion in emotions:
        if emotion['score'] > max_emotion_value:
            max_emotion_value = emotion['score']
            max_emotion_label = emotion['label']
    
    if max_emotion_label == 'neutral':
        emotions_text += 'нейтральный 🙂</b>\n' \
        'Это хороший результат! Чтобы ещё улучшить самопрезентацию, добавь немного акцентов и интонаций в свою речь 😃'
        overall_score += 1

    if max_emotion_label == 'positive':
        emotions_text += 'позитивный 😃</b>\n' \
        'Это отличный результат! Люди тянутся к таким людям как ты!'
        overall_score += 1

    if max_emotion_label != 'neutral' and max_emotion_label != 'positive':
        emotions_text += 'ближе к негативу 😔</b>\n' \
        'Возможно, просто день не задался или алгоритм глючит.\n' \
        'Не переживай! Вздохни полной грудью, помни что у тебя все получится и попробуй ещё раз 😤'

    overall_text = f'<b>Общая оценка самопрезетации: {overall_score} из 3</b>\n'
    if  overall_score <= 1:
        overall_text += 'Не переживай! У тебя все получится, главное продолжай пытаться и верь в себя 😤'
    else:
         overall_text += 'Отличный результат! Продолжай в том же духе, ты всё сможешь 😺'

    text += duration_text + '\n\n' + fillers_text + '\n\n' + emotions_text + '\n\n' + overall_text 
    return text

async def process_voice(update, file_arr, file_name):
    result = whisper_model.transcribe(file_name)

    res = {}
    fillers_count = 0
    data = result["text"].lower()
    
    for word in fillers:
        match_count = data.count(word)
        res[word] = match_count
        fillers_count += match_count

    text = nlp(result["text"])
    words_count = 0

    for token in text:
        if not token.is_punct and \
        not token.is_space and \
        not token.is_bracket and \
        not token.is_quote:
            words_count += 1
    
    duration = get_duration(file_name)

    emotions = pipe(file_name)

    report_text = get_report(duration, fillers_count, words_count, res, emotions)

    await update.message.reply_text(report_text, parse_mode='HTML')

async def voice(update: Update, context: CallbackContext) -> None:
    message = update.effective_message
    chat_id = update.effective_message.chat.id
    file_name = 'voices/%s_%s%s.ogg' % (chat_id, update.message.from_user.id, update.message.message_id)

    logger.info(file_name)

    file = await context.bot.get_file(message.voice.file_id)
    
    file_arr = await file.download_to_drive(file_name)
    
    await update.message.reply_text('Обрабатываю, ожидай 😎\nПримерное время ожидания - 10 минут ⏱')
   
    await process_voice(update, file_arr, file_name)

async def hello(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Привет 🤗\n\nЯ помогу тебе тренировать самопрезентацию 😤\nЗапиши голосовуху с самопрезентацией и получи отчет 📄')


logging.basicConfig(filename='bot.log', level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = 'token_here'

# ML INIT
whisper_model = whisper.load_model("medium")
nlp = spacy.load("ru_core_news_lg")
pipe = pipeline(model="KELONMYOSA/wav2vec2-xls-r-300m-emotion-ru", trust_remote_code=True)

app = ApplicationBuilder().token(TOKEN).build()

voice_handler = MessageHandler(filters.VOICE, voice)

app.add_handler(CommandHandler("start", hello))
app.add_handler(voice_handler)

app.run_polling()