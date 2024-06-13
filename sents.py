import io
import spacy


fillers = set([
    "знаешь",
    "знаете",
    "блин",
    "в натуре",
    "практически",
    "фактически",
    "как его",
    "ладно",
    "как сказать",
    "всё такое",
    "все такое",
    "допустим",
    "в самом деле",
    "значит",
    "как-то так",
    "походу",
    "типа",
    "как бы",
    "итак",
    "короче",
    "таки",
    "ну",
    "вот",
    "честно говоря",
    "грубо говоря",
    "мягко говоря",
    "собственно говоря",
    "на самом деле",
    "в общем",
    "в целом",
    "прикинь",
    "это самое",
    "например",
    "допустим",
    "как говорится",
    "в принципе",
    "так сказать",
    "прямо",
    "э-э",
    "кстати",
    "слушай",
    "понимаешь",
    "так вот",
    "ну-у",
    "ну",
    "конкретно",
    "а-а",
    "м-м",
    "и-и",
    "то есть",
    "вообще"
])

# nlp = spacy.load("ru_core_news_lg")

# data = None
# with io.open('artem_kazimir.txt','r', encoding='utf8') as f:
#     data = f.read()

# doc = nlp(data)

# # with io.open('./result.txt','w', encoding='utf8') as f:
# #     for sent in doc.sents:
# #         f.write(str(sent) + '\n')

# res = {}
# data = data.lower()
# for w in p:
#    res[w] = data.count(w)

# print(res)
nlp = spacy.load("ru_core_news_lg")

text = nlp('Ну типа привет как дела вообще ну как бы короче не знаю')#nlp(result["text"])
words_count = 0
for token in text:
    if not token.is_punct and \
       not token.is_space and \
       not token.is_bracket and \
       not token.is_quote:
        words_count += 1

print(words_count)
