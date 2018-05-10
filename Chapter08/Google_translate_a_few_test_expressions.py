#Google Translate
#Built with Google Translation tools
#Copyright 2018 Denis Rothman MIT License. See LICENSE.

from googleapiclient.discovery import build
import html.parser as htmlparser
parser = htmlparser.HTMLParser()

def g_translate(source,targetl):
    service = build('translate', 'v2',developerKey='Your Google API KEY')
    request = service.translations().list(q=source, target=targetl)
    response = request.execute()
    return response['translations'][0]['translatedText']

source='Hello. My name is Usty!'
targetl="fr"  
result = g_translate(source,targetl)
print("result:",parser.unescape(result))


source='The weather is nice today'
targetl="fr"  
result = g_translate(source,targetl)
print("result:",parser.unescape(result))


source='Ce professor me cherche des poux.'
targetl="en"  
result = g_translate(source,targetl)
print("result:",parser.unescape(result))

source='chercher des poux'
targetl="en"  
result = g_translate(source,targetl)
print("result:",parser.unescape(result))
  
source='Chercher des limites est intéressant.'
targetl="en"  
result = g_translate(source,targetl)
print("result:",parser.unescape(result))

source='Une SAS ne dispense pas de suivre les recommandations en vigueur autour des pratiques commerciales.'
targetl="en"  
result = g_translate(source,targetl)
print("result:",parser.unescape(result))

source='The project team is all ears.'
targetl="fr"  
result = g_translate(source,targetl)
print("result:",parser.unescape(result))

source='The coach stopped and everybody was complaning.'
targetl="fr"  
result = g_translate(source,targetl)
print("result:",parser.unescape(result))

source='The coach broke down and stopped and everybody was complaning.'
targetl="fr"  
result = g_translate(source,targetl)
print("result:",parser.unescape(result))




