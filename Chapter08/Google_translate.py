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

source='Google Translate is great!'
targetl="fr"  
result = g_translate(source,targetl)
print("result:",parser.unescape(result))
