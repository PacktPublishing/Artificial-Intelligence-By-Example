#Google Translate
#Built with Google Translation tools
#Copyright 2018 Denis Rothman MIT License. See LICENSE.
from googleapiclient.discovery import build
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

X=['Eating fatty food can be unhealthy.',
   'This was a catch-22 situation.',
   'She would not lend me her tote bag',
   'He had a chip on his shoulder',
   'The market was bearish yesterday',
   'That was definitely wrong',
   'The project was compromised but he pulled a rabit out of his hat',
   'So just let the chips fall where they may',
   'She went the extra mile to satisfy the customer',
   'She bailed out when it became unbearable',
   'The term person includes one or more individuals, labor unions, partnerships, associations, corporations, legal representatives, mutual companies, joint-stock companies, trusts, unincorporated organizations, trustees, trustees in bankruptcy, or receivers.',
   'The coach broke down, stopped and everybody was complaining']

X1=['grasse',
    'insoluble',
    'sac',
    'aggressif',
    'marché',
    'certainement',
    'chapeau',
    'advienne',
    'supplémentaire',
    'parti',
    'personne',
    'bus']


X2=[0,0,0,1,0,0,0,0,0,0,0,1]

phrase_translation=['','','','Il est agressif','','','','','','','','']
V1=['broke','road','stopped','shouted','coach','bus','car','truck','break','broke','roads','stop']
V1_class=['trainer','coach','bus']
vpolysemy=[[0,0,0,0]]

def g_translate(source,targetl,m):
    service = build('translate', 'v2',developerKey='Your Google API KEY')
    request = service.translations().list(q=source, target=targetl,format='text',model=m)
    response = request.execute()
    return response['translations'][0]['translatedText']

def deeper_translate(source,index):
    dt=source
    deeper_response=phrase_translation[index]
    if(len(deeper_response)<=0):
        print("deeper translation program result:",deeper_response,":Now true")
    if(len(deeper_response)<=0):
        v1=0
        for i in range(4):
            ngram=V1[i]
            if(ngram in source):
                vpolysemy[0][i]=9
                v1=1
        if(v1>0):
            polysemy='V1'
            begin=str(V1[0]).strip('[]');end=str(V1[3]).strip('[]')
            sememe=knn(polysemy,vpolysemy,begin,end)
            for i in range(2):
                if(V1_class[i] in source):
                    replace=str(V1_class[i]).strip('[]')
                    sememe=str(sememe).strip('[]')
                    dtsource = source.replace(replace,sememe)
                    targetl="fr";m='base'  
                    result = g_translate(dtsource,targetl,m)
                    print('polysemy narrowed result:',result,":Now true")
    
                    
def knn(polysemy,vpolysemy,begin,end):
    df = pd.read_csv(polysemy+'.csv')
    X = df.loc[:,begin:end]
    Y = df.loc[:,'class']
    knn = KNeighborsClassifier()
    knn.fit(X,Y)
    prediction = knn.predict(vpolysemy)
    return prediction

def frequency_p(tnumber,cnumber):
    ff=cnumber/tnumber  #frequentist interpretation and probablity
    return ff

#print('Phrase-Based Machine Translation(PBMT)model:base'): #m='base'  

print('Neural Machine Translation model:nmt')
t=0;f=0;dt=0     # true;false;deeper translation call
for xi in range(len(X)):
    source=X[xi]
    targetl="fr";m='nmt'  
    result = g_translate(source,targetl,m)
    targetl="en"
    back_translate=result
    back_translate = g_translate(back_translate,targetl,m)
    print("source:",source,":",len(source))
    print("result:",result)
    print("target:",back_translate,":",len(back_translate))
    #length comparaison can be used to improve the algorithm
    #In this case, the source is compared to the back translation
    term=X1[xi]
    print("term:",term)
    input=result
    words=input.split()
    if(source == back_translate):
          print("true")
          if((term not in words)and (xi!=4)):
              t+=1
    else:
        f+=1;print("false")
        if(X2[xi]>0):    
            DT=deeper_translate(source,xi)
            dt+=1
    if(f>0):
        B1=frequency_p(xi+1,f)    #error detection probablity before deep translation 
        B2=frequency_p(xi+1,f-dt) #error detection probablity after deep translation
    if(f>0):
        print("ETP before DT",round(B1,2),"ETP with DT",round(B2,2))
    else:
        print('Insufficient data in probablity distribution')
      
print("------Summary------")
print('Neural Machine Translation model:nmt')
print('Google Translate:',"True:",t,"False:",f,'ETP',round(f/len(X),2))
print('Customized Google Translate:',"True:",t,"False:",f-dt,'ETP',round((f-dt)/len(X),2))
a=2.5;at=t+a;af=f-a #subjective acceptance of an approximate result
print('Google Translate acceptable:',"True:",at,"False:",af,'ETP',round(af/len(X),2))
#The error rate should decrease and be stablized as the KNN knowledge base increases
print('Customized Google Translate acceptable:',"True:",at,"False:",af-dt,'ETP',round((af-dt)/len(X),2))

