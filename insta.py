import base64
import instaloader
import argparse
import sys
import time
import datetime
from itertools import dropwhile, takewhile
import json
import requests
import wget
import shutil
from model import make_img

def get_url(path):
    key_imgbb='6d45f0a683ee9271fd459ccdcf03767e'
    with open(path, "rb") as file:
        url = "https://api.imgbb.com/1/upload"
        payload = {
            "key": key_imgbb,
            "image": base64.b64encode(file.read()),
        }
        res = requests.post(url, payload)
    result=json.loads(res.text)
    print(result)
    if result['success']==True:
        myurl=result['data']['url']
        return myurl
    else:
        print('Faild')
        
def text_parser(text):
    myanimals= 'краб скумбрия барракуда удильщик осьминог мидии рыба-клоун хирург крылатка дракон дракон звезда еж угорь дракон окунь камбала медуза креветка ангел большерот медуза групер акула-призрак червь саблезуб червь хвостокол спинорог зубатка губка акула-молот форель кальмар ушки гребешок тунец треска уклейка сардина рыба-меч пикша осетр налим линь лакедра лангуст черепаха плоскохвост ламантин сом-касатка дельфин голотурия таракан ёрш нетопырь офиуры конек паук луна'.lower()
    myanimals = myanimals.split()
    text=text.lower()
    if any(word in text for word in myanimals):
        return [number+1 for number, animal in enumerate(myanimals) if animal in text], [animal for animal in myanimals if animal in text]
    else:
        return [],[]
      
def preparing_samples(classes,dirallsamples):
   
    num=1
    #dirallsamples= '/content/allanimals/'
    dirsamples= '/data/'
    if os.path.exists(dirsamples):
                     shutil.rmtree(dirsamples)
    if len(classes)>0:
     for cl in classes:
        if num<6:
            shutil.copytree(dirallsamples + str(cl), dirsamples + str(num))
            num+=1
        else: return True
    for i in range(num,6):
        shutil.copytree(dirallsamples + str(i), dirsamples + str(i))
    print("sampes prepared")
    return True
  
  
L = instaloader.Instaloader()
L.login('fish.poster', 'loveisfish')


insta_business_id='17841447575001964'
access_token='EAACfdmd7010BADkeJ8eEqc5OeYdocjxwZBZCy8tI43SEHMcWNOSa00s1BaT0oXjNzv2jsAPOn8qWw0HZCAQ8J6wgxdUShlNZCR9yZBqrrVLZBv8qhTjrA2QHH9UhmmZA7gQyPpQuZBIzvpK1XZBi3Ldd114hvkm7OjXEqT2kgUhvZC6vgeiBz7jBOm' 



def insta_post_image(image,mention):
    post_url='https://graph.facebook.com/{}/media'.format(insta_business_id)

    payload={
        'image_url':image,
        'caption':'@'+mention,
        'access_token': access_token
            }

    r=requests.post(post_url,data=payload)
    time.sleep(2)
    result=json.loads(r.text)
    print(result)

    if 'id' in result:
        creation_id=result['id']
    else:
        return 'failed'

    second_post_url='https://graph.facebook.com/{}/media_publish'.format(insta_business_id)
    second_payload={
        'creation_id': creation_id,
        'access_token': access_token
        }
    r=requests.post(second_post_url,data=second_payload)
    print(r.text)
    return 'posted'
def tags_cheker(tags,SINCE, UNTIL, dirallsamples):
    for tag in tags:
        HASHTAG=tag
        post_iterator = instaloader.NodeIterator(
            L.context, "9b498c08113f1e09617a1703c22b2f32",
            lambda d: d['data']['hashtag']['edge_hashtag_to_media'],
            lambda n: instaloader.Post(L.context, n),
            {'tag_name': HASHTAG},
            f"https://www.instagram.com/explore/tags/{HASHTAG}/"
        )
        #for post in post_iterator:
        for post in takewhile(lambda p: p.date > UNTIL, dropwhile(lambda p: p.date > SINCE, post_iterator)):
                print(post.date)
                print(post.url)
                print(post.caption)
                print(post.owner_profile.username)

                classes, animals = text_parser(post.caption)
                preparing_samples(classes,dirallsamples)

                prof_name=post.owner_profile.username
                if os.path.exists("/content/temp.jpg"):
                    os.remove("/content/temp.jpg")
                wget.download(post.url,out='/content/temp.jpg')

                myimg = make_img(options,'/content/temp.jpg')
                #
                print('gettig url')
                myurl=get_url(myimg)
                print('post into inst')
                res=insta_post_image(myurl,prof_name+' '+''.join(animals[0]))
                print(res)
                if res == 'posted':
                    time.sleep(60)
  

#TODO update access token

if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument ('--min',default='6')
     parser.add_argument ('--tag',default='biogeohub')
     parser.add_argument ('--dirallsamples',default='')
     namespace = parser.parse_args(sys.argv[1:])
     #datetime.datetime.now()
   # d =  datetime.date.today()+ datetime.timedelta(days=10) 
    mins=int(namespace.min)
    tags= namespace.tag.split()
    SINCE=datetime.datetime.now()
    UNTIL = datetime.datetime.now()-datetime.timedelta(seconds=mins*60)
    while True:
        tags_cheker(tags,SINCE,UNTIL,dirallsamples)
        print('Next check after',mins)
        time.sleep(mins*60)
        SINCE=datetime.datetime.now()
        UNTIL = datetime.datetime.now()-datetime.timedelta(seconds=mins*60)
