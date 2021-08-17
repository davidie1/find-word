from django.shortcuts import render
from django.http import HttpResponse
from src.model.model import model
import io
from PIL import Image
from django.views.decorators.csrf import csrf_exempt
from imageio import imread
from io import BytesIO
import base64
import json
import os
srcPath = os.path.abspath('./src')
@csrf_exempt
def detectWord(request):
    try:
        if request.method == 'POST':
            print('got post request for detectWord')

            image = request.FILES['imageFile'].read()
            imageFile = os.path.join(srcPath, r'image.jpg')
            print(imageFile)
            a = open(imageFile,'wb')
            a.write(image)
            a.close()
            word = request.POST.get("word")
            detector = model()
            detector.detectWord(word=word,source = imageFile)
            return HttpResponse(1)
        else:
            return HttpResponse('only post request allowed')
    except Exception as error:
        print(error)
        return HttpResponse(error)

@csrf_exempt
def home(request):
    print('got post request for home')
    print((request.get_host()))
    homeHttpFile = os.path.join(srcPath,r'web/index.html')
    httpRespon = open(homeHttpFile).read().replace('detectWordFunction',request.path)
    return HttpResponse(httpRespon)
# Create your views here.
@csrf_exempt
def show(request):
    imageFile = os.path.join(srcPath, r'resaultImage.jpg')
    a = open(imageFile, 'rb').read()
    return HttpResponse(a,"image/jpg")
# Create your views here.

