import os
srcPath = os.path.abspath('./src')
jpgFile = os.path.join(srcPath,"tanya5.jpg")
detect = os.path.join(srcPath,r'yolov5/detect.py')
class model():
    def __init__(self):
        print('constructor')
        self.bla = 2
        self.classIndex = {'6': 15, '5': 12, '700': 20, '3': 6, '10': 1, '4': 9, '1': 0, '50': 13, '7': 18, '20': 4, '8': 21,
                      '40': 10, '100': 2, '2': 3, '500': 14, '200': 5, '600': 17, '70': 19, '60': 16, '90': 25,
                      '300': 8, '400': 11, '30': 7, '9': 24, '80': 22, '800': 23, '900': 26}
        self.weights = os.path.join(srcPath,r'weights/last.pt')
        self.treshold = 0.3

    def gimatrya(self,word: str) -> set:
        gimHash = dict()
        gimHash["א"] = "1"
        gimHash["ב"] = "2"
        gimHash["ג"] = "3"
        gimHash["ד"] = "4"
        gimHash["ה"] = "5"
        gimHash["ו"] = "6"
        gimHash["ז"] = "7"
        gimHash["ח"] = "8"
        gimHash["ט"] = "9"
        gimHash["י"] = "10"
        gimHash["כ"] = "20"
        gimHash["ל"] = "30"
        gimHash["מ"] = "40"
        gimHash["נ"] = "50"
        gimHash["ס"] = "60"
        gimHash["ע"] = "70"
        gimHash["פ"] = "80"
        gimHash["צ"] = "90"
        gimHash["ק"] = "100"
        gimHash["ר"] = "200"
        gimHash["ש"] = "300"
        gimHash["ת"] = "400"
        gimHash["ך"] = "500"
        gimHash["ם"] = "600"
        gimHash["ן"] = "700"
        gimHash["ף"] = "800"
        gimHash["ץ"] = "900"
        chars = list(word)
        return {gimHash[x] for x in chars}


    # format chars to specific model classes
    def classesbyWord(self,word: str, classes: dict):
        gimclasses = self.gimatrya(word)
        return set([str(self.classIndex[x]) for x in gimclasses])


    def detectWord(self,word='ישראל',*, source = jpgFile):
        print('detectWord')
        #reset memory and detect:
        myclasses = " ".join(self.classesbyWord(word,self.classIndex))
        command = f'\
        python3 {detect}\
        --save-txt\
        --conf-thres {self.treshold}\
        --line-thickness 3\
        --source {source}\
        --img 2000\
        --weights {self.weights}\
        --classes {myclasses}\
        --word {word}'
        r = os.system(command)
        print(r)
        return r