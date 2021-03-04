from textblob import TextBlob
from pyspark import SparkConf, SparkContext
import re



def abb_en(line):
   abbreviation_en = {
    'u': 'you',
    'thr': 'there',
    'asap': 'as soon as possible',
    'lv' : 'love',    
    'c' : 'see'
   }
   
   abbrev = ' '.join (abbreviation_en.get(word, word) for word in line.split())
   return (abbrev)

def remove_features(data_str):
   
    url_re = re.compile(r'https?://(www.)?\w+\.\w+(/\w+)*/?')    
    mention_re = re.compile(r'@|#(\w+)')  
    RT_re = re.compile(r'RT(\s+)')
    num_re = re.compile(r'(\d+)')
    
    data_str = str(data_str)
    data_str = RT_re.sub(' ', data_str)  
    data_str = data_str.lower()  
    data_str = url_re.sub(' ', data_str)   
    data_str = mention_re.sub(' ', data_str)  
    data_str = num_re.sub(' ', data_str)
    return data_str


def sentiment_classification(x):
    if x==0:
        output = "neu"
    elif x>0:
        output = "+ve"
    else:
        output = "-ve"
        
    return output
   
  
   
#Write your main function here
def main(sc,filename):
    
    #maintable
    myrdd = sc.textFile(filename).map(lambda x:x.split(',')).filter(lambda x:len(x)==8).filter(lambda x:len(x[0])>1)
    
    #prepare tweet for sentiment analysis
    myrdd1 = myrdd.map(lambda x:x[4]).map(lambda x:remove_features(x)).map(lambda x:x.lower()).map(lambda x:abb_en(x))
    
    #sentiment analysis
    myrdd2 = myrdd1.map(lambda x:TextBlob(x)).map(lambda x:x.sentiment.polarity).map(lambda x:sentiment_classification(x))
    
    #sequence 1 
    mytable1 = myrdd.map(lambda x:x[0]+','+x[4]+','+x[2]+','+x[1])
    
    #sequence 2
    mytable2 = myrdd.map(lambda x:x[3]+','+x[5]+','+x[6]+','+x[7])
    
    #zip table 1
    mytable3 = mytable1.zip(myrdd2).map(lambda x:','.join(x))
    
    #zip table 2
    mytable4 = mytable3.zip(mytable2).map(lambda x:','.join(x))
    
    #remove quote
    mytablefinal = mytable4.map(lambda x:x.replace("'","")).map(lambda x:x.replace('"',''))
    
    #save file
    mytablefinal.saveAsTextFile('ADE_Haikal3')
    
    for x in mytablefinal.take(10):
        print(x)
   
   

  
   

if __name__ == "__main__":
    
    conf = SparkConf().setMaster("local[1]").setAppName("My Spark Application")
    sc = SparkContext(conf=conf)


    filename = 'starbucks_v1.csv'
    main(sc,filename)
 
    sc.stop()
