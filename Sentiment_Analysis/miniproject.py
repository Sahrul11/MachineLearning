import tweepy
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from textblob.np_extractors import ConllExtractor

consumer_key= 'HrzPwaMKMHYoYmheWYONkT9lj'
consumer_secret= 'QFkYY54uDjYY1X5HMjYHDQbVXcbihNu8qydG9eR1AVJn7NQmcI'
access_token='860834802689298432-jwoqdMmw2ydXbm53AWMxRCM71jpajeK'
access_token_secret='vKhASPKJPp3J42HvPA5vHpFpYGY6iBLHyZTpk3HYd4T1c'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


print('Choose an option (1 or 2): ')
print('1. Choose a topic to search tweets for. ')
print('2. Choose a Twitter Username to search tweets for. ')
input_data = input()

if input_data=='1':
    print('Enter a topic: ')
    topic_name=input()
    new_tweets = api.search(q=topic_name)
    for tweet in new_tweets:
        analysis = TextBlob(tweet.text, analyzer=NaiveBayesAnalyzer(), np_extractor= ConllExtractor())
        polarity = 'Positive'
        if (analysis.sentiment.p_pos < 0.50):
            polarity = 'Negative'
        print ("Sentiment Analysis and Topic of Interest")
        print ("Tweet : ",tweet.text)
        print ("Sentiment:",polarity)
        print ("Confidence :  Positive score: " ,analysis.sentiment.p_pos*100, "  Negative score: ", analysis.sentiment.p_neg*100 )
        print ("Areas of interest: ", analysis.noun_phrases)
        print ("---------------------------------------------------------------------------")

else:
    print('2. Enter a Twitter Username to search tweets for: ')
    screen_name=input()
    new_tweets = api.user_timeline(screen_name =screen_name,count=20)
    for tweet in new_tweets:
        analysis = TextBlob(tweet.text, analyzer=NaiveBayesAnalyzer(), np_extractor= ConllExtractor())
        polarity = 'Positive'
        if (analysis.sentiment.p_pos < 0.50):
            polarity = 'Negative'
        print ("Sentiment Analysis and Topic of Interest")
        print ("Tweet : ",tweet.text)
        print ("Sentiment:",polarity)
        print ("Confidence :  Positive score: " ,analysis.sentiment.p_pos*100, "  Negative score: ", analysis.sentiment.p_neg*100 )
        print ("Areas of interest: ", analysis.noun_phrases)
        print ("---------------------------------------------------------------------------")

