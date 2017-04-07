#Watson call and toneanzalizer 
import json
from watson_developer_cloud import ToneAnalyzerV3


#Generated in BlueMix Platform 
tone_analyzer = ToneAnalyzerV3(
    username='',
    password='',
    version='2016-02-11')

print(json.dumps(tone_analyzer.tone(text='Put the sentence to analyze here'), indent=2))

