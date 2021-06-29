Run in cmd line:  
```
rasa run actions
rasa shell
```

Train the model:  
```
rasa train
```

Run witn confidence of prediction:
```
rasa shell nlu
```

Run with interface:  
- run the following commands:
```
rasa run actions
rasa run --credentials ./credentials.yml  --enable-api --auth-token XYZ123 --model ./models --endpoints ./endpoints.yml --cors "*"
```
- run chatbot_apis/run_application_APIs.py  
- run web_app.py  
- acess it at http://127.0.0.1:8000/ 