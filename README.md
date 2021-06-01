Run with interface:  
```
rasa actions
rasa run --credentials ./credentials.yml  --enable-api --auth-token XYZ123 --model ./models --endpoints ./endpoints.yml --cors "*"
run web_api.py
```

Run in cmd line:  
```
rasa run actions
rasa shell
```

Train the model:  
```
rasa train
```