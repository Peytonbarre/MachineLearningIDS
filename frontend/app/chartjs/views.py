from django.shortcuts import render 
from django.views.generic import View 
from django.http import JsonResponse
   
from rest_framework.views import APIView 
from rest_framework.response import Response 

import requests
   
class HomeView(View): 
    def get(self, request, *args, **kwargs): 
        return render(request, 'chartjs/index.html') 
   
class ChartData(APIView): 
    authentication_classes = [] 
    permission_classes = [] 
   
    def get(self, request, format = None): 
        labels = [ 
            'January', 
            'February',  
            'March',  
            'April',  
            'May',  
            'June',  
            'July'
            ] 
        chartLabel = "my data"
        chartdata = [0, 10, 5, 2, 20, 30, 45] 
        data ={ 
                     "labels":labels, 
                     "chartLabel":chartLabel, 
                     "chartdata":chartdata, 
             } 
        return Response(data) 
    
def get_flask_data(request):
    flask_api_url = "http://127.0.0.1:5000/sendData"
    
    try:
        response = requests.get(flask_api_url)
        response.raise_for_status()  # Check for HTTP errors
        data_from_flask = response.json()
        return JsonResponse(data_from_flask)
    except requests.exceptions.RequestException as e:
        return JsonResponse({'error': str(e)}, status=500)