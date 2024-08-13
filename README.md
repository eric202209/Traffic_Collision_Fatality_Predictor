# Traffic_Collision_Fatality_Predictor

Steps

1. Git clone from github
2. Run "pip install -r requirements.txt"
3. Setup your vitual environment. See https://code.visualstudio.com/docs/python/environments
4. Run "python run.py"
5. Click on http://127.0.0.1:5000 or http://localhost:5000

Bluk Prediction format:
[
{
"ROAD_CLASS": 1,
"VISIBILITY": 1,
"LIGHT": 0,
"PEDESTRIANcounter": 1,
"DISTRICT": 0,
"ACCLOC": 2,
"TRAFFCTL": 1,
"RDSFCOND": 1
},
{
"ROAD_CLASS": 4,
"VISIBILITY": 3,
"LIGHT": 3,
"PEDESTRIANcounter": 10,
"DISTRICT": 2,
"ACCLOC": 4,
"TRAFFCTL": 0,
"RDSFCOND": 3
}
]
