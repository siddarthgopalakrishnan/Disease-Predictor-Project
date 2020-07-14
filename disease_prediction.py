############################################ IMPORTING LIBRARIES ############################################

from tkinter import *
import numpy as np
import pandas as pd

######################################### LIST OF SYMPTOMS IN DATA ##########################################

symptom_list = ['itching','skin_rash','nodal_skin_eruptions','back_pain','constipation','abdominal_pain','anxiety',
'continuous_sneezing','shivering','chills','joint_pain','stomach_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','acidity','ulcers_on_tongue','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation','weight_gain',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs','fatigue',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool','spotting_ urination',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs','cold_hands_and_feets',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','burning_micturition','nausea',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','weight_loss',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','restlessness',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness','muscle_wasting','vomiting','cough',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine','mood_swings','sweating',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','lethargy','sunken_eyes',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','high_fever',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','dehydration',
'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances','breathlessness',
'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding','patches_in_throat',
'distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum','indigestion',
'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring',
'skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','headache',
'red_sore_around_nose','yellow_crust_ooze','irregular_sugar_level','yellowish_skin','dark_urine',
'loss_of_appetite','pain_behind_the_eyes']

########################################## LIST OF DISEASES IN DATA #########################################

disease = ['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue',
'Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)','Heartattack','Varicoseveins','Hypothyroidism',
'Hyperthyroidism','Hypoglycemia','Osteoarthristis','Arthritis','(vertigo) Paroymsal  Positional Vertigo',
'Acne','Urinary tract infection','Psoriasis','Impetigo']

######################### FLAG LIST WHERE ELEMENT IS SET TO 1 FOR RESPECTIVE SYMPTOM ########################

flag_list = []
for x in range(0, len(symptom_list)):
	flag_list.append(0)

############################################### TRAINING DATA ###############################################

df = pd.read_csv("C:/Users/ASUS/Downloads/DC++ Downloads/Maxgen-Intern/Project/Training.csv")
df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,'Migraine':11,
'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,
'Typhoid':18,'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,
'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,
'Hypothyroidism':31,'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,'Impetigo':40}}, inplace = True)

X = df[symptom_list]
y = df[["prognosis"]]
np.ravel(y)

################################################ TEST DATA ##################################################

tr = pd.read_csv("C:/Users/ASUS/Downloads/DC++ Downloads/Maxgen-Intern/Project/Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,'Migraine':11,
'Cervical spondylosis':12,'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,
'Typhoid':18,'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,
'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,
'Hypothyroidism':31,'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,'Impetigo':40}}, inplace = True)

X_test = tr[symptom_list]
y_test = tr[["prognosis"]]
np.ravel(y_test)

############################################### DECISION TREE ###############################################

def DecisionTree():
	from sklearn import tree
	clf3 = tree.DecisionTreeClassifier()
	clf3 = clf3.fit(X, np.ravel(y))

	# CALCULATING ACCURACY
	from sklearn.metrics import accuracy_score
	y_pred = clf3.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	print(accuracy_score(y_test, y_pred, normalize = False))

	patient_symptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

	for k in range(0, len(symptom_list)):
		for z in patient_symptoms:
			if z == symptom_list[k]:
				flag_list[k]=1

	inputtest = [flag_list]
	predict = clf3.predict(inputtest)
	predicted = predict[0]

	flag = 'no'
	for dis in range(0, len(disease)):
		if(predicted == dis):
			flag = 'yes'
			break

	if flag == 'yes':
		t1.delete("1.0", END)
		t1.insert(END, disease[dis])
	else:
		t1.delete("1.0", END)
		t1.insert(END, "Not Found")

############################################### RANDOM FOREST ###############################################

def RandomForest():
	from sklearn.ensemble import RandomForestClassifier
	clf4 = RandomForestClassifier()
	clf4 = clf4.fit(X, np.ravel(y))

	# CALCULATING ACCURACY
	from sklearn.metrics import accuracy_score
	y_pred = clf4.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	print(accuracy_score(y_test, y_pred, normalize = False))

	patient_symptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

	for k in range(0,len(symptom_list)):
		for z in patient_symptoms:
			if z == symptom_list[k]:
				flag_list[k]=1

	inputtest = [flag_list]
	predict = clf4.predict(inputtest)
	predicted = predict[0]

	flag = 'no'
	for dis in range(0, len(disease)):
		if predicted == dis:
			flag = 'yes'
			break

	if flag == 'yes':
		t2.delete("1.0", END)
		t2.insert(END, disease[dis])
	else:
		t2.delete("1.0", END)
		t2.insert(END, "Not Found")

############################################### NAIVE BAYES ###############################################

def NaiveBayes():
	from sklearn.naive_bayes import GaussianNB
	gnb = GaussianNB()
	gnb = gnb.fit(X,np.ravel(y))

	# CALCULATING ACCURACY
	from sklearn.metrics import accuracy_score
	y_pred = gnb.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	print(accuracy_score(y_test, y_pred,normalize = False))

	patient_symptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
	for k in range(0, len(symptom_list)):
		for z in patient_symptoms:
			if z == symptom_list[k]:
				flag_list[k] = 1

	inputtest = [flag_list]
	predict = gnb.predict(inputtest)
	predicted = predict[0]

	flag = 'no'
	for dis in range(0,len(disease)):
		if predicted == dis:
			flag = 'yes'
			break

	if flag == 'yes':
		t3.delete("1.0", END)
		t3.insert(END, disease[dis])
	else:
		t3.delete("1.0", END)
		t3.insert(END, "Not Found")

############################################### TKINTER GUI ###############################################

root = Tk()
root.title('Disease Predictor')
root.configure(background = 'deep sky blue')

# ENTRY VARIABLES
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

# HEADING
heading_label = Label(root, text = "Disease Predictor", fg = "white", bg = "deep sky blue")
heading_label.config(font = ("comic sans ms", 25))
heading_label.grid(row = 1, column = 1)

# LABELS
name_label = Label(root, text = "Patient Name", fg = "yellow", bg = "black")
name_label.grid(row = 6, column = 0, pady = 15, sticky = W)
S1_label = Label(root, text = "Symptom 1", fg = "yellow", bg = "black")
S1_label.grid(row = 7, column = 0, pady = 10, sticky = W)
S2_label = Label(root, text = "Symptom 2", fg = "yellow", bg = "black")
S2_label.grid(row = 8, column = 0, pady = 10, sticky = W)
S3_label = Label(root, text = "Symptom 3", fg = "yellow", bg = "black")
S3_label.grid(row = 9, column = 0, pady = 10, sticky = W)
S4_label = Label(root, text = "Symptom 4", fg = "yellow", bg = "black")
S4_label.grid(row = 10, column = 0, pady = 10, sticky = W)
S5_label = Label(root, text = "Symptom 5", fg = "yellow", bg = "black")
S5_label.grid(row = 11, column = 0, pady = 10, sticky = W)

decisionTree_label = Label(root, text = "Decision Tree Result", fg = "white", bg = "green")
decisionTree_label.grid(row = 15, column = 0, pady = 10,sticky = W)
randomForest_label = Label(root, text="Random Forest Result", fg = "white", bg = "green")
randomForest_label.grid(row = 17, column = 0, pady = 10,sticky = W)
naiveBayes_label = Label(root, text="Naive Bayes Result", fg = "white", bg = "green")
naiveBayes_label.grid(row = 19, column = 0, pady = 10,sticky = W)

# ENTRIES
OPTIONS = sorted(symptom_list)
Name_entry = Entry(root, textvariable=Name, width=40)
Name_entry.grid(row = 6, column = 1)
S1_entry = OptionMenu(root, Symptom1, *OPTIONS)
S1_entry.grid(row = 7, column = 1)
S2_entry = OptionMenu(root, Symptom2, *OPTIONS)
S2_entry.grid(row = 8, column = 1)
S3_entry = OptionMenu(root, Symptom3, *OPTIONS)
S3_entry.grid(row = 9, column = 1)
S4_entry = OptionMenu(root, Symptom4, *OPTIONS)
S4_entry.grid(row = 10, column = 1)
S5_entry = OptionMenu(root, Symptom5, *OPTIONS)
S5_entry.grid(row = 11, column = 1)

decisionTree_entry = Button(root, text = "Decision Tree", command = DecisionTree, bg = "brown1", fg = "black")
decisionTree_entry.grid(row = 8, column = 3, padx = 10)
randomForest_entry = Button(root, text = "Random forest", command = RandomForest, bg = "brown1", fg = "black")
randomForest_entry.grid(row = 9, column = 3, padx = 10)
naiveBayes_entry = Button(root, text = "Naive Bayes", command = NaiveBayes, bg = "brown1", fg = "black")
naiveBayes_entry.grid(row = 10, column = 3, padx = 10)

# TEXT FIELDS
t1 = Text(root, height = 1, width = 30, bg = "orange", fg = "black")
t1.grid(row = 15, column = 1)
t2 = Text(root, height = 1, width = 30, bg = "orange", fg = "black")
t2.grid(row = 17, column = 1)
t3 = Text(root, height = 1, width = 30, bg = "orange", fg = "black")
t3.grid(row = 19, column = 1)

root.mainloop()