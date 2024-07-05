import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import plotly.graph_objects as go


df_cardio = pd.read_csv("cardio_train.csv", sep=",", index_col=0)

print(df_cardio.head())
print(df_cardio.info())
print(df_cardio.describe())

#ver quantos são cardiacos e quantos não

print(df_cardio["cardio"].value_counts())

#media da idade: 19468 / 365 = 53.33 anos
#com um desvio padrão de 2467.25 / 365 = 6.7 anos
#ou seja, são pacientes na metade da vida

#para saber se tem alguma variavel faltando podemos usar:
#sendo o True = 1 e nesse caso tudo esta 0, sendo False, ou seja, tudo ok.

print(df_cardio.isna().sum())

#ANÁLISE EXPLORATÓRIA DE DADOS

#Dados numericos

fig = make_subplots(rows=4, cols=1)
fig.add_trace(go.Box(x=df_cardio["age"]/365, name= "Idade"), row=1, col=1)
fig.add_trace(go.Box(x=df_cardio["weight"], name= "Peso"), row=2, col=1)
fig.add_trace(go.Box(x=df_cardio["ap_hi"], name= "Pressão sanguínea sistólica"), row=3, col=1)
fig.add_trace(go.Box(x=df_cardio["ap_lo"], name= "Pressão sanguínea diastólica"), row=4, col=1)

fig.update_layout(template="plotly_dark", height=700)
'''fig.show()'''

#Dados categóricos

fig = make_subplots(rows=2, cols=3)

fig.add_trace(go.Bar(y=df_cardio["gender"].value_counts(), x=["Feminino", "Masculino"], name="Genero"), row=1, col=1)
fig.add_trace(go.Bar(y=df_cardio["cholesterol"].value_counts(), x=["Normal", "Acima do Normal", "Muito acima do normal"], name="Cholesterol"), row=1, col=2)
fig.add_trace(go.Bar(y=df_cardio["gluc"].value_counts(), x=["Normal", "Acima do Normal", "Muito acima do normal"], name="Glicose"), row=1, col=3)
fig.add_trace(go.Bar(y=df_cardio["smoke"].value_counts(), x=["Não fumante", "Fumante"], name="Fumante"), row=2, col=1)
fig.add_trace(go.Bar(y=df_cardio["alco"].value_counts(), x=["Não Alcoólatra", "Alcoólatra"], name="Alcoólatra"), row=2, col=2)
fig.add_trace(go.Bar(y=df_cardio["active"].value_counts(), x=["Não Ativo", "Ativo"], name="Ativo"), row=2, col=3)

fig.update_layout(template="plotly_dark", height=700)
'''fig.show()'''


#MACHINE LEARNING

#Preparação dos dados

Y = df_cardio["cardio"]
X = df_cardio.loc[:, df_cardio.columns != 'cardio']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

#Treinamento do modelo

#aqui instancio o modelo e passo os parametros
ml_model = RandomForestClassifier(n_estimators=20, n_jobs=4, max_depth=4, )
ml_model.fit(x_train, y_train)

#aqui vamos pegar um paciente qualquer
x_test.iloc[0].to_frame().transpose()

ml_model.predict_proba(x_test.iloc[0].to_frame().transpose())[0][1] * 100

predictions = ml_model.predict(x_test)

#Avaliação do modelo

#aqui na matrix de confusão o modelo aprendeu alguma coisa, pois o f1-score é de 0.73
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

# Feature Importance

#ele vai começar a bagunaçr as variaveis e vai mensurar qual a queda de performance
#que o modelo tem ao bagunçar as variaveis; a variavel que implicar em uma maior queda 
#de performance é a variavel mais importante.

result = permutation_importance(ml_model, x_test, y_test, n_repeats=10, n_jobs=2)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
#esse ".T" é de transposição
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=x_test.columns[sorted_idx])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()

#nesse grafico a variavel que está destacada e separada e que é a que mais impactou no resultado foi a 
#pressão ap_hi.


import shap
explainer = shap.TreeExplainer(ml_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[:, :, 1], X)

#neste gráfico ele explana bem quais são as variáveis que mais impactam no modelo de classificação e só foi possível pois o modelo teve uma acurácia de 0.73 anteriormente
#Quanto mais pra direita, mais a variável impacta no modelo positivamente, enquanto mais para a esquerda seria a relação negativa.
#impactar positivamente neste modelo significa: QUE O RESULTADO SERÁ "1", OU SEJA, A PESSOA TEM PROBLEMA CARDÍACO.
#a partir do gráfico é possível notar que as pressões tem muito impacto positivo no modelo, assim como a idade e o colesterol.
#Portanto, quando a pressão for alta, aumenta a possibilidade de problemas cardíacos; a idade também influencia, quando for alta, aumenta a possíbilidade de problemas
#pressões baixa ou baixa idade não influenciam, mas pelo gráfico, a ausencia de atividade fisica pode influenciar.






