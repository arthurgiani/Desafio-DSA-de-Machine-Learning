# Prevendo a ocorrência de diabetes 

Projeto realizado na competição mensal de Machine Learning da Data Science Academy (DSA)

O conjunto de dados é do Instituto Nacional de Diabetes e Doenças Digestivas e Renais (National Institute of Diabetes and Digestive and Kidney Diseases). O objetivo é prever com base em medidas de diagnóstico se um paciente tem diabetes ou não. Várias restrições foram colocadas na seleção dessas instâncias de um banco de dados maior. Em particular, todos os 600 pacientes analisados aqui são do sexo feminino com pelo menos 21 anos de idade. 

### **Descrição dos Arquivos**

dataset.csv - arquivo contendo os dados necessários para a execução do projeto

### **Descrição dos Campos**

- num_gestacoes - Número de vezes grávida
- glicose - Concentração plasmática de glicose em teste oral de tolerância à glicose
- pressao_sanguinea - Pressão arterial diastólica em mm Hg
- grossura_pele - Espessura da dobra da pele do tríceps em mm
- insulina - Insulina em mu U / ml
- bmi - Índice de massa corporal medido em peso em kg / (altura em m) ^ 2
- indice_historico - Índice de histórico de diabetes (Pedigree Function)
- idade - Idade em anos
- classe - Classe (0 - não desenvolveu a doença / 1 - desenvolveu a doença)

 ![2019-02-04_16-39-30](https://user-images.githubusercontent.com/45671820/52230037-881ba380-289d-11e9-845e-4aaab42a342a.png)
 
 Como desejamos prever em qual classe o paciente está (0 ou 1), a variável foi transformada em fator para que a modelagem preditiva seja feita corretamente. Na parte de limpeza e pré processamento dos dados, nenhum valor NA foi encontrado, e os valores 0 das colunas que constam os dados a respeito de bmi, insulina, pressão sanguínea, grossura de pele e glicose foram substituídos pelo valor da média encontrada em cada uma das colunas.
 
 ![2](https://user-images.githubusercontent.com/45671820/52230486-a504a680-289e-11e9-8e3e-1deebccf3e9c.png)
 
 Como pode ser visto no gráfico abaixo, há uma porcentagem maior de pessoas registradas que não apresentam diabates (0). Isso pode influenciar o modelo preditivo futuro criando uma tendência de classificação de não diabéticos nos dados de teste. Sendo assim, é interessante utilizar alguma técnica para balancear estes dados para que a modelagem seja a mais precisa possível.
 
 ![3](https://user-images.githubusercontent.com/45671820/52231546-66242000-28a1-11e9-8604-8824b994483f.png)
 
 Para isso, foi utilizado a técnica SMOTE que realizou um oversampling igualando o números de pacientes de classe 0 e 1 (416 pacientes para cada tipo de classe)
 
 ![4](https://user-images.githubusercontent.com/45671820/52231703-c87d2080-28a1-11e9-9476-d7dd5117be67.png)
 
 Após essa etapa preliminar de limpeza e pré-processamento dos dados é necessário realizar a análise exploratória dos dados.
 
 ### **Análise Exploratória**
 
 A seguir, os histogramas de cada variável do estudo serão apresentados com o objetivo de obter uma melhor compreensão de como os dados estão distrbuídos e qual a relação destas variáveis com a provável ocorrência ou não de diabetes
 
 ![5](https://user-images.githubusercontent.com/45671820/52232152-e8f9aa80-28a2-11e9-842f-9a312afc21a9.png)

 
![6](https://user-images.githubusercontent.com/45671820/52232153-e9924100-28a2-11e9-97d3-dbe56d1baa47.png)
A maioria significativa dos pacientes possui os níveis de glicose acima da média (Normal <= 100mg/DL)


![8](https://user-images.githubusercontent.com/45671820/52232154-ea2ad780-28a2-11e9-8af0-acdda968d318.png)
Pressão sanguinea bem distribuída ao redor da medida central.

![9](https://user-images.githubusercontent.com/45671820/52232155-ea2ad780-28a2-11e9-83fb-f25d137b5ad2.png)
Grossura da pele baixa na maioria dos casos. Mulheres portadoras de diabetes possuem pele mais fina e ressecada do que a maioria das pessoas.

![10](https://user-images.githubusercontent.com/45671820/52232156-ea2ad780-28a2-11e9-9aa4-97c046f86f7c.png)
Valores acima de 230mU/DL indicam a chance de resistência a insulina.

![11](https://user-images.githubusercontent.com/45671820/52232157-ea2ad780-28a2-11e9-94fc-ed947a216669.png)
A maioria das pacientes analisadas está próxima de ultrapassar ou ultrapassou a barreira saudável do BMI (BMI saudável <= 30)

![12](https://user-images.githubusercontent.com/45671820/52232158-eac36e00-28a2-11e9-8028-73362bc25421.png)


![13](https://user-images.githubusercontent.com/45671820/52232159-eac36e00-28a2-11e9-8043-28ae7a596ba5.png)


### **Importância das variáveis para o modelo preditivo**

No plot abaixo mostra de forma visual o grau de importância de cada variável para a criação de um modelo preditivo consistente.

![14](https://user-images.githubusercontent.com/45671820/52232983-121b3a80-28a5-11e9-8458-8d78dc9209dd.png)
Como destaque podemos elencar as variáveis "glicose" e "bmi", ou seja, o nível de insulina detectado no paciente e a sua relação peso/altura são de extrema importância para determinar a ocorrência de diabetes.



### **Divisão do dataset**

Utilizando a função sample.split, o dataset foi dividido na proporção de 75% para os dados de treino do modelo e 25% para os dados de teste.

![15](https://user-images.githubusercontent.com/45671820/52233124-6c1c0000-28a5-11e9-9099-cb78ec1d9ddc.png)

### **Treinamento do modelo**

Considerando todas as variáveis independentes como importantes para a determinação da classe do paciente (diabético ou não), o modelo preditivo foi feito utilizando o algoritmo Random Forest, que gera múltiplas árvores de decisão que depois são utilizadas na classificação de novos objetos com a maior acurácia possível. Esse algoritmo pode ser usado tanto em tarefas de classificação quanto de regressão.

O modelo treinado com os 75% do dataset original obteve uma taxa média de erro de 13,46%, caracterizando assim uma acurácia de 86,54%.

![16](https://user-images.githubusercontent.com/45671820/52235908-87d6d480-28ac-11e9-853e-139bfc40fa3e.png)


### **Teste do Modelo**

Com o modelo treinado chegou a hora de testar a sua acurácia com os dados de teste (25% do dataset original).

![17](https://user-images.githubusercontent.com/45671820/52236085-06cc0d00-28ad-11e9-84e8-1c9724b9819a.png)

O modelo treinado foi capaz de prever a ocorrência de diabetes nos pacientes com 90,87% de acurácia, um resultado extremamente satisfatório.

### **Curva de Aprendizagem**

A imagem abaixo é a curva de aprendizagem do algoritmo Random Forest nesta modelagem. Como pode ser visto, a grande área abaixo da curva mostra que o algoritmo atingiu uma alta taxa de verdadeiros positivos. Quanto maior a área abaixo da curva, mais preciso é o seu modelo.

![18](https://user-images.githubusercontent.com/45671820/52236764-dedda900-28ae-11e9-8ad5-9ba1aab154e4.png)

O cientista de dados deve sempre fazer esforços para aproximar a curva do eixo y = 1, de forma que o algoritmo consiga realizar suas previsões de forma cada vez mais precisa. Para isso, ténicas de otimização podem ser utilizadas.

### **Otimização do modelo**

Neste estudo será utilizado o algoritmo C50 para a otimização do modelo preditivo. O pacote C50 permite que se construa uma matriz denominada Cost Funcion, que nada mais é do que uma ferramenta que da pesos diferentes para cada tipo de ocorrência possível na matriz de confusão gerada pelo modelo de classificação. As ocorrências são:

- Verdadeiro Positivo = O modelo apontou a ausência de diabetes de forma assertiva.
- Falso Positivo = O modelo apontou ausência e está errado, ou seja, o paciente realmente tem diabetes.
- Falso negativo = O modelo apontou a presença de diabetes mas na verdade o paciente não possui a doença.
- Verdadeiro negativo = O modelo apontou presença de diabetes de forma assertiva.

Especificamente neste exercício a ocorrência que pode apresentar alguma consequência grave para o paciente é a de falso positivo. Sendo assim, essa ocorrência apresentará peso 2 para a construção do modelo de forma que o mesmo apresente um resultado ponderado.

![19](https://user-images.githubusercontent.com/45671820/52238358-ee5ef100-28b2-11e9-8488-9bee99c1d826.png)

![20](https://user-images.githubusercontent.com/45671820/52238412-0cc4ec80-28b3-11e9-8f18-420a0679c09b.png)

Fazendo uso do pacote C50 foi possível prever os resultads dos dados de teste com 92,31% de acurácia contra os 90,87% da primeira modelagem.

### **Conclusão**

Um das importantes tarefas do cientista de dados é analisar o problema sob a ótica do negócio no qual o problema se encontra. Se tratando de um projeto de Data Science aplicado a área da saúde, é de extrema importância que se atinja um nível de acurácia altíssimo no modelo preditivo pois a não identificação de diabates em um paciente pode ser extremamente danoso, podendo até mesmo levar a morte do mesmo devido ao não tratamento da doença.

Sendo assim o projeto foi capaz de realizar a previsão da ocorrência de diabates com um nível de precisão de 92,31% através do algoritmo Random Forest, considerado então um modelo consistente e útil para o problema de negócio apresentado. 

Certamente sempre há espaço para a melhora nos algoritmos com o objetivo de aumentar sua acurácia, podendo ser utilizado nesse estudo recursos como: teste do modelo sob a perspectiva de diferentes algoritmos (Árvores de Decisão, Regressão logística etc), a criação de novas variáveis no dataset e até a obtenção de novos pacientes com o objetivo de aumentar a amostra do estudo.








 
 
 
 

 
 

