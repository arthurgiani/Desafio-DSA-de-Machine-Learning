
#Carregando pacotes
install.packages("ggplot2")
install.packages("dplyr")
install.packages("caret")
install.packages("readr")
install.packages("binaryLogic")
library(ggplot2)
library(dplyr)
library(caret)
library(readr)
library(binaryLogic)

#Carregando os dados
dataset <- read.csv("dataset.csv")


#Avaliando os dados
str(dataset)


# As etapas abaixo dizem respeito ao pré-processamento e limpeza dos dados

#Verificando valores N.A
colSums(is.na(dataset))

#Nenhum dado NA encontrado

#Verificando quantidade de valores Zero em cada coluna
colSums(dataset == 0)

#Realizando a substituição dos valores zero pela média de cada coluna.
dataset$glicose[dataset$glicose== 0] <- mean(dataset$glicose)
dataset$pressao_sanguinea[dataset$pressao_sanguinea == 0] <- mean(dataset$pressao_sanguinea)
dataset$insulina[dataset$insulina == 0] <- mean(dataset$insulina)
dataset$bmi[dataset$bmi == 0] <- mean(dataset$bmi)
dataset$grossura_pele[dataset$grossura_pele == 0] <- mean(dataset$grossura_pele)


#Transformando a variável classe em fator
dataset$classe <- factor(dataset$classe)
levels(dataset$classe)

ggplot(dataset, aes(x = classe)) + geom_bar(col = 'white', fill = 'yellow') + ggtitle('Proporção de pacientes por classe')


table(dataset$classe)

#No dataset de estudo, há uma porcentagem maior de pessoas registradas que não apresentam diabates (0).
#Isso pode influenciar o modelo preditivo futuro criando uma tendência de classificação de não diabéticos em futuros dados.
#Sendo assim é interessante utilizar alguma técnica para balancear estes dados.


#Balanceamento de dados através do comando SMOTE
install.packages("DMwR")
library(DMwR)
dataset <- SMOTE(classe~., dataset, perc.over = 100)
table(dataset$classe)

ggplot(dataset, aes(x = classe)) + geom_bar(col = 'white', fill = 'yellow') + ggtitle('Proporção de pacientes por classe')

#Análise explorátoria dos dados

summary(dataset)

hist(dataset$num_gestacoes, main = 'Número de gestações', xlab = "Nº Gestações")

hist(dataset$glicose, main = 'Concentração de glicose', xlab = "(mg/DL)")
#A maioria significativa dos pacientes possui os níveis de glicose acima da média (Normal <= 100mg/DL)

hist(dataset$pressao_sanguinea, main = 'Pressão Sanguínea', xlab = 'mm/Hg')
#Pressão sanguinea bem distribuída ao redor da medida central. 

hist(dataset$grossura_pele, main = 'Grossura da pele (Dobra do tríceps)', xlab = 'mm')
#Grossura da pele baixa na maioria dos casos. Mulheres portadoras de diabetes possuem pele mais fina e ressacada do que a maioria das pessoas.

hist(dataset$insulina, main = 'Nível de insulina', xlab = 'mU/dL')
#Valores acima de 230mU/DL indicam a chance de resistência a insulina.

hist(dataset$bmi , main = 'Índice de massa corpórea (BMI)', xlab = 'BMI')
#a maioria da amostra está próxima ou ultrapassou a barreira saudável do BMI (BMI saudável <= 30)

hist(dataset$idade, main = 'Idade', xlab = 'Anos')


hist(dataset$indice_historico, main = 'Índice histórico de diabetes (Pedigree Function)', xlab = 'Coeficiente')
#o estudo foi conduzido num cenário onde a maioria dos pacientes não possui alta tendencia de diabetes
#aumentando a chance da condução de um modelo preditivo não tendencioso.


#Analisando a relação entre as variáveis
library(randomForest)

ImportanciaVariáveis <- randomForest(classe ~., data = dataset, ntree=10, nodezise = 10, importance = TRUE)
varImpPlot(ImportanciaVariáveis)

#Dividindo os datasets

install.packages("caTools")
library(caTools)
set.seed(101) 
amostra <- sample.split(dataset$classe, SplitRatio = 0.75) 
summary(amostra)

# Criando dados de treino - 75% dos dados
dataset_treino = subset(dataset, amostra == TRUE)

# Criando dados de teste - 25% dos dados
dataset_teste = subset(dataset, amostra == FALSE)

#Criando o modelo

modelo1 <- randomForest(classe ~., data=dataset_treino)
modelo1
summary(modelo1)



#Realizando a previsão dos dados de teste

previsao <- predict(modelo1, newdata = dataset_teste)
previsao

comparativo <- data.frame('Real' = dataset_teste$classe, 'Previsto' = previsao)
comparativo

summary(comparativo)

#Matriz de confusão
install.packages('caret')
library (caret)

confusionMatrix(comparativo$Real, comparativo$Previsto)

#Geração da curva de aprendizagem

install.packages("ROCR")
library("ROCR")

# Gerando as classes de dados
class1 <- predict(modelo1, newdata = dataset_teste, type = 'prob')
class2 <- dataset_teste$classe
class1
class2

# Gerando a curva ROC
?prediction
?performance
pred <- prediction(class1[,2], class2) 
pred
perf <- performance(pred, "tpr","fpr") 
plot(perf, col = rainbow(10), xlab = 'Taxa de falsos positivos', ylab = 'Taxa de verdadeiros positivos')

###################################otimizando o algoritmo

#Tentativa de otimização por meio do comando c50

install.packages("C50")
library(C50)

Cost_func <- matrix(c(0, 2, 1, 0), nrow = 2, dimnames = list(c("0", "1"), c("0", "1")))


# Cria o modelo2
set.seed(300)
modelo2  <- C5.0(classe ~ ., data = dataset_treino, trials = 100,cost = Cost_func)
print(modelo2)
summary(modelo2)

previsao2 <- predict(modelo2, newdata = dataset_teste)
previsao2


comparativo2 <- data.frame('Real' = dataset_teste$classe, 'Previsto' = previsao2)
comparativo2

summary(comparativo2)

confusionMatrix(comparativo2$Real, comparativo2$Previsto)


###################################################

