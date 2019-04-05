# Sales Forecasting in a Pharmacy Group Store


In this folder you will find the jupyter notebook I submitted on kaggle.

Kernel Challenge Link: https://www.kaggle.com/arthurgiani/sales-forecasting

Considerations: The original dataset has 1 million observations in it composition. Using a Notebook with 8GB DDR3 Ram Memory,
an 3º Generation Core i5 and no dedicated GPU, the use of strategies like Deep Learning, XGBoost or advanced handling in algorithm parameters,
was totally unviable. 

Therefore, RandomForest was applied to predict Sales Amount causing a possible overfitting due the restrictions above.

Even so, the challenge submission shows an RMSPE (Root Mean Square Percentage Error) = 0.39556. This results represented a 14th place in the competiton among 64 other competitors.
